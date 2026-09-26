"""Incremental machine-readable results, human report and final archive."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import zipfile
from pathlib import Path

from spinq_audit.common import atomic_bytes, atomic_json, utc_now


CSV_FIELDS=("topic","method","block","phase","measurements","internal_repetitions",
            "requested_samples","wall_seconds","compute_seconds","error","uncertainty",
            "improvement_vs_baseline","status","reason")


class Results:
    def __init__(self,out:Path, config:dict):
        self.out=out
        out.mkdir(parents=True,exist_ok=True)
        (out/"data").mkdir(exist_ok=True);(out/"models").mkdir(exist_ok=True)
        (out/"plots").mkdir(exist_ok=True)
        target=out/"results.json"
        self.data=json.loads(target.read_text(encoding="utf-8")) if target.exists() else {
            "started_utc":utc_now(),"state":"running","config":config,"topics":{},"rows":[],
            "errors":[],"hardware_results_present":False,"study_phase":"pilot",
            "raw_adc_confirmed":False,"upload":{"status":"NOT_ATTEMPTED"}}
        if self.data["config"]!=config:
            # Only block count may increase when continuing a frozen plan.
            old=self.data["config"].copy();new=config.copy()
            old.pop("blocks",None);new.pop("blocks",None)
            if old!=new or config["blocks"]<self.data["config"]["blocks"]:
                raise ValueError("Resume configuration changed; only blocks may increase")
            self.data["config"]["blocks"]=config["blocks"]
        self.save()

    def save(self):
        atomic_json(self.out/"results.json",self.data)
        target=self.out/"comparison.csv.tmp"
        with target.open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=CSV_FIELDS,extrasaction="ignore")
            writer.writeheader();writer.writerows(self.data["rows"])
        os.replace(target,self.out/"comparison.csv")
        atomic_bytes(self.out/"REPORT.md",report(self.data).encode("utf-8"))

    def row(self,**values):
        row={k:values.get(k) for k in CSV_FIELDS}
        identity=(row["topic"],row["method"],row["block"],row["phase"])
        self.data["rows"]=[old for old in self.data["rows"] if
                           (old["topic"],old["method"],old["block"],old["phase"])!=identity]
        self.data["rows"].append(row)
        self.save()

    def topic(self,name,block,value):
        self.data["topics"].setdefault(name,{})[str(block)]=value
        self.save()


def aggregate_rows(rows, frozen_plan=None):
    """Experiment blocks, not FID samples, are the statistical units."""
    from scipy.stats import t as student_t
    groups={}
    for row in rows:
        if isinstance(row.get("error"),(int,float)) and math.isfinite(row["error"]):
            groups.setdefault((row["topic"],row["method"]),[]).append(row)
    targets={"calibration":(frozen_plan or {}).get("calibration_target_joint_error"),
             "acquisition":(frozen_plan or {}).get("acquisition_target_se_hz"),
             "pulse_tuning":(frozen_plan or {}).get("pulse_target_held_out_error"),
             "robust_pulse":(frozen_plan or {}).get("robust_target_error"),
             "denoising":(frozen_plan or {}).get("denoising_target_combined_error")}
    result=[]
    for (topic,method),items in sorted(groups.items()):
        errors=[float(x["error"]) for x in items]
        n=len(errors);mean=statistics.mean(errors)
        sd=statistics.stdev(errors) if n>1 else None
        ci=(student_t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
        target=targets.get(topic)
        successes=sum(e<=target for e in errors) if target is not None else None
        improvements=[float(x["improvement_vs_baseline"]) for x in items
                      if isinstance(x.get("improvement_vs_baseline"),(int,float))]
        measurement_costs=[float(x["measurements"]) for x in items if x.get("measurements") is not None]
        wall_costs=[float(x["wall_seconds"]) for x in items if x.get("wall_seconds") is not None]
        result.append({"topic":topic,"method":method,"blocks":n,"mean_error":mean,
                       "sd_between_blocks":sd,"mean_error_95pct_ci":[mean-ci,mean+ci] if ci is not None else None,
                       "mean_improvement_vs_baseline":statistics.mean(improvements) if improvements else None,
                       "frozen_success_target":target,"successes":successes,
                       "success_rate":successes/n if successes is not None else None,
                       "mean_measurements":statistics.mean(measurement_costs) if measurement_costs else None,
                       "mean_wall_seconds":statistics.mean(wall_costs) if wall_costs else None,
                       "status":"PILOT_NEEDS_REPLICATION" if n>=3 else "NEPRESVEDČIVÉ"})
    return result


def report(data):
    lines=["# SpinQ Gemini Lab — experimentálny benchmark", "",
           f"Začiatok: {data['started_utc']}; stav: **{data['state']}**; režim: **{data['study_phase']}**.",
           f"Skutočné hardvérové údaje: {'áno' if data['hardware_results_present'] else 'nie'}.",
           f"Upload: {data.get('upload',{}).get('status','NOT_ATTEMPTED')}.", "",
           "Výstupy `data/` sú dekódované grafy SpinQLabLink, nie potvrdený RAW ADC. "
           "Interný počet opakovaní a skryté prípravné RF pulzy SDK neoznamuje.",
           "Kladné zlepšenie znamená nižšiu chybu pri rovnakom definovanom rozpočte. "
           "Tri bloky sú pilot; intervaly medzi blokmi ostávajú široké.", "",
           "| Téma | Metóda | Blok | Merania | Čas (s) | Chyba | Neistota | Zlepšenie | Stav |",
           "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in data.get("rows",[]):
        def fmt(x):
            return "—" if x is None else f"{x:.4g}" if isinstance(x,float) and math.isfinite(x) else str(x)
        lines.append("| "+" | ".join(fmt(r.get(k)) for k in
            ("topic","method","block","measurements","wall_seconds","error","uncertainty",
             "improvement_vs_baseline","status"))+" |")
    lines.extend(["","## Súhrn po nezávislých blokoch", ""])
    for item in data.get("aggregate",[]):
        lines.append(f"- {item['topic']} / {item['method']}: {item['blocks']} bloky; "
                     f"priemerná chyba {item['mean_error']:.4g}; 95 % interval "
                     f"{item['mean_error_95pct_ci']}; úspechy {item['successes']}/{item['blocks']} "
                     f"pri vopred stanovenej hranici {item['frozen_success_target']}; {item['status']}.")
    lines.extend(["","## Metodika a obmedzenia", "",
        "Pri porovnaní kalibrácie sa spoločné pilotné merania započítajú každej metóde ako cena prvého použitia. "
        "Cieľové referencie pochádzajú z oddelených meraní a nesmú byť vstupom hodnotenej metódy.",
        "Zložky FID sa sledujú komplexným fitom a vyhodnocujú po blokoch. FID vzorky sa nerátajú ako nezávislé pokusy.",
        "Rotačná chyba z FID je iba experimentálny proxy merateľnej odozvy, nie procesná fidelita.",
        "Optimalizovaný GRAPE pulz používa lokálny jednojadrový model H; simulovaná fidelita je oddelená od merania. "
        "Nezmeraná väzba H–P je významné obmedzenie modelu.",
        "Voľný FID poskytuje T2*, nie echo T2. SDK 1.0.2 neodhaľuje jednotlivo nastaviteľné echo časy; "
        "adaptívne echo sa preto bez overenej podpory nevykoná.",
        "Hankel a Noise2Noise sa učia z nezávislých opakovaní; referenčný priemer nie je bezšumová pravda.",
        "Nie je preukázané, že serverová FFT sa dá vypnúť, takže jej prenosový čas môže zostať.",
        "", "## Témy", ""])
    for name,blocks in data.get("topics",{}).items():
        lines.extend([f"### {name}",""])
        for block,item in blocks.items():
            lines.append(f"- Blok {block}: {item.get('status','NEPRESVEDČIVÉ')}; {item.get('reason','')}" )
        lines.append("")
    if data.get("errors"):
        lines.extend(["## Chyby",""]+[f"- {e}" for e in data["errors"]]+[""])
    return "\n".join(lines)+"\n"


def plot_comparisons(out:Path,rows:list[dict],topics:dict|None=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for topic in sorted(set(r["topic"] for r in rows)):
            sub=[r for r in rows if r["topic"]==topic and isinstance(r.get("error"),(int,float))]
            if not sub:continue
            fig,ax=plt.subplots(figsize=(7,4))
            for method in sorted(set(r["method"] for r in sub)):
                points=sorted((r for r in sub if r["method"]==method),key=lambda r:(r["measurements"] or 0,r["block"] or 0))
                ax.plot([r["measurements"] for r in points],[r["error"] for r in points],"o-",label=method)
            ax.set(xlabel="Fyzické akvizície",ylabel="Chyba (podľa témy)",title=topic)
            ax.legend(fontsize=7);fig.tight_layout()
            fig.savefig(out/"plots"/(topic+"_error.png"),dpi=140)
            plt.close(fig)
            timed=[r for r in sub if isinstance(r.get("wall_seconds"),(int,float))]
            if timed:
                fig,ax=plt.subplots(figsize=(7,4))
                for method in sorted(set(r["method"] for r in timed)):
                    points=sorted((r for r in timed if r["method"]==method),key=lambda r:r["wall_seconds"])
                    ax.plot([r["wall_seconds"] for r in points],[r["error"] for r in points],"o-",label=method)
                ax.set(xlabel="Skutočný čas experimentov (s)",ylabel="Chyba (podľa témy)",title=topic)
                ax.legend(fontsize=7);fig.tight_layout()
                fig.savefig(out/"plots"/(topic+"_time.png"),dpi=140);plt.close(fig)
        topics=topics or {}
        robust=topics.get("robust_pulse",{})
        pulse=topics.get("pulse_tuning",{})
        if pulse:
            fig,ax=plt.subplots(figsize=(7,4))
            for method in sorted({m for block in pulse.values() for m in block.get("methods",{})}):
                histories=[block.get("methods",{}).get(method,{}).get("evaluations",[])
                           for block in pulse.values()]
                histories=[h for h in histories if h]
                if histories:
                    count=min(len(h) for h in histories)
                    ax.plot(range(1,count+1),[sum(min(e["loss"] for e in h[:k]) for h in histories)/len(histories)
                                                  for k in range(1,count+1)],"o-",label=method)
            ax.set(xlabel="Kandidáti meraní (2 akvizície na kandidáta)",ylabel="Najnižšia pozorovaná strata",
                   title="Konvergencia RF doladenia")
            ax.legend();fig.tight_layout();fig.savefig(out/"plots"/"pulse_convergence.png",dpi=140);plt.close(fig)
        if robust:
            fig,ax=plt.subplots(figsize=(7,4))
            for method in sorted({m for block in robust.values() for m in block.get("readings",{})}):
                pairs=[]
                for block in robust.values():
                    for r in block.get("readings",{}).get(method,[]):
                        pairs.append((r["detuning_hz"],r["error"]))
                if pairs:
                    xs=sorted(set(x for x,_ in pairs))
                    ax.plot(xs,[sum(y for x,y in pairs if x==v)/sum(x==v for x,_ in pairs)
                                for v in xs],"o-",label=method)
            ax.set(xlabel="Vysielacie rozladenie (Hz)",ylabel="Relatívna chyba komplexnej odozvy",
                   title="Robustnosť H pulzov")
            ax.legend();fig.tight_layout();fig.savefig(out/"plots"/"robustness.png",dpi=140);plt.close(fig)
        denoise_blocks=[b.get("metrics",{}) for b in topics.get("denoising",{}).values() if b.get("metrics")]
        if denoise_blocks:
            fig,ax=plt.subplots(figsize=(8,4))
            names=list(denoise_blocks[0])
            ax.bar(names,[sum(b[n]["combined_error"] for b in denoise_blocks)/len(denoise_blocks) for n in names])
            ax.set(ylabel="Kombinovaná chyba voči nezávislému priemeru",
                   title="Denoising: nezávislé testovacie rodiny")
            ax.tick_params(axis="x",labelrotation=25)
            fig.tight_layout();fig.savefig(out/"plots"/"denoising.png",dpi=140);plt.close(fig)
    except Exception:
        # Plot failure never destroys numerical results.
        pass


def bundle(out:Path):
    tmp=out/"results.zip.tmp"
    with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name not in ("results.zip","results.zip.tmp","events.jsonl") and not p.name.endswith(".tmp"):
                z.write(p,p.relative_to(out).as_posix())
    os.replace(tmp,out/"results.zip")
