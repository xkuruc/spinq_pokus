"""Incremental, honest output for the eight-module local benchmark."""

from __future__ import annotations

import csv
import json
import os
import zipfile
from pathlib import Path

import numpy as np

from spinq_audit.common import atomic_bytes, atomic_json, utc_now

MODULES = "ABCDEFGH"
STATUSES = {"PENDING", "SUCCESS_VALIDATED", "VALID_NEGATIVE_RESULT", "METHOD_FAILED",
            "DEPENDENCY_FAILED", "BUDGET_EXHAUSTED", "UNVERIFIED_TIMING",
            "REFERENCE_INADEQUATE", "RUNNING", "STOPPED_UNCERTAIN"}
FIELDS = ("module", "method", "baseline", "task", "block", "data_source", "acquisitions",
          "wall_seconds", "analysis_seconds", "design_seconds", "rf_duration_us",
          "error", "ci_low", "ci_high", "tolerance", "status", "reason")


class Results:
    def __init__(self, out: Path, config: dict, preflight: dict):
        self.out = out
        for folder in ("raw", "vendor_reference", "models", "plots"):
            (out/folder).mkdir(parents=True,exist_ok=True)
        target = out/"results.json"
        if target.is_file():
            self.data = json.loads(target.read_text(encoding="utf-8"))
            if self.data["config"] != config:
                raise ValueError("Resume configuration changed")
        else:
            self.data = {"started_utc":utc_now(),"state":"RUNNING","config":config,
                "preflight":preflight,"modules":{m:{"status":"PENDING"} for m in MODULES},
                "rows":[],"errors":[],"pilot":{},"frozen_plan":{},
                "budgets":{"acquisitions_used":0,"rf_duration_us_used":0.0},
                "raw_source":"exported complex FID; raw ADC unverified",
                "server_processing_offload_unverified":True,
                "hardware_results_present":False,"upload":{"status":"NOT_ATTEMPTED"}}
        self.save()

    def save(self):
        atomic_json(self.out/"results.json",self.data)
        temporary=self.out/"comparison.csv.tmp"
        with temporary.open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.data["rows"])
        os.replace(temporary,self.out/"comparison.csv")
        atomic_bytes(self.out/"REPORT.md",self.markdown().encode("utf-8"))

    def module(self,letter:str,status:str,reason:str="",**details):
        if letter not in MODULES or status not in STATUSES: raise ValueError("Invalid module state")
        self.data["modules"][letter]={"status":status,"reason":reason,**details}
        if status not in {"PENDING","RUNNING"}:
            for row in self.data["rows"]:
                if row["module"]==letter and row["status"]=="RUNNING":
                    row["status"]=status
                    if not row["reason"]:row["reason"]=reason
        self.save()
        if status!="RUNNING":
            print(f"MODULE {letter}: {status}: {reason}",flush=True)

    def row(self,**values):
        item={key:values.get(key) for key in FIELDS}
        identity=(item["module"],item["method"],item["task"],item["block"])
        self.data["rows"]=[r for r in self.data["rows"] if
            (r["module"],r["method"],r["task"],r["block"])!=identity]
        self.data["rows"].append(item)
        self.save()
        print(f"ROW {item['module']} block={item['block']} method={item['method']} "
              f"task={item['task']} acquisitions={item['acquisitions']} "
              f"error={item['error']} tolerance={item['tolerance']} "
              f"status={item['status']}",flush=True)

    def markdown(self):
        d=self.data
        lines=["# SpinQ Gemini Lab — lokálny low-level benchmark", "",
               f"Stav: **{d['state']}**; skutočné hardvérové údaje: **{'áno' if d['hardware_results_present'] else 'nie'}**.",
               "Všetky FFT, fity, modely a porovnania v tomto behu počíta lokálny Windows proces.",
               "Prijímaný FID môže byť predspracovaný serverom; surový ADC nebol potvrdený.",
               "Tri pilotné bloky nie sú silný dôkaz zlepšenia. Neznáme interné opakovania sú UNKNOWN.",
               "", "## Moduly", ""]
        for module in MODULES:
            item=d["modules"][module]
            lines.append(f"- **{module}**: {item['status']} — {item.get('reason','')}")
        if not d["rows"]:
            lines.extend(["", "Zatiaľ nevzniklo žiadne porovnanie. Ak hardvérová úloha skončila, "
                          "jej pôvodný dekódovaný výstup je v `data/<kľúč>.json` "
                          "a udalosti v `data/events.jsonl.gz`; opraviteľný beh pokračuje "
                          "s `--resume` bez opakovania dokončenej úlohy."])
        lines.extend(["", "## Porovnanie", "",
                      "| Modul | Metóda | Referencia | Úloha | Blok | Akvizície | Čas (s) | Chyba | CI | Tolerancia | Stav |",
                      "|---|---|---|---|---:|---:|---:|---:|---|---:|---|"])
        def fmt(v):
            if v is None: return "—"
            if isinstance(v,float): return f"{v:.5g}"
            return str(v).replace("|","/").replace("\n"," ")
        for r in d["rows"]:
            cells=("module","method","baseline","task","block","acquisitions",
                   "wall_seconds","error")
            lines.append("| " + " | ".join(fmt(r[k]) for k in cells) +
                         f" | [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | {fmt(r['tolerance'])} | {fmt(r['status'])} |")
        lines.extend(["", "## Plán a limity", "",
            "Rozpočty sú softvérová výskumná obálka odvodená z doteraz dokončených H meraní, "
            "nie certifikované hranice výrobcu. Neznáme J, časovanie P a riadenie tretieho spinu "
            "neboli dosadené nulou ani vyhlásené za overené.",
            "Echo T2 sa odlišuje od T2* z voľného FID. Simulovaná unitárna zhoda nie je "
            "experimentálna fidelita brány. Vendor grafy sú len v vendor_reference/.",
            "", "```json",json.dumps(d.get("frozen_plan",{}),ensure_ascii=False,indent=2),"```",""])
        if d["errors"]:
            lines.extend(["## Chyby",""]+[f"- {e}" for e in d["errors"]]+[""])
        if d.get("recovered_errors"):
            lines.extend(["## Opravené chyby pri pokračovaní",""]+
                         [f"- {e}" for e in d["recovered_errors"]]+[""])
        return "\n".join(lines)


def plots_from_results(out:Path,data:dict):
    """Rebuild plots from saved rows; never contacts hardware."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir=out/"plots";plot_dir.mkdir(exist_ok=True)
    for old in plot_dir.glob("*_comparison.png"): old.unlink()
    for module in MODULES:
        rows=[r for r in data["rows"] if r["module"]==module and
              isinstance(r.get("error"),(float,int)) and isinstance(r.get("acquisitions"),(float,int))]
        if not rows: continue
        fig,ax=plt.subplots(figsize=(7,4))
        for method in sorted({r["method"] for r in rows}):
            own=sorted((r for r in rows if r["method"]==method),key=lambda r:(r["acquisitions"],r["block"] or 0))
            ax.plot([r["acquisitions"] for r in own],[r["error"] for r in own],"o-",label=method)
        ax.set(xlabel="Fyzické akvizície",ylabel="Nezávislá chyba (jednotka podľa úlohy)",title=f"Modul {module}")
        ax.legend(fontsize=7);fig.tight_layout();fig.savefig(plot_dir/f"{module}_comparison.png",dpi=140);plt.close(fig)
    # Separate time and drift views use entire physical blocks, never the
    # thousands of correlated FID points as statistical repetitions.
    for module in MODULES:
        rows=[r for r in data["rows"] if r["module"]==module and
              isinstance(r.get("error"),(float,int)) and
              isinstance(r.get("wall_seconds"),(float,int))]
        if not rows:continue
        fig,ax=plt.subplots(figsize=(7,4))
        for method in sorted({r["method"] for r in rows}):
            own=sorted((r for r in rows if r["method"]==method),
                       key=lambda r:sum(x["wall_seconds"] for x in rows
                                        if x["method"]==method and str(x["block"])<=str(r["block"])))
            elapsed=np.cumsum([r["wall_seconds"] for r in own])
            ax.plot(elapsed,[r["error"] for r in own],"o-",label=method)
        ax.set(xlabel="Kumulatívny fyzický wall čas (s)",ylabel="Nezávislá chyba",
               title=f"Modul {module}: chyba vs. čas")
        ax.legend(fontsize=7);fig.tight_layout()
        fig.savefig(plot_dir/f"{module}_time_comparison.png",dpi=140);plt.close(fig)
    for module in ("C","E","G"):
        source=out/"models"/f"{module}_physical.json"
        if not source.is_file():continue
        physical=json.loads(source.read_text(encoding="utf-8"))
        rows=physical.get("rows",[])
        if not rows:continue
        fig,ax=plt.subplots(figsize=(7,4))
        methods=sorted({r["method"] for r in rows})
        for method in methods:
            own=[r for r in rows if r["method"]==method]
            ax.plot([r["block"] for r in own],[r["absolute_error"] for r in own],
                    "o-",label=method)
        ax.set(xlabel="Nezávislý merací blok",ylabel="Komplexná FID odchýlka",
               title=f"{module}: robustnosť medzi blokmi")
        ax.legend(fontsize=7);fig.tight_layout()
        fig.savefig(plot_dir/f"{module}_robustness.png",dpi=140);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,3))
        by_block={}
        for row in rows:by_block[row["block"]]=max(by_block.get(row["block"],0),
                                                    row["reference_drift"])
        ax.plot(sorted(by_block),[by_block[k] for k in sorted(by_block)],"o-")
        ax.set(xlabel="Merací blok",ylabel="Návratová FID referencia",
               title=f"{module}: drift referencie")
        fig.tight_layout();fig.savefig(plot_dir/f"{module}_drift.png",dpi=140);plt.close(fig)
    source=out/"models"/"vendor_fft_replica.json"
    if source.is_file():
        vendor=json.loads(source.read_text(encoding="utf-8"))
        rows=vendor.get("heldout_rows",[])
        if rows:
            fig,ax=plt.subplots(figsize=(7,3))
            x=np.arange(len(rows))
            ax.bar(x-.2,[r["relative_replica_error"] for r in rows],.4,label="zmrazená replika")
            ax.bar(x+.2,[r["relative_plain_fft_error_after_frozen_scale"] for r in rows],
                   .4,label="obyčajná FFT")
            ax.set_xticks(x,[r["task"] for r in rows],rotation=30,ha="right")
            ax.set(ylabel="Relatívne reziduum spektra",title="Vendor FFT: odložené FID")
            ax.legend(fontsize=7);fig.tight_layout()
            fig.savefig(plot_dir/"vendor_fft_residuals.png",dpi=140);plt.close(fig)
    source=out/"models"/"F_analysis.json"
    if source.is_file():
        denoise=json.loads(source.read_text(encoding="utf-8"))
        rows=denoise.get("test_blocks",[])
        if rows:
            fig,ax=plt.subplots(figsize=(8,4))
            for method in sorted({r["method"] for r in rows}):
                own=[r for r in rows if r["method"]==method]
                ax.plot(np.arange(len(own)),[r["complex_fid_rmse"] for r in own],
                        "o-",label=method)
            ax.set(xlabel="Odložený blok",ylabel="Komplexná FID RMSE",
                   title="F: denoising na reálnych opakovaniach")
            ax.legend(fontsize=7);fig.tight_layout()
            fig.savefig(plot_dir/"F_denoising.png",dpi=140);plt.close(fig)


def bundle_complete(out:Path) -> Path:
    """Full local delivery, including original exported FID and vendor reference."""
    temporary=out/"results.zip.tmp"
    with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
        for path in sorted(out.rglob("*")):
            if not path.is_file() or path.name in {"results.zip","results.zip.tmp","events.jsonl"} or path.name.endswith(".tmp"):
                continue
            # On a validated task, raw NPZ and events retain the arrays. If
            # validation failed, the decoded SDK task JSON is the only simple
            # browsable copy of that measurement and must remain in the ZIP.
            if path.parent == out/"data" and path.name.endswith(".json") and path.name not in {
                "hardware_journal.json","initial_telemetry.json","recorder_status.json"} and not (
                path.name.endswith(".error.json") or path.name.endswith(".payload_mismatch.json")):
                key=path.stem
                if (out/"raw"/f"{key}.json").is_file() and (out/"raw"/f"{key}.npz").is_file():
                    continue
            archive.write(path,path.relative_to(out).as_posix())
    os.replace(temporary,out/"results.zip")
    return out/"results.zip"
