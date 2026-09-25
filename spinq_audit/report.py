"""Self-contained offline report and shareable bundle."""

from __future__ import annotations

import json
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from .common import atomic_bytes, atomic_json, escaped, utc_now

QUESTIONS = """# Otázky pre SpinQ

1. Poskytuje server pre Gemini Lab výstup FID pred výpočtom FFT/fitov alebo overený FID-only režim? Ak áno, presný parameter a verzia servera?
2. Sú fidRe/fidIm pred alebo po downconversion, filtrácii, decimácii, priemerovaní, phase cyclingu a normalizácii? Jednotky osi a amplitúdy?
3. Aká je skutočná jednotka `sampleFre` a `relaxation_time` v payload-e fyzikálnej vrstvy? Ako sa správa `step=10000` pri zmene atribútu?
4. Koľko interných akvizícií a prípravných/meracích RF pulzov obsahuje jeden custom physical task pri `makePps=true`?
5. Existuje read-only údaj o firmware/server verzii, hardvérovom sériovom čísle, rade úloh a ich vlastníctve?
6. Ako sa synchronizujú H/P pulzy, ak SDK rozdelí zoznam na `hPulse` a `pPulse`? Aké sú limity segmentov a časové kvantovanie?
7. Má API potvrdený raw ADC/IQ stream, nastavenie prijímacieho zisku/AGC, receiver dead time a online callback počas sekvencie?
8. Ktoré hodnoty v `pulseParam`, `ppsParam`, `sampleParam`, `shimmingParam` a `lockParam` sú aktuálne merania, cache alebo uložená kalibrácia?
9. Aký je dokumentovaný bezpečný abort iba vlastnej úlohy a stav po strate spojenia?
10. Ktoré operácie zapisujú trvalú kalibráciu/shim a ako funguje readback/rollback? Audit ich zámerne nevolá.
"""

COMPUTE_TABLE = [
    ("Pulzy, RF časovanie, fyzická akvizícia", "zariadenie/server", "nedá sa nahradiť časovaním Pythonu"),
    ("Chart FID a FFT doručené SDK", "server -> notebook", "interný pôvod/predspracovanie neznáme"),
    ("FFT, validácia osi, charakteristiky FID", "notebook", "implementované z prijatých chart bodov"),
    ("Interná FFT/fit v SpinQ", "server alebo zariadenie", "zdokumentovaný vypínač neoverený"),
    ("Frekvenčný/pulzový návrh a výber kandidáta", "notebook", "budúca slučka iba cez schválený executor"),
]

READINESS = [
    ("Frekvenčná kalibrácia", "FID a telemetria frekvencií", "jednotky/predspracovanie, drift", "overiť baseline a fázu"),
    ("Pulzová kalibrácia", "Rabi amplitúdy, FID", "prípravné pulzy, RF záťaž", "schválené malé varianty"),
    ("Tvarované pulzy", "SDK typ ShapePulse", "RF výstup nie je meraný", "vendor limity segmentov"),
    ("Monitoring driftu", "lock/teplota/frekvencie", "cache a čerstvosť", "pasívny dlhší zber"),
    ("Denoising", "komplexný chart FID", "bez čistého ground truth", "relácie pre train/test; merať chybu fázy/frekvencie/šírky"),
    ("Shimovanie", "readback skupina shimmingParam", "trvalý účinok neznámy", "oddelené povolenie a rollback"),
    ("Koordinácia agentov", "lokálny plán a executor", "vlastníctvo fronty neoverené", "jediný writer a nezávislá safety kontrola"),
]


def make_summary(mode: str, env: dict[str, Any], capabilities: list[dict[str, Any]],
                 issues: list[dict[str, Any]], analysis: dict[str, Any],
                 blockers: list[str], tests: dict[str, Any] | None = None) -> dict[str, Any]:
    state_count = dict(Counter(item["status"] for item in capabilities))
    tests = tests or {"passed": 0, "failed": 0, "skipped": 0, "not_run": True}
    return {"mode": mode, "created_utc": utc_now(),
            "sdk_origin": env["spinqlablink"]["origin"],
            "sdk_version": env["spinqlablink"]["version"],
            "live_connection_performed": bool(analysis.get("connection_performed", False)),
            "hardware_experiments_executed": analysis.get("executed_experiments", 0),
            "hardware_experiments_attempted": analysis.get("attempted_experiments", 0),
            "hardware_experiments_planned": analysis.get("planned_experiments", 0),
            "hardware_experiments_not_performed": max(0, analysis.get("planned_experiments", 0) - analysis.get("executed_experiments", 0)),
            "charts_observed": analysis.get("chart_count", 0),
            "fid_pairs": sum(row.get("status") == "paired" for row in analysis.get("fid_pairs", [])),
            "raw_adc_confirmed": False, "capability_states": state_count,
            "synthetic_or_mock_data": analysis.get("synthetic_or_mock_data", False),
            "issue_count": len(issues), "blockers": blockers, "tests": tests,
            "next_steps": ["Spustiť offline audit v rovnakom interpreteri ako Windows SDK 1.0.2.",
                           "Pasívne čítanie až po overení lokálnej inštalácie a oprávnenia.",
                           "Schváliť baseline, jednotky, RF limity a presnú kópiu plánu pred meraním."]}


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    h = "".join(f"<th>{escaped(x)}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{escaped(v)}</td>" for v in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def build_report(out: Path, summary: dict[str, Any], env: dict[str, Any],
                 capabilities: list[dict[str, Any]], issues: list[dict[str, Any]],
                 analysis: dict[str, Any], plan: dict[str, Any] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cap_rows = [[item["id"], item["status"], item.get("api_path") or "—",
                 item.get("wire_name") or "—", item.get("units") or "neznáme",
                 ", ".join(e.get("ref", "") for e in item.get("evidence", [])) or "bez dôkazu"]
                for item in capabilities]
    issue_rows = [[issue["id"], issue["severity"], issue["detail"], ", ".join(issue["evidence"])]
                  for issue in issues]
    ready_rows = [list(row) for row in READINESS]
    computation = [list(row) for row in COMPUTE_TABLE]
    chart_count = summary["charts_observed"]
    fid_count = summary["fid_pairs"]
    history = json.loads((out / "user_reported_history.json").read_text(encoding="utf-8")) if (out / "user_reported_history.json").exists() else None
    history_note = ("Používateľov skorší výpis uvádzal Rabiho meranie v piatich bodoch a príjem 16 000 bodov FID; "
                    "v tomto pracovnom priestore chýba jeho pôvodný JSON, preto to nie je nezávisle prehraté meranie. "
                    "Časovanie: prvý FFT graf 3,845 s, obe FID zložky 4,084 s; SDK bufferovanie môže tieto časy ovplyvniť."
                    if history else "Žiadny historický výpis nebol priložený.")
    svg = (f'<svg viewBox="0 0 600 92" role="img" aria-label="Počet grafov a párov FID">'
           f'<text x="0" y="20">Grafy: {chart_count}</text><rect x="115" y="6" width="{min(450, chart_count*30)}" height="18" fill="#3996bd"/>'
           f'<text x="0" y="59">FID páry: {fid_count}</text><rect x="115" y="44" width="{min(450, fid_count*30)}" height="18" fill="#59ab77"/></svg>')
    blockers = "".join(f"<li>{escaped(item)}</li>" for item in summary["blockers"]) or "<li>Žiadne ďalšie blokátory v tomto režime.</li>"
    links = " ".join(f'<a href="{escaped(name)}">{escaped(name)}</a>' for name in
                     ("summary.json", "environment.json", "capabilities.json", "field_catalog.json",
                      "issues.json", "analysis.json", "original_scientific.json", "scientific_arrays.npz",
                      "events.jsonl.gz", "manifest.json"))
    html_doc = f"""<!doctype html><html lang="sk"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SpinQ audit</title><style>
body{{font:16px system-ui,sans-serif;max-width:1080px;margin:2rem auto;padding:0 1rem;color:#172c38;background:#f8fbfc;line-height:1.5}}
h1,h2{{color:#123b54}}nav a{{margin-right:1rem}}section{{background:white;padding:1rem 1.4rem;margin:1rem 0;border:1px solid #dce9ef;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{border:1px solid #dce9ef;padding:.5rem;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{background:#e8f3f8}}code{{background:#eef3f5;padding:.15rem}}
.tag{{display:inline-block;background:#dcefe5;padding:.25rem .5rem;border-radius:5px}}.warn{{background:#fff3db;padding:.6rem}}
</style></head><body><h1>Audit SpinQLabLink pre Gemini Lab</h1>
<p class="tag">Režim: {escaped(summary['mode'])} · SDK: {escaped(summary['sdk_version'] or 'nenainštalované')} · {escaped(summary['created_utc'])}</p>
<nav><a href="#odpovede">Odpovede</a><a href="#dovody">Dôkazy a blokátory</a><a href="#mapa">Mapa schopností</a><a href="#data">Dáta</a><a href="#vypocty">Výpočty</a><a href="#vyskum">Výskumné slučky</a><a href="#subory">Súbory</a></nav>
<section id="odpovede"><h2>Čo je teraz podložené</h2><ul>
<li>Čítať možno serverom posielané stavové, lock a parametrické správy cez SDK; konkrétna telemetria v tejto relácii závisí od pasívne prijatých udalostí.</li>
<li>Dočasné experimentálne parametre sú v klientskom fyzikálnom payload-e. Serverové prijatie a fyzický účinok každej zmeny vyžadujú samostatný test.</li>
<li>Trvalý zápis kalibrácie, shimov alebo firmware týmto auditom nebol povolený ani vykonaný.</li>
<li>Dostupný dátový typ je dekódovaný chart FID (ak bol prijatý). RAW ADC nie je potvrdené. Chýbajúce chart dáta neznamenajú fyzikálnu nemožnosť.</li>
<li>Lokálna FFT a kontroly dát sú implementované; vypnutie serverovej FFT/fitov cez zdokumentované API sa nepotvrdilo.</li>
</ul><p class="warn">Živé experimenty: {escaped(summary['hardware_experiments_executed'])}. Syntetické/mock dáta v tomto reporte: {escaped(summary['synthetic_or_mock_data'])}. Syntetické testy nie sú dôkazom zariadenia.</p></section>
<section id="dovody"><h2>Dôkazy a blokátory</h2><p>Zdroj SDK: {escaped(env['spinqlablink']['origin'])}; model „Gemini Lab“ je údaj používateľa, firmware a server verzia sú neznáme. Klientské device_id nie je sériové číslo. Úplná nemennosť lokálnych súborov a balíka nie je týmto auditom overená.</p><ul>{blockers}</ul>
{_table(['SDK nástraha','Úroveň','Zistenie','Zdroj'],issue_rows)}</section>
<section id="mapa"><h2>Mapa schopností</h2>{_table(['ID','Stav','API','Wire','Jednotka','Dôkaz'],cap_rows)}</section>
<section id="data"><h2>Prijaté dáta</h2>{svg}<p>Komplexný FID sa skladá iba z jednej dvojice Re/Im rovnakého tasku, kanála, kroku, bloku a osi. Počet párov: {fid_count}. Pôvodné číselné krivky sú celé v <code>original_scientific.json</code>. Pole <code>raw_adc_confirmed</code> zostáva false.</p><p>{escaped(history_note)}</p></section>
<section id="vypocty"><h2>Umiestnenie výpočtov</h2>{_table(['Úloha','Kde','Hranica dôkazu'],computation)}</section>
<section id="vyskum"><h2>Pripravenosť pre výskum</h2>{_table(['Úloha','Vstupy','Obmedzenie','Najbližšie overenie'],ready_rows)}
<p>Budúca slučka: lokálny návrh parametrov → nezávislá kontrola limitov → jediný hardvérový executor → pôvodné dáta → lokálna analýza → ďalší návrh. Agent nemá priamu cestu okolo kontroly.</p>
<p>Dataset delíme podľa meracích relácií alebo nezávislých akvizícií, nikdy náhodne po bodoch jedného FID. Denoising hodnotíme aj chybou fázy, frekvencie, amplitúdy, šírky čiary a následnej kalibrácie.</p></section>
<section id="subory"><h2>Podklady na offline kontrolu</h2><p>{links}</p><p>Úplnosť záznamu a preťaženie fronty sú v <code>recorder_status.json</code>. ZIP obsahuje zdieľateľné podklady bez autentizačných tajomstiev.</p></section></body></html>"""
    html_doc = html_doc.replace("</body></html>",
        '<section><h2>Zdroje</h2><p><a href="https://doc.spinq.cn/doc/SpinQLAB_Link/en/api/spinqlablink.html">Oficiálne API</a> · '
        '<a href="https://github.com/SpinQTech/spinqlablink/tree/8fe50f65bf87b97bf39dc4e1f8db9363801fd169">Oficiálny zdrojový commit</a> · '
        '<a href="https://github.com/SpinQTech/spinqlablink/blob/8fe50f65bf87b97bf39dc4e1f8db9363801fd169/spinqlablink/connection/message.proto">Wire schéma</a>. '
        'Dátum overenia dokumentácie: 2026-09-25. Odkazy sú informatívne; report funguje bez internetu.</p></section></body></html>')
    atomic_bytes(out / "REPORT.html", html_doc.encode("utf-8"))
    markdown = f"""# SpinQ audit — {summary['mode']}

- Vytvorené: {summary['created_utc']}
- SDK: {summary['sdk_version'] or 'nenainštalované'} ({summary['sdk_origin']})
- Živé merania: {summary['hardware_experiments_executed']}; grafy: {chart_count}; párované FID: {fid_count}.
- RAW ADC potvrdené: nie. Trvalé kalibrácie zmenené: nie.
- Plán: {(plan or {}).get('status', 'nevytvorený')}.

## Blokátory
""" + "".join(f"- {item}\n" for item in summary["blockers"]) + "\n## Súbory\n\nPozri REPORT.html, summary.json, capabilities.json, field_catalog.json a audit_bundle.zip.\n"
    atomic_bytes(out / "REPORT.md", markdown.encode("utf-8"))
    atomic_bytes(out / "vendor_questions.md", QUESTIONS.encode("utf-8"))


def finalize_bundle(out: Path) -> None:
    files = [p for p in out.rglob("*") if p.is_file() and p.name not in {"manifest.json", "audit_bundle.zip"}
             and not p.name.endswith(".tmp")]
    manifest = {"created_utc": utc_now(), "files": {str(p.relative_to(out)).replace("\\", "/"):
                                                {"bytes": p.stat().st_size}
                                                for p in sorted(files)}}
    atomic_json(out / "manifest.json", manifest)
    temporary = out / "audit_bundle.zip.tmp"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files + [out / "manifest.json"]):
            archive.write(path, str(path.relative_to(out)).replace("\\", "/"))
    os.replace(temporary, out / "audit_bundle.zip")
