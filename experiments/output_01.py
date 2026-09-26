"""Incremental outputs for the real Gemini Lab Bayesian calibration study.

This module never contacts the instrument.  Rows represent independent
acquisitions or completed method/block comparisons, never individual FID
samples.  The machine-readable JSON keeps every supplied field; the CSV and
Markdown report expose the principal comparisons without inventing missing
reference values.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

from spinq_audit.common import atomic_bytes, atomic_json, redact


ROW_FIELDS = (
    "method", "baseline", "block", "status", "reason", "acquisitions",
    "acquisition_seconds", "fitting_seconds", "inference_seconds",
    "design_seconds", "end_to_end_seconds", "frequency_error_hz",
    "t90_error_us", "phase_error_deg", "coverage_frequency",
    "coverage_t90", "coverage_phase", "reference_uncertainty",
    "estimate_uncertainty", "calibration_bias", "control_error",
)
SOURCE_FILES = (
    "experiments/__init__.py",
    "experiments/bayes_calibration.py", "experiments/smc.py",
    "experiments/design.py", "experiments/likelihood.py",
    "experiments/signal_01.py", "experiments/output_01.py",
    "experiments/publish_01.py", "experiments/01_sources.md",
    "experiments/01_reproduction_scope.md", "spinq_benchmark/__init__.py",
    "spinq_benchmark/hardware.py", "spinq_local/__init__.py",
    "spinq_local/core.py", "spinq_local/signal.py",
    "spinq_audit/__init__.py", "spinq_audit/adapter.py",
    "spinq_audit/common.py", "spinq_audit/discovery.py",
    "spinq_audit/probes.py", "spinq_audit/recorder.py",
    "spinq_audit/safety.py", "spinq_live_suite.py",
    "run_01_bayes_kalibracia.cmd", "bayes_01_windows.py",
    "requirements-01-bayes.txt",
    "config-01-bayes.json", "BAYES_01_README.md",
)


def prepare_output(out_dir: Path) -> Path:
    """Create the required output tree without deleting existing run data."""
    out = Path(out_dir)
    for name in ("raw", "data", "vendor_reference", "models", "pulses",
                 "plots", "source_snapshot"):
        (out / name).mkdir(parents=True, exist_ok=True)
    return out


def _jsonable(value: Any) -> Any:
    """Keep numerical arrays/scalars as numbers instead of ``default=str``."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, complex):
        return {"real": _jsonable(value.real), "imag": _jsonable(value.imag)}
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    raise TypeError(f"Result contains unsupported type: {type(value).__name__}")


def _csv_value(value: Any) -> str | int | float | None:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(redact(value), ensure_ascii=False, allow_nan=False,
                          sort_keys=True, default=str)
    return str(value)


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        return f"{value:.5g}"
    if isinstance(value, dict):
        return ", ".join(f"{key}: {_markdown_cell(val)}" for key, val in value.items())
    return str(value).replace("|", "/").replace("\r", " ").replace("\n", " ")


def render_report(data: dict[str, Any]) -> str:
    """Human-readable comparison, including explicit missing/failed statuses."""
    rows = data.get("rows") or []
    total = data.get("hardware_tasks_completed", data.get("measured_tasks", "unknown"))
    lines = [
        "# Gemini Lab: Bayesovská kalibrácia viacerých parametrov", "",
        f"Stav: **{_markdown_cell(data.get('state', 'UNKNOWN'))}**. "
        f"Dokončené fyzické úlohy: **{_markdown_cell(total)}**. "
        f"Porovnávacie riadky: **{len(rows)}**.",
        "Exportovaný komplexný FID nie je potvrdený RAW ADC. "
        "FID vzorky nie sú nezávislé qubitové shots; serverové FFT môže ďalej bežať.",
        "Referencie a kontrolné rotácie sú oddelené od výberu meraní. "
        "Chýbajúce referencie sa nenahrádzajú pilotným odhadom.", "",
        "## Porovnanie po blokoch", "",
        "| Metóda | Baseline | Blok | Akvizície | Akvizícia (s) | "
        "Fit (s) | Inferencia (s) | Návrh (s) | Koniec–koniec (s) | "
        "Δf (Hz) | Δt90 (µs) | Δφ (°) | Kontrolná chyba | Stav | Dôvod |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        cells = [row.get(key) for key in (
            "method", "baseline", "block", "acquisitions",
            "acquisition_seconds", "fitting_seconds", "inference_seconds",
            "design_seconds", "end_to_end_seconds", "frequency_error_hz",
            "t90_error_us", "phase_error_deg", "control_error", "status", "reason",
        )]
        lines.append("| " + " | ".join(_markdown_cell(x) for x in cells) + " |")
    if not rows:
        lines.extend(["", "Zatiaľ nevzniklo porovnanie. Pilot alebo bezpečnostná "
                      "podmienka zatiaľ nedovolili ďalšie merania."])
    lines += ["", "## Neistota a pokrytie", "",
              "| Metóda | Blok | Neistota referencie | Neistota odhadu | "
              "Kalibračný bias | Pokrytie f | Pokrytie t90 | Pokrytie φ |",
              "|---|---:|---|---|---|---|---|---|"]
    for row in rows:
        cells = [row.get(key) for key in (
            "method", "block", "reference_uncertainty", "estimate_uncertainty",
            "calibration_bias", "coverage_frequency", "coverage_t90",
            "coverage_phase",
        )]
        lines.append("| " + " | ".join(_markdown_cell(x) for x in cells) + " |")
    if not rows:
        lines.append("| — | — | — | — | — | — | — | — |")
    lines += ["", "Pokrytie sa vyhodnocuje iba pri dostupnej nezávislej "
              "referencii. Detailné polia sú v `comparison.csv` a `results.json`.", ""]
    if data.get("pilot"):
        pilot = data["pilot"]
        status = pilot.get("status", "RECORDED") if isinstance(pilot, dict) else "RECORDED"
        lines += ["## Pilot", "", f"Stav: **{_markdown_cell(status)}**; "
                  "podrobnosti sú v `results.json` a `plan.json`.", ""]
    if data.get("plan"):
        plan = data["plan"]
        status = plan.get("status", "RECORDED") if isinstance(plan, dict) else "RECORDED"
        lines += ["## Zmrazený plán", "", f"Stav: **{_markdown_cell(status)}**; "
                  "rozpočet a tolerancie sú v `plan.json`.", ""]
    if data.get("errors"):
        lines += ["## Chyby a obmedzenia", ""]
        for error in data["errors"]:
            lines.append("- " + _markdown_cell(error))
        lines.append("")
    lines += ["## Súbory", "",
              "`results.zip` obsahuje kompletný zachovaný export vrátane pôvodných "
              "FID NPZ, sanitizovaného denníka udalostí, vendor_reference, modelov, "
              "pulzov, grafov a snímky spusteného zdrojového kódu. "
              "Pri rozdelenom uploade sú diely a návod vo výsledkovej Git vetve.", ""]
    return "\n".join(lines)


def save_results(out_dir: Path, data: dict[str, Any]) -> None:
    """Write JSON, full-field CSV and report, preserving an existing raw tree."""
    out = prepare_output(out_dir)
    clean = redact(_jsonable(data))
    rows = clean.get("rows") or []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("results rows must be a list of dictionaries")
    atomic_json(out / "results.json", clean)
    extras = sorted({key for row in rows for key in row if key not in ROW_FIELDS})
    fields = (*ROW_FIELDS, *extras)
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key)) for key in fields})
    atomic_bytes(out / "comparison.csv", target.getvalue().encode("utf-8"))
    atomic_bytes(out / "REPORT.md", render_report(clean).encode("utf-8"))


def snapshot_sources(out_dir: Path, repo_dir: Path,
                     relative_paths: tuple[str, ...] = SOURCE_FILES) -> list[str]:
    """Copy a whitelist of relevant code/docs; never copy private config files."""
    out = prepare_output(out_dir)
    repo = Path(repo_dir).resolve()
    copied: list[str] = []
    for relative in relative_paths:
        source = (repo / relative).resolve()
        if not source.is_relative_to(repo):
            raise ValueError(f"Source snapshot path escapes repository: {relative}")
        if not source.is_file():
            continue
        destination = out / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == "config-01-bayes.json":
            # The live configuration may later gain local credentials. Keep
            # reproducible numerical settings while stripping secret fields.
            atomic_json(destination, json.loads(source.read_text(encoding="utf-8")))
        else:
            shutil.copyfile(source, destination)
        copied.append(relative)
    for source_name, output_name in (
        ("experiments/01_sources.md", "sources.md"),
        ("experiments/01_reproduction_scope.md", "reproduction_scope.md"),
    ):
        source = repo / source_name
        if source.is_file():
            shutil.copyfile(source, out / output_name)
    manifest = {"source_files": copied,
                "note": "Snapshot of benchmark-relevant source files; no secrets or credentials."}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True, check=False)
    if revision.returncode == 0:
        manifest["git_revision"] = revision.stdout.strip()
    atomic_json(out / "source_snapshot" / "manifest.json", manifest)
    return copied


def _ensure_compressed_events(out: Path) -> None:
    """Refresh raw/events from the newest sanitized journal after a resume.

    ``EventRecorder`` appends to data/events.jsonl on every connection and
    writes data/events.jsonl.gz at close.  A prior interrupted run may already
    have created raw/events.jsonl.gz, so its mere existence does not mean it
    contains the newly appended events.
    """
    raw_event = out / "raw" / "events.jsonl.gz"
    data_event = out / "data" / "events.jsonl.gz"
    plain_event = out / "data" / "events.jsonl"
    if not data_event.is_file() and not plain_event.is_file():
        # A finalized raw journal from a previous archival step is still
        # recoverable even when the data/ copy was subsequently removed.
        return
    temporary = out / "raw" / "events.jsonl.gz.tmp"
    # The plain append-only journal wins when it is as new as the compressed
    # one (ties also protect filesystems with coarse timestamp resolution).
    use_plain = plain_event.is_file() and (
        not data_event.is_file()
        or plain_event.stat().st_mtime_ns >= data_event.stat().st_mtime_ns
    )
    if use_plain:
        with plain_event.open("rb") as source, gzip.open(temporary, "wb") as destination:
            shutil.copyfileobj(source, destination)
        data_temporary = out / "data" / "events.jsonl.gz.tmp"
        shutil.copyfile(temporary, data_temporary)
        _replace_retry(data_temporary, data_event)
    else:
        shutil.copyfile(data_event, temporary)
    _replace_retry(temporary, raw_event)


def _replace_retry(source: Path, destination: Path) -> None:
    """Windows scanners may briefly deny replacement of a completed file."""
    for delay in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, None):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def archive_results(out_dir: Path) -> Path:
    """Create one complete ZIP; the publisher can split its bytes losslessly."""
    out = prepare_output(out_dir)
    _ensure_compressed_events(out)
    data_path = out / "results.json"
    if not data_path.is_file():
        raise FileNotFoundError("results.json must be saved before archiving")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    has_npz = any((out / "raw").glob("*.npz"))
    if data.get("hardware_results_present") and not has_npz:
        raise ValueError("Hardware results are reported, but no exported FID NPZ is present")
    if has_npz and not (out / "raw" / "events.jsonl.gz").is_file():
        raise ValueError("FID NPZ exists without a recoverable sanitized event journal")
    temporary = out / "results.zip.tmp"
    success = False
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for file in sorted(out.rglob("*")):
                if not file.is_file() or file.is_symlink():
                    continue
                relative = file.relative_to(out).as_posix()
                if relative == "results.zip" or relative.startswith("results.zip.part") \
                        or relative == "results.zip.tmp" or relative.endswith(".tmp"):
                    continue
                if relative == "data/events.jsonl" and (out / "raw" / "events.jsonl.gz").exists():
                    continue
                archive.write(file, relative)
        _replace_retry(temporary, out / "results.zip")
        success = True
    finally:
        if success:
            temporary.unlink(missing_ok=True)
    return out / "results.zip"


def plot_comparisons(out_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    """Build headless plots from saved method/block comparisons only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = prepare_output(out_dir) / "plots"
    made: list[str] = []
    for key, label in (("frequency_error_hz", "Absolútna chyba frekvencie [Hz]"),
                       ("t90_error_us", "Absolútna chyba t90 [µs]"),
                       ("phase_error_deg", "Absolútna chyba relatívnej fázy [°]"),
                       ("control_error", "Chyba kontrolnej rotácie")):
        valid = [row for row in rows if isinstance(row.get(key), (int, float))
                 and math.isfinite(float(row[key]))]
        if not valid:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plotted = False
        for method in sorted({str(row.get("method")) for row in valid}):
            own = [row for row in valid if str(row.get("method")) == method]
            for axis, xkey, xlabel in (
                (axes[0], "acquisitions", "Fyzické akvizície"),
                (axes[1], "end_to_end_seconds", "Koniec–koniec čas [s]"),
            ):
                points = [(float(row[xkey]), float(row[key])) for row in own
                          if isinstance(row.get(xkey), (int, float))
                          and math.isfinite(float(row[xkey]))]
                if points:
                    points.sort()
                    axis.plot([point[0] for point in points],
                              [point[1] for point in points], "o-", label=method)
                    plotted = True
                axis.set_xlabel(xlabel)
                axis.set_ylabel(label)
        if plotted:
            for axis in axes:
                if axis.has_data():
                    axis.legend()
            fig.tight_layout()
            filename = f"{key}.png"
            fig.savefig(plot_dir / filename, dpi=140)
            made.append(filename)
        plt.close(fig)
    return made
