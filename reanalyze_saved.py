"""Reanalyse a completed Windows benchmark from saved FIDs, without device access.

The original run remains untouched. Derived reports and models are written to a
new child directory; publishing the compact summary is opt-in.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from publish_results import publish_summary
from spinq_audit.common import utc_now
from spinq_local.analysis_runner import analyze_de_fg
from spinq_local.core import RawFIDRecord
from spinq_local.preflight import check as check_runtime
from spinq_local.report import Results, plots_from_results


ROOT = Path(__file__).resolve().parent
ROLES = ("pilot", "train", "validation", "test")


def reclassify_clipped_b_reference(data: dict) -> None:
    """Correct a legacy B label when its saved frequency reference was clipped."""
    pilot = data.get("pilot", {})
    reference = pilot.get("reference_frequency_hz")
    if not isinstance(reference, (int, float)) or not math.isfinite(reference):
        return
    clipped = any(
        len(band) == 2 and all(isinstance(v, (int, float)) for v in band) and
        float(band[0]) < float(band[1]) and
        min(abs(reference - float(band[0])), abs(reference - float(band[1]))) <
        .01 * (float(band[1]) - float(band[0]))
        for band in pilot.get("multiplet_bands_hz", ())
        if isinstance(band, (list, tuple))
    )
    if not clipped:
        return
    reason = ("Stored pilot frequency fit was clipped to its frozen component band; "
              "FID-length frequency accuracy has no valid independent reference. "
              "Saved B differences are diagnostic only; echo timing remains unverified")
    if "B" in data.get("modules", {}):
        old = data["modules"]["B"]
        data["modules"]["B"] = {**old, "status": "REFERENCE_INADEQUATE",
                                  "reason": reason,
                                  "prior_status": old.get("status"),
                                  "prior_reason": old.get("reason")}
    for row in data.get("rows", ()):
        if row.get("module") == "B":
            row["status"] = "REFERENCE_INADEQUATE"
            row["reason"] = reason
    data.setdefault("reanalysis", {})["legacy_b_reference_reclassified"] = True


def load_saved_records(source: Path) -> dict[str, list[RawFIDRecord]]:
    """Require each completed task's original FID and role metadata."""
    raw = source / "raw"
    journal = source / "data" / "hardware_journal.json"
    if not raw.is_dir() or not journal.is_file():
        raise ValueError("Saved raw/ and data/hardware_journal.json are required")
    history = json.loads(journal.read_text(encoding="utf-8"))
    completed = {key for key, entry in history.items()
                 if isinstance(entry, dict) and entry.get("phase") == "completed"}
    uncertain = {key: entry.get("phase") if isinstance(entry, dict) else "INVALID_ENTRY"
                 for key, entry in history.items()
                 if not isinstance(entry, dict) or entry.get("phase") != "completed"}
    if uncertain:
        raise ValueError(f"Unresolved hardware journal entries: {uncertain}")
    if not completed:
        raise ValueError("No completed hardware tasks in the saved journal")
    roles: dict[str, list[RawFIDRecord]] = {role: [] for role in ROLES}
    for key in sorted(completed):
        if not (raw / f"{key}.json").is_file() or not (raw / f"{key}.npz").is_file():
            raise ValueError(f"Completed task {key} lacks a saved FID pair")
        record = RawFIDRecord.load(raw, key)
        role = record.metadata.get("dataset_role")
        if role not in roles:
            raise ValueError(f"Saved task {key} has invalid dataset role {role!r}")
        roles[role].append(record)
    return roles


def reanalyze(source: Path, *, upload: bool = False) -> Path:
    source = source.resolve()
    original_path = source / "results.json"
    if not original_path.is_file():
        raise ValueError(f"No saved results.json in {source}")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if not original.get("state", "").startswith("COMPLETED"):
        raise ValueError("Reanalysis requires a completed source run")
    roles = load_saved_records(source)
    runtime = check_runtime()  # local imports and CPU smoke tests only
    print("OFFLINE: " + json.dumps({
        "saved_tasks": sum(map(len, roles.values())),
        "numeric": runtime.get("numeric", {}).get("ready"),
        "torch": runtime.get("torch", {}).get("ready"),
        "torch_reason": runtime.get("torch", {}).get("reason")
    }, ensure_ascii=False), flush=True)
    if not runtime.get("numeric", {}).get("ready"):
        raise RuntimeError("Local numerical runtime failed: " +
                           str(runtime.get("numeric", {}).get("reason")))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    out = source / f"reanalysis_{stamp}"
    out.mkdir(exist_ok=False)
    results = Results(out, original["config"], runtime)
    data = copy.deepcopy(original)
    data["state"] = "RUNNING_OFFLINE_REANALYSIS"
    data["preflight"] = runtime
    data["reanalysis"] = {
        "source_run": source.name,
        "source_raw": "../raw",
        "source_vendor_reference": "../vendor_reference",
        "offline_only": True,
        "created_utc": utc_now(),
        "source_started_utc": original.get("started_utc"),
        "saved_task_count": sum(map(len, roles.values())),
    }
    reclassify_clipped_b_reference(data)
    data["rows"] = [row for row in data.get("rows", [])
                    if row.get("module") not in {"D", "E", "F"} and not (
                        row.get("module") == "G" and
                        str(row.get("task", "")).startswith("heldout_analogue_readout:"))]
    for module in "DEF":
        data["modules"][module] = {"status": "PENDING"}
    data["upload"] = {"status": "NOT_ATTEMPTED"}
    data.pop("finished_utc", None)
    results.data = data
    results.save()
    try:
        summary = analyze_de_fg(roles, data["pilot"], out, runtime, results)
        results.data["reanalysis"]["offline_module_summary"] = summary
        results.data["state"] = "COMPLETED_OFFLINE_REANALYSIS_WITH_EXPLICIT_LIMITATIONS"
    except Exception as exc:
        results.data["state"] = "FAILED_OFFLINE_REANALYSIS"
        results.data["errors"].append(f"Offline reanalysis: {type(exc).__name__}: {exc}")
        results.save()
        raise
    results.data["finished_utc"] = utc_now()
    results.save()
    plots_from_results(out, results.data)
    if upload:
        branch = f"benchmark/{source.name}_{out.name}"
        results.data["upload"] = publish_summary(ROOT, out, branch)
        results.save()
    print(f"OFFLINE SUMMARY: D={results.data['modules']['D']['status']} "
          f"E={results.data['modules']['E']['status']} "
          f"F={results.data['modules']['F']['status']} "
          f"G={results.data['modules']['G']['status']} "
          f"upload={results.data['upload']['status']}", flush=True)
    if results.data["upload"]["status"] == "UPLOAD_FAILED":
        print(f"OFFLINE UPLOAD ERROR: {results.data['upload'].get('reason')}", flush=True)
    print(f"Report: {out / 'REPORT.md'}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Completed results/<run_id> directory")
    parser.add_argument("--upload", action="store_true",
                        help="Publish only the derived report, table, summary and plots")
    args = parser.parse_args()
    out = reanalyze(args.source, upload=args.upload)
    if args.upload:
        data = json.loads((out / "results.json").read_text(encoding="utf-8"))
        if data["upload"]["status"] != "UPLOAD_SUCCEEDED":
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
