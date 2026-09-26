"""Read-only Rabi diagnostic for an existing Windows benchmark directory.

This script loads saved complex FIDs and performs local numerical analysis.
It never creates a SpinQLabLink connection or submits an experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

from spinq_local.core import RawFIDRecord
from spinq_local.session import LocalSession, _jsonable, _rabi_period
from spinq_local.signal import estimate_noise, validate_axis


WIDTH_KEYS = ((40, "pilot_40_r0"), (80, "pilot_80"), (120, "pilot_120"),
              (160, "pilot_160"), (200, "pilot_200"))
TIMING_KEYS = ("timing_return_0", "timing_split_equal",
               "timing_split_opposite", "timing_return_1", "timing_gap_10",
               "timing_gap_20", "timing_terminal_idle", "timing_return_2")


def _timing_contrasts(values: list[complex]) -> dict:
    """Report the live gate's effect sizes without certifying device timing."""
    drift = max(abs(values[0]-values[3]), abs(values[3]-values[7]), 1e-9)
    contrasts = {
        "equal_split": abs(values[1]-values[0]),
        "opposite_phase": abs(values[2]-values[1]),
        "gap_10_vs_20_us": abs(values[5]-values[4]),
        "terminal_idle": abs(values[6]-values[3]),
    }
    thresholds = {
        "equal_split": max(3*drift, .3*abs(values[0])),
        "opposite_phase": max(3*drift, .1*abs(values[0])),
        "gap_10_vs_20_us": 3*drift,
        "terminal_idle": 3*drift,
    }
    return {"status": "MEASURED_CONTRASTS", "return_control_drift": float(drift),
            "coefficient_scale": float(abs(values[0])),
            "contrasts": {name: float(value) for name,value in contrasts.items()},
            "thresholds": {name: float(value) for name,value in thresholds.items()},
            "interpretation": "Contrast below drift threshold is inconclusive; "
                              "it does not prove the device ignored the sequence"}


def diagnose(directory: Path) -> dict:
    out = directory.resolve()
    raw = out / "raw"
    journal_path = out / "data" / "hardware_journal.json"
    journal_status = {"status": "MISSING", "completed_tasks": 0,
                      "uncertain_keys": [], "missing_completed_results": []}
    if journal_path.is_file():
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        uncertain = [key for key, row in journal.items() if row.get("phase") in
                     {"submission_attempted_unconfirmed", "sent_unconfirmed", "unknown"}]
        completed = [key for key, row in journal.items() if row.get("phase") == "completed"]
        missing = [key for key in completed if not (out / "data" / f"{key}.json").is_file()]
        journal_status = {"status": "LOCAL_RECORDS_CONSISTENT" if not uncertain and not missing
                          else "REVIEW_REQUIRED", "completed_tasks": len(completed),
                          "uncertain_keys": uncertain, "missing_completed_results": missing}
    records = {width: RawFIDRecord.load(raw, key) for width, key in WIDTH_KEYS}
    axes = {width: validate_axis(record) for width, record in records.items()}
    with tempfile.TemporaryDirectory(prefix="spinq_pilot_diagnostic_") as temporary:
        session = object.__new__(LocalSession)
        session.out = Path(temporary)
        session.pilot_records = {"pilot_40_r0": records[40]}
        coefficients = [session.coefficient(records[width]) for width, _ in WIDTH_KEYS]
        missing_timing = [key for key in TIMING_KEYS if not (raw / f"{key}.json").is_file()]
        if missing_timing:
            timing_result = {"status": "UNAVAILABLE", "missing_keys": missing_timing}
        else:
            try:
                timing_values = [session.coefficient(RawFIDRecord.load(raw, key))
                                 for key in TIMING_KEYS]
                timing_result = _timing_contrasts(timing_values)
            except Exception as exc:
                timing_result = {"status": "UNAVAILABLE",
                                 "reason": f"{type(exc).__name__}: {exc}"}
    fit = _rabi_period([width for width, _ in WIDTH_KEYS], coefficients)
    noise_result = None
    try:
        repeats = [RawFIDRecord.load(raw, f"pilot_40_r{i}") for i in range(3)]
        noise = estimate_noise(repeats)
        noise_result = {"status": "OK", "independent_repetitions": noise.repetitions,
                        "covariance": noise.re_im_covariance.tolist()}
    except Exception as exc:
        noise_result = {"status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}"}
    previous = {}
    summary = out / "results.json"
    if summary.is_file():
        previous = json.loads(summary.read_text(encoding="utf-8")).get("pilot",{}).get("failures",{})
    return {"status": "VALID", "source": "saved exported complex FIDs; no hardware contact",
            "directory": str(out),
            "records": {str(width): {"key": records[width].key,
                                     "task_id": records[width].task_id,
                                     "points": axes[width].point_count,
                                     "coefficient": _jsonable(coefficients[index])}
                        for index, (width, _) in enumerate(WIDTH_KEYS)},
            "rabi": fit, "noise": noise_result, "timing": timing_result,
            "journal": journal_status,
            "previous_pilot_failures": previous,
            "physical_adc_clock_verified": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only check of saved Gemini Lab Rabi pilot")
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("--json", action="store_true", help="Print all offline diagnostic details")
    args = parser.parse_args(argv)
    try:
        result = diagnose(args.results_directory)
    except Exception as exc:
        print(f"PILOT DIAGNOSTIC FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc(limit=6, file=sys.stdout)
        return 2
    fit=result["rabi"]
    journal=result["journal"]
    print("PILOT DIAGNOSTIC: saved FIDs only; no hardware command sent", flush=True)
    print(f"RABI: period_us={fit['period_us']:.1f} "
          f"t90_us={fit['t90_us']:.1f} R2={fit['signed_complex_r2']:.4f}",flush=True)
    print(f"NOISE: {result['noise']['status']}; "
          f"JOURNAL: {journal['status']} completed={journal['completed_tasks']} "
          f"uncertain={len(journal['uncertain_keys'])} "
          f"missing_results={len(journal['missing_completed_results'])}",flush=True)
    timing=result["timing"]
    if timing["status"]=="MEASURED_CONTRASTS":
        parts=[f"drift={timing['return_control_drift']:.4g}"]
        for name in ("opposite_phase","gap_10_vs_20_us","terminal_idle"):
            parts.append(f"{name}={timing['contrasts'][name]:.4g}/"
                         f"{timing['thresholds'][name]:.4g}")
        print("TIMING: " + " ".join(parts) + " (contrast/threshold)",flush=True)
    else:
        print(f"TIMING: {timing['status']}; saved timing FIDs incomplete",flush=True)
    if result["previous_pilot_failures"]:
        print(f"PREVIOUS PILOT FAILURES: {', '.join(result['previous_pilot_failures'])}",flush=True)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
