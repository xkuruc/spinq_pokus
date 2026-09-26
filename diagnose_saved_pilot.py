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


def diagnose(directory: Path) -> dict:
    out = directory.resolve()
    raw = out / "raw"
    records = {width: RawFIDRecord.load(raw, key) for width, key in WIDTH_KEYS}
    axes = {width: validate_axis(record) for width, record in records.items()}
    with tempfile.TemporaryDirectory(prefix="spinq_pilot_diagnostic_") as temporary:
        session = object.__new__(LocalSession)
        session.out = Path(temporary)
        session.pilot_records = {"pilot_40_r0": records[40]}
        coefficients = [session.coefficient(records[width]) for width, _ in WIDTH_KEYS]
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
            "rabi": fit, "noise": noise_result,
            "previous_pilot_failures": previous,
            "physical_adc_clock_verified": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only check of saved Gemini Lab Rabi pilot")
    parser.add_argument("results_directory", type=Path)
    args = parser.parse_args(argv)
    try:
        result = diagnose(args.results_directory)
    except Exception as exc:
        print(f"PILOT DIAGNOSTIC FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc(limit=6, file=sys.stdout)
        return 2
    print("PILOT DIAGNOSTIC: saved FIDs only; no hardware command sent", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
