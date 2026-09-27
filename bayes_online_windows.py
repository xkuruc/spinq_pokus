"""Live 01_bayes_online entry point: Windows or local-only macOS."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from bayes_online_core.artifacts import atomic_json, publish
from bayes_online_core.runner import OnlineRun, numeric_preflight, quick_task_plan, validate_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live Gemini Lab Bayesian online experiment")
    parser.add_argument("--task", choices=("frequency", "rabi", "coupling", "pps", "bell", "all"),
                        default="all")
    parser.add_argument("--exclusive-use-confirmed", action="store_true",
                        help="Confirm that the Gemini Lab is idle and reserved for this run")
    parser.add_argument("--quick", action="store_true",
                        help="Three short paired H-only B-vs-D blocks; no PPS/Bell tasks")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config-bayes-online.json"))
    parser.add_argument("--resume", type=Path,
                        help="Resume a stopped run with the identical configuration/task")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Retain local results only")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parent
    try:
        config = dict(json.loads(args.config.read_text(encoding="utf-8")))
        task = args.task
        if args.quick:
            if task not in ("all", "rabi", "frequency"):
                raise ValueError("--quick supports only Rabi/frequency calibration")
            task = "rabi" if task == "all" else task
            config.update(channels=["H"], blocks=3,
                          calibration_acquisitions_per_method=10,
                          reference_acquisitions_per_block=5,
                          max_tasks=120, max_requested_rf_us=30000,
                          pause_seconds=1)
        config = validate_config(config)
        quick_plan = quick_task_plan(config) if args.quick else None
        checks = numeric_preflight()
    except Exception as exc:
        print(f"01 ONLINE PREFLIGHT ERROR: {type(exc).__name__}: {exc}; no physical task sent", flush=True)
        return 2
    if os.name != "nt" and not (sys.platform == "darwin" and args.skip_upload):
        print("01 ONLINE ERROR: macOS live runs require --skip-upload; "
              "no connection opened", flush=True)
        return 2
    print("01 ONLINE PREFLIGHT OK: SpinQLabLink 1.0.2, numeric model and gate algebra", flush=True)
    if quick_plan:
        print(f"01 ONLINE QUICK PLAN: {quick_plan['planned_physical_tasks']} "
              "physical FIDs, H only, B versus D in three paired scenarios; "
              "roughly 30–40 minutes at previously observed task times, "
              "50-minute active runtime cap", flush=True)
    output = (args.resume.resolve() if args.resume else
              repo / "results" / "01_bayes_online" /
              (("quick_" if args.quick else "") +
               datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")))
    print(f"01 ONLINE RESULT DIRECTORY: {output}", flush=True)
    run = OnlineRun(repo, output, config, task,
                    exclusive_use_confirmed=args.exclusive_use_confirmed,
                    resume=bool(args.resume), quick=args.quick)
    atomic_json(output / "profiles" / "preflight.json", checks)
    state = run.execute()
    if args.skip_upload:
        upload = {"status": "UPLOAD_SKIPPED_BY_USER"}
    else:
        branch = f"benchmark/01_bayes_online/{output.name}"
        try:
            upload = publish(repo, output, branch)
        except Exception as exc:
            upload = {"status": "UPLOAD_FAILED", "branch": branch,
                      "reason": f"{type(exc).__name__}: {exc}"[:1000]}
    atomic_json(output / "upload.json", upload)
    print(f"01 ONLINE UPLOAD: {upload['status']}", flush=True)
    if upload.get("reason"):
        print(f"01 ONLINE UPLOAD REASON: {upload['reason']}", flush=True)
    return 0 if state["status"] == "COMPLETED_WITH_EXPLICIT_LIMITATIONS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
