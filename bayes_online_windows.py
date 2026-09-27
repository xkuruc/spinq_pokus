"""Windows-only entry point for independent, live 01_bayes_online."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from bayes_online_core.artifacts import atomic_json, publish
from bayes_online_core.runner import OnlineRun, numeric_preflight, validate_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live Gemini Lab Bayesian online experiment")
    parser.add_argument("--task", choices=("frequency", "rabi", "coupling", "pps", "bell", "all"),
                        default="all")
    parser.add_argument("--exclusive-use-confirmed", action="store_true",
                        help="Confirm that the Gemini Lab is idle and reserved for this run")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config-bayes-online.json"))
    parser.add_argument("--resume", type=Path,
                        help="Resume a stopped run with the identical configuration/task")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Retain local results only")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parent
    try:
        config = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
        config = dict(config)
        checks = numeric_preflight()
    except Exception as exc:
        print(f"01 ONLINE PREFLIGHT ERROR: {type(exc).__name__}: {exc}; no physical task sent", flush=True)
        return 2
    if os.name != "nt":
        print("01 ONLINE ERROR: live launcher is Windows-only; no connection opened", flush=True)
        return 2
    print("01 ONLINE PREFLIGHT OK: SpinQLabLink 1.0.2, numeric model and gate algebra", flush=True)
    output = (args.resume.resolve() if args.resume else
              repo / "results" / "01_bayes_online" /
              datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC"))
    print(f"01 ONLINE RESULT DIRECTORY: {output}", flush=True)
    run = OnlineRun(repo, output, config, args.task,
                    exclusive_use_confirmed=args.exclusive_use_confirmed,
                    resume=bool(args.resume))
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
