"""Windows command-line entry point for the real Gemini Lab experiment 01."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.bayes_calibration import (BayesRun, EXPERIMENT,
    offline_preflight, reanalyze_saved, validate_config)


def saved_config_for_reanalysis(out: Path) -> dict:
    """Use the run's frozen settings without consulting today's live config."""
    source = out / "results.json"
    if not source.is_file():
        raise FileNotFoundError(f"Saved results.json is missing: {source}")
    saved = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or saved.get("experiment") != EXPERIMENT:
        raise ValueError("Saved results are not from experiment 01")
    config = saved.get("config")
    if not isinstance(config, dict):
        raise ValueError("Saved run configuration is missing")
    return validate_config(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live, locally analyzed Bayesian SpinQ calibration")
    parser.add_argument("--config", type=Path, default=Path("config-01-bayes.json"))
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--resume", type=Path, help="Continue this saved run without repeating completed tasks")
    choice.add_argument("--reanalyze", type=Path, help="Only recompute reports from saved raw FIDs")
    choice.add_argument("--preflight", action="store_true", help="Check local SDK/numerics without connecting")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parent
    out = None
    if args.resume or args.reanalyze:
        supplied = args.resume or args.reanalyze
        out = supplied if supplied.is_absolute() else repo / supplied
        if not out.is_dir():
            print(f"01 ERROR: result directory does not exist: {out}", flush=True)
            return 2
    try:
        if args.reanalyze:
            config = saved_config_for_reanalysis(out)
        else:
            source = args.config if args.config.is_absolute() else repo / args.config
            config = validate_config(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        print(f"01 CONFIG FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 2
    try:
        preflight = offline_preflight()
    except Exception as exc:
        print(f"01 PREFLIGHT FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 2
    print("01 PREFLIGHT OK: SpinQLabLink 1.0.2 serializer and local numeric runtime", flush=True)
    if args.preflight:
        print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
        return 0
    if out is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
        out = repo / "results" / EXPERIMENT / run_id
    print(f"01 RESULT DIRECTORY: {out}", flush=True)
    if args.reanalyze:
        try:
            return reanalyze_saved(repo, out, config, preflight)
        except Exception as exc:
            print(f"01 REANALYSIS FAILED: {type(exc).__name__}: {exc}", flush=True)
            return 2
    try:
        session = BayesRun(repo, out, config, preflight, resume=bool(args.resume))
        return session.execute()
    except Exception as exc:
        print(f"01 START FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
