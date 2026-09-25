"""CLI for offline, passive, plan, active and replay audits."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
import sys
import tomllib
import unittest
from pathlib import Path
from typing import Any

from .analysis import analyze_events, compare_requested, import_legacy_result, read_events
from .common import atomic_json, redact, utc_now
from .discovery import discover
from .report import build_report, finalize_bundle, make_summary
from .safety import build_plan


def _load_json(path: Path | None) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path else {}


def _load_config(path: Path | None) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8")) if path else {}


def _offline_tests() -> dict[str, Any]:
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent.parent / "tests"), pattern="test_audit_*.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    return {"passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
            "failed": len(result.failures) + len(result.errors), "skipped": len(result.skipped),
            "not_run": False, "tests_run": result.testsRun,
            "failures": [str(test) for test, _ in result.failures + result.errors]}


def main() -> int:
    parser = argparse.ArgumentParser(description="SpinQLabLink audit; offline is default and never connects")
    parser.add_argument("mode", nargs="?", default="offline", choices=("offline", "passive", "plan", "active", "replay"))
    parser.add_argument("--out", type=Path, default=Path("audit_offline"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--sdk-wheel", type=Path, help="optional downloaded 1.0.2 wheel for static inspection only")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--approved-plan", type=Path, help="approval JSON with plan_file and an exact copy of the plan")
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--max-experiments", type=int, default=0)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Výstupný adresár už obsahuje dáta; zvoľ nový názov, aby sa originály neprepísali.")
    out.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    mode_details: dict[str, Any] = {}
    plan = None
    tests = None
    exit_code = 0
    try:
        environment, capabilities, issues, experiments = discover(args.sdk_wheel)
        atomic_json(out / "environment.json", environment)
        atomic_json(out / "capabilities.json", capabilities)
        atomic_json(out / "issues.json", issues)
        atomic_json(out / "validated_schemas.json", {"fields": environment["spinqlablink"]["fields"],
                                                      "experiment_implementations": experiments,
                                                      "wire_chart_schema": {"source": "official SpinQTech message.proto:41-54 at pinned main commit",
                                                                            "task_id": "string", "group": "string",
                                                                            "chart_name": "string", "path": "optional string",
                                                                            "qubit": "optional string", "step": "optional string",
                                                                            "points.x": "protobuf float32",
                                                                            "points.y": "protobuf float32"}})
        atomic_json(out / "experiments.json", [])
        atomic_json(out / "state_snapshots.json", {})
        history_path = Path(__file__).parent / "user_reported_history.json"
        if history_path.exists():
            shutil.copyfile(history_path, out / "user_reported_history.json")
        if args.mode == "offline":
            tests = _offline_tests()
            atomic_json(out / "test_results.json", tests)
            if tests["failed"]:
                blockers.append("Niektoré offline testy zlyhali; audit nemožno považovať za zelený.")
                exit_code = 2
            if environment["spinqlablink"]["origin"] != "installed_distribution":
                blockers.append("Tento interpreter nemá nainštalovaný SpinQLabLink; lokálny Windows SDK musí overiť offline režim tam.")
        elif args.mode == "plan":
            config = _load_config(args.config)
            baseline = _load_json(args.baseline)
            plan = build_plan(config, baseline)
            atomic_json(out / "plan.json", plan)
            approval = {"approved": False, "operator": "", "approved_utc": "",
                        "plan_file": str(out / "plan.json"), "approved_plan_snapshot": plan}
            atomic_json(out / "approval_template.json", approval)
            blockers.extend(plan["blockers"])
        elif args.mode == "passive":
            config = _load_config(args.config)
            from .probes import passive
            mode_details = passive(config, out, args.duration)
            atomic_json(out / "passive_summary.json", mode_details)
        elif args.mode == "active":
            if not args.allow_hardware:
                raise RuntimeError("Chýba --allow-hardware.")
            if args.max_experiments <= 0:
                raise RuntimeError("Chýba nenulový --max-experiments.")
            approval = _load_json(args.approved_plan)
            if not approval.get("plan_file"):
                raise RuntimeError("Schvaľovací súbor nemá plan_file.")
            plan_file = Path(approval["plan_file"])
            if not plan_file.is_absolute() and args.approved_plan:
                plan_file = args.approved_plan.parent / plan_file
            plan = _load_json(plan_file)
            atomic_json(out / "approved_plan.json", {"plan": plan, "approval": approval})
            config = _load_config(args.config)
            baseline = _load_json(args.baseline)
            from .probes import active
            mode_details = active(config, baseline, plan, approval, out,
                                  allow_hardware=True, max_experiments=args.max_experiments)
            atomic_json(out / "active_summary.json", mode_details)
            atomic_json(out / "approved_plan.json", {"plan": plan, "approval": approval})
        elif args.mode == "replay":
            if not args.input:
                raise RuntimeError("Replay potrebuje --input.")
            source = args.input.resolve()
            if source == out:
                raise RuntimeError("Replay výstup musí byť iný adresár ako vstup.")
            if source.is_file() and source.suffix == ".json":
                count = import_legacy_result(source, out)
                mode_details = {"imported_legacy_charts": count,
                                "provenance": "SDK aggregated result, not pre-handler transport"}
            else:
                events = list(read_events(source))
                with (out / "events.jsonl").open("w", encoding="utf-8") as target:
                    for event in events:
                        target.write(json.dumps(redact(event), ensure_ascii=False, allow_nan=True) + "\n")
                mode_details = {"replayed_events": len(events), "source": "local saved events"}
            atomic_json(out / "replay_summary.json", mode_details)
    except (Exception, KeyboardInterrupt) as exc:
        blockers.append(f"{type(exc).__name__}: {redact(str(exc))}")
        exit_code = 3 if not isinstance(exc, KeyboardInterrupt) else 130
    finally:
        # An interrupted/blocked run still gets a valid, explicit partial report.
        if not (out / "events.jsonl").exists() and not (out / "events.jsonl.gz").exists():
            (out / "events.jsonl").write_text("", encoding="utf-8")
        if (out / "events.jsonl").exists() and not (out / "events.jsonl.gz").exists():
            with (out / "events.jsonl").open("rb") as source, gzip.open(out / "events.jsonl.gz", "wb") as target:
                target.write(source.read())
        events = []
        try:
            events = list(read_events(out))
            analysis = analyze_events(events, out)
        except Exception as exc:
            blockers.append(f"Analýza neúplná: {type(exc).__name__}")
            analysis = {"chart_count": 0, "fid_pairs": [], "issues": ["analysis_failed"], "raw_adc_confirmed": False}
            atomic_json(out / "analysis.json", analysis)
            atomic_json(out / "field_catalog.json", [])
            atomic_json(out / "original_scientific.json", {"charts": [], "incomplete": True})
            (out / "scientific_arrays.npz").write_bytes(b"")
            exit_code = max(exit_code, 3)
        if (out / "recorder_status.json").exists():
            status = _load_json(out / "recorder_status.json")
            if not status.get("complete"):
                blockers.append("Recorder stratil udalosti alebo zlyhal; dataset je neúplný.")
        else:
            atomic_json(out / "recorder_status.json", {"complete": True, "events_enqueued": len(events),
                                                        "events_dropped": 0, "writer_error": None})
        if not (out / "test_results.json").exists():
            atomic_json(out / "test_results.json", {"passed": 0, "failed": 0, "skipped": 0, "not_run": True})
        if not (out / "plan.json").exists():
            atomic_json(out / "plan.json", plan or {"status": "not_created", "tests": []})
        environment = _load_json(out / "environment.json") if (out / "environment.json").exists() else {
            "spinqlablink": {"origin": "unavailable", "version": None}, "device_model": {"value": "Gemini Lab", "evidence": "user_declared"}}
        capabilities = json.loads((out / "capabilities.json").read_text()) if (out / "capabilities.json").exists() else []
        issues = json.loads((out / "issues.json").read_text()) if (out / "issues.json").exists() else []
        records = _load_json(out / "experiments.json") if (out / "experiments.json").exists() else []
        if not isinstance(records, list):
            records = []
        analysis["attempted_experiments"] = sum(row.get("phase") != "not_submitted" for row in records)
        analysis["executed_experiments"] = sum(row.get("phase") in {"confirmed_finished", "confirmed_failed"} for row in records)
        analysis["planned_experiments"] = len((plan or {}).get("tests", []))
        analysis["connection_performed"] = (out / "connection_status.json").exists()
        compare_requested(analysis, records, plan)
        atomic_json(out / "analysis.json", analysis)
        summary = make_summary(args.mode, environment, capabilities, issues, analysis, blockers, tests)
        atomic_json(out / "summary.json", summary)
        build_report(out, summary, environment, capabilities, issues, analysis, plan)
        finalize_bundle(out)
        print(f"Audit {args.mode}: {out / 'REPORT.html'}")
        print(f"Offline testy: {summary['tests'].get('passed', 0)} OK, {summary['tests'].get('failed', 0)} zlyhalo; merania: {summary['hardware_experiments_executed']}; blokátory: {len(blockers)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
