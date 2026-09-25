"""Independent, fail-closed policy for proposed physical-layer payloads."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from .common import redact

WIRE_KEYS = {"compute_type", "relaxation_time", "stepList", "samplePath",
             "h_freShift", "p_freShift", "h_freDemo", "p_freDemo", "makePps",
             "sampleFre", "sampleCount", "sampleDelay", "pulse", "gradient"}
PULSE_KEYS = {"width", "am", "phase", "freshift"}
LIMIT_KEYS = {"max_pulse_amplitude_pct", "max_pulse_width_us", "max_total_rf_us",
              "max_sequence_us", "max_cumulative_rf_us", "min_relaxation_us",
              "max_sample_count", "max_sample_frequency", "max_sample_delay_us",
              "max_acquisitions_per_request", "min_inter_experiment_seconds",
              "min_temperature_c", "max_temperature_c"}


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_payload(params: dict[str, Any] | None, baseline: dict[str, Any],
                     config: dict[str, Any], *, planned_count: int = 1) -> list[str]:
    problems: list[str] = []
    if not isinstance(params, dict):
        return ["chýba presný serializovaný payload"]
    if set(params) != WIRE_KEYS:
        problems.append("payload má chýbajúce alebo neznáme kľúče")
    if params.get("compute_type") != 0:
        problems.append("povolený je iba schválený custom physical baseline (compute_type=0)")
    if params.get("gradient") != []:
        problems.append("gradient/shim nie je v tomto audite schválený")
    if params.get("stepList") != []:
        problems.append("neoverený stepList nie je povolený")
    if params.get("samplePath") not in (0, 1):
        problems.append("samplePath musí byť jeden schválený kanál")
    for key in ("h_freShift", "p_freShift", "h_freDemo", "p_freDemo"):
        if params.get(key) != 0:
            problems.append(f"{key}: aktívna zmena frekvencie nie je schválená")
    if params.get("makePps") is not True:
        problems.append("zmena prípravy stavu makePps nie je schválená")
    pulses = params.get("pulse")
    if not isinstance(pulses, dict) or set(pulses) != {"hPulse", "pPulse"}:
        problems.append("neznáma štruktúra pulzov")
        pulses = {"hPulse": [], "pPulse": []}
    all_pulses = []
    for channel in ("hPulse", "pPulse"):
        if not isinstance(pulses.get(channel), list):
            problems.append(f"{channel} nie je zoznam")
            continue
        for pulse in pulses[channel]:
            if not isinstance(pulse, dict) or set(pulse) != PULSE_KEYS:
                problems.append(f"{channel}: neznámy pulzový payload")
                continue
            all_pulses.append(pulse)
    if not all_pulses:
        problems.append("chýba schválený pulz")

    limits = config.get("limits", {})
    missing = sorted(k for k in LIMIT_KEYS if not _number(limits.get(k)))
    if missing:
        problems.append("chýbajú prevádzkové limity: " + ", ".join(missing))
    if not baseline.get("approved") or not baseline.get("workstation_verified"):
        problems.append("baseline nebol explicitne schválený a overený na pracovisku")
    if not baseline.get("approved_by") or not baseline.get("approved_utc") or not baseline.get("approval_evidence"):
        problems.append("chýba identita, čas alebo dôkaz schválenia baseline")
    if not config.get("limits_approved_by") or not config.get("limits_approved_utc") or not config.get("limits_evidence"):
        problems.append("chýba identita, čas alebo pôvod schválených prevádzkových limitov")
    units = baseline.get("verified_units", {})
    if (units.get("relaxation_time") != "us" or units.get("sampleFre") != "Hz" or
            units.get("temperature") != "C"):
        problems.append("jednotky relaxácie, sampleFre alebo teploty nie sú potvrdené pre toto pracovisko")
    acq = baseline.get("internal_acquisitions_upper_bound")
    if not _number(acq) or acq < 1:
        problems.append("neznámy skrytý počet akvizícií; chýba overený horný odhad")
    prep_rf = baseline.get("preparation_rf_upper_bound_us")
    if not _number(prep_rf) or prep_rf < 0:
        problems.append("neznáma RF záťaž automatickej prípravy stavu")
    for key in ("relaxation_time", "sampleFre", "sampleCount", "sampleDelay"):
        if not _number(params.get(key)):
            problems.append(f"{key} nie je konečné číslo")
    for pulse in all_pulses:
        for key in PULSE_KEYS:
            if not _number(pulse.get(key)):
                problems.append(f"pulzové {key} nie je konečné číslo")
    if problems:
        return problems

    for pulse in all_pulses:
        if pulse["am"] < 0 or pulse["am"] > limits["max_pulse_amplitude_pct"]:
            problems.append("amplitúda mimo schváleného limitu")
        if pulse["width"] < 0 or pulse["width"] > limits["max_pulse_width_us"]:
            problems.append("šírka pulzu mimo schváleného limitu")
        if pulse["freshift"] != 0:
            problems.append("detuning potrebuje samostatne schválený variant")
    rf_per_request = (sum(p["width"] for p in all_pulses) + prep_rf) * acq
    if rf_per_request > limits["max_total_rf_us"]:
        problems.append("RF čas jednej požiadavky prekračuje limit")
    if rf_per_request * planned_count > limits["max_cumulative_rf_us"]:
        problems.append("kumulatívny RF čas plánu prekračuje limit")
    if sum(p["width"] for p in all_pulses) + prep_rf > limits["max_sequence_us"]:
        problems.append("sekvencia prekračuje limit")
    if params["relaxation_time"] < limits["min_relaxation_us"]:
        problems.append("relaxačná prestávka je pod schváleným minimom")
    if params["sampleCount"] < 1 or params["sampleCount"] > limits["max_sample_count"]:
        problems.append("počet bodov mimo limitu")
    if params["sampleFre"] < 1 or params["sampleFre"] > limits["max_sample_frequency"]:
        problems.append("vzorkovacia frekvencia mimo limitu")
    if params["sampleDelay"] < 0 or params["sampleDelay"] > limits["max_sample_delay_us"]:
        problems.append("oneskorenie akvizície mimo limitu")
    if acq > limits["max_acquisitions_per_request"]:
        problems.append("počet interných akvizícií prekračuje limit")
    if planned_count > 1 and not (0 < limits["min_inter_experiment_seconds"] <= 3600):
        problems.append("medzi meraniami chýba schválená nenulová relaxačná prestávka")
    if limits["min_temperature_c"] >= limits["max_temperature_c"]:
        problems.append("neplatný schválený interval teploty")
    if pulses["hPulse"] and pulses["pPulse"] and not baseline.get("two_channel_timing_verified"):
        problems.append("simultánnosť H/P nie je overená pre tento baseline")
    if params["sampleFre"] % 10000 != 0:
        problems.append("sampleFre nezodpovedá deklarovanému kroku SDK 10000")
    return problems


def build_plan(config: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    params = baseline.get("params")
    params_wire = json.dumps(params) if isinstance(params, dict) else None
    count = baseline.get("repeat_count", 2)
    if type(count) is not int or not 0 <= count <= 3:
        count = 2
    proposed = [{"id": "baseline_lifecycle", "hypothesis": "FID a metadáta prídu kompletné",
                 "params": deepcopy(params), "params_wire_json": params_wire,
                 "expected_evidence": "started/chart boundaries/fidRe/fidIm/finished",
                 "success": "COMPLETED a kompletné párované FID",
                 "deadline_seconds": 120, "stop": "lost lock, stale status, overflow, timeout"}]
    for index in range(count):
        proposed.append({"id": f"baseline_repeat_{index+1}",
                         "hypothesis": "hrubá kontrola opakovateľnosti a driftu, nie štatistika šumu",
                         "params": deepcopy(params), "params_wire_json": params_wire,
                         "expected_evidence": "samostatný task a FID",
                         "success": "samostatne dokončené meranie", "deadline_seconds": 120,
                         "stop": "lost lock, stale status, overflow, timeout"})
    for variant in baseline.get("approved_variants", []):
        if isinstance(variant, dict) and variant.get("approved") is True:
            proposed.append({"id": str(variant.get("id", "variant")),
                             "hypothesis": str(variant.get("hypothesis", "one variable change")),
                             "params": deepcopy(variant.get("params")),
                             "params_wire_json": json.dumps(variant["params"]) if isinstance(variant.get("params"), dict) else None,
                             "expected_evidence": "FID and one parameter effect",
                             "success": "accepted and measurable effect; not guaranteed",
                             "deadline_seconds": 120,
                             "stop": "lost lock, stale status, overflow, timeout"})
    blockers = []
    for test in proposed:
        blockers.extend(f"{test['id']}: {issue}" for issue in validate_payload(
            test["params"], baseline, config, planned_count=len(proposed)))
    feedback = baseline.get("feedback")
    feedback_proposal = None
    if isinstance(feedback, dict):
        candidates = feedback.get("candidates", [])
        feedback_proposal = {"status": "prepared_for_offline_selection_only_not_live_executor",
                             "metric": "first_complex_fid_amplitude",
                             "threshold": feedback.get("threshold"),
                             "candidates": candidates,
                             "rule": "choose below_id when metric<threshold; otherwise above_id",
                             "below_id": feedback.get("below_id"), "above_id": feedback.get("above_id"),
                             "requires_separate_exact_plan_approval": True}
        for candidate in candidates:
            if isinstance(candidate, dict):
                blockers.extend(f"feedback {candidate.get('id')}: {issue}" for issue in validate_payload(
                    candidate.get("params"), baseline, config, planned_count=len(proposed) + 1))
        if not _number(feedback.get("threshold")):
            blockers.append("feedback: prah lokálnej metriky chýba")
    return {"schema": 1, "config_snapshot": redact(deepcopy(config)),
            "baseline_snapshot": redact(deepcopy(baseline)),
            "baseline_id": baseline.get("id"), "tests": proposed,
            "experiment_budget_requested": len(proposed), "blockers": sorted(set(blockers)),
            "status": "ready_for_explicit_approval" if not blockers else "blocked_safety",
            "optional_feedback_proposal": feedback_proposal,
            "payload_semantics": "full physical params serialized exactly as SDK get_parameters()"}


def select_feedback_candidate(metric: float, proposal: dict[str, Any]) -> dict[str, Any]:
    """Pure offline selector. It never sends or expands a payload."""
    if not _number(metric) or not _number(proposal.get("threshold")):
        raise ValueError("Feedback metric/threshold must be finite")
    selected_id = proposal.get("below_id") if metric < proposal["threshold"] else proposal.get("above_id")
    candidates = [item for item in proposal.get("candidates", []) if item.get("id") == selected_id]
    if len(candidates) != 1:
        raise ValueError("Selected candidate is not uniquely preapproved")
    return candidates[0]


def verify_approval(approval: dict[str, Any], plan: dict[str, Any],
                    config: dict[str, Any], baseline: dict[str, Any], max_experiments: int) -> None:
    if approval.get("approved") is not True or not approval.get("operator") or not approval.get("approved_utc"):
        raise RuntimeError("Plán nebol explicitne schválený operátorom.")
    if approval.get("approved_plan_snapshot") != plan:
        raise RuntimeError("Plán sa po schválení zmenil.")
    if plan.get("config_snapshot") != redact(config) or plan.get("baseline_snapshot") != redact(baseline):
        raise RuntimeError("Konfigurácia alebo baseline sa po schválení zmenili.")
    if plan.get("blockers") or plan.get("status") != "ready_for_explicit_approval":
        raise RuntimeError("Plán obsahuje bezpečnostný blokátor.")
    if max_experiments <= 0 or len(plan.get("tests", [])) > max_experiments:
        raise RuntimeError("Nedostatočný nenulový rozpočet experimentov.")
    if config.get("active_enabled") is not True:
        raise RuntimeError("Aktívny režim je v konfigurácii vypnutý.")
    for test in plan["tests"]:
        issues = validate_payload(test["params"], baseline, config, planned_count=len(plan["tests"]))
        if issues:
            raise RuntimeError(f"Neplatný payload {test['id']}: {', '.join(issues)}")
        if test.get("params_wire_json") != json.dumps(test["params"]):
            raise RuntimeError(f"Serialized params were changed for {test['id']}.")


class HardwareLock:
    def __init__(self, path: Path):
        self.path = path.expanduser()
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(f"Lokálny hardvérový zámok už existuje: {self.path}") from exc
        os.write(self.fd, json.dumps({"pid": os.getpid()}).encode())
        return self

    def __exit__(self, *_):
        if self.fd is not None:
            os.close(self.fd)
            self.path.unlink(missing_ok=True)
