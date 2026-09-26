"""One-run, real Gemini Lab measurement suite through the installed SpinQLabLink.

This program never simulates a measurement. It records decoded server events
before the vendor experiment handlers simplify or replace chart data.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import sys
import time
import tomllib
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spinq_audit.adapter import AuditAdapter, verify_installed_sdk
from spinq_audit.analysis import _fft, analyze_events, field_catalog
from spinq_audit.common import atomic_bytes, atomic_json, redact, utc_now
from spinq_audit.discovery import discover
from spinq_audit.probes import _configure_physical, wait_terminal
from spinq_audit.recorder import EventRecorder
from spinq_audit.safety import HardwareLock
from publish_results import publish_results

REQUIRED_LIMITS = (
    "max_pulse_amplitude_pct", "max_single_pulse_width_us", "max_requested_rf_us_per_task",
    "max_cumulative_requested_rf_us", "min_relaxation_value", "max_relaxation_value", "max_sample_count",
    "max_sample_frequency_hz", "max_sample_delay_us", "max_abs_detuning_hz",
    "max_abs_frequency_shift", "max_abs_demod_shift", "min_temperature_c", "max_temperature_c",
)

# Exact physical-layer request that the operator already ran successfully on
# this Gemini Lab. This is an observation, not a certified operating envelope.
HISTORICAL_PHYSICAL_BASELINE: dict[str, Any] = {
    "compute_type": 0, "relaxation_time": 15.0, "stepList": [],
    "samplePath": 0, "h_freShift": 0, "p_freShift": 0,
    "h_freDemo": 0, "p_freDemo": 0, "makePps": True,
    "sampleFre": 10000, "sampleCount": 16000, "sampleDelay": 0,
    "pulse": {"hPulse": [{"width": 40.0, "am": 100.0,
                          "phase": 90.0, "freshift": 0.0}], "pPulse": []},
    "gradient": [],
}

# Research envelope for this one finite study, based on the operator's completed
# 40-200 µs/100% experiments and small proposed changes around the 40 µs point.
# These are TEST PLAN bounds, not a manufacturer's hardware safety ratings.
STUDY_MAX_REQUESTED_RF_US = 200.0
STUDY_MAX_CUMULATIVE_REQUESTED_RF_US = 6000.0
PREVIOUSLY_COMPLETED_RABI_WIDTHS_US = (40, 80, 120, 160, 200)


def number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def safe_metadata(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_metadata(item) for item in value]
    return value


def metadata_candidates(value: Any, path: str = "") -> list[dict[str, Any]]:
    candidates = []
    if isinstance(value, dict):
        for key, nested in value.items():
            child = f"{path}.{key}" if path else str(key)
            if any(name in str(key).lower() for name in ("model", "firmware", "serverversion", "deviceversion")):
                candidates.append({"path": child, "as_received": redact(nested),
                                   "interpretation": "candidate only; field meaning not confirmed"})
            if isinstance(nested, (dict, list)):
                candidates.extend(metadata_candidates(nested, child))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            if isinstance(nested, (dict, list)):
                candidates.extend(metadata_candidates(nested, f"{path}[{index}]"))
    return candidates


def load_config(path: Path) -> dict[str, Any]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Konfigurácia musí byť TOML tabuľka.")
    return config


def physical_baseline(config: dict[str, Any]) -> dict[str, Any]:
    b = config.get("baseline", {})
    if b.get("sample_path", 0) != 0:
        raise ValueError("H baseline vyžaduje sample_path=0; P má samostatný pracovný bod")
    return {
        "compute_type": 0, "relaxation_time": b.get("relaxation_delay_value"), "stepList": [],
        "samplePath": b.get("sample_path", 0), "h_freShift": 0, "p_freShift": 0,
        "h_freDemo": 0, "p_freDemo": 0, "makePps": b.get("make_pps"),
        "sampleFre": b.get("sample_frequency_hz"), "sampleCount": b.get("sample_count"),
        "sampleDelay": b.get("sample_delay_us"),
        "pulse": {"hPulse": [{"width": b.get("h_width_us"), "am": b.get("h_amplitude_pct"),
                              "phase": b.get("h_phase_deg"), "freshift": b.get("h_detuning_hz", 0)}],
                  "pPulse": []},
        "gradient": [],
    }


def make_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    base = physical_baseline(config)
    cases = [{"id": "nmr_basic", "kind": "nmr", "params": copy.deepcopy(base),
              "purpose": "základný NMR signál cez oficiálny typ"},
             {"id": "physical_baseline", "kind": "physical", "params": copy.deepcopy(base),
              "purpose": "kompletný FID fyzikálnej vrstvy"}]
    count = config.get("operation", {}).get("repeat_count", 20)
    if type(count) is not int or not 0 <= count <= 30:
        raise ValueError("repeat_count musí byť celé číslo 0..30")
    for index in range(count):
        cases.append({"id": f"repeat_{index+1:02d}", "kind": "physical", "params": copy.deepcopy(base),
                      "purpose": "identický baseline pre variabilitu a drift"})

    deltas = config.get("deltas", {})
    features = config.get("features", {})
    variations: list[tuple[str, dict[str, Any], bool, str]] = []

    def changed(name: str, change: Any, *, enabled: bool = True, reason: str = "") -> None:
        params = copy.deepcopy(base)
        change(params)
        variations.append((name, params, enabled, reason))

    hp = lambda p: p["pulse"]["hPulse"][0]
    for key, name, updater in (
        ("amplitude_pct", "pulse_amplitude", lambda p, d: hp(p).__setitem__("am", hp(p)["am"]-d)),
        ("phase_deg", "pulse_phase", lambda p, d: hp(p).__setitem__("phase", (hp(p)["phase"]+d)%360)),
        ("width_us", "pulse_width", lambda p, d: hp(p).__setitem__("width", hp(p)["width"]+d)),
        ("detuning_hz", "pulse_detuning", lambda p, d: hp(p).__setitem__("freshift", hp(p)["freshift"]+d)),
    ):
        delta = deltas.get(key)
        if number(delta) and delta > 0:
            changed(name, lambda p, d=delta, f=updater: f(p, d))
        else:
            changed(name, lambda p: None, enabled=False, reason=f"chýba kladná deltas.{key}")

    for relative in (90, 180, 270):
        changed(f"phase_relative_{relative}",
                lambda p, angle=relative: hp(p).__setitem__("phase", (hp(p)["phase"]+angle)%360))

    if features.get("multi_segment_verified") is True:
        def split_pulse(p: dict[str, Any]) -> None:
            first = p["pulse"]["hPulse"][0]
            second = copy.deepcopy(first)
            first["width"] = first["width"] / 2
            second["width"] = second["width"] / 2
            p["pulse"]["hPulse"].append(second)
        changed("multi_segment_h", split_pulse)
    else:
        changed("multi_segment_h", lambda p: None, enabled=False,
                reason="poradie a význam segmentov neboli potvrdené na pracovisku")

    for key, field, name, feature in (
        ("frequency_shift", "h_freShift", "h_frequency_shift", "frequency_shift_verified"),
        ("demod_shift", "h_freDemo", "h_demod_shift", "demod_shift_verified"),
        ("sample_frequency_hz", "sampleFre", "sample_frequency", "sampling_variants_verified"),
        ("sample_count", "sampleCount", "sample_count", "sampling_variants_verified"),
        ("sample_delay_us", "sampleDelay", "sample_delay", "sampling_variants_verified"),
        ("relaxation_delay_value", "relaxation_time", "relaxation_delay", "relaxation_variant_verified"),
    ):
        delta = deltas.get(key)
        if features.get(feature) is not True:
            changed(name, lambda p: None, enabled=False, reason=f"features.{feature} nie je potvrdené")
        elif not number(delta) or delta <= 0:
            changed(name, lambda p: None, enabled=False, reason=f"chýba kladná deltas.{key}")
        else:
            sign = -1 if field == "sampleCount" else 1
            changed(name, lambda p, k=field, d=delta, s=sign: p.__setitem__(k, p[k]+s*d))

    if features.get("state_initialization_off_verified") is True:
        changed("state_initialization_off", lambda p: p.__setitem__("makePps", False))
    else:
        changed("state_initialization_off", lambda p: None, enabled=False,
                reason="neznáme vnútorné prípravné pulzy a význam vypnutia makePps")
    pbase = config.get("phosphorus_baseline", {})
    if features.get("p_channel_verified") is True and all(number(pbase.get(k)) for k in
            ("width_us", "amplitude_pct", "phase_deg")):
        def set_p(p: dict[str, Any]) -> None:
            p["samplePath"] = 1
            p["pulse"]["hPulse"] = []
            p["pulse"]["pPulse"] = [{"width": pbase["width_us"], "am": pbase["amplitude_pct"],
                                       "phase": pbase["phase_deg"], "freshift": pbase.get("detuning_hz", 0)}]
        changed("p_channel_path", set_p)
    else:
        changed("p_channel_path", lambda p: None, enabled=False,
                reason="P kanál nemá overený pracovný bod")
    if features.get("no_rf_verified") is True and features.get("state_initialization_off_verified") is True:
        def no_rf(p: dict[str, Any]) -> None:
            p["pulse"] = {"hPulse": [], "pPulse": []}
            p["makePps"] = False
        changed("no_rf_candidate", no_rf)
    else:
        changed("no_rf_candidate", lambda p: None, enabled=False,
                reason="nulové RF vyžaduje overený postup bez skrytej prípravy")

    for name, params, enabled, reason in variations:
        cases.append({"id": name, "kind": "physical", "params": params,
                      "purpose": "jedna zmena voči baseline", "enabled": enabled, "skip_reason": reason,
                      "comparison": "baseline / change / return"})
        if enabled:
            cases.append({"id": name+"_return", "kind": "physical", "params": copy.deepcopy(base),
                          "purpose": "nezávislý návrat na baseline po zmene", "return_for": name})

    if features.get("frequency_scan_enabled") is True:
        for detuning in (-20, -10, 0, 10, 20):
            params = copy.deepcopy(base)
            hp(params)["freshift"] = detuning
            label = f"frequency_scan_{'m' if detuning < 0 else 'p'}{abs(detuning)}"
            cases.append({"id": label, "kind": "physical", "params": params,
                          "purpose": "päťbodová odozva na malé rozladenie H pulzu"})
        cases.append({"id": "frequency_scan_return", "kind": "physical", "params": copy.deepcopy(base),
                      "return_for": "frequency_scan_p20", "purpose": "návrat po frekvenčnom skene"})

    if features.get("rabi_scan_verified") is True:
        for label, width in zip(("low2", "low", "center", "high", "high2"),
                                PREVIOUSLY_COMPLETED_RABI_WIDTHS_US):
            params = copy.deepcopy(base)
            hp(params)["width"] = width
            cases.append({"id": "rabi_"+label, "kind": "rabi", "params": params,
                          "purpose": "krátky Rabi sken; výber ďalšieho bodu lokálne"})
    else:
        cases.append({"id": "rabi_scan", "kind": "rabi", "params": copy.deepcopy(base),
                      "enabled": False, "skip_reason": "Rabi rozsah nebol potvrdený"})
    return cases


def check_case(case: dict[str, Any], config: dict[str, Any], cumulative_requested_rf: float) -> float:
    if config.get("baseline_verified") is not True:
        raise ValueError("baseline_verified nie je true; známy pracovný bod nebol potvrdený")
    if config.get("historical_baseline_only") is True:
        if (case["id"] != "physical_baseline" or case["kind"] != "physical" or
                case["params"] != HISTORICAL_PHYSICAL_BASELINE or cumulative_requested_rf != 0):
            raise ValueError("bez potvrdených limitov je povolený iba jeden presný historický fyzikálny baseline")
        return 40.0
    if config.get("bounded_study_enabled") is True:
        return check_bounded_study_case(case, config, cumulative_requested_rf)
    limits = config.get("limits", {})
    missing = [key for key in REQUIRED_LIMITS if not number(limits.get(key))]
    if missing:
        raise ValueError("chýbajú prevádzkové limity: " + ", ".join(missing))
    p = case["params"]
    if p["compute_type"] != 0 or p["gradient"] != [] or p["stepList"] != []:
        raise ValueError("nepovolený režim, gradient alebo stepList")
    if p["samplePath"] not in (0, 1):
        raise ValueError("neznámy akvizičný kanál")
    pulses = p["pulse"]["hPulse"] + p["pulse"]["pPulse"]
    if not pulses and case["id"] != "no_rf_candidate":
        raise ValueError("chýba overený pulz")
    for pulse in pulses:
        if not all(number(pulse.get(k)) for k in ("width", "am", "phase", "freshift")):
            raise ValueError("pulz má chýbajúcu alebo nekonečnú hodnotu")
        if not (0 <= pulse["phase"] <= 360 and
                0 <= pulse["am"] <= limits["max_pulse_amplitude_pct"] and
                0 <= pulse["width"] <= limits["max_single_pulse_width_us"] and
                abs(pulse["freshift"]) <= limits["max_abs_detuning_hz"]):
            raise ValueError("pulz prekračuje potvrdený limit")
    requested_rf = sum(pulse["width"] for pulse in pulses)
    if requested_rf > limits["max_requested_rf_us_per_task"]:
        raise ValueError("RF čas požiadavky prekračuje limit")
    if cumulative_requested_rf + requested_rf > limits["max_cumulative_requested_rf_us"]:
        raise ValueError("kumulatívny požadovaný RF čas prekračuje limit")
    if type(p["makePps"]) is not bool:
        raise ValueError("makePps musí byť boolean")
    if type(p["sampleFre"]) is not int or type(p["sampleCount"]) is not int or type(p["sampleDelay"]) is not int:
        raise ValueError("sampleFre/sampleCount/sampleDelay musia byť celé čísla")
    if not all(number(p.get(key)) for key in ("relaxation_time", "sampleFre", "sampleCount", "sampleDelay")):
        raise ValueError("akvizičné hodnoty chýbajú alebo nie sú konečné")
    if (not limits["min_relaxation_value"] <= p["relaxation_time"] <= limits["max_relaxation_value"] or
            not 1 <= p["sampleCount"] <= limits["max_sample_count"] or
            not 1 <= p["sampleFre"] <= limits["max_sample_frequency_hz"] or
            not 0 <= p["sampleDelay"] <= limits["max_sample_delay_us"] or
            p["sampleFre"] % 10000 != 0):
        raise ValueError("akvizícia alebo relaxácia je mimo potvrdeného limitu")
    if abs(p["h_freShift"]) > limits["max_abs_frequency_shift"] or abs(p["p_freShift"]) > limits["max_abs_frequency_shift"]:
        raise ValueError("frekvenčný posun mimo limitu")
    if abs(p["h_freDemo"]) > limits["max_abs_demod_shift"] or abs(p["p_freDemo"]) > limits["max_abs_demod_shift"]:
        raise ValueError("demodulačný posun mimo limitu")
    if limits["min_temperature_c"] >= limits["max_temperature_c"]:
        raise ValueError("neplatný interval teploty")
    return requested_rf


def check_bounded_study_case(case: dict[str, Any], config: dict[str, Any],
                             cumulative_requested_rf: float) -> float:
    """Guard a finite research plan, without representing its bounds as device limits."""
    p = case["params"]
    if case["kind"] not in {"nmr", "physical", "rabi", "shape"}:
        raise ValueError("neznámy typ študijného experimentu")
    p_variants = config.get("_runtime_p_variants", {})
    if case["id"] in p_variants:
        expected_p = p_variants[case["id"]]
        if (case["kind"] != "physical" or p != expected_p or
                cumulative_requested_rf + expected_p["pulse"]["pPulse"][0]["width"] >
                STUDY_MAX_CUMULATIVE_REQUESTED_RF_US):
            raise ValueError("P pokus nie je presne odvodený z čerstvých kalibračných údajov")
        return expected_p["pulse"]["pPulse"][0]["width"]
    if (set(p) != set(HISTORICAL_PHYSICAL_BASELINE) or
            p["compute_type"] != 0 or p["stepList"] != [] or p["gradient"] != [] or
            p["samplePath"] != 0 or p["makePps"] is not True or
            p["relaxation_time"] != 15 or
            p["p_freShift"] != 0 or p["p_freDemo"] != 0 or
            p["h_freShift"] not in (0, 10) or p["h_freDemo"] not in (0, 10) or
            type(p["sampleFre"]) is not int or p["sampleFre"] not in (10000, 20000) or
            type(p["sampleCount"]) is not int or p["sampleCount"] not in (15000, 16000) or
            type(p["sampleDelay"]) is not int or p["sampleDelay"] not in (0, 100)):
        raise ValueError("požiadavka je mimo obmedzeného výskumného plánu")
    pulse_container = p["pulse"]
    if set(pulse_container) != {"hPulse", "pPulse"} or pulse_container["pPulse"] != []:
        raise ValueError("P kanál nemá potvrdený pracovný bod")
    pulses = pulse_container["hPulse"]
    if case["kind"] == "shape":
        expected = [100*math.exp(-((10*i-20)**2)/(2*10**2)) for i in range(4)]
        if (case["id"] != "shape_gaussian_h" or len(pulses) != 4 or
                any(set(pulse) != {"width", "am", "phase", "freshift"} or
                    not all(number(value) for value in pulse.values()) or
                    pulse["width"] != 10 or pulse["phase"] != 90 or
                    pulse["freshift"] != 0 or abs(pulse["am"]-expected[i]) > 1e-6
                    for i, pulse in enumerate(pulses))):
            raise ValueError("tvarovaný pulz nie je presný štvordielny plán")
        if cumulative_requested_rf + 40 > STUDY_MAX_CUMULATIVE_REQUESTED_RF_US:
            raise ValueError("vyčerpaný softvérový RF rozpočet")
        return 40.0
    if not isinstance(pulses, list) or not 1 <= len(pulses) <= 2:
        raise ValueError("výskumný plán povoľuje jeden alebo dva H pulzy")
    for pulse in pulses:
        if set(pulse) != {"width", "am", "phase", "freshift"} or not all(number(v) for v in pulse.values()):
            raise ValueError("pulz má neznáme alebo neplatné polia")
        allowed_width = (pulse["width"] in PREVIOUSLY_COMPLETED_RABI_WIDTHS_US
                         if case["kind"] == "rabi" else 20 <= pulse["width"] <= 44)
        if (not allowed_width or
                pulse["am"] not in (98, 100) or pulse["phase"] not in (0, 90, 95, 180, 270) or
                pulse["freshift"] not in (-20, -10, 0, 10, 20)):
            raise ValueError("pulz je mimo malých plánovaných zmien")
    requested_rf = sum(pulse["width"] for pulse in pulses)
    if requested_rf > STUDY_MAX_REQUESTED_RF_US or cumulative_requested_rf + requested_rf > STUDY_MAX_CUMULATIVE_REQUESTED_RF_US:
        raise ValueError("vyčerpaný softvérový rozpočet RF požiadaviek tejto série")
    return requested_rf


def device_snapshot(device: Any, latest: dict[str, tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    """Read SDK accessors while marking whether their source arrived this session."""
    origin = {"params": "s_post_device_param", "lock": "s_post_lock_data",
              "status": "s_post_device_info"}
    getters = {
        "to_dict": ("mixed", device.to_dict),
        "params.to_dict": ("params", device.params.to_dict),
        "get_frequencies": ("lock", device.get_frequencies),
        "get_temperature": ("status", device.get_temperature),
        "is_locked": ("status", device.is_locked),
        "get_pulse_amplitudes": ("params", device.get_pulse_amplitudes),
        "get_pulse_phases": ("params", device.get_pulse_phases),
        "get_qubit_params": ("params", device.get_qubit_params),
        "get_shimming_values": ("params", device.get_shimming_values),
        "lock_data": ("lock", lambda: vars(device.lock_data)),
        "status": ("status", lambda: vars(device.status)),
        "params.pulse_param": ("params", lambda: device.params.pulse_param),
        "params.pps_param": ("params", lambda: device.params.pps_param),
        "params.sample_param": ("params", lambda: device.params.sample_param),
        "params.shimming_param": ("params", lambda: device.params.shimming_param),
    }
    values = {}
    for name, (family, getter) in getters.items():
        required = list(origin.values()) if family == "mixed" else [origin[family]]
        observed = [key for key in required if key in latest]
        try:
            value = redact(getter())
            error = None
        except Exception as exc:
            value, error = None, str(redact(str(exc)))
        values[name] = {"value": value, "error": error,
                        "source_messages_this_session": observed,
                        "provenance": "server_update_seen" if len(observed) == len(required)
                        else "cached_or_sdk_default_possible"}
    return {"captured_utc": utc_now(), "device_id_candidate": redact(device.device_id),
            "device_type_candidate": redact(device.device_type), "accessors": values}


def append_p_channel_cases(cases: list[dict[str, Any]], config: dict[str, Any],
                           device: Any, latest: dict[str, tuple[int, dict[str, Any]]]) -> str:
    """Use only explicit fresh P calibration fields; otherwise document a skip."""
    if "s_post_device_param" not in latest or "s_post_lock_data" not in latest:
        return "P: v tejto relácii neprišli kalibračné parametre aj frekvencie"
    params = device.params.to_dict()
    pulse = params.get("pulseParam", {})
    pps = params.get("ppsParam", {})
    if not isinstance(pulse, dict) or not isinstance(pps, dict):
        return "P: kalibračné skupiny pulseParam/ppsParam nemajú očakávaný tvar"
    required = ((pulse, "am_Q2"), (pulse, "phaseQ2_0"), (pps, "width_Q2"))
    if any(key not in group or not number(group[key]) for group, key in required):
        return "P: chýba explicitná amplitúda, fáza alebo šírka v čerstvej kalibrácii"
    amplitude, phase, width = float(pulse["am_Q2"]), float(pulse["phaseQ2_0"]), float(pps["width_Q2"])
    frequency = device.get_frequencies().get("P")
    if not (0 < amplitude <= 100 and 0 <= phase <= 360 and 1 <= width <= 44 and
            number(frequency) and frequency > 0):
        return "P: kalibračné hodnoty nie sú konečné alebo sú mimo úzkeho plánu; jednotky/rozsah nemožno potvrdiť"
    base = physical_baseline(config)
    base["samplePath"] = 1
    base["pulse"] = {"hPulse": [], "pPulse": [{"width": width, "am": amplitude,
                                              "phase": phase, "freshift": 0.0}]}
    config["_runtime_p_baseline"] = base
    config["_runtime_p_variants"] = {"p_baseline": copy.deepcopy(base)}
    cases.append({"id": "p_baseline", "kind": "physical", "params": copy.deepcopy(base),
                  "purpose": "P pracovný bod odvodený z čerstvých parametrov; účinok sa ešte len overí"})
    for index in range(config["operation"]["repeat_count"]):
        repeat_id = f"p_repeat_{index+1:02d}"
        config["_runtime_p_variants"][repeat_id] = copy.deepcopy(base)
        cases.append({"id": repeat_id, "kind": "physical",
                      "params": copy.deepcopy(base), "requires_verified": "p_baseline",
                      "purpose": "identický P experiment pre drift a denoising"})
    candidates = []
    for label, field, value in (
            ("p_amplitude", "am", amplitude-2),
            ("p_phase", "phase", (phase+5)%360),
            ("p_width", "width", width+2),
            ("p_detuning", "freshift", 10),
            ("p_phase_relative_90", "phase", (phase+90)%360),
            ("p_phase_relative_180", "phase", (phase+180)%360),
            ("p_phase_relative_270", "phase", (phase+270)%360)):
        changed = copy.deepcopy(base)
        changed["pulse"]["pPulse"][0][field] = value
        valid = (0 < changed["pulse"]["pPulse"][0]["am"] <= 100 and
                 1 <= changed["pulse"]["pPulse"][0]["width"] <= 44)
        candidates.append((label, changed, valid))
    for label, field in (("p_frequency_shift", "p_freShift"),
                         ("p_demod_shift", "p_freDemo")):
        changed = copy.deepcopy(base)
        changed[field] = 10
        candidates.append((label, changed, True))
    for label, changed, valid in candidates:
        cases.append({"id": label, "kind": "physical", "params": changed,
                      "enabled": valid, "skip_reason": "zmena presahuje úzky P plán",
                      "requires_verified": "p_baseline", "purpose": "jedna malá P zmena voči baseline"})
        if valid:
            config["_runtime_p_variants"][label] = copy.deepcopy(changed)
            return_id = label+"_return"
            config["_runtime_p_variants"][return_id] = copy.deepcopy(base)
            cases.append({"id": return_id, "kind": "physical", "params": copy.deepcopy(base),
                          "return_for": label, "requires_verified": "p_baseline",
                          "purpose": "návrat na čerstvý P pracovný bod"})
    return "P: kandidát pracovného bodu odvodený z čerstvej telemetrie; nepovažovať za bezpečnostnú certifikáciu"


def append_frequency_cases(cases: list[dict[str, Any]], config: dict[str, Any],
                           device: Any, latest: dict[str, tuple[int, dict[str, Any]]]) -> str:
    if "s_post_lock_data" not in latest:
        return "custom_freq: chýbajú čerstvé H/P frekvencie"
    frequencies = device.get_frequencies()
    h, p = frequencies.get("H"), frequencies.get("P")
    if not (number(h) and number(p) and 1e6 < h < 100e6 and 1e6 < p < 100e6):
        return "custom_freq: H/P frekvencie nie sú v rozsahu vyžadovanom SDK"
    config["_runtime_frequency_hz"] = {"H": h, "P": p}
    base = physical_baseline(config)
    cases.append({"id": "nmr_custom_frequency", "kind": "nmr", "params": copy.deepcopy(base),
                  "custom_frequency_hz": {"H": h, "P": p},
                  "purpose": "výslovné použitie čerstvej aktuálnej frekvencie bez rozladenia"})
    cases.append({"id": "nmr_custom_frequency_return", "kind": "nmr",
                  "params": copy.deepcopy(base), "return_for": "nmr_custom_frequency",
                  "purpose": "návrat na automatický výber frekvencie"})
    return "custom_freq: pripravené z čerstvých frekvencií zariadenia"


def append_shape_case(cases: list[dict[str, Any]], config: dict[str, Any],
                      device: Any, latest: dict[str, tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    """Check SDK waveform math locally, then schedule one small real shape test."""
    try:
        from spinqlablink import WaveformGenerator
        generated = WaveformGenerator.generate(0, WaveformGenerator.GAUSSIAN,
                                               4, 40, 90, 100, 0)
        if len(generated) != 4:
            raise ValueError("generátor nevrátil štyri segmenty")
        expected = [100*math.exp(-((10*i-20)**2)/(2*10**2)) for i in range(4)]
        observed = [pulse.amplitude for pulse in generated]
        if any(abs(a-b) > 1e-6 for a, b in zip(expected, observed)):
            raise ValueError("generovaný tvar nesúhlasí s lokálnou Gaussovou funkciou")
        base = physical_baseline(config)
        base["pulse"]["hPulse"] = [pulse.to_dict() for pulse in generated]
        sample = device.params.sample_param if "s_post_device_param" in latest else {}
        sample_kind = next((sample[key] for key in ("calibrate_sample", "calibrateSample", "sampleType", "sample_type")
                            if isinstance(sample, dict) and key in sample), None)
        if sample_kind not in (0, "0", "CH3PO(CH2CH3)2"):
            cases.append({"id": "shape_gaussian_h", "kind": "shape", "params": base,
                          "enabled": False, "skip_reason": "typ vzorky pre SHAPE_PULSE calibrate_sample=0 nebol čerstvo potvrdený"})
            return {"status": "local_math_checked_live_skipped", "expected_amplitudes": expected,
                    "sdk_amplitudes": observed, "sample_kind_received": sample_kind,
                    "physical_rf_output_measured": False}
        cases.append({"id": "shape_gaussian_h", "kind": "shape", "params": base,
                      "purpose": "živé meranie štvordielneho H tvaru s celkovou šírkou 40 µs"})
        return {"status": "local_math_checked_not_rf_output", "expected_amplitudes": expected,
                "sdk_amplitudes": observed, "segment_width_us": 10}
    except Exception as exc:
        cases.append({"id": "shape_gaussian_h", "kind": "shape", "params": physical_baseline(config),
                      "enabled": False, "skip_reason": "WaveformGenerator zlyhal: "+str(redact(str(exc)))})
        return {"status": "NEOVERENÉ", "reason": str(redact(str(exc)))}


SKIPPED_CAPABILITIES: tuple[tuple[str, str, str], ...] = (
    ("pps_custom", "QUANTUM_SYSTEM_INITIALIZATION / using_custom_pps / pps_json",
     "čerstvá telemetria nedodáva úplnú overenú PPS sekvenciu; vlastný JSON by bol odhad"),
    ("pauli_type1", "PHYSICAL_LAYER_EXPERIMENT(type_setting=1, stepList)",
     "vnútorné prípravné/čítacie pulzy pre Pauliho krok nemajú známy RF rozpočet"),
    ("physical_tomography_type2", "PHYSICAL_LAYER_EXPERIMENT(type_setting=2)",
     "vnútorné kroky tomografie a ich RF rozpočet nie sú z klienta obmedziteľné"),
    ("state_tomography", "QUANTUM_STATE_TOMOGRAPHY",
     "nie je doložená bezpečná vlastná PPS a počet vnútorných meraní"),
    ("quantum_gates", "QUANTUM_GATES_AND_CIRCUIT / Circuit / Gate / CustomGate",
     "vnútorná príprava a mapovanie vlastnej brány na pulzy nie sú potvrdené"),
    ("circuit_layer", "CIRCUIT_LAYER_EXPERIMENT",
     "mapovanie obvodu a prípravných pulzov na hardvér nie je potvrdené"),
    ("numerical_optimization", "NUMERICAL_OPTIMIZATION_PULSE",
     "počet interných optimalizačných meraní nie je cez SDK obmedziteľný"),
    ("t1", "QUANTUM_DECOHERENCE_T1", "SDK neposkytuje voľbu jednotlivých oneskorení ani počet vnútorných bodov"),
    ("t2", "QUANTUM_DECOHERENCE_T2", "SDK neposkytuje voľbu jednotlivých oneskorení ani počet vnútorných bodov"),
    ("spin_echo", "SPIN_ECHO", "zoznam pulzov nešpecifikuje overené echo časovanie"),
    ("dynamic_decoupling", "DYNAMIC_DECOUPLING", "sekvencia a jej vnútorné opakovania nie sú potvrdené"),
    ("gradient_write", "Gradient / append_gradient", "mapovanie cievok a povolené napätie nemáme doložené"),
    ("shim_write", "set_device_params / shimmingParam", "lokálny setter nie je doložený hardvérový zápis; trvalé nastavenie sa nemení"),
    ("receiver_adc", "SpinQLabLink / Device", "v skúmanom SDK nie je doložený priamy ADC stream ani nastavenie prijímacieho zisku"),
    ("server_fft_disable", "SpinQLabLink / experiment parameters", "SDK neponúka overený prepínač serverovej FFT alebo fitovania"),
    ("sample_path_both", "PHYSICAL_LAYER_EXPERIMENT(samplePath=-1)",
     "súčasné časovanie H/P a význam oboch kanálov pri type_setting=0 nie sú potvrdené"),
    ("task_abort", "SpinQLabLink / deregister_experiment", "SDK nepreukazuje fyzické zastavenie úlohy; pri timeoute sa ďalšia neposiela"),
    ("realtime_feedback", "SpinQLabLink / experiment events", "SDK sprístupňuje priebežné udalosti, nie doloženú spätnú väzbu meniacu bežiacu úlohu"),
)


def wait_recorded(recorder: EventRecorder, seconds: float = 8) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and recorder.queue.unfinished_tasks:
        if not recorder.status()["complete"]:
            raise RuntimeError("záznamník stratil údaje")
        time.sleep(.02)
    if recorder.queue.unfinished_tasks:
        raise TimeoutError("záznamník nedokončil zápis včas")


def read_new_events(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    events = []
    with path.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        for line in stream:
            if line.strip():
                events.append(json.loads(line))
        return events, stream.tell()


def scientific_metrics(analysis: dict[str, Any], scientific_path: Path) -> dict[str, Any]:
    pairs = [p for p in analysis.get("fid_pairs", []) if p.get("status") == "paired"]
    if not pairs:
        return {"fid_status": "unpaired_or_absent", "raw_adc_confirmed": False}
    pair = pairs[0]
    data = json.loads(scientific_path.read_text(encoding="utf-8"))
    identity = tuple(pair["identity"])
    matching = [c for c in data["charts"] if (c.get("task_id"), c.get("group"), c.get("path"),
                c.get("qubit"), c.get("step"), c.get("block")) == identity]
    real_chart = next((c for c in matching if c.get("chart_name") == "fidRe"), None)
    real = real_chart["points"] if real_chart else None
    imag = next((c["points"] for c in matching if c.get("chart_name") == "fidIm"), None)
    if not real or not imag or len(real) != len(imag):
        return {"fid_status": "unpaired_or_absent", "raw_adc_confirmed": False}
    signal = [complex(re[1], im[1]) for re, im in zip(real, imag)]
    atomic_json(scientific_path.with_name("complex_fid.json"), {
        "identity": pair["identity"], "axis_unit": "NEOVERENÉ",
        "complex_format": "[real, imaginary] in received absolute scale",
        "axis_as_received": [point[0] for point in real],
        "re_im": [[z.real, z.imag] for z in signal]})
    length = 1 << (len(signal)-1).bit_length()
    spectrum = _fft(signal + [0j]*(length-len(signal)))
    magnitudes = [abs(v) for v in spectrum]
    peak = max(range(length), key=magnitudes.__getitem__)
    signed = peak if peak <= length//2 else peak-length
    half = magnitudes[peak]/2
    left, right = peak, peak
    while left > 0 and magnitudes[left-1] >= half:
        left -= 1
    while right < length-1 and magnitudes[right+1] >= half:
        right += 1
    tail = signal[3*len(signal)//4:]
    tail_mean = sum(tail)/len(tail) if tail else 0j
    tail_rms = math.sqrt(sum(abs(z-tail_mean)**2 for z in tail)/len(tail)) if tail else None
    background = statistics.median(magnitudes) if magnitudes else None
    return {"fid_status": "paired", "points": len(signal), "first_amplitude": abs(signal[0]),
            "first_phase_rad": math.atan2(signal[0].imag, signal[0].real),
            "axis_check_as_received": real_chart.get("axis_check"),
            "axis_unit": "NEZNÁME",
            "peak_signed_bin": signed, "peak_cycles_per_sample": signed/length,
            "peak_magnitude": magnitudes[peak], "line_width_fwhm_bins": right-left+1,
            "spectral_snr_proxy": magnitudes[peak]/background if background and background > 0 else None,
            "tail_complex_rms_proxy": tail_rms, "raw_adc_confirmed": False,
            "metric_note": "FFT forward, 1/N, rectangular; SNR=peak/median spectral magnitude. "
                           "Tail RMS may contain signal. Axis units and ADC preprocessing unknown."}


def preview_svg(scientific_path: Path, output: Path) -> None:
    data = json.loads(scientific_path.read_text(encoding="utf-8"))
    chart = next((c for c in data.get("charts", []) if c.get("chart_name") == "fidRe" and
                  c.get("axis_check", {}).get("valid_finite_pairs") and c.get("points")), None)
    if chart is None:
        return
    points = chart["points"]
    stride = max(1, len(points)//800)
    selected = points[::stride]
    ys = [p[1] for p in selected]
    lo, hi = min(ys), max(ys)
    span = hi-lo or 1
    polyline = " ".join(f"{i*780/max(1,len(selected)-1):.2f},{140-120*(y-lo)/span:.2f}"
                        for i, y in enumerate(ys))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 180">'
           f'<rect width="800" height="180" fill="#fff"/>'
           f'<text x="10" y="16" font-size="12">fidRe preview, every {stride}th point; full data in JSON</text>'
           f'<polyline transform="translate(10 20)" fill="none" stroke="#145c82" stroke-width="1" points="{polyline}"/>'
           f'</svg>')
    atomic_bytes(output, svg.encode("utf-8"))
    imag_chart = next((c for c in data.get("charts", []) if c.get("chart_name") == "fidIm" and
                       c.get("task_id") == chart.get("task_id") and c.get("group") == chart.get("group") and
                       c.get("path") == chart.get("path") and c.get("qubit") == chart.get("qubit") and
                       c.get("step") == chart.get("step") and c.get("block") == chart.get("block")), None)
    if not imag_chart or not imag_chart.get("axis_check", {}).get("valid_finite_pairs"):
        return
    imag_points = imag_chart["points"]
    if len(imag_points) != len(points) or any(a[0] != b[0] for a, b in zip(points, imag_points)):
        return
    signal = [complex(a[1], b[1]) for a, b in zip(points, imag_points)]
    length = 1 << (len(signal)-1).bit_length()
    magnitude = [abs(x) for x in _fft(signal+[0j]*(length-len(signal)))]
    step = max(1, length//800)
    values = magnitude[::step]
    upper = max(values) or 1
    fft_line = " ".join(f"{i*780/max(1,len(values)-1):.2f},{140-120*y/upper:.2f}"
                        for i, y in enumerate(values))
    fft_svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 180">'
               f'<rect width="800" height="180" fill="#fff"/>'
               f'<text x="10" y="16" font-size="12">local FFT magnitude, unshifted bin order; every {step}th bin</text>'
               f'<polyline transform="translate(10 20)" fill="none" stroke="#805214" stroke-width="1" points="{fft_line}"/>'
               f'</svg>')
    atomic_bytes(output.with_name("fft_preview.svg"), fft_svg.encode("utf-8"))


def effect(current: dict[str, Any], baseline: dict[str, Any] | None,
           returned: dict[str, Any] | None = None,
           repeat_variability: dict[str, Any] | None = None) -> dict[str, Any]:
    if not baseline or current.get("fid_status") != "paired" or baseline.get("fid_status") != "paired":
        return {"status": "NEOVERENÉ", "reason": "chýba kompatibilný FID alebo baseline"}
    keys = ("first_amplitude", "first_phase_rad", "peak_signed_bin", "peak_magnitude", "points")
    differences = {key: current[key]-baseline[key] for key in keys if number(current.get(key)) and number(baseline.get(key))}
    for name, item in (("current", current), ("baseline", baseline)):
        if name == "current":
            current_step = item.get("axis_check_as_received", {}).get("median_axis_step")
        else:
            baseline_step = item.get("axis_check_as_received", {}).get("median_axis_step")
    if number(current_step) and number(baseline_step):
        differences["axis_step_as_received"] = current_step-baseline_step
    if "first_phase_rad" in differences:
        differences["first_phase_rad"] = math.atan2(math.sin(differences["first_phase_rad"]),
                                                     math.cos(differences["first_phase_rad"]))
    status = "NEOVERENÉ"
    reason = "numerický rozdiel od baseline; kauzálny účinok vyžaduje návrat a odhad variability"
    if returned and returned.get("fid_status") == "paired":
        significant = []
        variability = (repeat_variability or {}).get("metrics", {})
        for key, delta in differences.items():
            if key == "axis_step_as_received":
                return_step = returned.get("axis_check_as_received", {}).get("median_axis_step")
                reference = baseline_step
                var_key = "axis_step_as_received"
            else:
                return_step = returned.get(key)
                reference = baseline.get(key)
                var_key = key
            if not number(return_step) or not number(reference):
                continue
            return_drift = abs(return_step-reference)
            sd = variability.get(var_key, {}).get("population_sd")
            if key == "first_phase_rad":
                return_drift = abs(math.atan2(math.sin(return_step-reference), math.cos(return_step-reference)))
            floor = .5 if key in ("points", "peak_signed_bin") else 1e-9*max(1, abs(reference))
            if number(sd) and abs(delta) > max(3*sd, 2*return_drift, floor):
                significant.append(key)
        if significant:
            status = "OVERENÉ"
            reason = "zmena prekročila 3× variabilitu opakovaní aj 2× návratový drift: " + ", ".join(significant)
        else:
            reason = "zmena neprevýšila dostupnú variabilitu a návratový drift"
    return {"status": status, "differences": differences, "reason": reason,
            "claim_scope": "observed decoded FID change, not RF waveform or persistent calibration"}


def repeat_statistics(result: dict[str, Any], prefix: str = "repeat_") -> dict[str, Any]:
    rows = [row.get("returned", {}).get("metrics", {}) for row in result.get("tests", [])
            if row.get("id", "").startswith(prefix)]
    rows = [row for row in rows if row.get("fid_status") == "paired"]
    if len(rows) < 2:
        return {"status": "NEOVERENÉ", "paired_repeats": len(rows)}
    columns = {key: [row[key] for row in rows if number(row.get(key))]
               for key in ("first_amplitude", "first_phase_rad", "peak_signed_bin",
                           "peak_magnitude", "spectral_snr_proxy", "line_width_fwhm_bins", "points")}
    columns["axis_step_as_received"] = [row.get("axis_check_as_received", {}).get("median_axis_step")
                                        for row in rows if number(row.get("axis_check_as_received", {}).get("median_axis_step"))]
    return {"status": "observed_variability_not_noise_ground_truth", "paired_repeats": len(rows),
            "metrics": {key: {"mean": statistics.mean(values),
                              "population_sd": statistics.pstdev(values),
                              "first_minus_last": values[0]-values[-1]}
                        for key, values in columns.items() if len(values) >= 2},
            "independence_of_internal_acquisitions": "unknown"}


def readiness_summary(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {row["id"]: row for row in result.get("tests", [])}
    def verified(*names: str) -> list[str]:
        return [name for name in names if rows.get(name, {}).get("status") == "OVERENÉ"]
    h_repeats = len(verified(*(f"repeat_{i:02d}" for i in range(1, 31))))
    p_repeats = len(verified(*(f"p_repeat_{i:02d}" for i in range(1, 31))))
    return {
        "adaptive_calibration": {"evidence": verified("rabi_low2", "rabi_low", "rabi_center",
                                                       "rabi_high", "rabi_high2", "rabi_control"),
                                 "missing": "identifikovateľnosť a zlepšenie nad prirodzenou variabilitou",},
        "complex_fid_denoising": {"evidence": {"h_repeats": h_repeats, "p_repeats": p_repeats},
                                  "missing": "nezávislá bezšumová pravda a známe interné spracovanie"},
        "learned_pps": {"evidence": verified("physical_baseline"),
                        "missing": "overená kompletná PPS sekvencia a nezávislá tomografia"},
        "robust_pulses": {"evidence": verified("multi_segment_h", "shape_gaussian_h",
                                                  "phase_relative_90", "phase_relative_180", "phase_relative_270"),
                          "missing": "meranie skutočného RF výstupu a viac stavov"},
        "drift_compensation": {"evidence": verified("physical_baseline", "final_reference"),
                               "missing": "dlhšie časové rady a oddelenie driftu od zmeny vzorky"},
        "ai_shimming": {"evidence": "read-only shim hodnoty iba ak dorazili v tejto relácii",
                        "missing": "mapovanie ciest, povolené napätia a bezpečný reverzibilný zápis"},
        "agent_control": {"evidence": "sekvenčné merania s návratovým stavom a lokálnym výberom Rabi bodu",
                          "missing": "spoľahlivé prerušenie úlohy, globálna RF záťaž a hardvérové limity"},
    }


def frequency_response(metrics_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    offsets = (-20, -10, 0, 10, 20)
    names = [f"frequency_scan_{'m' if offset < 0 else 'p'}{abs(offset)}" for offset in offsets]
    values = [metrics_by_id.get(name, {}).get("peak_magnitude") for name in names]
    response = {"detuning_hz_requested": offsets, "local_fft_peak_magnitude": values,
                "uncertainty": "NEOVERENÉ: päť bodov, neznáma os kalibrácie a interné priemerovanie"}
    if not all(number(value) for value in values):
        response.update(status="NEOVERENÉ", reason="chýba niektorý párovaný FID")
        return response
    index = max(range(5), key=lambda i: values[i])
    response["best_measured_offset_hz"] = offsets[index]
    if index in (0, 4):
        response.update(status="NEOVERENÉ", reason="maximum je na okraji skenu; rezonancia nie je ohraničená")
        return response
    left, center, right = values[index-1:index+2]
    denominator = left-2*center+right
    if denominator >= 0 or denominator == 0:
        response.update(status="NEOVERENÉ", reason="lokálna odozva nie je konkávne maximum")
        return response
    vertex = offsets[index] + 5*(left-right)/denominator
    if not offsets[index-1] <= vertex <= offsets[index+1]:
        response.update(status="NEOVERENÉ", reason="parabolický vrchol je mimo susedných bodov")
        return response
    response.update(status="exploratory_local_fit", peak_offset_hz_estimate=vertex,
                    reason="len lokálny parabolický odhad, nie kalibrovaná rezonancia")
    return response


def markdown_report(result: dict[str, Any]) -> str:
    lines = ["# Séria testov SpinQ Gemini Lab", "", f"Začiatok: {result['started_utc']}",
             f"Stav série: {result.get('state', 'running')}",
             f"Rozsah: {result.get('execution_scope', 'NEOVERENÉ')}",
             f"Odoslanie výsledkov: {result.get('upload', {}).get('status', 'NEZAČATÉ')}; "
             f"dôvod: {result.get('upload', {}).get('reason', '—')}",
             f"SDK: {result.get('environment', {}).get('sdk_version') or 'NEZNÁME'}", "",
             f"Pokusy o skutočné experimenty: {result.get('real_hardware_attempts', 0)}; potvrdene dokončené: {result.get('real_hardware_completed', 0)}.",
             "Ak je počet 0, prístroj sa týmto programom nemeral. Prijaté údaje sú dekódované chart body; RAW ADC nie je potvrdené.",
             "40 µs baseline bol na tomto prístroji úspešný. Hranice výskumnej série sú softvérový rozpočet, nie certifikované limity prístroja.",
             "", "## Testy", ""]
    for row in result.get("tests", []):
        lines += [f"### {row['id']} — {row.get('status', 'NEOVERENÉ')}", "",
                  f"- API: {row.get('api', 'NEOVERENÉ')}",
                  f"- Poslané: `{json.dumps(row.get('sent'), ensure_ascii=False, default=str)}`" if row.get("sent") is not None else "- Poslané: nič",
                  f"- Vrátené: `{json.dumps(row.get('returned'), ensure_ascii=False, default=str)}`" if row.get("returned") is not None else "- Vrátené: nič",
                  f"- Účinok: `{json.dumps(row.get('effect'), ensure_ascii=False, default=str)}`" if row.get("effect") is not None else "- Účinok: NEOVERENÉ",
                  f"- Dôvod: {row.get('reason', '—')}", ""]
    lines += ["## Čo vieme ovládať a získať", "",
              "Fyzikálne účinky sú uvedené pri jednotlivých testoch. Skutočne prijaté polia sú v data/*/field_catalog.json.",
              "Úplné pôvodné dekódované číselné grafy sú v data/*/original_scientific.json a data/events.jsonl.gz.",
              "Serverová FFT/fit sa podľa overeného SDK 1.0.2 nedá preukázateľne vypnúť; lokálna FFT a metriky sú oddelené.",
              "Interná akvizícia, príprava, ADC reťazec a fyzický RF výstup zostávajú NEZNÁME bez ďalších dôkazov.",
              "Opakovania nedávajú bezšumovú pravdu; denoising treba hodnotiť na nezávislých meraniach.",
              "", "## Variabilita opakovaní", "",
              "`"+json.dumps(result.get("repeat_statistics", {}), ensure_ascii=False, default=str)+"`",
              "", "## Lokálna kalibračná slučka", "",
              "`"+json.dumps(result.get("calibration_selection", {}), ensure_ascii=False, default=str)+"`",
              "", "## Lokálne fity", "",
              "Rabi: `"+json.dumps(result.get("rabi_fit", {}), ensure_ascii=False, default=str)+"`",
              "Frekvenčná odozva: `"+json.dumps(result.get("frequency_response", {}), ensure_ascii=False, default=str)+"`",
              "", "## Chyby a zastavenia", ""]
    lines += [f"- {item}" for item in result.get("errors", [])] or ["- Žiadne zaznamenané."]
    lines += ["", "## Pripravenosť na AI/ML výskum", ""]
    for name, item in result.get("readiness", {}).items():
        lines += [f"- **{name}**: dôkazy `{json.dumps(item.get('evidence'), ensure_ascii=False)}`; "
                  f"chýba {item.get('missing', 'NEOVERENÉ')}"]
    return "\n".join(lines)+"\n"


def checkpoint(out: Path, result: dict[str, Any]) -> None:
    safe = safe_metadata(result)
    atomic_json(out/"results.json", safe)
    atomic_bytes(out/"REPORT.md", markdown_report(safe).encode("utf-8"))


def bundle(out: Path) -> None:
    archive_path = out/"results.zip"
    temporary = out/"results.zip.tmp"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if (path.is_file() and path != archive_path and path != temporary and
                    path.name != "events.jsonl" and not path.name.endswith(".tmp") and
                    not (path == out/"events.jsonl.gz" and (out/"data"/"events.jsonl.gz").exists())):
                archive.write(path, str(path.relative_to(out)).replace("\\", "/"))
    os.replace(temporary, archive_path)


def case_api(case: dict[str, Any]) -> str:
    return case.get("api") or {
        "physical": "register_experiment(PHYSICAL_LAYER_EXPERIMENT) / run_experiment",
        "nmr": "register_experiment(NMR_PHENOMENON_AND_SIGNAL) / run_experiment",
        "rabi": "register_experiment(RABI_OSCILLATIONS) / run_experiment",
        "shape": "register_experiment(SHAPE_PULSE) / run_experiment",
    }.get(case["kind"], "NEOVERENÉ")


def main() -> int:
    parser = argparse.ArgumentParser(description="Real Gemini Lab measurement suite via SpinQLabLink")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    out = (args.out or Path("results")/f"{stamp}_live_suite").resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Výstupný adresár už obsahuje dáta; pôvodné merania sa neprepisujú.")
    out.mkdir(parents=True, exist_ok=True)
    atomic_bytes(out/"errors.log", b"")
    result: dict[str, Any] = {"started_utc": utc_now(), "state": "starting", "tests": [],
                              "errors": [], "real_hardware_attempts": 0, "real_hardware_completed": 0,
                              "raw_adc_confirmed": False}
    checkpoint(out, result)
    link = adapter = recorder = device = device_observer = None
    connected = False
    exit_code = 0
    cases: list[dict[str, Any]] = []
    try:
        config = load_config(args.config)
        config.pop("_runtime_p_baseline", None)
        config.pop("_runtime_p_variants", None)
        config.pop("_runtime_frequency_hz", None)
        cases = make_cases(config)
        result["planned_tests"] = [case["id"] for case in cases]
        checkpoint(out, result)
        operation = config.get("operation", {})
        if not config.get("host") or not config.get("account") or type(config.get("port")) is not int:
            raise ValueError("host, port a account chýbajú")
        sdk_version = importlib.metadata.version("spinqlablink")
        result["environment"] = {"python": sys.version.split()[0], "os": platform.platform(),
                                  "architecture": platform.machine(), "sdk_version": sdk_version,
                                  "connection": "tablet SpinQLabLink TCP; device USB link reported by operator",
                                  "model": "Gemini Lab (operator reported)",
                                  "server_version": None, "firmware_version": None}
        checkpoint(out, result)
        sdk_env, sdk_capabilities, sdk_issues, sdk_experiments = discover()
        atomic_json(out/"sdk_environment.json", sdk_env)
        atomic_json(out/"sdk_capabilities.json", sdk_capabilities)
        atomic_json(out/"sdk_issues.json", sdk_issues)
        atomic_json(out/"sdk_experiments.json", sdk_experiments)
        verify_installed_sdk()  # This capture adapter is verified for installed SDK 1.0.2.
        from spinqlablink import ExperimentType, Pulse, SpinQLabLink
        result["supported_experiment_types_in_client"] = [name for name in dir(ExperimentType) if name.isupper()]
        result["static_findings"] = {
            "raw_adc": "NEZNÁME; decoded chart float32 is not ADC proof",
            "server_fft_disable": "no verified option in installed SDK 1.0.2",
            "pulse_domain_analysis": "SDK utility opens a Qt event loop; omitted from unattended run; generated waveform checked against local Gaussian math, not physical RF output",
            "operating_limits": "manufacturer/device operating limits not supplied or established; bounded study uses a finite software test plan, not a hardware safety rating",
            "bounded_study_requested_rf_budget_us": STUDY_MAX_CUMULATIVE_REQUESTED_RF_US,
            "requested_rf_caveat": "known configured H pulse widths only; hidden preparation and internal repetitions are not counted",
            "relaxation_delay_units": "official experiment page says seconds; SDK source describes microseconds; numeric baseline value 15 is unchanged",
            "internal_repeat_count": "NEZNÁME",
            "receiver_gain_and_filters": "NEZNÁME",
            "gradient_and_shim_write": "not attempted; units and limits unconfirmed",
            "gradient_client_fields": "Gradient(path 0..13, voltage -1..1, duration µs) is an SDK declaration only; no coil mapping or hardware operating limit established",
            "shim_read": "Device.get_shimming_values reads path0..13 from received shimmingParam when present; missing fields are not treated as zero measurements",
            "t1_t2_delay_control": "SDK 1.0.2 experiment parameters do not expose per-point delays or internal acquisition count",
            "phase_controls": "relative +90/+180/+270 degrees are scheduled around the observed H baseline; outcome requires measured FID",
            "cross_channel_timing": "NEZNÁME; a separate P path is not proof of simultaneous H/P timing",
            "t1_t2": "client experiment classes exist; live long scan not in this budget",
        }
        checkpoint(out, result)
        result["execution_scope"] = ("one_exact_historical_physical_baseline" if
                                     config.get("historical_baseline_only") is True else
                                     "finite_bounded_research_study" if
                                     config.get("bounded_study_enabled") is True else
                                     "configured_series_requiring_operating_limits")
        checkpoint(out, result)
        password = os.environ.get(config.get("password_env", "SPINQ_AUDIT_PASSWORD"))
        result["login_source"] = "environment" if password else "unavailable"
        if not password and config.get("use_existing_demo_login") is True and config.get("account") == "anyword":
            # This is the exact public fallback in the already working
            # spinq_lab_control.py, not a newly guessed device credential.
            password = "anyword"
            result["login_source"] = "existing_spinq_lab_control_default"
        if not password:
            raise RuntimeError("Chýba neinteraktívne prihlásenie: nastav premennú hesla alebo použi existujúce demo prihlásenie")
        recorder = EventRecorder(out, max_events=256)
        link = SpinQLabLink(config["host"], config["port"], config["account"], password)
        del password
        device = link.get_device()
        result["observer_updates"] = []
        def observed_device(_device: Any, update_type: str) -> None:
            result["observer_updates"].append({"received_utc": utc_now(),
                                               "type": update_type,
                                               "source": "SDK register_observer callback"})
        device_observer = observed_device
        device.register_observer(device_observer)
        adapter = AuditAdapter(link, recorder, mode="active", owns_connection=True)
        adapter.attach()
        link.connect()  # SDK 1.0.2 returns None on successful connect().
        if not link.get_connection() or not link.wait_for_login(timeout=10):
            raise RuntimeError("Spojenie alebo prihlásenie zlyhalo; nič sa nemeralo")
        connected = True
        result["state"] = "connected"
        checkpoint(out, result)
        warmup = time.monotonic()+20
        while time.monotonic() < warmup and (adapter.latest.get("s_post_device_info") is None or adapter.queue is None):
            time.sleep(.1)
        result["device_observed"] = {kind: redact(body) for kind, (_at, body) in adapter.latest.items()
                                     if kind != "s_post_exp_queue_update"}
        result["device_version_candidates"] = metadata_candidates(result["device_observed"])
        atomic_json(out/"telemetry_field_catalog.json", field_catalog(
            {"message_type": kind, "as_received": body} for kind, body in result["device_observed"].items()))
        result["queue_observed"] = adapter.queue is not None
        result["device_accessors"] = device_snapshot(device, adapter.latest)
        result["telemetry_provenance"] = {kind: {"received_monotonic_ns": when,
                                                  "source": "decoded server message",
                                                  "snapshot_utc": utc_now()}
                                          for kind, (when, _body) in adapter.latest.items()}
        result["custom_frequency_basis"] = append_frequency_cases(cases, config, device, adapter.latest)
        if "_runtime_frequency_hz" not in config:
            cases.append({"id": "nmr_custom_frequency", "kind": "nmr", "params": physical_baseline(config),
                          "enabled": False, "api": "NMR_PHENOMENON_AND_SIGNAL(custom_freq)",
                          "skip_reason": result["custom_frequency_basis"]})
        result["waveform_local_check"] = append_shape_case(cases, config, device, adapter.latest)
        cases.append({"id": "final_reference", "kind": "physical", "params": physical_baseline(config),
                      "purpose": "záverečný H referenčný FID pre oddelenie driftu"})
        result["p_channel_basis"] = append_p_channel_cases(cases, config, device, adapter.latest)
        if "_runtime_p_baseline" not in config:
            cases.append({"id": "p_baseline", "kind": "physical", "params": physical_baseline(config),
                          "enabled": False, "api": "PHYSICAL_LAYER_EXPERIMENT(samplePath=1)",
                          "skip_reason": result["p_channel_basis"]})
        for capability_id, api, reason in SKIPPED_CAPABILITIES:
            cases.append({"id": capability_id, "kind": "capability", "api": api,
                          "params": None, "enabled": False, "skip_reason": reason,
                          "purpose": "preskúmanie dostupnosti bez neovereného hardvérového zásahu"})
        result["planned_tests"] = [case["id"] for case in cases]
        checkpoint(out, result)
        max_experiments = operation.get("max_experiments", 0)
        pause = operation.get("pause_seconds", 0)
        timeout = operation.get("timeout_seconds", 0)
        if type(max_experiments) is not int or not 1 <= max_experiments <= 120:
            raise ValueError("operation.max_experiments musí byť 1..120")
        if not number(pause) or not 1 <= pause <= 3600 or not number(timeout) or not 20 <= timeout <= 3600:
            raise ValueError("neplatná prestávka alebo deadline")
        if operation.get("max_experiments") is None:
            raise ValueError("explicitný hardvérový rozpočet chýba")
        cumulative_rf = 0.0
        attempted = 0
        file_position = 0
        metrics_by_id: dict[str, dict[str, Any]] = {}
        last_baseline: dict[str, Any] | None = None
        last_finished = 0.0
        lock_path = Path(config.get("hardware_lock_path", "~/.spinq_live_gemini.lock"))
        with HardwareLock(lock_path):
            for case in cases:
                row: dict[str, Any] = {"id": case["id"], "kind": case["kind"], "purpose": case.get("purpose"),
                                       "api": case_api(case),
                                       "status": "NEOVERENÉ", "sent": None, "returned": None,
                                       "effect": None, "reason": "nezačaté"}
                result["tests"].append(row)
                checkpoint(out, result)
                if case.get("enabled") is False:
                    row["reason"] = case.get("skip_reason", "vypnuté v konfigurácii")
                    checkpoint(out, result)
                    continue
                if case.get("requires_verified"):
                    required = next((prior for prior in result["tests"]
                                     if prior["id"] == case["requires_verified"]), None)
                    if required is None or required["status"] != "OVERENÉ":
                        row["reason"] = "predchádzajúci pracovný bod nemá overený signál"
                        checkpoint(out, result)
                        continue
                if case.get("return_for"):
                    parent = next((prior for prior in result["tests"] if prior["id"] == case["return_for"]), None)
                    if parent is None or parent.get("sent") is None:
                        row["reason"] = "zmena sa neodoslala; návratové meranie netreba"
                        checkpoint(out, result)
                        continue
                if attempted >= max_experiments:
                    row["reason"] = "vyčerpaný max_experiments; nič sa neposlalo"
                    checkpoint(out, result)
                    continue
                try:
                    requested_rf = check_case(case, config, cumulative_rf)
                except (ValueError, KeyError, TypeError) as exc:
                    row["reason"] = str(exc)
                    checkpoint(out, result)
                    continue
                if last_finished:
                    remaining = pause-(time.monotonic()-last_finished)
                    if remaining > 0:
                        time.sleep(remaining)
                max_status_age = operation.get("status_max_age_seconds", 120)
                max_queue_age = operation.get("queue_max_age_seconds", 120)
                status = adapter.fresh("s_post_device_info", max_status_age)
                if not status or status.get("connected") is not True or status.get("lockState") is not True:
                    raise RuntimeError("čerstvý stav/lock chýba alebo nevyhovuje; séria zastavená")
                temperature = status.get("temperature")
                row["preflight_temperature_c"] = temperature
                if not number(temperature):
                    raise RuntimeError("čerstvá teplota chýba alebo nie je konečná")
                if (config.get("historical_baseline_only") is not True and
                        config.get("bounded_study_enabled") is not True):
                    limits = config["limits"]
                    if not limits["min_temperature_c"] <= temperature <= limits["max_temperature_c"]:
                        raise RuntimeError("teplota je mimo potvrdeného rozsahu")
                if adapter.lock_lost_observed or adapter.decoder_failures or not recorder.status()["complete"]:
                    raise RuntimeError("strata locku alebo prijatých udalostí; séria zastavená")
                queue_fresh = (adapter.queue is not None and
                               (time.monotonic_ns()-adapter.queue[0])/1e9 <= max_queue_age)
                if not queue_fresh and operation.get("exclusive_use_confirmed") is not True:
                    raise RuntimeError("fronta chýba alebo je stará; potvrď výhradné používanie v konfigurácii")
                if not queue_fresh:
                    row["queue_note"] = "fronta neoverená; obsluha potvrdila výhradné používanie"
                if adapter.queue is not None and queue_fresh and adapter.queue[1].get("queue") != []:
                    raise RuntimeError("fronta nie je prázdna; cudzia úloha sa neovláda")
                experiment_type = (ExperimentType.PHYSICAL_LAYER_EXPERIMENT if case["kind"] == "physical"
                                   else ExperimentType.RABI_OSCILLATIONS if case["kind"] == "rabi"
                                   else ExperimentType.SHAPE_PULSE if case["kind"] == "shape"
                                   else ExperimentType.NMR_PHENOMENON_AND_SIGNAL)
                experiment, params = link.register_experiment(experiment_type)
                submitted = False
                terminal = False
                try:
                    desired = case["params"]
                    if case["kind"] == "physical":
                        _configure_physical(params, desired)
                    else:
                        params.pulses = [Pulse(path=path, width=pulse["width"], amplitude=pulse["am"],
                                               phase=pulse["phase"], detuning=pulse["freshift"])
                                         for path, channel in ((0, "hPulse"), (1, "pPulse"))
                                         for pulse in desired["pulse"][channel]]
                        params.samplePath = desired["samplePath"]
                        if case["kind"] == "shape":
                            params.sampleFre = desired["sampleFre"]
                            params.sampleCount = desired["sampleCount"]
                            params.sampleDelay = desired["sampleDelay"]
                            params.h_freShift = desired["h_freShift"]
                            params.h_freDemo = desired["h_freDemo"]
                        else:
                            params.makePps = desired["makePps"]
                            frequencies = case.get("custom_frequency_hz")
                            params.custom_freq = frequencies is not None
                            if frequencies is not None:
                                if frequencies != config.get("_runtime_frequency_hz"):
                                    raise ValueError("vlastné frekvencie sa líšia od čerstvej telemetrie")
                                params.freq_h = frequencies["H"]/1e6
                                params.freq_p = frequencies["P"]/1e6
                    wire = experiment.get_experiment_parameter()
                    actual = json.loads(wire["params"])
                    if case["kind"] == "physical" and actual != desired:
                        raise ValueError("SDK serializoval iný fyzikálny payload")
                    if actual.get("pulse") != desired["pulse"] or actual.get("samplePath") != desired["samplePath"]:
                        raise ValueError("finálny pulz/kanál sa zmenil pri serializácii")
                    if case["kind"] == "shape" and (actual.get("sampleFre") != desired["sampleFre"] or
                                                   actual.get("sampleCount") != desired["sampleCount"]):
                        raise ValueError("SHAPE_PULSE serializoval iné vzorkovanie")
                    if case.get("custom_frequency_hz") and (actual.get("custom_freq") is not True or
                            abs(actual.get("freq_h", 0)-case["custom_frequency_hz"]["H"]) > 1 or
                            abs(actual.get("freq_p", 0)-case["custom_frequency_hz"]["P"]) > 1):
                        raise ValueError("SDK serializoval inú vlastnú frekvenciu")
                    row["sent"] = {"experiment_type": str(experiment_type), "params": actual,
                                   "sdk_task_id_before_ack": str(experiment.id),
                                   "phase": "prepared_not_sent"}
                    row["reason"] = "payload pripravený"
                    checkpoint(out, result)
                    adapter.own_task_ids.add(str(experiment.id))
                    adapter.pending_own_ack = True
                    adapter.ack_mismatch = False
                    row["sent"]["phase"] = "submission_attempted_unconfirmed"
                    checkpoint(out, result)
                    submitted = True  # From here a send failure is an uncertain hardware state.
                    attempted += 1
                    result["real_hardware_attempts"] = attempted
                    send_started_ns = time.monotonic_ns()
                    link.run_experiment()
                    cumulative_rf += requested_rf
                    row["sent"]["phase"] = "send_enqueued_server_unconfirmed"
                    checkpoint(out, result)
                    state = wait_terminal(experiment, link.get_connection, recorder, timeout,
                                          on_poll=lambda _: (_ for _ in ()).throw(RuntimeError("nesúlad ACK sequence_id"))
                                          if adapter.ack_mismatch else None)
                    terminal = True
                    last_finished = time.monotonic()
                    row["returned"] = {"task_id": str(experiment.id), "state": state,
                                       "server_ack_observed": not adapter.pending_own_ack,
                                       "requested_rf_us_known_only": requested_rf}
                    row["sent"]["phase"] = "server_finished" if state == "COMPLETED" else "server_failed"
                    if state != "COMPLETED":
                        row["status"] = "ZLYHALO"
                        row["reason"] = "server nahlásil FAILED; ďalšie požiadavky sa neposielajú"
                        checkpoint(out, result)
                        raise RuntimeError(row["reason"])
                    result["real_hardware_completed"] += 1
                    measurement = out/"data"/case["id"]
                    measurement.mkdir(parents=True, exist_ok=True)
                    sdk_result = experiment.get_result()
                    atomic_bytes(measurement/"sdk_result.json", json.dumps(redact(sdk_result),
                                 ensure_ascii=False, allow_nan=True, default=str).encode("utf-8"))
                    wait_recorded(recorder)
                    events, file_position = read_new_events(out/"events.jsonl", file_position)
                    own_events = [event for event in events if
                                  (event.get("payload", {}).get("chart_data") or
                                   event.get("payload", {}).get("json_data") or {}).get("taskId") == experiment.id]
                    def arrival(kind: str, last: bool = False) -> float | None:
                        stamps = [event.get("received_monotonic_ns") for event in own_events
                                  if event.get("kind") == kind and type(event.get("received_monotonic_ns")) is int]
                        if not stamps:
                            return None
                        return round(((stamps[-1] if last else stamps[0])-send_started_ns)/1e9, 6)
                    row["returned"]["client_arrival_seconds_after_send_start"] = {
                        "server_ack": arrival("s_add_exp_task_res"),
                        "experiment_started": arrival("s_post_exp_started"),
                        "first_chart": arrival("s_post_exp_chart_updated"),
                        "last_chart": arrival("s_post_exp_chart_updated", last=True),
                        "experiment_finished": arrival("s_post_exp_finished"),
                        "clock_basis": "one local monotonic clock; server timestamps not subtracted"}
                    analysis = analyze_events(own_events, measurement)
                    metric = scientific_metrics(analysis, measurement/"original_scientific.json")
                    if case["kind"] == "rabi":
                        real = sdk_result.get("result", {}).get("real")
                        if number(real):
                            metric["rabi_real_server_result"] = real
                    metrics_by_id[case["id"]] = metric
                    preview_svg(measurement/"original_scientific.json", measurement/"fid_preview.svg")
                    row["returned"].update({"chart_count": analysis["chart_count"],
                                             "chart_names": sorted({c.get("chart_name") for c in
                                                 json.loads((measurement/"original_scientific.json").read_text(encoding="utf-8"))["charts"]
                                                 if c.get("chart_name")}),
                                             "fid_pairs": len([x for x in analysis["fid_pairs"] if x.get("status") == "paired"]),
                                             "files": str(measurement.relative_to(out)),
                                             "transport_complete_confirmed": analysis["transport_complete_confirmed"],
                                             "metrics": metric,
                                             "local_analysis_seconds": analysis["analysis_wall_seconds"]})
                    if case["kind"] == "physical":
                        row["returned"]["requested_vs_received_points"] = {
                            "requested": desired["sampleCount"], "received_fid": metric.get("points"),
                            "equal": metric.get("points") == desired["sampleCount"]
                            if number(metric.get("points")) else None}
                    if number(metric.get("axis_check_as_received", {}).get("median_axis_step")):
                        row["returned"]["sample_timing_comparison"] = {
                            "requested_hz": desired["sampleFre"],
                            "expected_seconds_per_sample": 1/desired["sampleFre"],
                            "received_axis_step": metric["axis_check_as_received"]["median_axis_step"],
                            "received_axis_unit": "NEOVERENÉ; porovnanie čísiel nie je dôkaz jednotky"}
                    if not row["returned"]["server_ack_observed"]:
                        raise RuntimeError("potvrdenie servera sa nezachytilo; ďalšie merania sa neposielajú")
                    if case["id"] in {"physical_baseline", "p_baseline"}:
                        last_baseline = metric
                    if case["id"] == "final_reference":
                        row["effect"] = effect(metric, metrics_by_id.get("physical_baseline"),
                                               repeat_variability=result.get("repeat_statistics"))
                    if case["id"].startswith("repeat_") and metric.get("fid_status") == "paired":
                        last_baseline = metric
                        result["repeat_statistics"] = repeat_statistics(result)
                    if case["id"].startswith("p_repeat_") and metric.get("fid_status") == "paired":
                        last_baseline = metric
                        result["p_repeat_statistics"] = repeat_statistics(result, "p_repeat_")
                    if case.get("return_for"):
                        original = next((r for r in result["tests"] if r["id"] == case["return_for"]), None)
                        if original and original.get("returned"):
                            parent_baseline = (metrics_by_id.get("physical_baseline")
                                               if case["return_for"] == "nmr_custom_frequency"
                                               else last_baseline)
                            original["effect"] = effect(original["returned"].get("metrics", {}),
                                                        parent_baseline, metric, result.get("repeat_statistics"))
                            if (original["returned"].get("transport_complete_confirmed") and
                                    analysis["transport_complete_confirmed"]):
                                original["status"] = original["effect"]["status"]
                            else:
                                original["status"] = "NEOVERENÉ"
                                original["effect"]["reason"] += "; prenos kompletnosti grafov nie je potvrdený"
                        last_baseline = metric
                    elif case["id"] not in {"physical_baseline", "p_baseline", "final_reference"} and not case["id"].startswith(("repeat_", "p_repeat_")):
                        h_reference = metrics_by_id.get("physical_baseline")
                        comparison = (h_reference if case["id"].startswith("nmr_custom_frequency") or
                                      case["kind"] == "shape" else last_baseline)
                        row["effect"] = effect(metric, comparison)
                    complete_chart = analysis["transport_complete_confirmed"]
                    data_verified = (complete_chart and analysis["chart_count"] and
                                     (case["kind"] not in {"physical", "shape"} or metric.get("fid_status") == "paired"))
                    if case["kind"] == "rabi" and number(metric.get("rabi_real_server_result")):
                        data_verified = True  # Scope: a server-derived scalar, not a complete FID.
                    row["status"] = "OVERENÉ" if data_verified else "NEOVERENÉ"
                    if case["id"] == "rabi_control":
                        selected_id = result.get("calibration_selection", {}).get("selected")
                        prior = metrics_by_id.get(selected_id, {})
                        criterion_key = result.get("calibration_selection", {}).get("metric_key", "peak_magnitude")
                        result["calibration_selection"]["control_measurement"] = {
                            "status": "measured" if number(metric.get(criterion_key)) else "metric_unavailable",
                            "metric_key": criterion_key,
                            "selected_value": prior.get(criterion_key),
                            "control_value": metric.get(criterion_key),
                            "difference": metric.get(criterion_key)-prior.get(criterion_key)
                            if number(metric.get(criterion_key)) and number(prior.get(criterion_key)) else None}
                    row["reason"] = ("reálne meranie dokončené; účinok presne podľa uvedených metrík"
                                     if row["status"] == "OVERENÉ" else "meranie skončilo, ale kompletný FID nebol potvrdený")
                    checkpoint(out, result)
                    print(f"{case['id']}: {row['status']} ({analysis['chart_count']} kriviek)", flush=True)
                    if case["id"] == "frequency_scan_p20":
                        result["frequency_response"] = frequency_response(metrics_by_id)
                        checkpoint(out, result)
                    if case["id"] == "rabi_high2":
                        scan = [(name, metrics_by_id.get(name, {})) for name in
                                ("rabi_low2", "rabi_low", "rabi_center", "rabi_high", "rabi_high2")]
                        if all(number(m.get("rabi_real_server_result")) for _, m in scan):
                            try:
                                from analyze_results import fit_rabi
                                result["rabi_fit"] = fit_rabi([
                                    {"width_us": width, "real": metrics_by_id.get(name, {}).get("rabi_real_server_result")}
                                    for name, width in zip(("rabi_low2", "rabi_low", "rabi_center",
                                                            "rabi_high", "rabi_high2"),
                                                           PREVIOUSLY_COMPLETED_RABI_WIDTHS_US)])
                                result["rabi_fit"]["uncertainty"] = (
                                    "NEOVERENÉ: päť bodov a neznámy interný priemer; perióda je exploratívny lokálny fit")
                            except Exception as exc:
                                result["rabi_fit"] = {"status": "NEOVERENÉ", "reason": str(redact(str(exc)))}
                        metric_key = ("peak_magnitude" if all(number(m.get("peak_magnitude")) for _, m in scan)
                                      else "rabi_real_server_result" if all(number(m.get("rabi_real_server_result")) for _, m in scan)
                                      else None)
                        if metric_key:
                            selected = max(scan, key=lambda item: abs(item[1][metric_key]))[0]
                            chosen = next(item for item in cases if item["id"] == selected)
                            cases.insert(cases.index(case)+1,
                                         {"id": "rabi_control", "kind": "rabi",
                                          "params": copy.deepcopy(chosen["params"]),
                                          "purpose": "nezávislá kontrola lokálne vybraného bodu"})
                            result["planned_tests"].append("rabi_control")
                            result["calibration_selection"] = {
                                "criterion": ("largest absolute locally computed FID FFT peak" if metric_key == "peak_magnitude"
                                              else "largest absolute server Rabi scalar, selected locally"),
                                "metric_key": metric_key,
                                "selected": selected, "control_measurement": "pending",
                                "claim": "exploratory, not calibrated optimal pulse"}
                        else:
                            result["calibration_selection"] = {"status": "NEOVERENÉ",
                                "reason": "Rabi sken neposkytol párovaný FID pre lokálny výber"}
                        checkpoint(out, result)
                except (ValueError, TypeError, KeyError) as exc:
                    if submitted:
                        raise
                    row["reason"] = f"SDK odmietlo pripravenie pred odoslaním: {exc}"
                    checkpoint(out, result)
                    continue
                finally:
                    if submitted and not terminal:
                        row["status"] = "NEOVERENÉ"
                        row["reason"] = "koniec úlohy nepotvrdený; môže stále bežať, žiadny automatický retry"
                        checkpoint(out, result)
                    if not submitted or terminal:
                        link.deregister_experiment()
        result["state"] = "completed_with_skips" if attempted else "no_experiments_sent"
        if not attempted:
            exit_code = 2
    except KeyboardInterrupt:
        result["state"] = "interrupted_task_may_continue"
        result["errors"].append("Ctrl+C: ďalšie požiadavky zastavené; bežiaca úloha môže pokračovať na serveri")
        exit_code = 130
    except Exception as exc:
        result["state"] = "stopped"
        result["errors"].append(f"{type(exc).__name__}: {redact(str(exc))}")
        atomic_bytes(out/"errors.log", redact(traceback.format_exc(limit=4)).encode("utf-8"))
        exit_code = 1
    finally:
        seen_ids = {row["id"] for row in result["tests"]}
        for case in cases:
            if case["id"] not in seen_ids:
                result["tests"].append({"id": case["id"], "kind": case["kind"],
                    "api": case_api(case), "purpose": case.get("purpose"),
                    "status": "NEOVERENÉ", "sent": None, "returned": None, "effect": None,
                    "reason": (case.get("skip_reason", "vypnuté") if case.get("enabled") is False else
                               "séria sa zastavila pred týmto testom")})
        if connected and link is not None:
            try:
                link.disconnect()  # Own client only; this does not abort a task on hardware.
            except Exception as exc:
                result["errors"].append("disconnect: "+redact(str(exc)))
        if device is not None and adapter is not None:
            result["device_accessors_final"] = device_snapshot(device, adapter.latest)
        if device is not None and device_observer is not None:
            device.unregister_observer(device_observer)
        if adapter is not None:
            result["capture"] = {"decoder_failures": adapter.decoder_failures,
                                 "outgoing_types": sorted(set(adapter.outgoing))}
            adapter.detach()
        if recorder is not None:
            result["recorder"] = recorder.close()
            recorded = out/"events.jsonl.gz"
            if recorded.exists():
                (out/"data").mkdir(exist_ok=True)
                shutil.copyfile(recorded, out/"data"/"events.jsonl.gz")
        if result["errors"]:
            prior = (out/"errors.log").read_text(encoding="utf-8")
            atomic_bytes(out/"errors.log", (prior+"\n"+"\n".join(result["errors"])+"\n").encode("utf-8"))
        result["finished_utc"] = utc_now()
        result["readiness"] = readiness_summary(result)
        checkpoint(out, result)
        result["upload"] = {"status": "PENDING"}
        checkpoint(out, result)
        bundle(out)
        result["upload"] = publish_results(Path(__file__).resolve().parent,
                                             out/"results.zip", "results/"+stamp)
        checkpoint(out, result)
        bundle(out)
        print(f"Výsledky: {out/'results.zip'}; Git: {result['upload']['status']}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
