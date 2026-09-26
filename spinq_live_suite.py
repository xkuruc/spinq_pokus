"""One-run, real Gemini Lab measurement suite through the installed SpinQLabLink.

This program never simulates a measurement. It records decoded server events
before the vendor experiment handlers simplify or replace chart data.
"""

from __future__ import annotations

import argparse
import copy
import getpass
import importlib.metadata
import json
import math
import os
import platform
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
    count = config.get("operation", {}).get("repeat_count", 10)
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

    width_delta = deltas.get("rabi_width_us")
    if number(width_delta) and width_delta > 0 and features.get("rabi_scan_verified") is True:
        for label, width in (("low", hp(base)["width"]-width_delta),
                             ("center", hp(base)["width"]), ("high", hp(base)["width"]+width_delta)):
            params = copy.deepcopy(base)
            hp(params)["width"] = width
            cases.append({"id": "rabi_"+label, "kind": "rabi", "params": params,
                          "purpose": "krátky Rabi sken; výber ďalšieho bodu lokálne"})
    else:
        cases.append({"id": "rabi_scan", "kind": "rabi", "params": copy.deepcopy(base),
                      "enabled": False, "skip_reason": "Rabi rozsah nebol potvrdený alebo chýba deltas.rabi_width_us"})
    return cases


def check_case(case: dict[str, Any], config: dict[str, Any], cumulative_requested_rf: float) -> float:
    if config.get("baseline_verified") is not True:
        raise ValueError("baseline_verified nie je true; známy pracovný bod nebol potvrdený")
    if config.get("historical_baseline_only") is True:
        if (case["id"] != "physical_baseline" or case["kind"] != "physical" or
                case["params"] != HISTORICAL_PHYSICAL_BASELINE or cumulative_requested_rf != 0):
            raise ValueError("bez potvrdených limitov je povolený iba jeden presný historický fyzikálny baseline")
        return 40.0
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


def repeat_statistics(result: dict[str, Any]) -> dict[str, Any]:
    rows = [row.get("returned", {}).get("metrics", {}) for row in result.get("tests", [])
            if row.get("id", "").startswith("repeat_")]
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


def markdown_report(result: dict[str, Any]) -> str:
    lines = ["# Séria testov SpinQ Gemini Lab", "", f"Začiatok: {result['started_utc']}",
             f"Stav série: {result.get('state', 'running')}",
             f"Rozsah: {result.get('execution_scope', 'NEOVERENÉ')}",
             f"SDK: {result.get('environment', {}).get('sdk_version') or 'NEZNÁME'}", "",
             f"Pokusy o skutočné experimenty: {result.get('real_hardware_attempts', 0)}; potvrdene dokončené: {result.get('real_hardware_completed', 0)}.",
             "Ak je počet 0, prístroj sa týmto programom nemeral. Prijaté údaje sú dekódované chart body; RAW ADC nie je potvrdené.",
             "Historický 40 µs baseline je iba skorší úspešný pokus na tomto prístroji, nie schválený limit pre ďalšiu sériu.",
             "", "## Testy", ""]
    for row in result.get("tests", []):
        lines += [f"### {row['id']} — {row.get('status', 'NEOVERENÉ')}", "",
                  f"- Poslané: `{json.dumps(row.get('sent'), ensure_ascii=False, default=str)}`" if row.get("sent") is not None else "- Poslané: nič",
                  f"- Vrátené: `{json.dumps(row.get('returned'), ensure_ascii=False, default=str)}`" if row.get("returned") is not None else "- Vrátené: nič",
                  f"- Účinok: `{json.dumps(row.get('effect'), ensure_ascii=False, default=str)}`" if row.get("effect") is not None else "- Účinok: NEOVERENÉ",
                  f"- Dôvod: {row.get('reason', '—')}", ""]
    lines += ["## Čo vieme ovládať a získať", "",
              "Fyzikálne účinky sú uvedené pri jednotlivých testoch. Skutočne prijaté polia sú v measurement/*/field_catalog.json.",
              "Úplné pôvodné dekódované číselné grafy sú v measurement/*/original_scientific.json a events.jsonl.gz.",
              "Serverová FFT/fit sa podľa overeného SDK 1.0.2 nedá preukázateľne vypnúť; lokálna FFT a metriky sú oddelené.",
              "Interná akvizícia, príprava, ADC reťazec a fyzický RF výstup zostávajú NEZNÁME bez ďalších dôkazov.",
              "Opakovania nedávajú bezšumovú pravdu; denoising treba hodnotiť na nezávislých meraniach.",
              "", "## Variabilita opakovaní", "",
              "`"+json.dumps(result.get("repeat_statistics", {}), ensure_ascii=False, default=str)+"`",
              "", "## Lokálna kalibračná slučka", "",
              "`"+json.dumps(result.get("calibration_selection", {}), ensure_ascii=False, default=str)+"`",
              "", "## Chyby a zastavenia", ""]
    lines += [f"- {item}" for item in result.get("errors", [])] or ["- Žiadne zaznamenané."]
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
            if path.is_file() and path != archive_path and path != temporary and path.name != "events.jsonl" and not path.name.endswith(".tmp"):
                archive.write(path, str(path.relative_to(out)).replace("\\", "/"))
    os.replace(temporary, archive_path)


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
    link = adapter = recorder = None
    connected = False
    exit_code = 0
    cases: list[dict[str, Any]] = []
    try:
        config = load_config(args.config)
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
            "operating_limits": "not supplied or established; historical mode permits only one exact operator-reported completed request",
            "relaxation_delay_units": "official experiment page says seconds; SDK source describes microseconds; numeric baseline value 15 is unchanged",
            "internal_repeat_count": "NEZNÁME",
            "receiver_gain_and_filters": "NEZNÁME",
            "gradient_and_shim_write": "not attempted; units and limits unconfirmed",
            "cross_channel_timing": "NEZNÁME; a separate P path is not proof of simultaneous H/P timing",
            "t1_t2": "client experiment classes exist; live long scan not in this budget",
        }
        checkpoint(out, result)
        result["execution_scope"] = ("one_exact_historical_physical_baseline" if
                                     config.get("historical_baseline_only") is True else
                                     "configured_series_requiring_operating_limits")
        checkpoint(out, result)
        password = os.environ.get(config.get("password_env", "SPINQ_AUDIT_PASSWORD"))
        if password is None:
            password = getpass.getpass("Heslo SpinQ (iba v pamäti): ")
        recorder = EventRecorder(out, max_events=256)
        link = SpinQLabLink(config["host"], config["port"], config["account"], password)
        del password
        adapter = AuditAdapter(link, recorder, mode="active", owns_connection=True)
        adapter.attach()
        if not link.connect() or not link.wait_for_login(timeout=10):
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
        checkpoint(out, result)
        max_experiments = operation.get("max_experiments", 0)
        pause = operation.get("pause_seconds", 0)
        timeout = operation.get("timeout_seconds", 0)
        if type(max_experiments) is not int or not 1 <= max_experiments <= 100:
            raise ValueError("operation.max_experiments musí byť 1..100")
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
                                       "status": "NEOVERENÉ", "sent": None, "returned": None,
                                       "effect": None, "reason": "nezačaté"}
                result["tests"].append(row)
                checkpoint(out, result)
                if case.get("enabled") is False:
                    row["reason"] = case.get("skip_reason", "vypnuté v konfigurácii")
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
                if config.get("historical_baseline_only") is not True:
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
                        params.makePps = desired["makePps"]
                        params.samplePath = desired["samplePath"]
                        params.custom_freq = False
                    wire = experiment.get_experiment_parameter()
                    actual = json.loads(wire["params"])
                    if case["kind"] == "physical" and actual != desired:
                        raise ValueError("SDK serializoval iný fyzikálny payload")
                    if actual.get("pulse") != desired["pulse"] or actual.get("samplePath") != desired["samplePath"]:
                        raise ValueError("finálny pulz/kanál sa zmenil pri serializácii")
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
                    measurement = out/"measurement"/case["id"]
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
                    if not row["returned"]["server_ack_observed"]:
                        raise RuntimeError("potvrdenie servera sa nezachytilo; ďalšie merania sa neposielajú")
                    if case["id"] == "physical_baseline":
                        last_baseline = metric
                    if case["id"].startswith("repeat_") and metric.get("fid_status") == "paired":
                        last_baseline = metric
                        result["repeat_statistics"] = repeat_statistics(result)
                    if case.get("return_for"):
                        original = next((r for r in result["tests"] if r["id"] == case["return_for"]), None)
                        if original and original.get("returned"):
                            original["effect"] = effect(original["returned"].get("metrics", {}),
                                                        last_baseline, metric, result.get("repeat_statistics"))
                            if (original["returned"].get("transport_complete_confirmed") and
                                    analysis["transport_complete_confirmed"]):
                                original["status"] = original["effect"]["status"]
                            else:
                                original["status"] = "NEOVERENÉ"
                                original["effect"]["reason"] += "; prenos kompletnosti grafov nie je potvrdený"
                        last_baseline = metric
                    elif case["id"] not in {"physical_baseline"} and not case["id"].startswith("repeat_"):
                        row["effect"] = effect(metric, last_baseline)
                    complete_chart = analysis["transport_complete_confirmed"]
                    data_verified = (complete_chart and analysis["chart_count"] and
                                     (case["kind"] != "physical" or metric.get("fid_status") == "paired"))
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
                    if case["id"] == "rabi_high":
                        scan = [(name, metrics_by_id.get(name, {})) for name in
                                ("rabi_low", "rabi_center", "rabi_high")]
                        metric_key = ("peak_magnitude" if all(number(m.get("peak_magnitude")) for _, m in scan)
                                      else "rabi_real_server_result" if all(number(m.get("rabi_real_server_result")) for _, m in scan)
                                      else None)
                        if metric_key:
                            selected = max(scan, key=lambda item: abs(item[1][metric_key]))[0]
                            chosen = next(item for item in cases if item["id"] == selected)
                            cases.append({"id": "rabi_control", "kind": "rabi",
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
                    "status": "NEOVERENÉ", "sent": None, "returned": None, "effect": None,
                    "reason": "séria sa zastavila pred týmto testom"})
        if connected and link is not None:
            try:
                link.disconnect()  # Own client only; this does not abort a task on hardware.
            except Exception as exc:
                result["errors"].append("disconnect: "+redact(str(exc)))
        if adapter is not None:
            result["capture"] = {"decoder_failures": adapter.decoder_failures,
                                 "outgoing_types": sorted(set(adapter.outgoing))}
            adapter.detach()
        if recorder is not None:
            result["recorder"] = recorder.close()
        if result["errors"]:
            prior = (out/"errors.log").read_text(encoding="utf-8")
            atomic_bytes(out/"errors.log", (prior+"\n"+"\n".join(result["errors"])+"\n").encode("utf-8"))
        result["finished_utc"] = utc_now()
        checkpoint(out, result)
        bundle(out)
        print(f"Výsledky: {out/'results.zip'}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
