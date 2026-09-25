"""Local scientific checks. Chart points are never called ADC samples."""

from __future__ import annotations

import gzip
import json
import math
import os
import statistics
import struct
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import atomic_bytes, atomic_json, redact, utc_now

STATIC_FIELD_SCHEMA = {
    "metadata.sequence_id": ("int64", "server message metadata; unit unknown"),
    "metadata.timestamp": ("int64", "server timestamp; unit unknown until confirmed"),
    "chart_data.taskId": ("string", "task identity"),
    "chart_data.group": ("string", "experiment group"),
    "chart_data.chart_name": ("string", "curve name, e.g. fidRe; not proof of raw ADC"),
    "chart_data.path": ("optional string decoded with absent/default ambiguity", "chart channel"),
    "chart_data.qubit": ("optional string decoded with absent/default ambiguity", "qubit label"),
    "chart_data.step": ("optional string decoded with absent/default ambiguity", "observation step"),
    "chart_data.points": ("array[N,2] of decoded protobuf float32", "original plotted x,y values"),
    "chart_data.points[]": ("array[2]", "one x,y point"),
    "chart_data.points[][]": ("float32 decoded to Python float", "one coordinate"),
    "json_data.connected": ("bool when present", "device status, received not default"),
    "json_data.lockState": ("bool when present", "device lock state, received not default"),
    "json_data.temperature": ("number when present", "units unknown until verified"),
    "json_data.pulseParam": ("object when present", "parameter group, not direct RF output measurement"),
    "json_data.ppsParam": ("object when present", "state-preparation parameter group"),
    "json_data.sampleParam": ("object when present", "sampling parameter group"),
    "json_data.shimmingParam": ("object when present", "shim parameter group, read only in audit"),
    "json_data.lockParam": ("object when present", "lock parameter group"),
}


def read_events(path: Path) -> Iterable[dict[str, Any]]:
    source = path / "events.jsonl.gz" if path.is_dir() else path
    if source.is_dir():
        source = source / "events.jsonl.gz"
    if not source.exists() and path.is_dir():
        source = path / "events.jsonl"
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Neplatná udalosť na riadku {number}") from exc


def field_catalog(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}

    def visit(value: Any, path: str):
        kind = type(value).__name__
        item = catalog.setdefault(path, {"path": path, "types": {}, "occurrences": 0,
                                         "sample_shape": None})
        item["types"][kind] = item["types"].get(kind, 0) + 1
        item["occurrences"] += 1
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(nested, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            if item["sample_shape"] is None:
                item["sample_shape"] = [len(value)]
                if value and isinstance(value[0], list):
                    item["sample_shape"].append(len(value[0]))
            # Schema/catalog only needs a representative nested member; full
            # numerical arrays remain in original_scientific.json/NPZ.
            if value:
                visit(value[0], f"{path}[]")

    for value in values:
        visit(value, "")
    for path, (type_name, meaning) in STATIC_FIELD_SCHEMA.items():
        entry = catalog.setdefault(path, {"path": path, "types": {}, "occurrences": 0,
                                          "sample_shape": None})
        entry["static_type"] = type_name
        entry["meaning"] = meaning
        entry["source"] = "official SpinQTech message.proto / SDK 1.0.2 source"
    for entry in catalog.values():
        entry["observed"] = entry["occurrences"] > 0
    return sorted(catalog.values(), key=lambda item: item["path"])


def _axis_info(points: list[Any]) -> dict[str, Any]:
    valid = all(isinstance(p, list) and len(p) == 2 and
                all(type(v) in (int, float) and math.isfinite(v) for v in p)
                for p in points)
    if not valid:
        return {"valid_finite_pairs": False, "count": len(points)}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    diffs = [b - a for a, b in zip(xs, xs[1:])]
    return {"valid_finite_pairs": True, "count": len(points),
            "x_min": min(xs) if xs else None, "x_max": max(xs) if xs else None,
            "y_min": min(ys) if ys else None, "y_max": max(ys) if ys else None,
            "strictly_increasing": all(d > 0 for d in diffs),
            "duplicate_x": len(xs) - len(set(xs)),
            "median_axis_step": statistics.median(diffs) if diffs else None}


def _fft(samples: list[complex]) -> list[complex]:
    n = len(samples)
    if not n or n & (n - 1):
        raise ValueError("FFT length must be power of two")
    data = samples[:]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            data[i], data[j] = data[j], data[i]
    length = 2
    while length <= n:
        angle = -2 * math.pi / length
        base = complex(math.cos(angle), math.sin(angle))
        for start in range(0, n, length):
            factor = 1 + 0j
            half = length // 2
            for offset in range(half):
                even = data[start + offset]
                odd = data[start + offset + half] * factor
                data[start + offset] = even + odd
                data[start + offset + half] = even - odd
                factor *= base
        length *= 2
    return [value / n for value in data]


def _fit_log_envelope(signal: list[complex]) -> dict[str, Any]:
    """Exploratory log-linear magnitude fit; never a calibrated T2 claim."""
    stop = min(len(signal), max(8, len(signal) // 4))
    points = [(i, math.log(abs(value))) for i, value in enumerate(signal[:stop]) if abs(value) > 0]
    if len(points) < 3:
        return {"status": "insufficient_nonzero_points", "window_indices": [0, stop]}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xbar, ybar = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x-xbar)**2 for x in xs)
    slope = sum((x-xbar)*(y-ybar) for x, y in points) / denominator
    intercept = ybar - slope*xbar
    predicted = [intercept + slope*x for x in xs]
    residual = sum((y-yhat)**2 for y, yhat in zip(ys, predicted))
    total = sum((y-ybar)**2 for y in ys)
    return {"status": "exploratory_fit", "model": "log(abs(FID)) = intercept + slope * sample_index",
            "window_indices": [0, stop], "points_used": len(points),
            "slope_per_sample": slope, "intercept": intercept,
            "r_squared": 1 - residual/total if total > 0 else None,
            "decay_constant_samples_if_negative_slope": -1/slope if slope < 0 else None,
            "physical_T2_claim": False}


def _npy_f64(values: list[float]) -> bytes:
    # NumPy v1.0 .npy header, little-endian float64, no object dtype/pickle.
    header = repr({"descr": "<f8", "fortran_order": False, "shape": (len(values),)}).encode("ascii")
    padding = (-(10 + len(header) + 1)) % 16
    header += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack(
        f"<{len(values)}d", *values)


def _extract_charts(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    charts = []
    issues = []
    blocks: dict[tuple[Any, Any], int] = defaultdict(int)
    open_blocks: set[tuple[Any, Any]] = set()
    for event in events:
        body = event.get("payload", {}).get("chart_data") or event.get("payload", {}).get("json_data")
        if not isinstance(body, dict):
            continue
        key = (body.get("taskId"), body.get("group"))
        kind = event.get("kind")
        if kind == "s_post_exp_chart_updated_started":
            blocks[key] += 1
            open_blocks.add(key)
        elif kind == "s_post_exp_chart_updated_finished":
            open_blocks.discard(key)
        elif kind == "s_post_exp_chart_updated":
            chart = {"seq": event.get("seq"), "task_id": body.get("taskId"),
                     "group": body.get("group"), "chart_name": body.get("chart_name"),
                     "path": body.get("path"), "qubit": body.get("qubit"),
                     "step": body.get("step"), "block": blocks[key] or None,
                     "boundary_observed": key in open_blocks,
                     "server_metadata": redact(event.get("payload", {}).get("metadata", {})),
                     "points": body.get("points", []),
                     "provenance": event.get("source", "unknown")}
            chart["axis_check"] = _axis_info(chart["points"])
            if not chart["axis_check"]["valid_finite_pairs"]:
                issues.append(f"curve {chart['seq']}: malformed/non-finite points")
            charts.append(chart)
    return charts, issues


def compare_requested(analysis: dict[str, Any], records: list[dict[str, Any]],
                      plan: dict[str, Any] | None) -> None:
    if not plan:
        return
    test_params = {test.get("id"): test.get("params", {}) for test in plan.get("tests", [])}
    by_task = {record.get("task_id"): test_params.get(record.get("id"), {})
               for record in records if record.get("task_id")}
    for pair in analysis.get("fid_pairs", []):
        task_id = pair.get("identity", [None])[0]
        params = by_task.get(task_id)
        if not params:
            pair["requested_vs_received"] = {"status": "request_or_task_id_unavailable"}
            continue
        received = pair.get("points")
        expected = params.get("sampleCount")
        pair["requested_vs_received"] = {
            "requested_sampleCount": expected, "received_fid_points": received,
            "point_count_equal": received == expected if received is not None else None,
            "requested_sampleFre": params.get("sampleFre"),
            "axis_step_comparison": "unresolved: chart axis unit and server resampling not confirmed"}


def analyze_events(events: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    started = time.monotonic()
    charts, issues = _extract_charts(events)
    groups: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for chart in charts:
        key = (chart["task_id"], chart["group"], chart["path"], chart["qubit"],
               chart["step"], chart["block"])
        groups[key][chart["chart_name"]].append(chart)
    pair_results = []
    arrays: dict[str, list[float]] = {}
    for index, (key, curves) in enumerate(groups.items()):
        real = curves.get("fidRe", [])
        imag = curves.get("fidIm", [])
        if not real and not imag:
            continue
        entry: dict[str, Any] = {"identity": list(key), "fidRe_count": len(real),
                                 "fidIm_count": len(imag), "status": "unpaired"}
        if len(real) != 1 or len(imag) != 1:
            entry["reason"] = "missing or repeated curve in same task/channel/step/block"
            issues.append(f"FID group {index}: {entry['reason']}")
        else:
            rp, ip = real[0]["points"], imag[0]["points"]
            if not real[0]["axis_check"]["valid_finite_pairs"] or not imag[0]["axis_check"]["valid_finite_pairs"]:
                entry["reason"] = "non-finite or malformed chart"
            elif len(rp) != len(ip) or any(a[0] != b[0] for a, b in zip(rp, ip)):
                entry["reason"] = "real/imaginary axes differ; no interpolation performed"
                issues.append(f"FID group {index}: axes mismatch")
            else:
                signal = [complex(a[1], b[1]) for a, b in zip(rp, ip)]
                n = len(signal)
                if n:
                    fft_size = 1 << (n - 1).bit_length()
                    spectrum = _fft(signal + [0j] * (fft_size - n))
                    peak = max(range(fft_size), key=lambda i: abs(spectrum[i]))
                    signed_peak = peak if peak <= fft_size // 2 else peak - fft_size
                    dt = real[0]["axis_check"]["median_axis_step"]
                    entry.update({"status": "paired", "points": n,
                                  "axis_unit": "unconfirmed", "axis_step_as_received": dt,
                                  "amplitude_first": abs(signal[0]), "phase_first_rad": math.atan2(signal[0].imag, signal[0].real),
                                  "fft": {"convention": "forward exp(-2pi i kn/N), 1/N, rectangular window, zero padded",
                                          "size": fft_size, "peak_signed_bin": signed_peak,
                                          "peak_cycles_per_sample": signed_peak / fft_size,
                                          "peak_magnitude": abs(spectrum[peak]),
                                          "integrated_magnitude": sum(abs(v) for v in spectrum)},
                                  "envelope_fit": _fit_log_envelope(signal),
                                  "tail_variability_not_noise_truth": statistics.pstdev(
                                      abs(v) for v in signal[n * 3 // 4:]) if n > 3 else None})
                    server_mod = curves.get("fftMod", [])
                    if len(server_mod) == 1 and server_mod[0]["axis_check"]["valid_finite_pairs"]:
                        server_points = server_mod[0]["points"]
                        server_peak = max(range(len(server_points)), key=lambda k: server_points[k][1]) if server_points else None
                        entry["server_fft_comparison"] = {
                            "server_points": len(server_points), "local_fft_points": fft_size,
                            "server_peak_array_index": server_peak,
                            "local_peak_array_index": peak,
                            "index_equal_without_axis_alignment": server_peak == peak if server_peak is not None else None,
                            "interpretation": "chart ordering, centering, scale and preprocessing unconfirmed"}
                    arrays[f"fid_{index}_x"] = [float(p[0]) for p in rp]
                    arrays[f"fid_{index}_re"] = [float(p[1]) for p in rp]
                    arrays[f"fid_{index}_im"] = [float(p[1]) for p in ip]
                    arrays[f"fid_{index}_fft_re"] = [v.real for v in spectrum]
                    arrays[f"fid_{index}_fft_im"] = [v.imag for v in spectrum]
                    entry["npz_prefix"] = f"fid_{index}"
                else:
                    entry["reason"] = "empty FID"
        pair_results.append(entry)
    out.mkdir(parents=True, exist_ok=True)
    # JSON is the primary unchanged numerical export; array conversion is a
    # separate convenience copy in IEEE float64 (not extra physical precision).
    atomic_bytes(out / "original_scientific.json", json.dumps(redact(
        {"label": "as_received_decoded_chart_values", "raw_adc_confirmed": False,
         "transport_point_dtype": "protobuf float32 for SDK 1.0.2 ChartData; decoded into Python float",
         "nonfinite_encoding": "Python JSON NaN/Infinity tokens if present", "charts": charts}),
        ensure_ascii=False, allow_nan=True, separators=(",", ":")).encode("utf-8"))
    npz_temp = out / "scientific_arrays.npz.tmp"
    with zipfile.ZipFile(npz_temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, values in arrays.items():
            archive.writestr(name + ".npy", _npy_f64(values))
    os.replace(npz_temp, out / "scientific_arrays.npz")
    result = {"created_utc": utc_now(), "event_count": len(events), "chart_count": len(charts),
              "fid_pairs": pair_results, "issues": issues,
              "raw_adc_confirmed": False,
              "synthetic_or_mock_data": any("synthetic" in str(chart["provenance"]).lower() or
                                            "mock" in str(chart["provenance"]).lower() for chart in charts),
              "npz_conversion": "decoded numeric chart values -> float64; no added physical precision",
              "analysis_wall_seconds": round(time.monotonic() - started, 6)}
    finished_tasks = {event.get("payload", {}).get("json_data", {}).get("taskId")
                      for event in events if event.get("kind") == "s_post_exp_finished"}
    closed_blocks = {(event.get("payload", {}).get("json_data", {}).get("taskId"),
                      event.get("payload", {}).get("json_data", {}).get("group"))
                     for event in events if event.get("kind") == "s_post_exp_chart_updated_finished"}
    result["transport_complete_confirmed"] = bool(charts) and all(
        chart["boundary_observed"] and chart["task_id"] in finished_tasks and
        (chart["task_id"], chart["group"]) in closed_blocks for chart in charts)
    result["transport_completeness_note"] = (
        "SDK message boundaries and final task events observed" if result["transport_complete_confirmed"]
        else "not confirmed: missing boundaries/final event or only aggregated legacy result")
    cadence = {}
    arrivals: dict[str, list[int]] = defaultdict(list)
    for event in events:
        if event.get("kind", "").startswith("s_post_device") or event.get("kind") == "s_post_lock_data":
            stamp = event.get("received_monotonic_ns")
            if type(stamp) is int:
                arrivals[event["kind"]].append(stamp)
    for kind, stamps in arrivals.items():
        gaps = [(b-a)/1e9 for a, b in zip(stamps, stamps[1:]) if b >= a]
        cadence[kind] = {"observed_messages": len(stamps),
                         "median_gap_seconds": statistics.median(gaps) if gaps else None,
                         "scope": "observed client arrivals, not guaranteed server schedule"}
    result["telemetry_cadence"] = cadence
    atomic_json(out / "analysis.json", result)
    atomic_json(out / "field_catalog.json", field_catalog(
        event.get("payload", {}) for event in events))
    return result


def import_legacy_result(source: Path, out: Path) -> int:
    """Import SDK result graphs, preserving lower provenance than pre-handler events."""
    data = json.loads(source.read_text(encoding="utf-8"))
    provenance = "synthetic_test" if data.get("data_origin") in {"synthetic_test", "mock"} else "legacy_sdk_result_not_prehandler_unverified"
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    if isinstance(data.get("physical_result"), dict):
        reports.append(data["physical_result"])
    reports.extend(row.get("raw_result", {}) for row in data.get("measurements", []) if isinstance(row, dict))
    count = 0
    with (out / "events.jsonl").open("w", encoding="utf-8") as target:
        for report_index, report in enumerate(reports):
            graphs = report.get("result", {}).get("graph", [])
            for block, graph in enumerate(graphs):
                if not isinstance(graph, dict):
                    continue
                for name, points in graph.items():
                    count += 1
                    event = {"seq": count, "received_utc": None, "received_monotonic_ns": None,
                             "kind": "s_post_exp_chart_updated", "source": provenance,
                             "payload": {"chart_data": {"taskId": report.get("id"), "group": None,
                                                        "chart_name": name, "path": None, "qubit": None,
                                                        "step": block, "points": points}}}
                    target.write(json.dumps(event, ensure_ascii=False, allow_nan=True) + "\n")
    return count
