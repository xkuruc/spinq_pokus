"""Static SDK inventory. Importing the vendor package is intentionally avoided."""

from __future__ import annotations

import ast
import importlib.metadata as metadata
import json
import platform
import sys
import zipfile
from pathlib import Path
from typing import Any

from .common import utc_now

REFERENCE = json.loads((Path(__file__).parent / "sdk_reference.json").read_text())
DOC = "https://doc.spinq.cn/doc/SpinQLAB_Link/en/api/spinqlablink.html"


class SourceSet:
    def __init__(self, wheel: Path | None = None):
        self.wheel = wheel
        self.dist = None
        self.version = None
        self.origin = "unavailable"
        self.editable = None
        self._zip = None
        if wheel is not None:
            self._zip = zipfile.ZipFile(wheel)
            self.origin = "downloaded_reference_wheel_not_installed"
            self.version = "1.0.2" if wheel.name.startswith("spinqlablink-1.0.2-") else None
        else:
            try:
                self.dist = metadata.distribution("spinqlablink")
                self.version = self.dist.version
                self.origin = "installed_distribution"
                direct = self.dist.read_text("direct_url.json")
                if direct:
                    self.editable = json.loads(direct).get("dir_info", {}).get("editable", False)
            except metadata.PackageNotFoundError:
                pass

    def read(self, relative: str) -> str | None:
        try:
            if self._zip:
                return self._zip.read(relative).decode("utf-8")
            if self.dist:
                path = Path(self.dist.locate_file(relative))
                if path.is_file():
                    return path.read_bytes().decode("utf-8")
        except (KeyError, OSError, UnicodeDecodeError):
            pass
        return None

    def inventory(self) -> dict[str, Any]:
        files = {}
        for path in REFERENCE["files"]:
            source = self.read(path)
            files[path] = {"available": source is not None,
                           "bytes": len(source.encode("utf-8")) if source is not None else None}
        dependencies = {}
        for package in ("numpy", "pydantic", "protobuf", "scipy", "matplotlib", "PyQt5", "pyqtgraph"):
            try:
                dependencies[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                dependencies[package] = None
        return {"version": self.version, "origin": self.origin, "editable": self.editable,
                "installed_location": str(self.dist.locate_file("")) if self.dist else None,
                "interpreter_dependencies": dependencies,
                "files": files, "local_source_modification_status": "not_verified"}


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.unparse(node)


def _model_fields(source: str, relative: str) -> list[dict[str, Any]]:
    result = []
    tree = ast.parse(source)
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for item in cls.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            value = item.value
            if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.func.id != "Field":
                continue
            options = {kw.arg: _literal(kw.value) for kw in value.keywords if kw.arg}
            result.append({"class": cls.name, "name": item.target.id,
                           "type": ast.unparse(item.annotation), "default": options.get("default"),
                           "validation": {k: v for k, v in options.items() if k in {"ge", "le", "gt", "lt", "step"}},
                           "description": options.get("description"),
                           "source": f"{relative}:{item.lineno}"})
    return result


def _symbols(source: str, relative: str) -> list[dict[str, Any]]:
    result = []
    for cls in (n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)):
        result.append({"symbol": cls.name, "source": f"{relative}:{cls.lineno}", "kind": "class"})
        for method in cls.body:
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result.append({"symbol": f"{cls.name}.{method.name}",
                               "source": f"{relative}:{method.lineno}", "kind": "method",
                               "signature_ast": ast.unparse(method.args)})
    return result


def _experiment_map(source: str) -> list[dict[str, Any]]:
    for cls in (n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "ExperimentManager"):
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name == "__init__":
                for node in ast.walk(method):
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "EXPERIMENT_TYPE_MAP" for t in node.targets) and isinstance(node.value, ast.Dict):
                        rows = []
                        for key, value in zip(node.value.keys, node.value.values):
                            module, impl, params = _literal(value)
                            rows.append({"type_symbol": ast.unparse(key), "module": module,
                                         "implementation": impl, "parameter_class": params,
                                         "source": f"spinqlablink/experiment/ExperimentManager.py:{node.lineno}"})
                        return rows
    return []


FIELD_PATHS = [
    ("pulse.path", "pulse_channel", "spinqlablink/utils/pulse.py", "Pulse", "hPulse/pPulse", "channel index"),
    ("pulse.amplitude", "pulse_amplitude", "spinqlablink/utils/pulse.py", "Pulse", "am", "percent, SDK description"),
    ("pulse.phase", "pulse_phase", "spinqlablink/utils/pulse.py", "Pulse", "phase", "degree, SDK description"),
    ("pulse.width", "pulse_width", "spinqlablink/utils/pulse.py", "Pulse", "width", "µs, SDK description"),
    ("pulse.detuning", "pulse_detuning", "spinqlablink/utils/pulse.py", "Pulse", "freshift", "Hz, SDK description"),
    ("gradient.voltage", "temporary_gradient_voltage", "spinqlablink/utils/pulse.py", "Gradient", "value", "V, SDK description"),
    ("gradient.duration", "temporary_gradient_duration", "spinqlablink/utils/pulse.py", "Gradient", "delay", "µs, SDK description"),
    ("physical.sampleFre", "sampling_frequency", "spinqlablink/experiment/exp_layer_physical.py", "ExpLayerPhysicalParameters", "sampleFre", "unknown physical unit; SDK description is frequency"),
    ("physical.sampleCount", "sampling_count", "spinqlablink/experiment/exp_layer_physical.py", "ExpLayerPhysicalParameters", "sampleCount", "points, SDK description"),
    ("physical.sampleDelay", "sampling_delay", "spinqlablink/experiment/exp_layer_physical.py", "ExpLayerPhysicalParameters", "sampleDelay", "µs, SDK description"),
    ("physical.samplePath", "sampling_channel", "spinqlablink/experiment/exp_layer_physical.py", "ExpLayerPhysicalParameters", "samplePath", "channel selector"),
    ("physical.type_setting", "compute_type", "spinqlablink/experiment/exp_layer_physical.py", "ExpLayerPhysicalParameters", "compute_type", "enum, source"),
    ("physical.relaxation_delay", "relaxation_delay", "spinqlablink/experiment/exp_layer_physical.py", "ExpLayerPhysicalParameters", "relaxation_time", "µs in Field; other examples may conflict"),
    ("physical.state_initialization", "state_initialization", "spinqlablink/experiment/exp_layer_physical.py", "ExpLayerPhysicalParameters", "makePps", "boolean"),
    ("physical.stepList", "step_list", "spinqlablink/experiment/exp_layer_physical.py", "ExpLayerPhysicalParameters", "stepList", "unknown"),
    ("physical.h_freShift", "hydrogen_shift", "spinqlablink/experiment/exp_layer_physical.py", "ExpLayerPhysicalParameters", "h_freShift", "unknown, SDK description"),
    ("physical.p_freShift", "phosphorus_shift", "spinqlablink/experiment/exp_layer_physical.py", "ExpLayerPhysicalParameters", "p_freShift", "unknown, SDK description"),
    ("physical.h_freDemo", "hydrogen_demodulation", "spinqlablink/experiment/exp_layer_physical.py", "ExpLayerPhysicalParameters", "h_freDemo", "unknown, SDK description"),
    ("physical.p_freDemo", "phosphorus_demodulation", "spinqlablink/experiment/exp_layer_physical.py", "ExpLayerPhysicalParameters", "p_freDemo", "unknown, SDK description"),
    ("rabi.freq_h", "rabi_h_frequency", "spinqlablink/experiment/exp_rabi.py", "ExpRabiParameters", "freq_h", "MHz client; wire Hz after x1e6"),
    ("rabi.freq_p", "rabi_p_frequency", "spinqlablink/experiment/exp_rabi.py", "ExpRabiParameters", "freq_p", "MHz client; wire Hz after x1e6"),
    ("rabi.custom_freq", "rabi_custom_frequency_switch", "spinqlablink/experiment/exp_rabi.py", "ExpRabiParameters", "custom_freq", "boolean"),
    ("rabi.makePps", "rabi_state_preparation", "spinqlablink/experiment/exp_rabi.py", "ExpRabiParameters", "makePps", "boolean"),
    ("shape.calibrate_sample", "shaped_pulse_sample_kind", "spinqlablink/experiment/exp_shapePulse.py", "ExpShapePulseParameters", "extra", "enum; source"),
]


def discover(wheel: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sources = SourceSet(wheel)
    if sources.origin == "unavailable" and wheel is None:
        snapshot = json.loads((Path(__file__).parent / "reference_snapshot.json").read_text(encoding="utf-8"))
        inventory = snapshot["inventory"]
        inventory["origin"] = "bundled_pypi_1_0_2_snapshot_not_installed"
        inventory["version"] = "1.0.2 reference only"
        environment = {"checked_utc": utc_now(), "python": sys.version.split()[0],
                       "os": platform.platform(), "architecture": platform.machine(),
                       "spinqlablink": inventory,
                       "device_model": {"value": "Gemini Lab", "evidence": "declared_by_user_not_device_confirmed"},
                       "firmware_version": None, "server_version": None,
                       "client_device_id_is_hardware_serial": False,
                       "official_repository_main_commit_checked": "8fe50f65bf87b97bf39dc4e1f8db9363801fd169",
                       "reference_comparison": REFERENCE["comparison_scope"],
                       "reference_python_source_matches_main_after_crlf_normalization": True,
                       "official_docs_checked_utc": "2026-09-25", "offline_import_of_vendor_sdk": False}
        for capability in snapshot["capabilities"]:
            for evidence in capability.get("evidence", []):
                if evidence.get("level") == "source_inspected":
                    evidence["level"] = "reference_wheel_source_inspected_not_local_installation"
            capability["client_exists"] = None
            if capability["status"] == "partially_verified":
                capability["status"] = "not_tested"
        for issue in snapshot["issues"]:
            if issue.get("severity") == "confirmed_in_source":
                issue["severity"] = "confirmed_in_reference_source_not_local_installation"
        return environment, snapshot["capabilities"], snapshot["issues"], snapshot["experiments"]
    inventory = sources.inventory()
    fields: list[dict[str, Any]] = []
    symbols: list[dict[str, Any]] = []
    for relative in inventory["files"]:
        source = sources.read(relative)
        if source:
            fields.extend(_model_fields(source, relative))
            symbols.extend(_symbols(source, relative))
    manager = sources.read("spinqlablink/experiment/ExperimentManager.py")
    experiments = _experiment_map(manager) if manager else []
    for row in experiments:
        module_file = row["module"].replace(".", "/") + ".py"
        source = sources.read(module_file)
        row["implementation_present"] = source is not None and f"class {row['implementation']}(" in source
        row["parameter_class_present"] = source is not None and f"class {row['parameter_class']}(" in source
    inventory["fields"] = fields
    inventory["symbols"] = symbols
    inventory["experiments"] = experiments
    inventory["public_parameter_name_search"] = {
        keyword: [f"{row['class']}.{row['name']}" for row in fields if keyword in row["name"].lower()]
        for keyword in ("fft", "fit", "raw", "adc", "fid_only", "stream")}
    environment = {"checked_utc": utc_now(), "python": sys.version.split()[0], "os": platform.platform(),
                   "architecture": platform.machine(), "spinqlablink": inventory,
                   "device_model": {"value": "Gemini Lab", "evidence": "declared_by_user_not_device_confirmed"},
                   "firmware_version": None, "server_version": None,
                   "client_device_id_is_hardware_serial": False,
                   "official_repository_main_commit_checked": "8fe50f65bf87b97bf39dc4e1f8db9363801fd169",
                   "reference_comparison": REFERENCE["comparison_scope"],
                   "reference_python_source_matches_main_after_crlf_normalization": True,
                   "official_docs_checked_utc": "2026-09-25", "offline_import_of_vendor_sdk": False}
    capabilities = []
    for path, ident, source_file, cls, wire, units in FIELD_PATHS:
        found = next((f for f in fields if f["class"] == cls and f["name"] == path.split(".")[-1] and f["source"].startswith(source_file)), None)
        capabilities.append({"id": ident, "category": path.split(".")[0], "purpose": path,
                             "api_path": f"{cls}.{path.split('.')[-1]}", "signature": found["type"] if found else None,
                             "data_type": found["type"] if found else None,
                             "default": found["default"] if found else None,
                             "validation": found["validation"] if found else None,
                             "wire_name": wire, "units": units,
                             "units_source": found["source"] if found else None,
                             "permission": "active_plan_only", "effect_scope": "experiment_request_not_persistent_calibration",
                             "safety": "requires approved baseline and operating limits",
                             "evidence": ([{"level": "source_inspected", "ref": found["source"]}] if found else []),
                             "documented": None, "client_exists": bool(found), "server_accepted": None,
                             "physical_effect_observed": None,
                             "status": "partially_verified" if found else "not_tested"})
    for ident, category, purpose in [
        ("fid_chart", "acquisition", "complex FID chart data, processing unknown"),
        ("raw_adc", "acquisition", "ADC codes or direct ADC stream"),
        ("fid_only", "server_processing", "disable server FFT and return FID only"),
        ("persistent_shim_write", "calibration", "write stored shim calibration"),
        ("persistent_pulse_calibration", "calibration", "write stored pulse calibration"),
        ("intra_sequence_feedback", "control", "feedback inside a running hardware sequence"),
    ]:
        capabilities.append({"id": ident, "category": category, "purpose": purpose,
                             "api_path": None, "signature": None, "data_type": None,
                             "default": None, "validation": None, "wire_name": None,
                             "units": None, "units_source": None, "permission": "not_authorized",
                             "effect_scope": "unknown", "safety": "no register probing or guessing",
                             "evidence": [], "documented": None, "client_exists": None,
                             "server_accepted": None, "physical_effect_observed": None,
                             "status": "not_tested"})
    for ident, wire, ref in [
        ("device_status_read", "s_post_device_info", "spinqlablink/spinqlablink.py:144-148"),
        ("device_lock_read", "s_post_lock_data", "spinqlablink/spinqlablink.py:140-142"),
        ("device_parameter_read", "s_post_device_param", "spinqlablink/spinqlablink.py:136-138"),
        ("sample_calibration_read", "s_post_sample_calibration_data", "spinqlablink/spinqlablink.py:150-152"),
        ("queue_update_read", "s_post_exp_queue_update", "spinqlablink/spinqlablink.py:154-159"),
        ("pulseParam_read", "pulseParam", "spinqlablink/device/device.py:69-72"),
        ("ppsParam_read", "ppsParam", "spinqlablink/device/device.py:64-67"),
        ("sampleParam_read", "sampleParam", "spinqlablink/device/device.py:74-77"),
        ("shimmingParam_read", "shimmingParam", "spinqlablink/device/device.py:79-82"),
        ("lockParam_read", "lockParam", "spinqlablink/device/device.py:59-62"),
    ]:
        capabilities.append({"id": ident, "category": "telemetry", "purpose": "read received server field without treating defaults as measurement",
                             "api_path": "Device / decoded server message", "signature": None,
                             "data_type": "dict", "default": None, "validation": None,
                             "wire_name": wire, "units": "field-specific or unknown", "units_source": None,
                             "permission": "passive_read", "effect_scope": "local cache / received message",
                             "safety": "no polling of undocumented endpoint", "evidence": [
                                 {"level": "source_inspected", "ref": ref}] if sources.read(ref.split(":")[0]) else [],
                             "documented": None, "client_exists": bool(sources.read(ref.split(":")[0])),
                             "server_accepted": None, "physical_effect_observed": None,
                             "status": "partially_verified" if sources.read(ref.split(":")[0]) else "not_tested"})
    fid = next(item for item in capabilities if item["id"] == "fid_chart")
    if protocol_source := sources.read("spinqlablink/connection/protocol.py"):
        if 'dict_data["chart_data"] = chart_data' in protocol_source:
            fid["client_exists"] = True
            fid["data_type"] = "protobuf float32 x/y decoded to Python float"
            fid["evidence"] = [{"level": "source_inspected", "ref": "spinqlablink/connection/protocol.py:129-142"},
                               {"level": "documented_wire", "ref": "spinqlablink/connection/message.proto:41-54 (official main)"}]
            fid["status"] = "partially_verified"
    issues = []
    if manager and "get_experiment_status" not in manager:
        link = sources.read("spinqlablink/spinqlablink.py") or ""
        if "self.expMgr.get_experiment_status()" in link:
            issues.append({"id": "sdk_status_missing_delegate", "severity": "confirmed_in_source",
                           "evidence": ["spinqlablink/spinqlablink.py:247-248", "spinqlablink/experiment/ExperimentManager.py"],
                           "detail": "Public wrapper delegates to absent manager method; existing controller uses experiment.get_status()."})
    if manager and "if status == ExperimentState.COMPLETED or status == ExperimentState.FAILED" in manager and "return True" in manager:
        issues.append({"id": "sdk_wait_true_on_failed", "severity": "confirmed_in_source",
                       "evidence": ["spinqlablink/experiment/ExperimentManager.py:132-137"],
                       "detail": "wait_for_experiment_completion returns True for FAILED; inspect state separately."})
    protocol = sources.read("spinqlablink/connection/protocol.py") or ""
    link = sources.read("spinqlablink/spinqlablink.py") or ""
    if "self.remaining_data = self.next_data" in protocol and "deserialize_message(data)" in link and "deserialize_message(b'')" not in link:
        issues.append({"id": "sdk_coalesced_frame_delay", "severity": "confirmed_in_source",
                       "evidence": ["spinqlablink/connection/protocol.py:147", "spinqlablink/spinqlablink.py:78"],
                       "detail": "Decoder retains later complete frames; link decodes once per TCP recv, so a final frame may remain pending."})
    if 'chart_data["path"] = message.chart_data.path' in protocol:
        issues.append({"id": "sdk_optional_chart_presence_lost", "severity": "confirmed_in_source",
                       "evidence": ["spinqlablink/connection/protocol.py:132-142", "spinqlablink/connection/message.proto:41-54 (official main)"],
                       "detail": "Decoded chart preserves values but optional path/qubit/step presence is reduced to default strings; pre-handler capture is not full wire capture."})
    physical = sources.read("spinqlablink/experiment/exp_layer_physical.py") or ""
    if 'self.step_lines[data["chart_name"]] = data["points"]' in physical:
        issues.append({"id": "sdk_chart_replacement", "severity": "confirmed_in_source",
                       "evidence": ["spinqlablink/experiment/exp_layer_physical.py:200"],
                       "detail": "Repeated chart_name overwrites prior points in SDK result; pre-handler recording required."})
    if "current_experiment.id" in link:
        issues.append({"id": "sdk_queue_without_experiment", "severity": "confirmed_in_source",
                       "evidence": ["spinqlablink/spinqlablink.py:154-159"],
                       "detail": "Queue update dereferences current_experiment without a null check."})
    pulse_source = sources.read("spinqlablink/utils/pulse.py") or ""
    if '"freshift": self.detuning' in pulse_source and '"am": self.amplitude' in pulse_source:
        issues.append({"id": "pulse_wire_mapping", "severity": "confirmed_in_source",
                       "evidence": ["spinqlablink/utils/pulse.py:27-39", "spinqlablink/experiment/exp_layer_physical.py:85-93"],
                       "detail": "Pulse becomes width/am/phase/freshift and is split by path into hPulse/pPulse; cross-channel timing is not specified."})
    if '"value": self.voltage' in pulse_source and '"delay": self.duration' in pulse_source:
        issues.append({"id": "gradient_wire_mapping", "severity": "confirmed_in_source",
                       "evidence": ["spinqlablink/utils/pulse.py:41-60"],
                       "detail": "Gradient voltage/duration serialize as value/delay; no live gradient write is authorized."})
    device = sources.read("spinqlablink/device/device.py") or ""
    if 'self._params = DeviceParams(device_params)' in device:
        issues.append({"id": "set_device_params_is_local_cache", "severity": "confirmed_in_source",
                       "evidence": ["spinqlablink/device/device.py:168-177"],
                       "detail": "set_device_params updates the local wrapper and observers; this is not a hardware calibration write."})
    if 'locking.get("frequency0", 0.0)' in device:
        issues.append({"id": "telemetry_defaults_not_measurements", "severity": "confirmed_in_source",
                       "evidence": ["spinqlablink/device/device.py:179-211", "spinqlablink/device/device.py:256-260"],
                       "detail": "Missing received fields become zero in SDK cache/getters; auditor preserves original field presence."})
    if '"SpinQLabLink-" + str(uuid.uuid4())' in link:
        issues.append({"id": "client_device_id_not_serial", "severity": "confirmed_in_source",
                       "evidence": ["spinqlablink/spinqlablink.py:50"],
                       "detail": "device_id is generated by the client, not a confirmed hardware serial number."})
    if 'def _handle_sample_calibration_post' in link and 'def _handle_exp_queue_update_post' in link:
        issues.append({"id": "sample_calibration_discarded", "severity": "confirmed_in_source",
                       "evidence": ["spinqlablink/spinqlablink.py:150-152"],
                       "detail": "SDK sample calibration handler only passes; pre-handler capture retains received fields."})
    if 'step=10000' in physical:
        issues.append({"id": "sample_frequency_step_unverified", "severity": "needs_offline_runtime_test",
                       "evidence": ["spinqlablink/experiment/exp_layer_physical.py:75"],
                       "detail": "Field declares step=10000; whether installed Pydantic enforces it on assignment remains unverified here."})
    if 'description="Relaxation delay (µs)"' in physical:
        issues.append({"id": "relaxation_unit_requires_workstation_confirmation", "severity": "blocked_safety",
                       "evidence": ["spinqlablink/experiment/exp_layer_physical.py:68", "spinqlablink/experiment/exp_layer_physical.py:105-106"],
                       "detail": "SDK labels relaxation_delay in µs and sends it as relaxation_time, while prior local script used 15 from an example; exact behavior requires confirmation."})
    if sources.origin != "installed_distribution":
        for capability in capabilities:
            capability["client_exists"] = None
            if capability["status"] == "partially_verified":
                capability["status"] = "not_tested"
            for evidence in capability.get("evidence", []):
                if evidence.get("level") == "source_inspected":
                    evidence["level"] = "reference_wheel_source_inspected_not_local_installation"
        for issue in issues:
            if issue.get("severity") == "confirmed_in_source":
                issue["severity"] = "confirmed_in_reference_source_not_local_installation"
    return environment, capabilities, issues, experiments
