"""Durable evidence, reports and isolated result publishing for experiment 01.

The learner receives only explicitly returned complex FID arrays. Original
transport events and applied payloads are retained for audit in separate files.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np


PART_BYTES = 80_000_000


def jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, complex):
        return {"re": float(value.real), "im": float(value.imag)}
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(jsonable(data), ensure_ascii=False, indent=2,
                         allow_nan=False).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending_",
                                     suffix=".json", delete=False) as temp:
        temp.write(payload)
        pending = Path(temp.name)
    try:
        for attempt in range(6):
            try:
                os.replace(pending, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(.05 * (attempt + 1))
    finally:
        pending.unlink(missing_ok=True)


def _sanitized(value: Any) -> Any:
    """Redact credential fields without altering measured numerical arrays."""
    if isinstance(value, dict):
        return {str(key): ("[REDACTED]" if any(token in str(key).lower()
                  for token in ("password", "token", "secret", "credential", "account"))
                  else _sanitized(item)) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_sanitized(item) for item in value]
    return jsonable(value)


class RunArtifacts:
    def __init__(self, path: Path):
        self.path = Path(path)
        for subdir in ("data", "evaluator_only", "profiles", "models", "plots",
                       "sanitized_logs", "source_snapshot", "vendor_reference"):
            (self.path / subdir).mkdir(parents=True, exist_ok=True)

    def save_acquisition(self, key: str, result: Any, logical_request: Any,
                         *, role: str, channel: str | None = None,
                         block: int | None = None, method: str | None = None) -> dict:
        if not key.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Unsafe acquisition key")
        path = self.path / "data" / f"{key}.npz"
        if path.exists():
            raise FileExistsError(f"Original FID already saved: {path}")
        time_s = np.asarray(result.time_s, float)
        fid = np.asarray(result.fid_complex, complex)
        if time_s.ndim != 1 or fid.shape != time_s.shape or len(fid) < 32:
            raise ValueError("Incomplete FID/time axis")
        if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(fid)):
            raise ValueError("Nonfinite FID")
        if not np.all(np.diff(time_s) > 0):
            raise ValueError("FID time axis is not strictly increasing")
        exported_x = np.asarray(result.exported_axis, float)
        if exported_x.shape != time_s.shape or not np.all(np.isfinite(exported_x)):
            raise ValueError("Original exported x-axis is incomplete")
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending_fid_",
                                         suffix=".npz", delete=False) as temporary:
            np.savez_compressed(temporary, time_s=time_s, exported_x=exported_x,
                                re=fid.real, im=fid.imag)
            pending_fid = Path(temporary.name)
        try:
            for attempt in range(6):
                try:
                    os.replace(pending_fid, path)
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(.05 * (attempt + 1))
        finally:
            pending_fid.unlink(missing_ok=True)
        raw_meta = {"key": key, "task_id": result.task_id, "status": result.status,
                    "role": role, "channel": channel, "block": block, "method": method,
                    "elapsed_s": result.elapsed_s,
                    "requested_logical_command": jsonable(logical_request),
                    "device_snapshot": _sanitized(result.device_snapshot),
                    "timing_evidence": _sanitized(result.timing_evidence),
                    "points": len(fid), "data_file": f"data/{key}.npz",
                    "exported_axis_file": f"data/{key}.npz:exported_x",
                    "source_type": "exported complex FID; ADC provenance unverified"}
        atomic_json(self.path / "data" / f"{key}.json", raw_meta)
        atomic_json(self.path / "evaluator_only" / f"{key}_applied.json", {
            "requested_payload": _sanitized(result.requested_payload),
            "sent_payload": _sanitized(result.sent_payload)})
        atomic_json(self.path / "vendor_reference" / f"{key}.json",
                    _sanitized(result.vendor_reference))
        # Chart/update events are task-filtered by the transport. They are
        # audit evidence; no login or generic socket traffic is written here.
        with (self.path / "sanitized_logs" / f"{key}_events.jsonl").open(
                "w", encoding="utf-8") as stream:
            for event in result.events:
                stream.write(json.dumps(_sanitized(event), ensure_ascii=False,
                                        allow_nan=False) + "\n")
        return raw_meta

    def load_fid(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self.path / "data" / f"{key}.npz", allow_pickle=False) as arrays:
            time_s = np.asarray(arrays["time_s"], float).copy()
            fid = np.asarray(arrays["re"], float) + 1j * np.asarray(arrays["im"], float)
        return time_s, fid

    def write_state(self, state: dict) -> None:
        atomic_json(self.path / "results.json", state)

    def snapshot_sources(self, repo: Path, filenames: list[str]) -> None:
        for filename in filenames:
            origin = repo / filename
            if not origin.is_file():
                continue
            destination = self.path / "source_snapshot" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)

    def comparison_csv(self, rows: list[dict]) -> None:
        columns = ("block", "scenario", "task", "channel", "method", "status",
                   "design_acquisitions", "sequential_stop_check_acquisitions",
                   "stopping_decision", "control_acquisitions", "total_acquisitions",
                   "shared_anchor_acquisitions", "shared_evaluator_acquisitions",
                   "cold_start_acquisitions", "cold_start_seconds",
                   "training_seconds", "validation_seconds", "end_to_end_seconds",
                   "frequency_error_hz", "rf_scale_error", "phase_error_deg",
                   "heldout_complex_error", "reason")
        with (self.path / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: json.dumps(jsonable(row.get(name)), ensure_ascii=False)
                                 if isinstance(row.get(name), (dict, list, tuple)) else
                                 row.get(name) for name in columns})

    def report(self, state: dict) -> None:
        rows = state.get("comparison", [])
        lines = ["# Gemini Lab — 01_bayes_online", "",
                 f"Stav: **{state.get('status', 'UNKNOWN')}**; fyzické akvizície: "
                 f"**{len(state.get('acquisitions', {}))}**; porovnávacie riadky: **{len(rows)}**.", "",
                 "Primárny vstup je exportovaný komplexný FID. Vzorky FID nie sú "
                 "nezávislé kvantové shots. Výrobné FFT a skóre nie sú vstupom učenia. "
                 "Kalibračné H/P FID používajú spoločnú serverovú prípravu "
                 "(`makePps=True`); samostatná vlastná PPS vetva ju vypína.", "",
                 "## Porovnanie", "",
                 "| Blok | Úloha | Kanál | Metóda | Stav | Akvizície ramena | Samostatný štart¹ | Čas ramena (s) | Samostatný štart¹ (s) | "
                 "Chyba df (Hz) | Chyba RF | Chyba fázy (°) | Kontrolná chyba |",
                 "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        if state.get("profile") == "quick":
            lines[4:4] = ["**Krátky protokol:** iba H kanál, tri scenáre a priamo "
                          "spárované B (klasický komplexný fit) proti D (adaptívny Bayes). "
                          "A, C, väzba, PPS a Bell nie sú v tomto behu merané. "
                          "Jeden nominálny anchor stojí najviac 18 FID; "
                          "po šiestich FID bez koherentného signálu sa beh zastaví. "
                          "Dve platené kontroly skorého zastavenia na rameno "
                          "sú oddelené od finálnych kontrol. "
                          "v každom bloku sú nové referencie, tréningy a kontrolné FID. "
                          "Tri páry predstavujú predbežné porovnanie, nie univerzálnu štatistickú výhodu.", ""]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(key, "—")) for key in
                ("block", "task", "channel", "method", "status", "total_acquisitions",
                 "cold_start_acquisitions", "end_to_end_seconds", "cold_start_seconds",
                 "frequency_error_hz", "rf_scale_error",
                 "phase_error_deg", "heldout_complex_error")) + " |")
        if not rows:
            lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
        lines += ["", "¹ Samostatný štart = rameno + spoločný pilot + nové referencie. "
                  "Je to porovnávací náklad jedného ramena, nie skutočný súčet behu. "
                  "Pilot sa fyzicky meria iba raz; skutočný počet úloh je uvedený na začiatku reportu. "
                  "Probe a kontroly driftu sa uvádzajú v celkovom počte úloh."]
        lines += ["", "## Párové porovnanie voči klasickému fitu B", "",
                  "Číselné rozdiely zahŕňajú aj neúspešné ramená a sú deskriptívne. "
                  "Na tvrdenie o výhode sú spôsobilé iba páry z rovnakého bloku/kanála, "
                  "v ktorých obe metódy splnili zmrazené kontroly a nebol zistený drift.", "",
                  "| Metóda | Zhodné bloky/kanály | Číselné páry | Platné páry | Cieľ metóda/B | Priemer metóda−B z platných párov | Deskriptívny priemer metóda−B zo všetkých číselných párov | "
                  "Priemerná zmena akvizícií | Priemerná zmena času (s) | Záver |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for method, item in state.get("paired_summary", {}).items():
            def show(value: Any) -> str:
                return "—" if value is None else str(value)
            lines.append("| " + " | ".join((method,
                     show(item.get("matched_block_channel_pairs", 0)),
                     show(item.get("independent_block_channel_pairs", 0)),
                     show(item.get("valid_target_pairs", 0)),
                     f"{item.get('method_target_reached',0)}/{item.get('classical_target_reached',0)}",
                     show(item.get("paired_valid_heldout_error_difference_mean")),
                     show(item.get("paired_heldout_error_difference_mean")),
                     show(item.get("paired_acquisition_difference_mean")),
                     show(item.get("paired_total_time_difference_mean_s")),
                     str(item.get("conclusion")))) + " |")
        if not state.get("paired_summary"):
            lines.append("| — | 0 | 0 | 0 | 0/0 | — | — | — | — | Zatiaľ bez párov |")
        if state.get("profile") == "quick":
            lines += ["", "### Jednotlivé predbežné páry B verzus D", "",
                      "Záporný rozdiel znamená nižšiu kontrolnú chybu D. "
                      "Neplatný pár zostáva viditeľný, ale nesmie podporiť tvrdenie o výhode.", "",
                      "| Blok | B chyba | D chyba | D − B | B stav | D stav | Platný pár |",
                      "|---:|---:|---:|---:|---|---|---|"]
            for row in state.get("paired_summary", {}).get("D_adaptive_bayes", {}).get("per_block", []):
                lines.append("| " + " | ".join(str(row.get(name, "—")) for name in
                    ("block", "classical_heldout_error", "method_heldout_error",
                     "paired_error_difference", "classical_status", "method_status",
                     "valid_for_advantage_claim")) + " |")
        lines += ["", "## Schopnosti a obmedzenia", ""]
        for name, item in state.get("capabilities", {}).items():
            lines.append(f"- {name}: {item}")
        for error in state.get("errors", []):
            lines.append(f"- Chyba: {error}")
        lines += ["", "Samotný úspech fitu nie je dôkaz výhody. "
                  "Výhoda vyžaduje rovnakú nezávisle overenú kvalitu a nižší počet "
                  "akvizícií alebo celý čas naprieč blokmi. Efektívna NMR matica "
                  "nie je dôkaz prepletenia úplného tepelného súboru.", ""]
        (self.path / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        self._plots(rows)

    def _plots(self, rows: list[dict]) -> None:
        points = [row for row in rows if isinstance(row.get("heldout_complex_error"),
                                                   (int, float)) and
                  math.isfinite(row["heldout_complex_error"])]
        for key, name, label in (("cold_start_acquisitions", "quality_vs_acquisitions.svg",
                                  "Cold-start physical acquisitions"),
                                 ("cold_start_seconds", "quality_vs_time.svg",
                                  "Cold-start measured time (s)")):
            subset = [row for row in points if isinstance(row.get(key), (int, float))
                      and math.isfinite(row[key])]
            width, height = 860, 490
            left, top, right, bottom = 75, 40, 25, 70
            x_max = max([float(row[key]) for row in subset] + [1.]) * 1.05
            y_max = max([float(row["heldout_complex_error"]) for row in subset] + [.1]) * 1.1
            x_scale = (width-left-right)/x_max
            y_scale = (height-top-bottom)/y_max
            colors = {"A_prior_only": "#5d6d7e", "B_classical": "#d35400",
                      "C_fixed_bayes": "#2374ab", "D_adaptive_bayes": "#7b2cbf"}
            svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
                   '<rect width="100%" height="100%" fill="white"/>',
                   f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
                   f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
                   f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="15">{label}</text>',
                   f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-size="15">Heldout relative complex FID error</text>',
                   f'<text x="{left}" y="{height-bottom+23}" font-size="12">0</text>',
                   f'<text x="{width-right-35}" y="{height-bottom+23}" font-size="12">{x_max:.1f}</text>',
                   f'<text x="{left-50}" y="{top+5}" font-size="12">{y_max:.2f}</text>']
            for row in subset:
                x = left + float(row[key])*x_scale
                y = height-bottom-float(row["heldout_complex_error"])*y_scale
                color = colors.get(row.get("method"), "#000")
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" opacity="0.78"/>')
            for i, (method, color) in enumerate(colors.items()):
                x = left + i*184
                svg.append(f'<circle cx="{x}" cy="18" r="5" fill="{color}"/>')
                svg.append(f'<text x="{x+9}" y="22" font-size="12">{method}</text>')
            if not subset:
                svg.append(f'<text x="{width/2}" y="{height/2}" text-anchor="middle" font-size="16">No validated result points</text>')
            svg.append('</svg>')
            (self.path / "plots" / name).write_text("\n".join(svg), encoding="utf-8")

    def archive(self) -> Path:
        target = self.path / "results.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for path in sorted(self.path.rglob("*")):
                if path.is_file() and path != target:
                    archive.write(path, path.relative_to(self.path).as_posix())
        return target


def publish(repo: Path, out: Path, branch: str) -> dict:
    """Push this run's report and complete raw archive to its own branch."""
    if not branch.startswith("benchmark/01_bayes_online/"):
        raise ValueError("Result branch is outside this experiment")
    import re
    if not re.fullmatch(r"benchmark/01_bayes_online/[A-Za-z0-9._-]+", branch):
        raise ValueError("Unsafe result branch")
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"

    def git(*args: str, cwd: Path = repo, timeout: int | None = None) -> str:
        proc = subprocess.run(["git", *args], cwd=cwd, env=env,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True,
                              timeout=timeout or 900)
        if proc.returncode:
            raise RuntimeError(f"git {args[0]} failed: {(proc.stderr or proc.stdout)[-800:]}")
        return proc.stdout.strip()

    remote = git("remote", "get-url", "origin")
    parsed = urlsplit(remote)
    if parsed.scheme not in ("https", "ssh") or parsed.hostname != "github.com" or \
            parsed.password or parsed.username not in (None, "git"):
        raise ValueError("Origin must be a credential-free GitHub URL")
    token = env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")
    with tempfile.TemporaryDirectory(prefix="spinq_online_publish_") as scratch:
        scratch_path = Path(scratch)
        if token:
            if sys.platform == "win32":
                askpass = scratch_path / "askpass.cmd"
                askpass.write_text('@echo off\r\n'
                    'echo %~1 | findstr /I /C:"Username" >nul\r\n'
                    'if not errorlevel 1 (echo x-access-token& exit /b 0)\r\n'
                    'if defined GH_TOKEN (echo %GH_TOKEN%& exit /b 0)\r\n'
                    'if defined GITHUB_TOKEN (echo %GITHUB_TOKEN%& exit /b 0)\r\n'
                    'exit /b 1\r\n', encoding="ascii")
            else:
                askpass = scratch_path / "askpass.sh"
                askpass.write_text('#!/bin/sh\ncase "$1" in *Username*) echo x-access-token;; '
                    '*) if [ -n "$GH_TOKEN" ]; then echo "$GH_TOKEN"; '
                    'else echo "$GITHUB_TOKEN"; fi;; esac\n', encoding="ascii")
                askpass.chmod(0o700)
            env["GIT_ASKPASS"] = str(askpass)
            env["GIT_ASKPASS_REQUIRE"] = "force"
        work = scratch_path / "worktree"
        git("worktree", "add", "--detach", str(work), "HEAD")
        try:
            probe = subprocess.run(["git", "ls-remote", "--heads", "origin", branch],
                cwd=work, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                text=True, timeout=120)
            if probe.returncode:
                raise RuntimeError(f"git ls-remote failed: {probe.stderr[-600:]}")
            if probe.stdout.strip():
                git("fetch", "origin", f"refs/heads/{branch}", cwd=work)
                git("checkout", "--detach", "FETCH_HEAD", cwd=work)
            else:
                git("checkout", "--orphan", "online-results-" + uuid.uuid4().hex[:12], cwd=work)
            if git("ls-files", cwd=work):
                git("rm", "-r", "--cached", ".", cwd=work)
            for child in work.iterdir():
                if child.name == ".git":
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            for name in ("REPORT.md", "comparison.csv", "results.json"):
                source = out / name
                if source.is_file():
                    shutil.copy2(source, work / name)
            archive = out / "results.zip"
            with archive.open("rb") as incoming:
                for index in range(1, 10000):
                    chunk = incoming.read(PART_BYTES)
                    if not chunk:
                        break
                    filename = "results.zip" if archive.stat().st_size <= PART_BYTES else \
                        f"results.zip.part{index:04d}"
                    (work / filename).write_bytes(chunk)
            if archive.stat().st_size > PART_BYTES:
                (work / "HOW_TO_JOIN.txt").write_text(
                    "PowerShell: $p=Get-ChildItem results.zip.part* | Sort-Object Name; "
                    "$o=[IO.File]::Create('results.zip'); try { foreach($f in $p) "
                    "{ $i=[IO.File]::OpenRead($f.FullName); try {$i.CopyTo($o)} "
                    "finally {$i.Dispose()} } } finally {$o.Dispose()}\n",
                    encoding="utf-8")
            description = ("REPORT.md and comparison.csv are summaries. Join ZIP parts "
                           "if necessary; results.zip contains original exported FID, "
                           "sanitized events and source snapshot.\n")
            (work / "README.md").write_text(
                "# 01_bayes_online results\n\n" + description,
                encoding="utf-8")
            git("add", "-A", cwd=work)
            if git("status", "--porcelain", cwd=work):
                git("-c", "user.name=SpinQ Online Results",
                    "-c", "user.email=spinq-online@users.noreply.github.com",
                    "commit", "-m", f"Results {out.name}", cwd=work)
                git("push", "origin", f"HEAD:refs/heads/{branch}", cwd=work)
            return {"status": "UPLOAD_SUCCEEDED", "branch": branch}
        except Exception as exc:
            reason = str(exc)
            for secret in (env.get("GH_TOKEN"), env.get("GITHUB_TOKEN")):
                if secret:
                    reason = reason.replace(secret, "[REDACTED]")
            return {"status": "UPLOAD_FAILED", "branch": branch,
                    "reason": reason[:1000]}
        finally:
            try:
                git("worktree", "remove", "--force", str(work))
            except Exception:
                pass
