"""Publish final benchmark summaries without touching the working checkout.

No credentials are written to the result or to the temporary Git repository.
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from spinq_audit.common import redact

GIT_TIMEOUT_SECONDS = 900
PART_BYTES = 80 * 1024 * 1024


def _git(args: list[str], cwd: Path, env: dict[str, str]) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, env=env,
                               text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=GIT_TIMEOUT_SECONDS,
                               stdin=subprocess.DEVNULL, check=False)
    if completed.returncode:
        detail = redact((completed.stderr or completed.stdout).strip())
        raise RuntimeError(f"git {args[0]} zlyhal ({completed.returncode}): {detail[:1000]}")
    return completed.stdout.strip()


def _remote(repo: Path, env: dict[str, str]) -> str:
    url = _git(["remote", "get-url", "origin"], repo, env).strip()
    if not url:
        raise RuntimeError("Git remote origin chýba")
    if "://" in url:
        parsed = urlsplit(url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise RuntimeError("Git remote obsahuje prihlasovacie údaje alebo nečakané parametre")
        if parsed.scheme not in {"https", "ssh"}:
            raise RuntimeError("Git remote nepoužíva HTTPS alebo SSH")
    elif not url.startswith("git@github.com:"):
        raise RuntimeError("Neznámy tvar Git remote; automatický push sa nevykoná")
    return url


def _summary_files(out: Path, destination: Path) -> list[str]:
    """Copy only analysis outputs; FID charts and models stay on the Windows PC."""
    data=json.loads((out/"results.json").read_text(encoding="utf-8"))
    if str(data.get("state","")).upper()=="RUNNING":
        raise ValueError("Benchmark is still running; summary upload refused")
    names=[]
    for name in ("REPORT.md","comparison.csv"):
        source=out/name
        if not source.is_file(): raise FileNotFoundError(source)
        shutil.copyfile(source,destination/name)
        names.append(name)
    summary={key:value for key,value in data.items() if key not in ("upload","environment")}
    (destination/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    names.append("summary.json")
    charts=sorted((out/"plots").glob("*.png")) if (out/"plots").exists() else []
    if charts:
        (destination/"plots").mkdir()
        for chart in charts:
            shutil.copyfile(chart,destination/"plots"/chart.name)
            names.append("plots/"+chart.name)
    (destination/"README.md").write_text(
        "# Gemini Lab benchmark: analyzed results\n\n"
        "This branch contains the final report, comparison table, machine-readable "
        "summary and plots. Measured FIDs, decoded events and models remain "
        "on the acquisition computer in the matching results directory.\n",
        encoding="utf-8")
    names.append("README.md")
    return names


def _archive_parts(zip_path: Path, destination: Path) -> list[str]:
    """Publish browsable results plus recoverable full-FID archive parts."""
    out=zip_path.parent
    names=[]
    for name in ("REPORT.md","comparison.csv","results.json",
                 "computation_map.md","reproduction_scope.md","sources.md"):
        source=out/name
        if source.is_file():
            shutil.copyfile(source,destination/name)
            names.append(name)
    plots=out/"plots"
    if plots.is_dir():
        for source in sorted(plots.glob("*.png")):
            target=destination/"plots"/source.name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
            names.append("plots/"+source.name)
    if zip_path.stat().st_size<=PART_BYTES:
        shutil.copyfile(zip_path,destination/"results.zip")
        names.append("results.zip")
        (destination/"README.md").write_text(
            "# Gemini Lab benchmark\n\nREPORT.md and comparison.csv are browsable here. "
            "results.zip contains original exported FID, event journal, vendor references, models and plots.\n",
            encoding="utf-8")
        names.append("README.md")
        return names
    with zip_path.open("rb") as source:
        for index in range(1,10000):
            chunk=source.read(PART_BYTES)
            if not chunk: break
            name=f"results.zip.part{index:04d}"
            (destination/name).write_bytes(chunk)
            names.append(name)
    (destination/"HOW_TO_JOIN.txt").write_text(
        "Spoj results.zip.part0001, part0002, ... v číselnom poradí do results.zip.\n"
        "Windows PowerShell: $parts = Get-ChildItem results.zip.part* | Sort-Object Name; "
        "$out = [IO.File]::Create('results.zip'); try { foreach ($part in $parts) { "
        "$inputFile = [IO.File]::OpenRead($part.FullName); try { $inputFile.CopyTo($out) } "
        "finally { $inputFile.Dispose() } } } finally { $out.Dispose() }\n",
        encoding="utf-8")
    names.append("HOW_TO_JOIN.txt")
    (destination/"README.md").write_text(
        "# Gemini Lab benchmark\n\nREPORT.md and comparison.csv are browsable here. "
        "All original exported FID and events are in numbered results.zip parts; "
        "see HOW_TO_JOIN.txt.\n",encoding="utf-8")
    names.append("README.md")
    return names


def _publish(repo: Path, branch: str, payload) -> dict[str, object]:
    env = os.environ.copy()
    env.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"})
    # Keep the normal Git credential configuration. Askpass is a noninteractive
    # fallback for an already supplied GH_TOKEN/GITHUB_TOKEN.
    try:
        remote = _remote(repo, env)
        with tempfile.TemporaryDirectory(prefix="spinq_results_push_") as name:
            work = Path(name) / "checkout"
            work.mkdir()
            askpass = Path(name) / "askpass.cmd"
            askpass.write_text(
                '@echo off\r\n'
                'echo %~1 | findstr /I /C:"Username" >nul\r\n'
                'if not errorlevel 1 (echo x-access-token& exit /b 0)\r\n'
                'if defined GH_TOKEN (echo %GH_TOKEN%& exit /b 0)\r\n'
                'if defined GITHUB_TOKEN (echo %GITHUB_TOKEN%& exit /b 0)\r\n'
                'exit /b 1\r\n', encoding="ascii")
            if sys.platform == "win32" and (env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")):
                env["GIT_ASKPASS"] = str(askpass)
            _git(["init", "-q"], work, env)
            _git(["remote", "add", "origin", remote], work, env)
            existing = _git(["ls-remote", "--heads", "origin", f"refs/heads/{branch}"], work, env)
            if existing:
                # A resumed analysis creates a fast-forward update to the same
                # dedicated branch. Never force-push or alter the main checkout.
                _git(["fetch", "-q", "--depth=1", "origin", f"refs/heads/{branch}"], work, env)
                _git(["checkout", "-q", "-b", branch, "FETCH_HEAD"], work, env)
                for old in work.iterdir():
                    if old.name==".git": continue
                    if old.is_dir(): shutil.rmtree(old)
                    else: old.unlink()
            else:
                _git(["checkout", "-q", "-b", branch], work, env)
            files = payload(work)
            _git(["add", "-A"], work, env)
            unchanged = subprocess.run(["git", "diff", "--cached", "--quiet"],
                cwd=work, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0
            if existing and unchanged:
                return {"status": "UPLOAD_SUCCEEDED", "branch": branch,
                        "files": files, "remote_source": "git remote origin",
                        "note": "identical publication already present"}
            _git(["-c", "user.name=SpinQ Results", "-c",
                  "user.email=spinq-results@users.noreply.github.com", "commit", "-q",
                  "-m", f"Gemini Lab results {branch.rsplit('/', 1)[-1]}"], work, env)
            push_prefix = (["-c", "credential.helper="] if env.get("GIT_ASKPASS") else [])
            try:
                _git([*push_prefix, "push", "--porcelain", "origin",
                      f"HEAD:refs/heads/{branch}"], work, env)
            except (RuntimeError, subprocess.TimeoutExpired):
                # HTTP may fail after the remote accepted the commit. Confirm
                # the exact remote HEAD before reporting a failed upload.
                local_head = _git(["rev-parse", "HEAD"], work, env)
                remote_head = _git(["ls-remote", "--heads", "origin",
                                    f"refs/heads/{branch}"], work, env)
                if not remote_head or remote_head.split()[0] != local_head:
                    raise
            return {"status": "UPLOAD_SUCCEEDED", "branch": branch,
                    "files": files, "remote_source": "git remote origin"}
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        reason = str(redact(str(exc)))
        for secret in (env.get("GH_TOKEN"), env.get("GITHUB_TOKEN")):
            if secret:
                reason = reason.replace(secret, "[REDACTED_TOKEN]")
        return {"status": "UPLOAD_FAILED", "branch": branch,
                "reason": reason[:1200]}


def publish_summary(repo: Path, out: Path, branch: str) -> dict[str, object]:
    """Publish report, table, plots and compact JSON; never upload raw charts."""
    return _publish(repo,branch,lambda work:_summary_files(out,work))


def publish_results(repo: Path, zip_path: Path, branch: str) -> dict[str, object]:
    """Compatibility API for the separate legacy live suite."""
    return _publish(repo,branch,lambda work:_archive_parts(zip_path,work))
