"""Publish a completed results ZIP to a new branch without touching the checkout.

No credentials are written to the result or to the temporary Git repository.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from spinq_audit.common import redact

PART_BYTES = 80 * 1024 * 1024
GIT_TIMEOUT_SECONDS = 120


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


def _parts(zip_path: Path, destination: Path) -> list[str]:
    if zip_path.stat().st_size <= PART_BYTES:
        shutil.copyfile(zip_path, destination / "results.zip")
        return ["results.zip"]
    names = []
    with zip_path.open("rb") as source:
        index = 1
        while True:
            chunk = source.read(PART_BYTES)
            if not chunk:
                break
            name = f"results.zip.part{index:04d}"
            (destination / name).write_bytes(chunk)
            names.append(name)
            index += 1
    (destination / "HOW_TO_JOIN.txt").write_text(
        "Súbory results.zip.part0001, part0002, ... spoj v číselnom poradí "
        "do results.zip. Lokálny pôvodný results.zip zostáva celý.\n",
        encoding="utf-8")
    names.append("HOW_TO_JOIN.txt")
    return names


def publish_results(repo: Path, zip_path: Path, branch: str) -> dict[str, object]:
    """Push only the completed result archive; return status without raising."""
    env = os.environ.copy()
    env.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"})
    # Keep the normal Git credential configuration. Askpass is a noninteractive
    # fallback for an already supplied GH_TOKEN/GITHUB_TOKEN.
    try:
        remote = _remote(repo, env)
        with tempfile.TemporaryDirectory(prefix="spinq_results_push_") as name:
            work = Path(name)
            askpass = work / "askpass.cmd"
            askpass.write_text(
                '@echo off\r\n'
                'echo %~1 | findstr /I /C:"Username" >nul\r\n'
                'if not errorlevel 1 (echo x-access-token& exit /b 0)\r\n'
                'if defined GH_TOKEN (echo %GH_TOKEN%& exit /b 0)\r\n'
                'if defined GITHUB_TOKEN (echo %GITHUB_TOKEN%& exit /b 0)\r\n'
                'exit /b 1\r\n', encoding="ascii")
            if sys.platform == "win32" and (env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")):
                env["GIT_ASKPASS"] = str(askpass)
            files = _parts(zip_path, work)
            _git(["init", "-q"], work, env)
            _git(["checkout", "-q", "-b", branch], work, env)
            _git(["remote", "add", "origin", remote], work, env)
            _git(["add", "--", *files], work, env)
            _git(["-c", "user.name=SpinQ Results", "-c",
                  "user.email=spinq-results@users.noreply.github.com", "commit", "-q",
                  "-m", f"Gemini Lab results {branch.rsplit('/', 1)[-1]}"], work, env)
            push_prefix = (["-c", "credential.helper="] if env.get("GIT_ASKPASS") else [])
            _git([*push_prefix, "push", "--porcelain", "origin",
                  f"HEAD:refs/heads/{branch}"], work, env)
            return {"status": "UPLOAD_SUCCEEDED", "branch": branch,
                    "files": files, "remote_source": "git remote origin"}
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        reason = str(redact(str(exc)))
        for secret in (env.get("GH_TOKEN"), env.get("GITHUB_TOKEN")):
            if secret:
                reason = reason.replace(secret, "[REDACTED_TOKEN]")
        return {"status": "UPLOAD_FAILED", "branch": branch,
                "reason": reason[:1200]}
