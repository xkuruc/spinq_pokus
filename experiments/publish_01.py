"""Publish a complete Bayesian-calibration result from an isolated Git worktree.

The active source checkout stays on its original branch.  No credentials are
stored in Git URLs, code, result files or logs.  An upload failure is returned
as a result and never changes the acquisition outcome or removes local ZIPs.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from spinq_audit.common import redact


PART_BYTES = 80_000_000  # safely below GitHub's 100 MB single-file rejection
GIT_TIMEOUT_SECONDS = 900
BRANCH_PREFIX = "benchmark/01_bayes_kalibracia/"


def _safe_reason(exc: BaseException, env: dict[str, str]) -> str:
    reason = str(redact(str(exc)))
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        secret = env.get(name)
        if secret:
            reason = reason.replace(secret, "[REDACTED_TOKEN]")
    return reason[:1200]


def _git(args: list[str], *, cwd: Path, env: dict[str, str],
         timeout_s: int = GIT_TIMEOUT_SECONDS) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        errors="replace", timeout=timeout_s, check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"git {args[0]} failed ({completed.returncode}): "
                           f"{_safe_reason(RuntimeError(detail), env)}")
    return completed.stdout.strip()


def _remote_url(repo: Path, env: dict[str, str]) -> str:
    url = _git(["remote", "get-url", "origin"], cwd=repo, env=env).strip()
    if not url:
        raise ValueError("Git origin remote is missing")
    if "://" in url:
        parsed = urlsplit(url)
        if (parsed.scheme not in ("https", "ssh") or parsed.hostname != "github.com"
                or parsed.username not in (None, "git") or parsed.password
                or parsed.query or parsed.fragment):
            raise ValueError("Git origin is not a credential-free GitHub HTTPS/SSH URL")
    elif not url.startswith("git@github.com:"):
        raise ValueError("Git origin is not a GitHub HTTPS/SSH URL")
    return url


def _validate_branch(branch: str) -> None:
    if not branch.startswith(BRANCH_PREFIX):
        raise ValueError(f"Result branch must begin {BRANCH_PREFIX}")
    suffix = branch[len(BRANCH_PREFIX):]
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", suffix)
            or ".." in suffix or suffix.endswith(".")):
        raise ValueError("Unsafe result branch name")


def _askpass_env(directory: Path, source: dict[str, str]) -> dict[str, str]:
    env = source.copy()
    env.update({"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"})
    if not (source.get("GH_TOKEN") or source.get("GITHUB_TOKEN")):
        return env  # Retain the user's existing noninteractive credential helper.
    if sys.platform == "win32":
        askpass = directory / "git_askpass.cmd"
        askpass.write_text(
            '@echo off\r\n'
            'echo %~1 | findstr /I /C:"Username" >nul\r\n'
            'if not errorlevel 1 (echo x-access-token& exit /b 0)\r\n'
            'if defined GH_TOKEN (echo %GH_TOKEN%& exit /b 0)\r\n'
            'if defined GITHUB_TOKEN (echo %GITHUB_TOKEN%& exit /b 0)\r\n'
            'exit /b 1\r\n', encoding="ascii")
    else:
        askpass = directory / "git_askpass.sh"
        askpass.write_text(
            '#!/bin/sh\n'
            'case "$1" in *Username*) printf "%s\\n" x-access-token;;\n'
            '*) if [ -n "$GH_TOKEN" ]; then printf "%s\\n" "$GH_TOKEN"; '
            'elif [ -n "$GITHUB_TOKEN" ]; then printf "%s\\n" "$GITHUB_TOKEN"; '
            'else exit 1; fi;; esac\n', encoding="ascii")
        askpass.chmod(0o700)
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_ASKPASS_REQUIRE"] = "force"
    return env


def _clear_worktree(work: Path) -> None:
    for child in work.iterdir():
        if child.name == ".git":
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def _copy_chunks(source: Path, destination: Path, max_part_bytes: int) -> list[str]:
    size = source.stat().st_size
    if size <= max_part_bytes:
        shutil.copyfile(source, destination / "results.zip")
        return ["results.zip"]
    names: list[str] = []
    with source.open("rb") as incoming:
        for index in range(1, 100_000):
            name = f"results.zip.part{index:04d}"
            with (destination / name).open("wb") as part:
                remaining = max_part_bytes
                while remaining:
                    piece = incoming.read(min(1024 * 1024, remaining))
                    if not piece:
                        break
                    part.write(piece)
                    remaining -= len(piece)
            if (destination / name).stat().st_size == 0:
                (destination / name).unlink()
                break
            names.append(name)
        else:
            raise ValueError("Too many ZIP parts")
    return names


def _copy_payload(out: Path, work: Path, max_part_bytes: int) -> list[str]:
    archive = out / "results.zip"
    if not archive.is_file():
        raise FileNotFoundError("Complete results.zip is required before upload")
    names: list[str] = []
    for filename in ("REPORT.md", "comparison.csv", "results.json",
                     "plan.json", "calibration.json", "sources.md",
                     "reproduction_scope.md"):
        source = out / filename
        if not source.is_file():
            if filename in ("REPORT.md", "comparison.csv", "results.json"):
                raise FileNotFoundError(f"Missing required result file: {filename}")
            continue
        shutil.copyfile(source, work / filename)
        names.append(filename)
    for source in sorted((out / "plots").glob("*.png")):
        destination = work / "plots" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        names.append(destination.relative_to(work).as_posix())
    parts = _copy_chunks(archive, work, max_part_bytes)
    names.extend(parts)
    if len(parts) > 1:
        (work / "HOW_TO_JOIN.txt").write_text(
            "The numbered files are consecutive byte ranges of the full results.zip.\n"
            "Windows PowerShell, in this directory:\n"
            "$parts = Get-ChildItem results.zip.part* | Sort-Object Name\n"
            "$out = [IO.File]::Create('results.zip')\n"
            "try { foreach ($part in $parts) { $inputFile = "
            "[IO.File]::OpenRead($part.FullName); "
            "try { $inputFile.CopyTo($out) } finally { $inputFile.Dispose() } } } "
            "finally { $out.Dispose() }\n"
            "On macOS/Linux: cat results.zip.part* > results.zip\n",
            encoding="utf-8",
        )
        names.append("HOW_TO_JOIN.txt")
    (work / "README.md").write_text(
        "# Gemini Lab Bayesian calibration results\n\n"
        "REPORT.md and comparison.csv are browsable summaries. "
        "The complete original exported FID, sanitized SDK event journal, "
        "vendor references, model histories, pulses, plots, plan, calibration "
        "and source snapshot are in results.zip"
        + (" parts; see HOW_TO_JOIN.txt.\n" if len(parts) > 1 else ".\n"),
        encoding="utf-8",
    )
    names.append("README.md")
    return names


def publish_results(repo_dir: Path, out_dir: Path, branch: str, *,
                    max_part_bytes: int = PART_BYTES,
                    timeout_s: int = GIT_TIMEOUT_SECONDS) -> dict[str, object]:
    """Push a complete archive to an isolated result branch, never force-push.

    Existing credential configuration is used.  GH_TOKEN/GITHUB_TOKEN are only
    read from the environment as a noninteractive fallback.  On a partial or
    failed upload, local files are untouched and ``UPLOAD_FAILED`` is returned.
    """
    repo = Path(repo_dir).resolve()
    out = Path(out_dir).resolve()
    env = os.environ.copy()
    work: Path | None = None
    temporary: Path | None = None
    scratch: str | None = None
    try:
        _validate_branch(branch)
        if not isinstance(max_part_bytes, int) or not 1 <= max_part_bytes <= PART_BYTES:
            raise ValueError("ZIP part size must be 1..80,000,000 bytes")
        if not (repo / ".git").exists():
            raise ValueError("Source checkout has no .git directory")
        temporary = Path(tempfile.mkdtemp(prefix="spinq_bayes_publish_"))
        env = _askpass_env(temporary, env)
        _remote_url(repo, env)
        auth_prefix = ["-c", "credential.helper="] if env.get("GIT_ASKPASS") else []
        remote = _git([*auth_prefix, "ls-remote", "--heads", "origin",
                       f"refs/heads/{branch}"], cwd=repo, env=env, timeout_s=timeout_s)
        if remote:
            _git([*auth_prefix, "fetch", "--no-tags", "origin", f"refs/heads/{branch}"],
                 cwd=repo, env=env, timeout_s=timeout_s)
            base = "FETCH_HEAD"
        else:
            base = "HEAD"
        work = temporary / "worktree"
        _git(["worktree", "add", "--detach", str(work), base],
             cwd=repo, env=env, timeout_s=timeout_s)
        if not remote:
            scratch = "spinq-bayes-publish-" + uuid.uuid4().hex[:12]
            _git(["switch", "--orphan", scratch], cwd=work, env=env,
                 timeout_s=timeout_s)
        _clear_worktree(work)
        files = _copy_payload(out, work, max_part_bytes)
        _git(["add", "-A"], cwd=work, env=env, timeout_s=timeout_s)
        change = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=work, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=timeout_s, check=False,
        )
        if change.returncode not in (0, 1):
            raise RuntimeError("Cannot inspect staged result changes")
        if remote and change.returncode == 0:
            return {"status": "UPLOAD_SUCCEEDED", "branch": branch,
                    "files": files, "note": "identical result already published"}
        commit_args = ["-c", "user.name=SpinQ Results", "-c",
                       "user.email=spinq-results@users.noreply.github.com",
                       "commit", "-q", "-m",
                       f"Gemini Lab Bayesian calibration {branch.rsplit('/', 1)[-1]}"]
        _git(commit_args, cwd=work, env=env, timeout_s=timeout_s)
        push_args = [*auth_prefix, "push", "--porcelain", "origin",
                     f"HEAD:refs/heads/{branch}"]
        try:
            _git(push_args, cwd=work, env=env, timeout_s=timeout_s)
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            # A timed-out client can still have successfully updated the remote.
            expected = _git(["rev-parse", "HEAD"], cwd=work, env=env,
                            timeout_s=timeout_s)
            observed = _git([*auth_prefix, "ls-remote", "--heads", "origin",
                             f"refs/heads/{branch}"], cwd=work, env=env,
                            timeout_s=timeout_s)
            if not observed or observed.split()[0] != expected:
                raise
        return {"status": "UPLOAD_SUCCEEDED", "branch": branch,
                "files": files}
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {"status": "UPLOAD_FAILED", "branch": branch,
                "reason": _safe_reason(exc, env)}
    finally:
        if work is not None and work.exists():
            try:
                _git(["worktree", "remove", "--force", str(work)], cwd=repo,
                     env=env, timeout_s=min(timeout_s, 60))
            except Exception:
                # Preserve the primary error/success. A later `git worktree prune`
                # can remove stale metadata; the source branch is untouched.
                pass
        if scratch:
            try:
                _git(["branch", "-D", scratch], cwd=repo, env=env,
                     timeout_s=min(timeout_s, 60))
            except Exception:
                pass
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
