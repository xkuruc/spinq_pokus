"""Result publishing keeps the original FID in the uploaded archive."""

from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bayes_online_core.artifacts import RunArtifacts, publish


def git(*args: str, cwd: Path | None = None) -> bytes:
    return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL)


class CompletePublishTests(unittest.TestCase):
    def test_quick_run_branch_contains_complete_fid_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            remote, repo, out = base / "remote.git", base / "repo", base / "run"
            git("init", "--bare", str(remote))
            git("init", str(repo))
            (repo / "source.py").write_text("pass\n", encoding="utf-8")
            git("add", "source.py", cwd=repo)
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "commit", "-m", "initial", cwd=repo)
            git("remote", "add", "origin", str(remote), cwd=repo)
            out.mkdir()
            (out / "data").mkdir()
            (out / "data" / "original_fid.npz").write_bytes(b"original complex FID")
            for name in ("REPORT.md", "comparison.csv", "results.json"):
                (out / name).write_text("complete\n", encoding="utf-8")
            RunArtifacts(out).archive()
            github_url = SimpleNamespace(scheme="https", hostname="github.com",
                                         password=None, username=None)
            with patch("bayes_online_core.artifacts.urlsplit", return_value=github_url):
                result = publish(repo, out, "benchmark/01_bayes_online/quick_test")
            self.assertEqual(result["status"], "UPLOAD_SUCCEEDED", result)
            uploaded = git("--git-dir", str(remote), "show",
                           "refs/heads/benchmark/01_bayes_online/quick_test:results.zip")
            with zipfile.ZipFile(io.BytesIO(uploaded)) as archive:
                self.assertEqual(archive.read("data/original_fid.npz"),
                                 b"original complex FID")


if __name__ == "__main__":
    unittest.main()
