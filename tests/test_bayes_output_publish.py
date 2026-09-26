"""Offline checks for complete result recovery and isolated Git publication."""

from __future__ import annotations

import csv
import gzip
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.output_01 import (archive_results, save_results,
                                   snapshot_sources)
from experiments.publish_01 import publish_results


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True,
                                   stderr=subprocess.DEVNULL).strip()


class BayesOutputPublishingTests(unittest.TestCase):
    def _result(self, out: Path) -> Path:
        (out / "raw").mkdir(parents=True, exist_ok=True)
        (out / "data").mkdir(parents=True, exist_ok=True)
        (out / "raw" / "fid_1.npz").write_bytes(b"complex FID original")
        (out / "data" / "events.jsonl").write_text(
            '{"kind":"s_post_exp_data","payload":{}}\n', encoding="utf-8")
        (out / "plan.json").write_text('{"frozen":true}', encoding="utf-8")
        (out / "calibration.json").write_text('{"t90_us":40}', encoding="utf-8")
        (out / "vendor_reference").mkdir(exist_ok=True)
        (out / "vendor_reference" / "fft.json").write_text("{}", encoding="utf-8")
        save_results(out, {"state": "BUDGET_EXHAUSTED",
                           "hardware_results_present": True,
                           "rows": [{"method": "D", "baseline": "B", "block": 1,
                                     "acquisitions": 24, "frequency_error_hz": 1.25,
                                     "reference_uncertainty": {"frequency_hz": 0.8},
                                     "status": "BUDGET_EXHAUSTED",
                                     "reason": "Tolerance not independently verified",
                                     "extra_cost": 9}],
                           "errors": []})
        return archive_results(out)

    def test_complete_archive_csv_fields_and_source_snapshot(self):
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            repo, out = base / "repo", base / "results"
            repo.mkdir(); out.mkdir()
            (repo / "code.py").write_text("print('test')\n", encoding="utf-8")
            copied = snapshot_sources(out, repo, ("code.py", "missing.py"))
            self.assertEqual(copied, ["code.py"])
            archive = self._result(out)
            with zipfile.ZipFile(archive) as opened:
                members = set(opened.namelist())
                self.assertTrue({"raw/fid_1.npz", "raw/events.jsonl.gz",
                                 "vendor_reference/fft.json", "plan.json",
                                 "calibration.json", "source_snapshot/code.py",
                                 "REPORT.md", "comparison.csv",
                                 "results.json"}.issubset(members))
                self.assertEqual(opened.read("raw/fid_1.npz"), b"complex FID original")
            with (out / "comparison.csv").open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["extra_cost"], "9")
            self.assertEqual(row["reference_uncertainty"], '{"frequency_hz": 0.8}')
            self.assertEqual(json.loads((out / "results.json").read_text())["state"],
                             "BUDGET_EXHAUSTED")

    def test_branch_upload_parts_resume_and_primary_checkout_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            bare, repo, out = base / "remote.git", base / "repo", base / "results"
            git("init", "--bare", str(bare))
            git("init", str(repo))
            (repo / "main_source.py").write_text("kept\n", encoding="utf-8")
            git("add", "main_source.py", cwd=repo)
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "commit", "-m", "main", cwd=repo)
            git("remote", "add", "origin", str(bare), cwd=repo)
            original_head = git("rev-parse", "HEAD", cwd=repo)
            original_branch = git("symbolic-ref", "--short", "HEAD", cwd=repo)
            out.mkdir()
            complete_zip = self._result(out)
            branch = "benchmark/01_bayes_kalibracia/offline-test"
            with patch("experiments.publish_01._remote_url", return_value=str(bare)):
                first = publish_results(repo, out, branch, max_part_bytes=100)
                self.assertEqual(first["status"], "UPLOAD_SUCCEEDED", first)
                second = publish_results(repo, out, branch, max_part_bytes=100)
                self.assertEqual(second["status"], "UPLOAD_SUCCEEDED", second)
                (out / "REPORT.md").write_text("Updated after reanalysis\n", encoding="utf-8")
                complete_zip = archive_results(out)
                third = publish_results(repo, out, branch, max_part_bytes=100)
                self.assertEqual(third["status"], "UPLOAD_SUCCEEDED", third)
            self.assertEqual(git("rev-parse", "HEAD", cwd=repo), original_head)
            self.assertEqual(git("symbolic-ref", "--short", "HEAD", cwd=repo),
                             original_branch)
            self.assertEqual((repo / "main_source.py").read_text(), "kept\n")
            self.assertEqual(git("worktree", "list", "--porcelain", cwd=repo).count("worktree "), 1)
            files = git("--git-dir", str(bare), "ls-tree", "-r", "--name-only",
                        f"refs/heads/{branch}").splitlines()
            self.assertIn("REPORT.md", files)
            self.assertIn("HOW_TO_JOIN.txt", files)
            self.assertNotIn("main_source.py", files)
            self.assertTrue(any(name.startswith("results.zip.part") for name in files))
            # Use binary subprocess output for archive chunks.
            joined = b"".join(subprocess.check_output(
                ["git", "--git-dir", str(bare), "show",
                 f"refs/heads/{branch}:{filename}"])
                for filename in files if filename.startswith("results.zip.part"))
            self.assertEqual(joined, complete_zip.read_bytes())
            with zipfile.ZipFile(complete_zip) as opened:
                for filename in ("REPORT.md", "comparison.csv", "results.json"):
                    branch_copy = subprocess.check_output(
                        ["git", "--git-dir", str(bare), "show",
                         f"refs/heads/{branch}:{filename}"])
                    self.assertEqual(opened.read(filename), branch_copy)

    def test_resume_refreshes_archived_event_journal(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name) / "results"; out.mkdir()
            first_archive = self._result(out)
            with zipfile.ZipFile(first_archive) as opened:
                first = gzip.decompress(opened.read("raw/events.jsonl.gz"))
            plain = out / "data" / "events.jsonl"
            with plain.open("ab") as destination:
                destination.write(b'{"kind":"later_resumed_task"}\n')
            # Ensure the append is newer than the old compressed journal even
            # on filesystems with coarse wall-clock resolution.
            previous_ns = (out / "data" / "events.jsonl.gz").stat().st_mtime_ns
            os.utime(plain, ns=(previous_ns + 1_000_000_000,
                                previous_ns + 1_000_000_000))
            second_archive = archive_results(out)
            with zipfile.ZipFile(second_archive) as opened:
                second_raw = gzip.decompress(opened.read("raw/events.jsonl.gz"))
                second_data = gzip.decompress(opened.read("data/events.jsonl.gz"))
            self.assertTrue(second_raw.startswith(first))
            self.assertIn(b"later_resumed_task", second_raw)
            self.assertEqual(second_raw, second_data)

    def test_missing_archive_does_not_change_measurement(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name) / "out"; out.mkdir()
            save_results(out, {"state": "COMPLETED", "rows": []})
            result = publish_results(Path(name), out,
                                     "benchmark/01_bayes_kalibracia/missing")
            self.assertEqual(result["status"], "UPLOAD_FAILED")
            self.assertEqual(json.loads((out / "results.json").read_text())["state"],
                             "COMPLETED")

    def test_archive_refuses_claimed_hardware_results_without_fid(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name) / "out"; out.mkdir()
            save_results(out, {"state": "COMPLETED", "rows": [],
                               "hardware_results_present": True})
            with self.assertRaisesRegex(ValueError, "no exported FID NPZ"):
                archive_results(out)
            self.assertFalse((out / "results.zip").exists())

    def test_source_snapshot_redacts_runtime_config_credentials(self):
        with tempfile.TemporaryDirectory() as name:
            repo = Path(name) / "repo"; repo.mkdir()
            out = Path(name) / "out"; out.mkdir()
            (repo / "config-01-bayes.json").write_text(
                '{"host":"127.0.0.1","password":"example-secret"}', encoding="utf-8")
            snapshot_sources(out, repo, ("config-01-bayes.json",))
            saved = json.loads((out / "source_snapshot" / "config-01-bayes.json")
                               .read_text(encoding="utf-8"))
            self.assertEqual(saved["host"], "127.0.0.1")
            self.assertEqual(saved["password"], "[REDACTED]")

    def test_results_keep_numpy_values_numeric(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name) / "out"
            save_results(out, {"state": "PILOT", "rows": [],
                               "history": {"means": np.array([1.25, 2.5]),
                                           "cost": np.float32(3.5)}})
            saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["history"], {"means": [1.25, 2.5],
                                                 "cost": 3.5})


if __name__ == "__main__":
    unittest.main()
