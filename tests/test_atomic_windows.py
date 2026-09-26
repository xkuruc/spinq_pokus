"""Offline regression tests for transient Windows file replacement locks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spinq_audit import common


class AtomicWriteTests(unittest.TestCase):
    def test_transient_permission_error_retries_same_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "hardware_journal.json"
            target.write_bytes(b"old")
            real_replace = common.os.replace
            attempts = []

            def intermittently_locked(source, destination):
                attempts.append((source, destination))
                if len(attempts) <= 2:
                    self.assertEqual(target.read_bytes(), b"old")
                    raise PermissionError(13, "temporarily locked", str(target))
                return real_replace(source, destination)

            with patch.object(common.os, "replace", side_effect=intermittently_locked), \
                 patch.object(common.time, "sleep") as sleep:
                common.atomic_bytes(target, b"new")

            self.assertEqual(target.read_bytes(), b"new")
            self.assertEqual(len(attempts), 3)
            self.assertEqual(len({source for source, _ in attempts}), 1)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_persistent_permission_error_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "hardware_journal.json"
            target.write_bytes(b"old")
            with patch.object(common.os, "replace",
                              side_effect=PermissionError(13, "locked", str(target))) as replace, \
                 patch.object(common.time, "sleep") as sleep:
                with self.assertRaises(PermissionError):
                    common.atomic_bytes(target, b"new")

            self.assertEqual(target.read_bytes(), b"old")
            self.assertEqual(replace.call_count, len(common._REPLACE_RETRY_DELAYS) + 1)
            self.assertEqual(sleep.call_count, len(common._REPLACE_RETRY_DELAYS))
            self.assertEqual(list(Path(directory).iterdir()), [target])


if __name__ == "__main__":
    unittest.main()
