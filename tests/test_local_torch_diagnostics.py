"""Keep Windows CPU PyTorch failures visible without contacting hardware."""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spinq_local import cde, preflight


class TorchDiagnosticsTests(unittest.TestCase):
    def test_runtime_import_error_keeps_windows_loader_reason(self):
        original_import = builtins.__import__

        def fail_torch(name, *args, **kwargs):
            if name == "torch":
                raise OSError("[WinError 126] A required DLL was not found")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_torch):
            with self.assertRaisesRegex(RuntimeError, r"WinError 126") as caught:
                cde._require_torch()
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_preflight_console_prints_optional_torch_failure(self):
        result = {"sdk": {"ready": True}, "numeric": {"ready": True},
                  "torch": {"ready": False, "installed_version": "2.x",
                            "reason": "OSError: [WinError 126] DLL missing"}}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "preflight.json"
            output = io.StringIO()
            with patch.object(preflight, "check", return_value=result), \
                 patch.object(sys, "argv", ["preflight", "--output", str(target)]), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(preflight.main(), 0)
            self.assertEqual(json.loads(target.read_text())["torch"], result["torch"])
            self.assertIn("WinError 126", output.getvalue())


if __name__ == "__main__":
    unittest.main()
