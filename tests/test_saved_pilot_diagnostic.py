"""The operator's saved-data check must work without instrument access."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from diagnose_saved_pilot import diagnose, main
from spinq_local.core import RawFIDRecord


class SavedPilotDiagnosticTests(unittest.TestCase):
    def test_saved_rabi_fit_is_read_only_and_signed(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            raw = out / "raw"
            raw.mkdir()
            fs, n = 10000, 1024
            t = np.arange(n) / fs
            base = np.exp((-5 + 2j * np.pi * 170) * t)
            axis = np.asarray(np.arange(n) * .1, dtype=np.float32).astype(float)
            rng = np.random.default_rng(26)
            for width in (40, 80, 120, 160, 200):
                signal = np.sin(2 * np.pi * width / 156) * base
                for repeat in (range(3) if width == 40 else range(1)):
                    key = f"pilot_40_r{repeat}" if width == 40 else f"pilot_{width}"
                    observed = signal + .002 * (rng.normal(size=n) + 1j * rng.normal(size=n))
                    RawFIDRecord(key=key, task_id=f"task_{key}", group="g", path="0",
                                 qubit="0", step="NMRSIG", axis_original=axis,
                                 time_seconds=t, re=observed.real, im=observed.imag,
                                 parameters_sent={"sampleFre": fs, "sampleCount": n}).save(raw)
            before = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
            with patch("spinq_benchmark.hardware.LiveHardware.__enter__",
                       side_effect=AssertionError("hardware must not be contacted")):
                result = diagnose(out)
                with redirect_stdout(io.StringIO()) as output:
                    code = main([str(out)])
            after = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
            self.assertEqual(before, after)
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "VALID")
            self.assertAlmostEqual(result["rabi"]["period_us"], 156, delta=2)
            self.assertGreater(result["rabi"]["signed_complex_r2"], .99)
            self.assertIn("no hardware command sent", output.getvalue())


if __name__ == "__main__":
    unittest.main()
