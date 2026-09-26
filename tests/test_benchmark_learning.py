"""Exercise the Windows DLL fallback without installing or importing PyTorch."""

import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    from spinq_benchmark.learning import apply_complex_denoiser,train_complex_denoiser
except ImportError:
    np=None


@unittest.skipIf(np is None,"NumPy not installed")
class NumpyNoise2NoiseTests(unittest.TestCase):
    def test_torch_import_failure_uses_phase_equivariant_learned_filter(self):
        rng=np.random.default_rng(4)
        t=np.arange(2048)
        clean=np.exp((-0.0004+0.04j)*t)
        def pair(phase):
            z=clean*np.exp(1j*phase)
            noise=lambda: 0.5*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
            return z+noise(),z+noise()
        train=[pair(i*.4) for i in range(4)]
        validation=[pair(i*.7) for i in range(2)]
        original_import=builtins.__import__
        def fail_torch(name,*args,**kwargs):
            if name=="torch" or name.startswith("torch."):
                raise OSError("simulated c10.dll failure")
            return original_import(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as d, patch("builtins.__import__",side_effect=fail_torch):
            info=train_complex_denoiser(train,validation,Path(d)/"model.pt",seed=2)
            self.assertEqual(info["backend"],"numpy_complex_fir_noise2noise")
            model=Path(info["model_path"])
            self.assertTrue(model.is_file())
            source,_=pair(.9)
            output=apply_complex_denoiser(source,model)
            rotated=apply_complex_denoiser(1j*source,model)
            self.assertEqual(len(output),len(source))
            self.assertLess(np.linalg.norm(rotated-1j*output),1e-9)
            self.assertLess(np.mean(np.abs(output-clean*np.exp(.9j))**2),
                            np.mean(np.abs(source-clean*np.exp(.9j))**2))


if __name__=="__main__":unittest.main()
