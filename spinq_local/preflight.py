"""Separate-process CPU/runtime check executed before any RF request."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from spinq_audit.common import atomic_json, utc_now


def check() -> dict:
    result = {"checked_utc": utc_now(), "python": sys.version.split()[0],
              "sdk": {"ready": False}, "numeric": {"ready": False},
              "torch": {"ready": False}}
    try:
        from spinq_audit.adapter import verify_installed_sdk
        verify_installed_sdk()
        result["sdk"] = {"ready": True,
                         "version": importlib.metadata.version("spinqlablink")}
    except Exception as exc:
        result["sdk"]["reason"] = f"{type(exc).__name__}: {exc}"
    try:
        import numpy as np
        from scipy.optimize import least_squares
        from sklearn.gaussian_process import GaussianProcessRegressor
        import matplotlib
        matplotlib.use("Agg")
        signal = np.exp(2j * np.pi * 3 * np.arange(64) / 64)
        peak = int(np.argmax(np.abs(np.fft.fft(signal))))
        fit = least_squares(lambda x: np.array([x[0] - 1]), [0.0])
        gp = GaussianProcessRegressor(alpha=1e-3).fit(np.arange(3)[:, None], [0.0, 1.0, 0.0])
        if peak != 3 or abs(fit.x[0]-1) > 1e-7 or not np.isfinite(gp.predict([[1.5]])[0]):
            raise RuntimeError("CPU numerical smoke test failed")
        result["numeric"] = {"ready": True, "numpy": np.__version__,
                             "scipy": importlib.metadata.version("scipy"),
                             "scikit_learn": importlib.metadata.version("scikit-learn"),
                             "matplotlib": matplotlib.__version__}
    except Exception as exc:
        result["numeric"]["reason"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch
        torch.set_num_threads(2)
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = ((x * x).sum())
        loss.backward()
        if x.grad is None or not torch.allclose(x.grad, torch.tensor([2.0, 4.0])):
            raise RuntimeError("PyTorch backward pass failed")
        result["torch"] = {"ready": True, "version": torch.__version__,
                           "device": "cpu", "backward_verified": True}
    except Exception as exc:
        result["torch"]["reason"] = f"{type(exc).__name__}: {exc}"
        try:
            result["torch"]["installed_version"] = importlib.metadata.version("torch")
        except Exception:
            result["torch"]["installed_version"] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check()
    atomic_json(args.output, result)
    summary = {k: v.get("ready") for k, v in result.items() if isinstance(v, dict)}
    if not result["torch"]["ready"]:
        summary["torch_reason"] = result["torch"].get("reason")
        summary["torch_installed_version"] = result["torch"].get("installed_version")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if result["sdk"]["ready"] and result["numeric"]["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
