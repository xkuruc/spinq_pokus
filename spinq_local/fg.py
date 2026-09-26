"""Local F/G algorithms for SpinQ NMR, with no device or network access.

F is a TVCondNet-method reimplementation (real spectrum) plus a distinct
complex-spectrum adaptation. G is an NMR adaptation of the public Fire Opal
workflow, not a copy of its unavailable production backend. All fitted scales,
regularizers, network weights, and circuit costs must come from training/pilot
data, never held-out validation runs.

Primary references:
Zou et al. https://arxiv.org/html/2405.11064v1
Qiu et al. https://arxiv.org/abs/2001.11815
Lehtinen et al. https://proceedings.mlr.press/v80/lehtinen18a.html
Mundada et al. https://arxiv.org/pdf/2209.06864 (Appendix C)
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.signal import fftconvolve
from numpy.lib.stride_tricks import sliding_window_view


def _array_1d(values, *, complex_ok=True):
    arr = np.asarray(values, dtype=np.complex128 if complex_ok else np.float64)
    if arr.ndim != 1 or len(arr) < 4 or not np.all(np.isfinite(arr)):
        raise ValueError("Expected a finite one-dimensional array of at least four points")
    return arr


def grouped_split(records: Sequence[Mapping], *, seed: int = 42,
                  fractions=(0.6, 0.2, 0.2)) -> dict[str, list[int]]:
    """Split whole connected session/setting-family groups, without leakage.

    Required metadata: session_id, setting_family, acquisition_id. A reference
    average must also list all constituent acquisition IDs in reference_ids.
    This conservative graph split fails if the data cannot support 3 groups.
    """
    if len(records) < 3 or len(fractions) != 3 or any(x <= 0 for x in fractions):
        raise ValueError("Three nonempty split fractions and records are required")
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    owner = {}
    acquisition_ids = set()
    for i, row in enumerate(records):
        for key in ("session_id", "setting_family", "acquisition_id"):
            if not row.get(key):
                raise ValueError(f"Missing {key}; leakage-safe split cannot be made")
        aid = str(row["acquisition_id"])
        if aid in acquisition_ids:
            raise ValueError(f"Duplicate acquisition_id {aid}")
        acquisition_ids.add(aid)
        refs = {str(x) for x in row.get("reference_ids", ())}
        if aid in refs:
            raise ValueError("A tested input cannot contribute to its own reference average")
        tokens=[(key,str(row[key])) for key in ("session_id","setting_family")]
        tokens.extend(("reference_id",ref) for ref in refs)
        for token in tokens:
            if token in owner:
                union(i, owner[token])
            else:
                owner[token] = i
    # Connected reference records must stay in the same split as their targets.
    by_aid = {str(r["acquisition_id"]): i for i, r in enumerate(records)}
    for i, row in enumerate(records):
        for ref in row.get("reference_ids", ()):
            if str(ref) in by_aid:
                union(i, by_aid[str(ref)])
    components: dict[int, list[int]] = {}
    for i in range(len(records)):
        components.setdefault(find(i), []).append(i)
    if len(components) < 3:
        raise ValueError("At least three disjoint session/family components are required")
    rng = np.random.default_rng(seed)
    groups = list(components.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    names = ("train", "validation", "test")
    result = {x: [] for x in names}
    total = float(sum(fractions))
    target = np.array(fractions, dtype=float) / total * len(records)
    # Seed each partition so even an imbalanced pilot cannot silently omit test.
    for name, group in zip(names, groups[:3]):
        result[name].extend(group)
    for group in groups[3:]:
        score = [len(result[name]) / target[j] for j, name in enumerate(names)]
        result[names[int(np.argmin(score))]].extend(group)
    return {key: sorted(value) for key, value in result.items()}


def tv_denoise(spectrum, lam: float, *, epsilon: float = 1e-4,
               maxiter: int = 120):
    """Solve 1/2||x-y||² + lam Σ sqrt(|D x|² + epsilon²).

    Complex input couples real and imaginary finite differences through their
    joint magnitude. The real branch uses the same well-defined objective.
    """
    original = np.asarray(spectrum)
    y = _array_1d(original, complex_ok=np.iscomplexobj(original))
    if lam < 0 or epsilon <= 0 or maxiter < 1:
        raise ValueError("Invalid TV regularization or solver settings")
    if lam == 0:
        return y.copy()
    is_complex = np.iscomplexobj(y)
    n = len(y)
    yvec = np.concatenate((y.real, y.imag)) if is_complex else y.real

    def unpack(v):
        return v[:n] + 1j * v[n:] if is_complex else v

    def objective(v):
        x = unpack(v)
        difference = x - y
        dx = np.diff(x)
        rad = np.sqrt(np.abs(dx) ** 2 + epsilon ** 2)
        value = 0.5 * np.vdot(difference, difference).real + lam * np.sum(rad)
        flux = dx / rad
        grad = np.asarray(difference).copy()
        grad[:-1] -= lam * flux
        grad[1:] += lam * flux
        gradient = np.r_[grad.real, grad.imag] if is_complex else grad.real
        return float(value), np.asarray(gradient, dtype=np.float64)

    fit = minimize(objective, yvec.astype(np.float64), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-9})
    if not np.all(np.isfinite(fit.x)):
        raise RuntimeError("TV solver produced non-finite values")
    return unpack(fit.x)


def select_tv_lambda(validation_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
                     candidates=(0.01, 0.03, 0.1, 0.3, 1.0), *, maxiter=80) -> dict:
    """Select one frozen regularizer using validation targets only."""
    if not validation_pairs:
        raise ValueError("Validation data required to choose TV regularization")
    rows = []
    for lam in candidates:
        errors = []
        for noisy, target in validation_pairs:
            y, x = _array_1d(noisy), _array_1d(target)
            if len(y) != len(x):
                raise ValueError("Validation pair axes differ")
            errors.append(float(np.mean(np.abs(tv_denoise(y, lam, maxiter=maxiter) - x) ** 2)))
        rows.append({"lambda": float(lam), "mse": float(np.mean(errors))})
    chosen = min(rows, key=lambda row: row["mse"])
    return {"lambda": chosen["lambda"], "validation_mse": chosen["mse"], "curve": rows}


def average_fids(independent_records: Sequence[np.ndarray], *, axes: Sequence[np.ndarray]):
    """Average independent acquisitions only after verifying identical axes."""
    if len(independent_records) < 2:
        raise ValueError("Averaging requires at least two independent acquisitions")
    values = [_array_1d(x) for x in independent_records]
    if len({len(x) for x in values}) != 1:
        raise ValueError("Acquisitions must have the same verified time axis")
    if axes is None or len(axes)!=len(values):
        raise ValueError("Each acquisition needs its verified time axis")
    baseline=np.asarray(axes[0],float)
    if baseline.shape!=values[0].shape or not np.all(np.isfinite(baseline)):
        raise ValueError("Invalid acquisition time axis")
    for axis in axes[1:]:
        other=np.asarray(axis,float)
        if other.shape!=baseline.shape or not np.allclose(other,baseline,rtol=1e-6,atol=1e-9):
            raise ValueError("Independent acquisition time axes do not match")
    return np.mean(np.stack(values), axis=0)


def hankel_low_rank_fid(fid, rank: int, *, window: int | None = None,
                        max_elements: int = 4_000_000, seed: int = 0):
    """Memory-bounded randomized Hankel/Cadzow-style rank projection.

    This is a strong classical low-rank baseline, but is not Qiu's exact
    auto-parameter algorithm. Rank must be frozen using pilot/validation data.
    """
    y = _array_1d(fid)
    n = len(y)
    window = min(256, n // 2, max_elements // max(n, 1)) if window is None else int(window)
    if window < 4 or window > n // 2 or (n-window+1)*window > max_elements:
        raise ValueError("Hankel window exceeds memory cap or is too small")
    if not 1 <= rank < window:
        raise ValueError("Rank must be between one and window-1")
    h = sliding_window_view(y, window)
    k = min(window, rank + 8)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((window, k)) + 1j*rng.standard_normal((window, k))
    q, _ = np.linalg.qr(h @ omega, mode="reduced")
    b = q.conj().T @ h
    ub, singular, vh = np.linalg.svd(b, full_matrices=False)
    q = q @ ub[:, :rank]
    # Anti-diagonal averaging is convolution of each rank-one factor. The
    # denominator counts exact overlap; no dense reconstructed Hankel matrix.
    reconstruction = np.zeros(n, dtype=np.complex128)
    for j in range(rank):
        reconstruction += fftconvolve(q[:, j] * singular[j], vh[j, :], mode="full")[:n]
    counts = np.convolve(np.ones(h.shape[0]), np.ones(window))
    return reconstruction / counts


def select_hankel_rank(validation_pairs, candidates=(1, 2, 3, 4, 6), **kwargs):
    """Tune rank on held-out pilot pairs; never on final test references."""
    if not validation_pairs:
        raise ValueError("Validation pairs required")
    rows = []
    for rank in candidates:
        errors = [float(np.mean(np.abs(hankel_low_rank_fid(noisy, rank, **kwargs)-target)**2))
                  for noisy, target in validation_pairs]
        rows.append({"rank": int(rank), "mse": float(np.mean(errors))})
    winner = min(rows, key=lambda row: row["mse"])
    return {"rank": winner["rank"], "validation_mse": winner["mse"], "curve": rows}


def fit_exponential_fid(fid, sample_hz: float, frequencies_hz: Sequence[float],
                        *, decay_bounds_hz=(0.1, 2_000.0)):
    """Classical damped-multiplet fit with fixed pilot-derived peak identities.

    This is a local reference, not the vendor fit. The returned reconstruction
    retains complex phase and physical amplitude; no global argmax is used.
    """
    y = _array_1d(fid)
    f = np.asarray(frequencies_hz, dtype=float)
    if sample_hz <= 0 or len(f) < 1 or len(f) > 8 or not np.all(np.isfinite(f)):
        raise ValueError("Verified sample rate and 1–8 pilot frequencies required")
    if np.max(np.abs(f)) >= sample_hz / 2:
        raise ValueError("Pilot frequency aliases at this sampling rate")
    t = np.arange(len(y))/sample_hz
    low, high = decay_bounds_hz
    if not 0 <= low < high:
        raise ValueError("Invalid decay range")
    # Variable projection removes all linear complex amplitudes and DC offset.
    def design(decays):
        return np.column_stack([np.ones(len(t)), *[
            np.exp((-decay+2j*np.pi*freq)*t) for decay, freq in zip(decays, f)]])

    def residual(decays):
        matrix = design(decays)
        coeff = np.linalg.lstsq(matrix, y, rcond=None)[0]
        r = matrix @ coeff - y
        return np.r_[r.real, r.imag]

    starts = (np.full(len(f), max(low, min(high, 1/max(t[-1], 1e-9)))),
              np.full(len(f), low+(high-low)*0.2))
    fits = [least_squares(residual, x, bounds=(np.full(len(f), low),
                  np.full(len(f), high)), max_nfev=80) for x in starts]
    best = min(fits, key=lambda x: float(np.sum(x.fun*x.fun)))
    matrix = design(best.x)
    coeff = np.linalg.lstsq(matrix, y, rcond=None)[0]
    fitted = matrix @ coeff
    return {"reconstruction": fitted, "frequencies_hz": f.tolist(),
            "decay_hz": best.x.tolist(), "coefficients": coeff[1:].tolist(),
            "offset": complex(coeff[0]),
            "residual_rms": float(np.sqrt(np.mean(np.abs(fitted-y)**2))),
            "model_scope": "fixed pilot frequencies; local variable-projection fit"}


def _torch_import():
    try:
        import torch
        from torch import nn
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(f"TORCH_UNAVAILABLE: {type(exc).__name__}: {exc}") from exc
    return torch, nn


def _unet_factory(nn, input_channels: int, output_channels: int):
    class Block(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.net = nn.Sequential(nn.Conv1d(a, b, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(b, b, 5, padding=2), nn.GELU())

        def forward(self, x):
            return self.net(x)

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = Block(input_channels, 16)
            self.e2 = Block(16, 32)
            self.mid = Block(32, 64)
            self.d2 = Block(64+32, 32)
            self.d1 = Block(32+16, 16)
            self.out = nn.Conv1d(16, output_channels, 1)

        def forward(self, x):
            import torch
            from torch.nn import functional as functional
            a = self.e1(x)
            b = self.e2(functional.avg_pool1d(a, 2, ceil_mode=True))
            c = self.mid(functional.avg_pool1d(b, 2, ceil_mode=True))
            up_b = functional.interpolate(c, size=b.shape[-1], mode="linear", align_corners=False)
            d = self.d2(torch.cat((up_b, b), dim=1))
            up_a = functional.interpolate(d, size=a.shape[-1], mode="linear", align_corners=False)
            return self.out(self.d1(torch.cat((up_a, a), dim=1)))

    return SmallUNet()


def _spectrum_channels(fid, variant):
    s = np.fft.fft(_array_1d(fid), norm="ortho")
    if variant == "real":
        return np.asarray([s.real], dtype=np.float32)
    if variant == "complex":
        return np.asarray([s.real, s.imag], dtype=np.float32)
    raise ValueError("variant must be 'real' or 'complex'")


def _prepare_spectral_pair(noisy, target, variant, lam, scale, conditioned):
    y = _spectrum_channels(noisy, variant).astype(np.float64)
    x = _spectrum_channels(target, variant).astype(np.float64)
    if y.shape != x.shape:
        raise ValueError("Input and target axes differ")
    z = y[0] if variant == "real" else y[0]+1j*y[1]
    c = tv_denoise(z, lam, maxiter=80)
    condition = np.asarray([c], dtype=np.float64) if variant == "real" else np.asarray([c.real,c.imag])
    inp = np.concatenate((y, condition), axis=0) if conditioned else y
    return (inp/scale).astype(np.float32), ((y-x)/scale).astype(np.float32)


def train_tvcondnet(train_examples: Sequence[Mapping], validation_examples: Sequence[Mapping],
                    model_path: Path, *, variant="complex", conditioned=True,
                    tv_lambda=0.1, epochs=30, patience=6, seed=42,
                    learning_rate=1e-3):
    """Train residual noise predictor Y-R(Y,TV(Y)) using local FID FFT.

    Examples must contain `noisy_fid`, `target_fid`, `reference_kind`,
    `acquisition_id`, `reference_ids`, `session_id`, `setting_family`. The caller must pass
    different whole session/family groups to train and validation. A single
    global training-only scale preserves between-acquisition amplitudes.
    """
    if not train_examples or not validation_examples or epochs < 1:
        raise ValueError("Train/validation examples and positive epochs required")
    if variant not in ("real", "complex") or tv_lambda < 0:
        raise ValueError("Invalid branch or TV lambda")
    all_examples = list(train_examples)+list(validation_examples)
    for row in all_examples:
        if not all(row.get(k) for k in ("acquisition_id", "session_id", "setting_family",
                                        "reference_kind")):
            raise ValueError("Explicit acquisition/session/family/reference-kind metadata is required")
        if str(row["acquisition_id"]) in {str(x) for x in row.get("reference_ids", ())}:
            raise ValueError("Tested input contributes to its own reference")
        if row.get("reference_kind")=="independent_noisy_acquisition" and not (
                row.get("signal_stability_verified") and row.get("noise_independence_verified")):
            raise ValueError("Noise2Noise needs verified stable signal and independent noise")
    for key in ("session_id", "setting_family"):
        if {str(x[key]) for x in train_examples} & {str(x[key]) for x in validation_examples}:
            raise ValueError(f"Train/validation {key} leakage")
    train_ids={str(x["acquisition_id"]) for x in train_examples}
    validation_ids={str(x["acquisition_id"]) for x in validation_examples}
    if train_ids & validation_ids or any(
            train_ids.intersection(str(x) for x in row.get("reference_ids", ()))
            for row in validation_examples) or any(
            validation_ids.intersection(str(x) for x in row.get("reference_ids", ()))
            for row in train_examples):
        raise ValueError("Train/validation acquisition or target-reference leakage")
    torch, nn = _torch_import()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(seed)
    scale = float(np.median([np.sqrt(np.mean(np.abs(_spectrum_channels(r["noisy_fid"],variant))**2))
                             for r in train_examples]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid training-only normalization scale")
    train = [_prepare_spectral_pair(r["noisy_fid"],r["target_fid"],variant,tv_lambda,scale,conditioned)
             for r in train_examples]
    val = [_prepare_spectral_pair(r["noisy_fid"],r["target_fid"],variant,tv_lambda,scale,conditioned)
           for r in validation_examples]
    out_channels = 1 if variant == "real" else 2
    network = _unet_factory(nn, out_channels*(2 if conditioned else 1), out_channels).cpu()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    best = float("inf")
    state = None
    stale = 0
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        network.train()
        for index in rng.permutation(len(train)):
            x, residual = train[int(index)]
            xx = torch.from_numpy(x[None]); rr = torch.from_numpy(residual[None])
            optimizer.zero_grad()
            loss = ((network(xx)-rr)**2).mean()
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            val_mse = float(np.mean([((network(torch.from_numpy(x[None]))-
                                     torch.from_numpy(r[None]))**2).mean().item()
                                     for x,r in val]))
        history.append({"epoch":epoch+1,"validation_residual_mse":val_mse})
        if val_mse < best - 1e-8:
            best = val_mse
            state = copy.deepcopy(network.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if state is None:
        raise RuntimeError("Neural denoiser did not produce a finite validation score")
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict":state,"variant":variant,"conditioned":bool(conditioned),
                "tv_lambda":float(tv_lambda),"scale":scale,
                "architecture":"local 1D U-Net 16/32/64 residual-noise predictor",
                "history":history,"train_examples":len(train),"validation_examples":len(val)}, model_path)
    return {"status":"TRAINED", "branch":"TVCondNet_real_spectrum_reimplementation" if variant=="real"
            else "TVCondNet_complex_NMR_adaptation", "model_path":str(model_path),
            "conditioned":bool(conditioned),"validation_residual_mse":best,
            "scale":scale,"history":history}


def apply_tvcondnet(fid, model_path: Path):
    """Apply saved model; complex branch returns reconstructed complex FID."""
    torch, nn = _torch_import()
    saved = torch.load(Path(model_path), map_location="cpu", weights_only=False)
    variant = saved["variant"]
    conditioned = saved["conditioned"]
    channels = _spectrum_channels(fid, variant).astype(np.float64)
    z = channels[0] if variant=="real" else channels[0]+1j*channels[1]
    c = tv_denoise(z, saved["tv_lambda"], maxiter=80)
    cond = np.asarray([c],dtype=float) if variant=="real" else np.asarray([c.real,c.imag])
    inp = np.concatenate((channels,cond)) if conditioned else channels
    network = _unet_factory(nn, channels.shape[0]*(2 if conditioned else 1), channels.shape[0])
    network.load_state_dict(saved["state_dict"])
    network.eval()
    with torch.no_grad():
        residual = network(torch.from_numpy((inp/saved["scale"]).astype(np.float32)[None])).numpy()[0]
    estimate = channels - saved["scale"]*residual
    if variant=="real":
        # The source paper estimates only real spectrum. Preserve the original
        # imaginary channel rather than falsely claim complex FID denoising.
        source = np.fft.fft(_array_1d(fid), norm="ortho")
        return {"real_spectrum":estimate[0],"spectrum":estimate[0]+1j*source.imag,
                "fid":np.fft.ifft(estimate[0]+1j*source.imag,norm="ortho"),
                "imaginary_channel":"unchanged from input"}
    spectrum = estimate[0]+1j*estimate[1]
    return {"spectrum":spectrum,"fid":np.fft.ifft(spectrum,norm="ortho")}


def phase_equivariance_error(denoise: Callable, fid, *, angles=(0.3,1.0,2.0)) -> float:
    """Relative equivariance error; must be measured, never assumed for U-Net."""
    x = _array_1d(fid)
    reference = _array_1d(denoise(x))
    errors = [np.linalg.norm(denoise(x*np.exp(1j*a))-reference*np.exp(1j*a)) /
              max(np.linalg.norm(reference),1e-12) for a in angles]
    return float(max(errors))


def denoising_metrics(estimate, target, *, signal_mask=None, noise_mask=None):
    """Complex RMSE, phase and amplitude bias, explicit optional residual SNR."""
    x, y = _array_1d(estimate), _array_1d(target)
    if len(x)!=len(y):
        raise ValueError("Denoising output/target axes differ")
    result = {"complex_rmse":float(np.sqrt(np.mean(np.abs(x-y)**2))),
              "relative_l2_error":float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-12)),
              "amplitude_ratio":float(np.linalg.norm(x)/max(np.linalg.norm(y),1e-12)),
              "phase_offset_deg":float(np.degrees(np.angle(np.vdot(y,x)))),
              "reference_kind":"supplied_target_not_ideal_ground_truth"}
    if signal_mask is not None and noise_mask is not None:
        sm = np.asarray(signal_mask,bool); nm=np.asarray(noise_mask,bool)
        if sm.shape!=x.shape or nm.shape!=x.shape or not sm.any() or not nm.any():
            raise ValueError("SNR masks must select valid signal/noise bins")
        power=float(np.mean(np.abs(x[sm])**2))
        noise=float(np.mean(np.abs(x[nm])**2))
        result["spectral_power_snr_db"] = 10*math.log10(max(power,1e-30)/max(noise,1e-30))
        result["snr_definition"] = "mean complex spectral power on signal bins / noise bins"
    return result


def benchmark_f_methods(records: Sequence[Mapping], model_dir: Path, *,
                        seed=42, tv_candidates=(0.01,0.1,1.0),
                        hankel_ranks=(1,2,3), epochs=30,
                        pilot_frequencies_hz: Sequence[float] | None = None,
                        sample_hz: float | None = None) -> dict:
    """Run paired local F baselines and both labelled neural branches.

    Each row needs noisy_fid, target_fid and grouped_split metadata. Targets
    should be independent high-quality measured records or deliberately
    noise-corrupted copies of a measured pilot, with `reference_kind` recorded.
    No RF call is made here. Test examples select neither hyperparameters nor
    training weights. Timing includes method inference, not training.
    """
    import time
    if any("noisy_fid" not in row or "target_fid" not in row or
           "reference_kind" not in row for row in records):
        raise ValueError("FID arrays and explicit reference_kind required")
    if (pilot_frequencies_hz is None)!=(sample_hz is None):
        raise ValueError("Physical-fit baseline requires both pilot frequencies and verified sample rate")
    split = grouped_split(records, seed=seed)
    train = [records[i] for i in split["train"]]
    val = [records[i] for i in split["validation"]]
    test = [records[i] for i in split["test"]]
    validation_spectral = [(np.fft.fft(_array_1d(r["noisy_fid"]),norm="ortho"),
                            np.fft.fft(_array_1d(r["target_fid"]),norm="ortho")) for r in val]
    tv = select_tv_lambda(validation_spectral,tv_candidates)
    hankel = select_hankel_rank([(r["noisy_fid"],r["target_fid"]) for r in val],hankel_ranks)
    models = {}
    model_dir = Path(model_dir)
    for variant, conditioned in (("real",True),("complex",True),("complex",False)):
        name = f"{variant}_{'conditioned' if conditioned else 'plain_unet'}"
        try:
            models[name] = train_tvcondnet(train,val,model_dir/f"{name}.pt",
                variant=variant,conditioned=conditioned,tv_lambda=tv["lambda"],
                epochs=epochs,seed=seed)
        except RuntimeError as exc:
            if "TORCH_UNAVAILABLE" not in str(exc):
                raise
            models[name] = {"status":"DEPENDENCY_FAILED","reason":str(exc)}
    rows=[]
    for index, row in zip(split["test"],test):
        noisy=_array_1d(row["noisy_fid"])
        target=_array_1d(row["target_fid"])
        if len(noisy)!=len(target):
            raise ValueError("Test target axis differs")
        methods={
            "unchanged":lambda:noisy,
            "complex_TV":lambda:np.fft.ifft(tv_denoise(np.fft.fft(noisy,norm="ortho"),tv["lambda"]),norm="ortho"),
            "randomized_Hankel_rank":lambda:hankel_low_rank_fid(noisy,hankel["rank"]),
        }
        if row.get("average_inputs"):
            ids={str(x) for x in row.get("average_input_ids",())}
            if len(ids)!=len(row["average_inputs"]) or str(row["acquisition_id"]) not in ids:
                raise ValueError("Average inputs need distinct IDs including the tested acquisition")
            if ids.intersection(str(x) for x in row.get("reference_ids",())):
                raise ValueError("Reference acquisitions cannot be reused in test-input average")
            methods["independent_acquisition_average"] = lambda:average_fids(
                row["average_inputs"],axes=row.get("average_axes"))
        if pilot_frequencies_hz is not None:
            methods["fixed_pilot_complex_multiplet_fit"] = lambda:fit_exponential_fid(
                noisy,float(sample_hz),pilot_frequencies_hz)["reconstruction"]
        for name, info in models.items():
            if info["status"]=="TRAINED":
                methods[name] = lambda path=info["model_path"]:apply_tvcondnet(noisy,Path(path))["fid"]
        for name, method in methods.items():
            start=time.perf_counter()
            estimate=method()
            elapsed=time.perf_counter()-start
            rows.append({"test_record_index":index,"acquisition_id":str(row["acquisition_id"]),
                         "reference_kind":str(row["reference_kind"]),"method":name,
                         "local_inference_seconds":elapsed,
                         **denoising_metrics(estimate,target)})
    return {"split_indices":split,"tv_selection":tv,"hankel_selection":hankel,
            "models":models,"test_rows":rows,
            "claim_scope":"local same-FID analysis; target need not be noiseless physical truth"}


# G1: a small circuit IR. A gate is a logical operation until the hardware
# adapter has explicitly mapped it to a verified low-level pulse sequence.
@dataclass(frozen=True)
class Gate:
    kind: str
    qubits: tuple[int, ...]
    angle_rad: float = 0.0
    phase_rad: float = 0.0

    def __post_init__(self):
        if self.kind not in ("RXY","RZ","CZ","RZZ"):
            raise ValueError("Only RXY, RZ, CZ and RZZ logical gates are supported")
        if len(self.qubits) != (1 if self.kind in ("RXY","RZ") else 2):
            raise ValueError("Wrong number of qubits for logical gate")
        if len(set(self.qubits)) != len(self.qubits) or min(self.qubits)<0:
            raise ValueError("Qubit indices must be distinct nonnegative integers")
        if not math.isfinite(self.angle_rad) or not math.isfinite(self.phase_rad):
            raise ValueError("Gate angles must be finite")


def _pauli_at(n, q, axis):
    i=np.eye(2,dtype=complex)
    mats={"X":np.array([[0,1],[1,0]],complex),
          "Y":np.array([[0,-1j],[1j,0]],complex),
          "Z":np.diag([1.,-1.]).astype(complex)}
    matrix=np.array([[1.]],complex)
    for index in range(n):
        matrix=np.kron(matrix,mats[axis] if index==q else i)
    return matrix


def circuit_unitary(gates: Sequence[Gate], n_qubits: int):
    """Exact ideal logical unitary including spectator identities, n<=3."""
    if not 1<=n_qubits<=3:
        raise ValueError("This small reference circuit supports one to three spins")
    dimension=2**n_qubits
    unitary=np.eye(dimension,dtype=complex)
    identity=np.eye(dimension,dtype=complex)
    for gate in gates:
        if max(gate.qubits)>=n_qubits:
            raise ValueError("Gate addresses a spin outside circuit dimension")
        if gate.kind=="RXY":
            x=_pauli_at(n_qubits,gate.qubits[0],"X")
            y=_pauli_at(n_qubits,gate.qubits[0],"Y")
            p=math.cos(gate.phase_rad)*x+math.sin(gate.phase_rad)*y
            matrix=math.cos(gate.angle_rad/2)*identity-1j*math.sin(gate.angle_rad/2)*p
        elif gate.kind=="RZ":
            z=_pauli_at(n_qubits,gate.qubits[0],"Z")
            matrix=math.cos(gate.angle_rad/2)*identity-1j*math.sin(gate.angle_rad/2)*z
        elif gate.kind=="CZ":
            z1=_pauli_at(n_qubits,gate.qubits[0],"Z")
            z2=_pauli_at(n_qubits,gate.qubits[1],"Z")
            p11=(identity-z1)@(identity-z2)/4
            matrix=identity-2*p11
        else:
            zz=_pauli_at(n_qubits,gate.qubits[0],"Z")@_pauli_at(n_qubits,gate.qubits[1],"Z")
            matrix=math.cos(gate.angle_rad/2)*identity-1j*math.sin(gate.angle_rad/2)*zz
        unitary=matrix@unitary
    return unitary


def unitary_equivalent(a, b, *, atol=1e-9):
    """Compare ideal operations up to a global phase, never measured fidelity."""
    u,v=np.asarray(a,complex),np.asarray(b,complex)
    if u.shape!=v.shape or u.ndim!=2 or u.shape[0]!=u.shape[1]:
        return False
    phase=np.vdot(v,u)
    if abs(phase)<1e-12:
        return False
    return bool(np.linalg.norm(u-v*phase/abs(phase))<=atol*np.sqrt(u.size))


def simplify_circuit(gates: Sequence[Gate], n_qubits: int):
    """Remove adjacent inverse rotations and merge same-axis rotations.

    Every pass is checked against an ideal spectator-aware unitary; an invalid
    rewrite fails closed rather than silently changing the algorithm.
    """
    original=list(gates)
    stack=[]
    for gate in original:
        if stack and gate.kind in ("RXY","RZ","RZZ"):
            previous=stack[-1]
            same=(previous.kind==gate.kind and previous.qubits==gate.qubits and
                  (gate.kind!="RXY" or abs(np.angle(np.exp(1j*(previous.phase_rad-gate.phase_rad))))<1e-10))
            if same:
                stack.pop()
                merged=previous.angle_rad+gate.angle_rad
                if abs(np.remainder(merged+2*math.pi,4*math.pi)-2*math.pi)>1e-10:
                    stack.append(Gate(gate.kind,gate.qubits,merged,gate.phase_rad))
                continue
        if gate.kind in ("RXY","RZ","RZZ") and abs(gate.angle_rad)<1e-12:
            continue
        stack.append(gate)
    if not unitary_equivalent(circuit_unitary(original,n_qubits),circuit_unitary(stack,n_qubits)):
        raise RuntimeError("Circuit simplification failed ideal-unitary verification")
    return stack


def compile_virtual_z(gates: Sequence[Gate], n_qubits: int):
    """Commute Z rotations into RF phases, retaining final frame corrections.

    The returned `pulse_gates` omit physical Z pulses. `final_frames_rad` MUST
    be handled by the readout basis or materialized as logical RZ operations.
    """
    frames=np.zeros(n_qubits)
    pulses=[]
    for gate in gates:
        if gate.kind=="RZ":
            frames[gate.qubits[0]]+=gate.angle_rad
        elif gate.kind=="RXY":
            q=gate.qubits[0]
            pulses.append(Gate("RXY",gate.qubits,gate.angle_rad,gate.phase_rad-frames[q]))
        else:
            pulses.append(gate)  # CZ/RZZ commute with all single-spin RZ.
    materialized=pulses+[Gate("RZ",(q,),float(angle)) for q,angle in enumerate(frames) if abs(angle)>1e-12]
    if not unitary_equivalent(circuit_unitary(gates,n_qubits),circuit_unitary(materialized,n_qubits)):
        raise RuntimeError("Virtual-Z frame rewrite failed ideal-unitary verification")
    return {"pulse_gates":pulses,"final_frames_rad":frames.tolist(),
            "materialized_for_verification":materialized,
            "frame_policy":"readout must absorb final frame or apply equivalent logical RZ"}


def rank_mappings(gates: Sequence[Gate], logical_qubits: int, physical_qubits: Sequence[int],
                  *, one_qubit_error: Mapping[int,float], two_qubit_error: Mapping[tuple[int,int],float],
                  one_qubit_duration_s: Mapping[int,float], two_qubit_duration_s: Mapping[tuple[int,int],float],
                  idle_error_per_s=0.0):
    """Enumerate small supported layouts with pilot-derived error/time costs."""
    if logical_qubits>3 or len(physical_qubits)<logical_qubits:
        raise ValueError("Only supported small physical mappings may be enumerated")
    results=[]
    for physical in itertools.permutations(physical_qubits,logical_qubits):
        mapping=dict(enumerate(physical))
        cost=0.0; duration=0.0; supported=True
        for gate in gates:
            if gate.kind in ("RXY","RZ"):
                q=mapping[gate.qubits[0]]
                if q not in one_qubit_error or q not in one_qubit_duration_s:
                    supported=False; break
                if gate.kind=="RXY":
                    cost+=float(one_qubit_error[q])
                    duration+=float(one_qubit_duration_s[q])
            else:
                pair=tuple(sorted(mapping[q] for q in gate.qubits))
                if pair not in two_qubit_error or pair not in two_qubit_duration_s:
                    supported=False; break
                cost+=float(two_qubit_error[pair])
                duration+=float(two_qubit_duration_s[pair])
        if supported:
            results.append({"mapping":mapping,"pilot_error_sum":cost,
                            "duration_s":duration,"score":cost+idle_error_per_s*duration})
    return sorted(results,key=lambda row:row["score"])


def bounded_anneal(objective: Callable[[np.ndarray],float], initial,
                   bounds: Sequence[tuple[float,float]], *, evaluations=32,
                   chains=3, seed=42, pilot_repeat=3):
    """Multiple *sequential* bounded SA chains over one shared call budget.

    The supplied objective may measure hardware, so this routine never calls
    it concurrently. Temperature starts at the measured pilot cost standard
    deviation (or a small floor). All proposals and rejections are logged.
    """
    x0=np.asarray(initial,float)
    limit=np.asarray(bounds,float)
    if x0.ndim!=1 or limit.shape!=(len(x0),2) or np.any(limit[:,0]>=limit[:,1]):
        raise ValueError("Initial point and finite low/high bounds required")
    if np.any(~np.isfinite(limit)) or np.any(x0<limit[:,0]) or np.any(x0>limit[:,1]):
        raise ValueError("Initial pulse lies outside explicit supported bounds")
    if chains<1 or evaluations<chains+pilot_repeat or pilot_repeat<2:
        raise ValueError("Budget must cover pilot repeats and every chain")
    rng=np.random.default_rng(seed)
    trace=[]; calls=0

    def evaluate(point, chain, reason):
        nonlocal calls
        value=float(objective(np.asarray(point,float).copy()))
        if not math.isfinite(value):
            raise ValueError("Measured cost must be finite")
        calls+=1
        trace.append({"evaluation":calls,"chain":chain,"reason":reason,
                      "parameters":np.asarray(point,float).tolist(),"cost":value})
        return value

    pilot=np.asarray([evaluate(x0,-1,"pilot_variability") for _ in range(pilot_repeat)])
    sigma=float(np.std(pilot,ddof=1))
    temperature=max(2*sigma,1e-6)
    state=[x0.copy() for _ in range(chains)]
    values=[evaluate(x0,i,"chain_initial") for i in range(chains)]
    best_index=int(np.argmin(values)); best_x=state[best_index].copy(); best_cost=values[best_index]
    step=0
    while calls<evaluations:
        index=step%chains
        fraction=step/max(1,evaluations-pilot_repeat-chains)
        proposal=state[index]+rng.normal(0,0.08*(limit[:,1]-limit[:,0])*(1-0.7*fraction))
        # Reflect, do not silently clip an out-of-range commanded pulse.
        span=limit[:,1]-limit[:,0]
        proposal=limit[:,0]+span-np.abs((proposal-limit[:,0])%(2*span)-span)
        cost=evaluate(proposal,index,"proposal")
        delta=cost-values[index]
        current_temperature=max(temperature*(0.03**fraction),1e-9)
        accept=delta<=0 or rng.random()<math.exp(-min(delta/current_temperature,700))
        trace[-1]["accepted"] = bool(accept)
        trace[-1]["temperature"] = current_temperature
        if accept:
            state[index]=proposal; values[index]=cost
        if cost<best_cost:
            best_cost=cost;best_x=proposal.copy()
        step+=1
    return {"best_parameters":best_x,"best_cost":float(best_cost),
            "evaluations":calls,"pilot_cost_sd":sigma,"initial_temperature":temperature,
            "trace":trace,"hardware_calls_sequential":True,
            "cost_scope":"caller-supplied multi-sequence measured objective"}


def grid_and_nelder_mead(objective: Callable, initial, bounds, *, evaluations=32, seed=42):
    """Equal-call classical scan plus Nelder–Mead baseline for G2."""
    x0=np.asarray(initial,float); lim=np.asarray(bounds,float)
    if evaluations<4 or lim.shape!=(len(x0),2) or np.any(lim[:,0]>=lim[:,1]) or np.any(x0<lim[:,0]) or np.any(x0>lim[:,1]):
        raise ValueError("Insufficient budget or invalid bounds")
    rng=np.random.default_rng(seed)
    trace=[]
    scan_budget=max(2,evaluations//2)
    points=[x0.copy()]
    for i in range(scan_budget-1):
        q=i%len(x0)
        v=x0.copy(); v[q]=lim[q,0]+(lim[q,1]-lim[q,0])*(i+0.5)/scan_budget
        points.append(v)
    best=None
    def call(x, label):
        if len(trace)>=evaluations:
            raise StopIteration
        x=np.asarray(x,float)
        if np.any(x<lim[:,0]) or np.any(x>lim[:,1]):
            return float("inf")
        y=float(objective(x.copy()))
        trace.append({"parameters":x.tolist(),"cost":y,"phase":label})
        return y
    for p in points:
        v=call(p,"coordinate_scan")
        if best is None or v<best[0]:best=(v,p.copy())
    def wrapped(x):
        try:return call(x,"nelder_mead")
        except StopIteration:return float("inf")
    minimize(wrapped,best[1],method="Nelder-Mead",options={"maxfev":evaluations-len(trace),
              "xatol":1e-4,"fatol":1e-6})
    while len(trace)<evaluations:
        # An early NM stop does not grant the baseline fewer measurements.
        point=lim[:,0]+rng.random(len(x0))*(lim[:,1]-lim[:,0])
        call(point,"budget_fill_scan")
    winner=min(trace,key=lambda r:r["cost"])
    return {"best_parameters":np.asarray(winner["parameters"]),
            "best_cost":winner["cost"],"evaluations":len(trace),"trace":trace,
            "hardware_calls_sequential":True}


def gate_error_trend(repetitions, measured_errors, *, max_error=0.15,
                     min_points=3, min_r2=0.9):
    """Use linear error-per-gate slope only in a checked low-error regime."""
    x=np.asarray(repetitions,float); y=np.asarray(measured_errors,float)
    if len(x)!=len(y) or len(x)<min_points or np.any(x<=0) or np.any(y<0):
        raise ValueError("Independent repeated-sequence measurements required")
    if np.max(y)>max_error:
        return {"status":"NONLINEAR_OR_HIGH_ERROR", "reason":"outside frozen low-error regime"}
    coeff=np.polyfit(x,y,1)
    predicted=np.polyval(coeff,x)
    ss=float(np.sum((y-y.mean())**2))
    r2=1-float(np.sum((y-predicted)**2))/max(ss,1e-15)
    if coeff[0]<0 or r2<min_r2:
        return {"status":"LINEAR_FIT_UNSUPPORTED","r2":r2}
    return {"status":"LINEAR_TREND_SUPPORTED","error_per_gate_slope":float(coeff[0]),
            "intercept":float(coeff[1]),"r2":r2,
            "claim_scope":"sequence error trend, not process fidelity"}


# G3: commuting Z and ZZ model only. Finite-width pulse effects and non-Z
# couplings require separate physical validation by the orchestrator.
@dataclass(frozen=True)
class DDPulse:
    time_s: float
    qubit: int
    axis: str = "X"

    def __post_init__(self):
        if self.time_s <= 0 or not math.isfinite(self.time_s) or self.qubit < 0:
            raise ValueError("DD pulse must be inside a positive-time idle interval")
        if self.axis not in ("X","Y"):
            raise ValueError("DD pulse must be a verified X or Y pi rotation")


def toggling_integrals(duration_s: float, pulses: Sequence[DDPulse],
                       qubits: Sequence[int], zz_pairs: Sequence[tuple[int,int]] = ()):
    """Exact toggling integrals for ideal instantaneous pi pulses.

    A simultaneous pi on both spins changes y_i and y_j together, leaving
    y_i*y_j unchanged. This matters for unwanted ZZ suppression.
    """
    if duration_s<=0 or not math.isfinite(duration_s):
        raise ValueError("Idle interval must be positive")
    qs=tuple(qubits)
    if len(set(qs))!=len(qs):
        raise ValueError("Qubit labels must be unique")
    for p in pulses:
        if p.qubit not in qs or p.time_s>=duration_s:
            raise ValueError("DD pulse outside specified qubits/idle interval")
    for a,b in zz_pairs:
        if a==b or a not in qs or b not in qs:
            raise ValueError("ZZ pair must contain two specified qubits")
    z={q:0.0 for q in qs}
    zz={tuple(sorted(pair)):0.0 for pair in zz_pairs}
    signs={q:1 for q in qs}
    grouped={}
    for pulse in pulses:
        grouped.setdefault(float(pulse.time_s),[]).append(pulse.qubit)
    current=0.0
    for next_time in sorted([*grouped,duration_s]):
        dt=next_time-current
        for q in qs:z[q]+=signs[q]*dt
        for (a,b) in zz:zz[(a,b)]+=signs[a]*signs[b]*dt
        for q in grouped.get(next_time,()):signs[q]*=-1
        current=next_time
    return {"single_z_s":z,"zz_s":zz,"final_signs":signs,
            "model":"commuting Z/ZZ, instantaneous perfect pi pulses"}


def _standard_dd(qubit: int, duration_s: float, name: str, offset=0.0):
    if name=="none":return []
    if name=="hahn":fractions=[0.5]; axes=["X"]
    elif name=="cpmg2":fractions=[0.25,0.75];axes=["X","X"]
    elif name=="xy4":fractions=[0.125,0.375,0.625,0.875];axes=["X","Y","X","Y"]
    else:raise ValueError("Unknown standard DD family")
    times=[(f+offset)*duration_s for f in fractions]
    if any(t<=0 or t>=duration_s for t in times):
        return None
    return [DDPulse(t,qubit,axis) for t,axis in zip(times,axes)]


def design_contextual_dd(duration_s: float, qubits: Sequence[int], *,
                         single_z_weights: Mapping[int,float],
                         unwanted_zz_weights: Mapping[tuple[int,int],float],
                         desired_zz_pairs: Sequence[tuple[int,int]] = (),
                         timing_verified: bool,
                         pi_pulse_error: float,
                         pulse_duration_s: float,
                         desired_tolerance_fraction=0.05):
    """Search Hahn/CPMG/XY4 and offsets while preserving intended ZZ.

    This returns a *candidate* pulse schedule. The orchestrator must still
    verify actual timing, pulse widths, and ideal propagator for its NMR model.
    Missing timing is an explicit blocker, not a guessed free-precession delay.
    """
    if not timing_verified:
        raise RuntimeError("UNVERIFIED_TIMING: DD needs hardware-verified in-sequence idle delays")
    if pi_pulse_error<0 or pulse_duration_s<0:
        raise ValueError("Pulse cost and duration must be nonnegative pilot values")
    qs=tuple(qubits)
    if not 1<=len(qs)<=3:
        raise ValueError("One to three identified spins required")
    pairs=set(tuple(sorted(x)) for x in (*unwanted_zz_weights,*desired_zz_pairs))
    options=[]
    for q in qs:
        seqs=[]
        for family in ("none","hahn","cpmg2","xy4"):
            for offset in ((0.,) if family=="none" else (-0.06,0.,0.06)):
                pulses=_standard_dd(q,duration_s,family,offset)
                if pulses is not None and len(pulses)*pulse_duration_s<0.25*duration_s:
                    seqs.append((family,offset,pulses))
        options.append(seqs)
    best=None
    baseline=toggling_integrals(duration_s,[],qs,list(pairs))
    for selection in itertools.product(*options):
        pulses=list(itertools.chain.from_iterable(option[2] for option in selection))
        ints=toggling_integrals(duration_s,pulses,qs,list(pairs))
        if any(abs(ints["zz_s"][tuple(sorted(pair))]-duration_s)>
               desired_tolerance_fraction*duration_s for pair in desired_zz_pairs):
            continue
        # Even pulse parity keeps the idle-frame logical operation unchanged.
        if any(ints["final_signs"][q]!=1 for q in qs):
            continue
        loss=sum(float(weight)*abs(ints["single_z_s"][q])/duration_s
                 for q,weight in single_z_weights.items())
        loss+=sum(float(weight)*abs(ints["zz_s"][tuple(sorted(pair))])/duration_s
                  for pair,weight in unwanted_zz_weights.items())
        loss+=pi_pulse_error*len(pulses)
        if best is None or loss<best[0]:
            best=(loss,selection,pulses,ints)
    if best is None:
        return {"status":"NO_DD_PRESERVES_DESIRED_EVOLUTION",
                "baseline_integrals":baseline}
    loss,selection,pulses,integrals=best
    return {"status":"CANDIDATE_REQUIRES_PHYSICAL_VALIDATION",
            "families":{q:{"name":choice[0],"offset_fraction":choice[1]}
                        for q,choice in zip(qs,selection)},
            "pulses":pulses,"serialized_pulses":[{"time_s":p.time_s,"qubit":p.qubit,"axis":p.axis}
                                                  for p in pulses],
            "score":float(loss),"integrals":integrals,
            "baseline_integrals":baseline,"desired_zz_preserved_in_ideal_model":True,
            "finite_width_and_noncommuting_model_verified":False}


# G4: correction of calibrated analogue observables, never bitstring shots.
@dataclass
class AnalogReadoutModel:
    mode: str
    coefficients: np.ndarray  # real output coordinates = [real inputs, imag inputs, 1] @ B
    groups: tuple[tuple[int,...], ...]
    features: int
    ridge: float
    condition_number: float
    preparation_uncertainty: float | None


def _as_complex_matrix(values):
    arr=np.asarray(values,dtype=np.complex128)
    if arr.ndim!=2 or arr.shape[0]<2 or arr.shape[1]<1 or not np.all(np.isfinite(arr)):
        raise ValueError("Expected finite calibration acquisitions × complex observables")
    return arr


def fit_analog_readout(measured, targets, *, ridge=1e-3, mode="full",
                       groups: Sequence[Sequence[int]] | None = None,
                       preparation_uncertainty: float | None = None):
    """Fit affine regularized inverse response, full or grouped blocks.

    All calibration targets require independent state-preparation uncertainty.
    This is a local classical reference, not receiver-only error attribution.
    """
    m=_as_complex_matrix(measured); t=_as_complex_matrix(targets)
    if m.shape!=t.shape or ridge<0 or mode not in ("full","grouped"):
        raise ValueError("Matched complex calibration data and nonnegative ridge required")
    if preparation_uncertainty is not None and preparation_uncertainty<0:
        raise ValueError("Preparation uncertainty cannot be negative")
    d=m.shape[1]
    if mode=="full":
        group_tuple=(tuple(range(d)),)
    else:
        if groups is None:
            raise ValueError("Grouped model requires explicit observable groups")
        group_tuple=tuple(tuple(int(v) for v in group) for group in groups)
        flattened=[x for g in group_tuple for x in g]
        if sorted(flattened)!=list(range(d)):
            raise ValueError("Groups must partition all observable components exactly once")
    input_real=np.column_stack((m.real,m.imag))
    output_real=np.column_stack((t.real,t.imag))
    correction=np.zeros((2*d+1,2*d),dtype=float)
    condition=0.0
    for group in group_tuple:
        cols=list(group)+[d+j for j in group]
        design=np.column_stack((input_real[:,cols],np.ones(len(m))))
        if np.linalg.matrix_rank(design)<len(cols)+1:
            raise ValueError("Calibration states do not span independent complex observables")
        gram=design.T@design
        reg=np.eye(len(cols)+1)*ridge
        reg[-1,-1]=0.0
        prior=np.zeros((len(cols)+1,len(cols)))
        prior[:len(cols),:]=np.eye(len(cols))
        coeff=np.linalg.solve(gram+reg,design.T@output_real[:,cols]+reg@prior)
        correction[np.ix_(cols,cols)]=coeff[:-1]
        correction[-1,cols]=coeff[-1]
        condition=max(condition,float(np.linalg.cond(gram+reg)))
    return AnalogReadoutModel(mode,correction,group_tuple,d,float(ridge),
                              condition,preparation_uncertainty)


def apply_analog_readout(model: AnalogReadoutModel, measured):
    """Apply correction to complex deviation coefficients; no simplex imposed."""
    values=np.asarray(measured,dtype=np.complex128)
    one=values.ndim==1
    if one:values=values[None,:]
    if values.ndim!=2 or values.shape[1]!=model.features or not np.all(np.isfinite(values)):
        raise ValueError("Measured observable dimensions do not match calibration")
    z=np.column_stack((values.real,values.imag,np.ones(len(values))))@model.coefficients
    out=z[:,:model.features]+1j*z[:,model.features:]
    return out[0] if one else out


def compare_readout_models(train_measured, train_targets, validation_measured,
                           validation_targets, *, ridge_candidates=(1e-5,1e-3,0.1),
                           groups=None, preparation_uncertainty=None):
    """Select strong full/grouped affine baselines on validation states."""
    rows=[]; fitted={}
    for mode in (("full","grouped") if groups is not None else ("full",)):
        for ridge in ridge_candidates:
            model=fit_analog_readout(train_measured,train_targets,ridge=ridge,
                    mode=mode,groups=groups,preparation_uncertainty=preparation_uncertainty)
            val=apply_analog_readout(model,validation_measured)
            loss=float(np.mean(np.abs(val-np.asarray(validation_targets))**2))
            key=(mode,float(ridge))
            fitted[key]=model
            rows.append({"mode":mode,"ridge":float(ridge),"validation_complex_mse":loss,
                         "condition_number":model.condition_number})
    winner=min(rows,key=lambda x:x["validation_complex_mse"])
    return {"winner":winner,"model":fitted[(winner["mode"],winner["ridge"])],
            "validation_curve":rows}


def _readout_net_factory(nn, features):
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.one=nn.Linear(2*features, max(8,2*features))
            self.two=nn.Linear(max(8,2*features),2*features)
            nn.init.zeros_(self.two.weight)
            nn.init.zeros_(self.two.bias)

        def forward(self, x):
            import torch
            return x+self.two(torch.tanh(self.one(x)))
    return Residual()


def train_residual_readout(train_measured, train_targets, validation_measured,
                           validation_targets, baseline: AnalogReadoutModel,
                           model_path: Path, *, epochs=80, patience=10, seed=42,
                           learning_rate=1e-3,
                           train_state_ids: Sequence[str] | None = None,
                           validation_state_ids: Sequence[str] | None = None):
    """Train small near-identity residual on top of frozen affine baseline."""
    train_m=_as_complex_matrix(train_measured);train_t=_as_complex_matrix(train_targets)
    val_m=_as_complex_matrix(validation_measured);val_t=_as_complex_matrix(validation_targets)
    if train_m.shape!=train_t.shape or val_m.shape!=val_t.shape or train_m.shape[1]!=baseline.features:
        raise ValueError("Train/validation states and calibrated features must align")
    if train_state_ids is None or validation_state_ids is None:
        raise ValueError("Independent calibration-state IDs are required for residual-network split")
    if len(train_state_ids)!=len(train_m) or len(validation_state_ids)!=len(val_m) or (
            {str(x) for x in train_state_ids} & {str(x) for x in validation_state_ids}):
        raise ValueError("Train/validation calibration states must have disjoint matching IDs")
    torch, nn=_torch_import()
    torch.set_num_threads(min(torch.get_num_threads(),4))
    torch.manual_seed(seed)
    base_train=apply_analog_readout(baseline,train_m)
    base_val=apply_analog_readout(baseline,val_m)
    def realify(z):return np.column_stack((z.real,z.imag)).astype(np.float32)
    x=torch.from_numpy(realify(base_train));y=torch.from_numpy(realify(train_t))
    vx=torch.from_numpy(realify(base_val));vy=torch.from_numpy(realify(val_t))
    net=_readout_net_factory(nn,baseline.features).cpu()
    opt=torch.optim.Adam(net.parameters(),lr=learning_rate)
    best=float("inf");state=None;stale=0;history=[]
    for epoch in range(epochs):
        net.train();opt.zero_grad()
        loss=((net(x)-y)**2).mean()
        loss.backward();opt.step()
        net.eval()
        with torch.no_grad():score=float(((net(vx)-vy)**2).mean().item())
        history.append({"epoch":epoch+1,"validation_mse":score})
        if score<best-1e-9:
            best=score;state=copy.deepcopy(net.state_dict());stale=0
        else:
            stale+=1
            if stale>=patience:break
    if state is None:
        raise RuntimeError("Residual correction did not produce a finite score")
    model_path=Path(model_path);model_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":state,"features":baseline.features,"history":history,
                "scope":"NMR analogue residual correction after affine model"},model_path)
    return {"status":"TRAINED","model_path":str(model_path),"validation_mse":best,
            "history":history,"preparation_uncertainty":baseline.preparation_uncertainty}


def apply_residual_readout(measured, baseline: AnalogReadoutModel, model_path: Path):
    torch, nn=_torch_import()
    saved=torch.load(Path(model_path),map_location="cpu",weights_only=False)
    if saved["features"]!=baseline.features:
        raise ValueError("Residual model and affine calibration dimensions differ")
    base=np.asarray(apply_analog_readout(baseline,measured))
    one=base.ndim==1
    if one:base=base[None,:]
    x=np.column_stack((base.real,base.imag)).astype(np.float32)
    net=_readout_net_factory(nn,baseline.features)
    net.load_state_dict(saved["state_dict"]);net.eval()
    with torch.no_grad():out=net(torch.from_numpy(x)).numpy()
    result=out[:,:baseline.features]+1j*out[:,baseline.features:]
    return result[0] if one else result


def g_ablation_plan(*, verified_one_qubit: bool, verified_two_qubit: bool,
                    timing_verified: bool, readout_calibrated: bool):
    """Declare executable G1–G4/leave-one-out cells without inventing 2q control."""
    if not verified_one_qubit:
        return {"status":"DEPENDENCY_FAILED","reason":"one-qubit low-level control unverified"}
    tasks=["one_qubit_sequences","held_out_one_qubit_random_sequences"]
    blocked=[]
    if verified_two_qubit:
        tasks.extend(("bell_preparation","two_qubit_qft_or_bv"))
    else:
        blocked.append({"tasks":["bell_preparation","two_qubit_qft_or_bv"],
                        "status":"DEPENDENCY_FAILED","reason":"low-level 2q pulse implementation unverified"})
    components=["G1_compilation","G2_bounded_annealing"]
    if timing_verified:components.append("G3_contextual_DD")
    else:blocked.append({"component":"G3_contextual_DD","status":"UNVERIFIED_TIMING"})
    if readout_calibrated:components.append("G4_analogue_readout")
    else:blocked.append({"component":"G4_analogue_readout","status":"DEPENDENCY_FAILED"})
    expert=["simple_sequence_optimization"]
    if timing_verified:expert.append("standard_DD")
    if readout_calibrated:expert.append("affine_readout")
    return {"status":"PLAN_READY","tasks":tasks,"conditions":["basic_low_level",
            "expert_"+"_plus_".join(expert),*components,
            "all_available_components",*[f"all_except_{x}" for x in components]],
            "blocked":blocked,"measurement_order":"balanced paired blocks; one worker"}
