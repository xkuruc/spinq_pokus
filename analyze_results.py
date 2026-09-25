"""Analyze saved SpinQLabLink results locally; never connects to the device."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def fit_rabi(measurements):
    widths = np.asarray([entry["width_us"] for entry in measurements], dtype=float)
    values = np.asarray([entry["real"] for entry in measurements], dtype=float)
    if len(widths) < 4 or not np.all(np.isfinite(values)):
        raise ValueError("Rabi fit potrebuje aspoň štyri konečné merania.")
    order = np.argsort(widths)
    widths, values = widths[order], values[order]
    spacing = np.diff(widths)
    if np.any(spacing <= 0):
        raise ValueError("Šírky Rabi pulzov musia byť rôzne.")

    # For five 40-us-spaced measurements, periods below 80 us are aliased.
    period_min = 2.01 * float(np.min(spacing))
    period_max = max(5 * float(np.ptp(widths)), 3 * period_min)
    best = None
    for period in np.linspace(period_min, period_max, 8000):
        omega = 2 * np.pi / period
        design = np.column_stack((np.sin(omega * widths),
                                  np.cos(omega * widths),
                                  np.ones_like(widths)))
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        residual = float(np.sum((values - design @ coefficients) ** 2))
        if best is None or residual < best[0]:
            best = (residual, period, coefficients)

    residual, period, coefficients = best
    sine, cosine, offset = (float(value) for value in coefficients)
    total = float(np.sum((values - np.mean(values)) ** 2))
    return {
        "period_us": float(period),
        "pi_half_us": float(period / 4),
        "pi_us": float(period / 2),
        "amplitude": float(np.hypot(sine, cosine)),
        "phase_rad": float(np.arctan2(cosine, sine)),
        "offset": offset,
        "r_squared": float(1 - residual / total) if total > 0 else None,
        "widths_us": widths.tolist(),
        "measured": values.tolist(),
        "fitted": (sine * np.sin(2 * np.pi * widths / period)
                   + cosine * np.cos(2 * np.pi * widths / period)
                   + offset).tolist(),
    }


def save_rabi_plot(fit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(fit["widths_us"])
    x_smooth = np.linspace(x.min(), x.max(), 500)
    y_smooth = (fit["amplitude"] *
                np.sin(2 * np.pi * x_smooth / fit["period_us"]
                       + fit["phase_rad"]) + fit["offset"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, fit["measured"], label="meranie")
    ax.plot(x_smooth, y_smooth, label="lokálny fit")
    ax.set_xlabel("Šírka pulzu (µs)")
    ax.set_ylabel("Signál (jednotky SpinQ)")
    ax.set_title("Rabiho oscilácia")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fid_fft(step):
    if "fidRe" not in step or "fidIm" not in step:
        return None
    real = np.asarray(step["fidRe"], dtype=float)
    imag = np.asarray(step["fidIm"], dtype=float)
    if (real.ndim != 2 or imag.ndim != 2 or real.shape[1] != 2 or
            imag.shape != real.shape or len(real) < 16):
        raise ValueError("FID dáta majú neočakávaný tvar.")
    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
        raise ValueError("FID dáta obsahujú neplatné čísla.")
    if not np.allclose(real[:, 0], imag[:, 0], rtol=1e-6, atol=1e-9):
        raise ValueError("Časové osi fidRe a fidIm sa líšia.")
    intervals = np.diff(real[:, 0])
    if np.any(intervals <= 0) or not np.allclose(
            intervals, np.median(intervals), rtol=0.02, atol=1e-9):
        raise ValueError("FID časová os nie je rovnomerne vzorkovaná.")

    # SpinQ's own graph labels the FID x-axis as time in milliseconds.
    dt_seconds = float(np.median(intervals)) / 1000
    signal = real[:, 1] + 1j * imag[:, 1]
    windowed = (signal - np.mean(signal)) * np.hanning(len(signal))
    spectrum = np.fft.fftshift(np.fft.fft(windowed))
    frequencies = np.fft.fftshift(np.fft.fftfreq(len(signal), d=dt_seconds))
    magnitude = np.abs(spectrum)
    peak_index = int(np.argmax(magnitude))
    return frequencies, spectrum, magnitude, {
        "samples": int(len(signal)),
        "sample_rate_hz_from_chart": float(1 / dt_seconds),
        "peak_offset_hz": float(frequencies[peak_index]),
    }


def graph_groups(report):
    for measurement in report.get("measurements", []):
        width = measurement["width_us"]
        for index, step in enumerate(measurement.get("raw_result", {})
                                     .get("result", {}).get("graph", []), 1):
            yield f"rabi_{width}us_step{index}", step
    physical = report.get("physical_result", {})
    for index, step in enumerate(physical.get("result", {}).get("graph", []), 1):
        yield f"physical_step{index}", step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path, help="uložený výsledok z results/")
    args = parser.parse_args()
    with args.result_json.open(encoding="utf-8") as handle:
        report = json.load(handle)
    output_dir = args.result_json.parent / (args.result_json.stem + "_analysis")
    output_dir.mkdir(exist_ok=True)

    if report.get("measurements"):
        fit = fit_rabi(report["measurements"])
        (output_dir / "rabi_fit.json").write_text(
            json.dumps(fit, ensure_ascii=False, indent=2), encoding="utf-8")
        save_rabi_plot(fit, output_dir / "rabi_fit.png")
        print("Rabi lokálny fit: perióda {:.1f} µs; π/2 {:.1f} µs; π {:.1f} µs"
              .format(fit["period_us"], fit["pi_half_us"], fit["pi_us"]))

    fft_count = 0
    for label, step in graph_groups(report):
        if not isinstance(step, dict):
            continue
        transformed = fid_fft(step)
        if transformed is None:
            print(f"{label}: FID nie je dostupný; rady: {', '.join(sorted(step))}")
            continue
        frequencies, spectrum, magnitude, summary = transformed
        csv_path = output_dir / (label + "_local_fft.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("frequency_offset_hz", "real", "imag", "magnitude"))
            for frequency, value, size in zip(frequencies, spectrum, magnitude):
                writer.writerow((float(frequency), float(value.real),
                                 float(value.imag), float(size)))
        fft_count += 1
        print(f"{label}: lokálna FFT z {summary['samples']} bodov; "
              f"vrchol pri {summary['peak_offset_hz']:.1f} Hz; {csv_path}")

    print(f"Hotovo: {output_dir} (FFT: {fft_count}). Žiadny príkaz sa neposlal prístroju.")


if __name__ == "__main__":
    main()
