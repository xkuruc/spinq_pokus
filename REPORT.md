# Gemini Lab: Bayesovská kalibrácia viacerých parametrov

Stav: **COMPLETED**. Dokončené fyzické úlohy: **219**. Porovnávacie riadky: **15**.
Exportovaný komplexný FID nie je potvrdený RAW ADC. FID vzorky nie sú nezávislé qubitové shots; serverové FFT môže ďalej bežať.
Referencie a kontrolné rotácie sú oddelené od výberu meraní. Chýbajúce referencie sa nenahrádzajú pilotným odhadom.

## Porovnanie po blokoch

| Metóda | Baseline | Blok | Akvizície | Akvizícia (s) | Fit (s) | Inferencia (s) | Návrh (s) | Koniec–koniec (s) | Δf (Hz) | Δt90 (µs) | Δφ (°) | Kontrolná chyba | Stav | Dôvod |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| A | none | 1 | 10 | 162.75 | 0.59992 | 0 | 0 | 228.7 | 11.011 | 0.0088942 | 2.5676 | 0.26892 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |
| B | A | 1 | 18 | 291.66 | 1.549 | 0 | 0 | 371.76 | 8.6257 | 0.061149 | 1.6585 | 0.52869 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |
| C | none | 1 | 10 | 161.97 | 0.68038 | 0.32784 | 0 | 228.66 | 12.968 | 0.066003 | 2.7437 | 0.22778 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |
| D | C and B | 1 | 10 | 162.37 | 0.33853 | 0.31155 | 0.92186 | 227.61 | 8.5212 | 0.27873 | 0.096249 | 0.21366 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |
| E | C and B | 1 | 10 | 154.89 | 0.34449 | 0.30552 | 0.93655 | 221.19 | 3.094 | 0.048094 | 0.052408 | 0.33376 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |
| A | none | 2 | 10 | 162.12 | 0.90229 | 0 | 0 | 227.01 | 2.7882 | 0.063239 | 1.4818 | 0.29348 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |
| B | A | 2 | 18 | 291.82 | 1.55 | 0 | 0 | 367.27 | 12.601 | 0.11678 | 1.8813 | 0.327 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |
| C | none | 2 | 10 | 162.27 | 0.33311 | 0.2335 | 0 | 227 | 2.984 | 0.080992 | 1.644 | 0.33002 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |
| D | C and B | 2 | 10 | 162.51 | 0.35304 | 0.29032 | 0.93544 | 224.76 | 2.8881 | 0.241 | 0.82658 | 0.34182 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |
| E | C and B | 2 | 10 | 154.07 | 0.63731 | 0.35168 | 1.2276 | 216.28 | 9.1545 | 0.12508 | 0.010366 | 0.32581 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |
| A | none | 3 | 10 | 161.48 | 0.94703 | 0 | 0 | 224.06 | 16.955 | 0.013474 | 2.0046 | 0.2536 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |
| B | A | 3 | 18 | 291.06 | 1.6957 | 0 | 0 | 365.95 | 6.2165 | 0.23952 | 0.53827 | 0.40038 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |
| C | none | 3 | 10 | 163.22 | 0.31779 | 0.17187 | 0 | 227.62 | 6.7401 | 0.37595 | 1.1587 | 0.17714 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |
| D | C and B | 3 | 10 | 162.16 | 0.49523 | 0.22988 | 0.87544 | 222.23 | 13.428 | 0.47827 | 2.529 | 0.39323 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |
| E | C and B | 3 | 10 | 154.49 | 0.32418 | 0.22638 | 0.99775 | 214.26 | 2.7562 | 0.33016 | 1.8275 | 0.29707 | SUCCESS_VALIDATED | independent reference and heldout controls resolve frozen tolerances |

## Neistota a pokrytie

| Metóda | Blok | Neistota referencie | Neistota odhadu | Kalibračný bias | Pokrytie f | Pokrytie t90 | Pokrytie φ |
|---|---:|---|---|---|---|---|---|
| A | 1 | delta_hz: 6.1972, t90_us: 0.72093, phase_deg: 3.9397 | delta_hz: 6.521, t90_us: 0.74759, phase_deg: 4.1334 | frequency_hz: -11.011, t90_us: -0.0088942, relative_phase_deg: 2.5676 | True | True | True |
| B | 1 | delta_hz: 6.1972, t90_us: 0.72093, phase_deg: 3.9397 | delta_hz: 4.0331, t90_us: 0.74123, phase_deg: 2.5668 | frequency_hz: -8.6257, t90_us: 0.061149, relative_phase_deg: 1.6585 | True | True | True |
| C | 1 | delta_hz: 6.1972, t90_us: 0.72093, phase_deg: 3.9397 | delta_hz: 6.5178, t90_us: 0.74644, phase_deg: 4.1336 | frequency_hz: -12.968, t90_us: -0.066003, relative_phase_deg: 2.7437 | True | True | True |
| D | 1 | delta_hz: 6.1972, t90_us: 0.72093, phase_deg: 3.9397 | delta_hz: 5.2576, t90_us: 0.84081, phase_deg: 3.2871 | frequency_hz: 8.5212, t90_us: -0.27873, relative_phase_deg: -0.096249 | True | True | True |
| E | 1 | delta_hz: 6.1972, t90_us: 0.72093, phase_deg: 3.9397 | delta_hz: 5.2304, t90_us: 0.82138, phase_deg: 3.274 | frequency_hz: -3.094, t90_us: -0.048094, relative_phase_deg: 0.052408 | True | True | True |
| A | 2 | delta_hz: 6.1888, t90_us: 0.72487, phase_deg: 3.9368 | delta_hz: 6.5327, t90_us: 0.74708, phase_deg: 4.134 | frequency_hz: -2.7882, t90_us: -0.063239, relative_phase_deg: 1.4818 | True | True | True |
| B | 2 | delta_hz: 6.1888, t90_us: 0.72487, phase_deg: 3.9368 | delta_hz: 4.0268, t90_us: 0.74143, phase_deg: 2.5666 | frequency_hz: -12.601, t90_us: -0.11678, relative_phase_deg: 1.8813 | True | True | True |
| C | 2 | delta_hz: 6.1888, t90_us: 0.72487, phase_deg: 3.9368 | delta_hz: 6.5309, t90_us: 0.75224, phase_deg: 4.1331 | frequency_hz: -2.984, t90_us: 0.080992, relative_phase_deg: 1.644 | True | True | True |
| D | 2 | delta_hz: 6.1888, t90_us: 0.72487, phase_deg: 3.9368 | delta_hz: 5.2225, t90_us: 0.85237, phase_deg: 3.3175 | frequency_hz: 2.8881, t90_us: -0.241, relative_phase_deg: -0.82658 | True | True | True |
| E | 2 | delta_hz: 6.1888, t90_us: 0.72487, phase_deg: 3.9368 | delta_hz: 5.2263, t90_us: 0.86302, phase_deg: 3.2359 | frequency_hz: -9.1545, t90_us: -0.12508, relative_phase_deg: -0.010366 | True | True | True |
| A | 3 | delta_hz: 6.1694, t90_us: 0.7312, phase_deg: 3.9319 | delta_hz: 6.5031, t90_us: 0.76292, phase_deg: 4.1369 | frequency_hz: -16.955, t90_us: -0.013474, relative_phase_deg: 2.0046 | True | True | True |
| B | 3 | delta_hz: 6.1694, t90_us: 0.7312, phase_deg: 3.9319 | delta_hz: 4.0264, t90_us: 0.74396, phase_deg: 2.5676 | frequency_hz: -6.2165, t90_us: -0.23952, relative_phase_deg: -0.53827 | True | True | True |
| C | 3 | delta_hz: 6.1694, t90_us: 0.7312, phase_deg: 3.9319 | delta_hz: 6.5359, t90_us: 0.74218, phase_deg: 4.1352 | frequency_hz: 6.7401, t90_us: -0.37595, relative_phase_deg: -1.1587 | True | True | True |
| D | 3 | delta_hz: 6.1694, t90_us: 0.7312, phase_deg: 3.9319 | delta_hz: 5.2109, t90_us: 0.86037, phase_deg: 3.2861 | frequency_hz: -13.428, t90_us: -0.47827, relative_phase_deg: -2.529 | True | True | True |
| E | 3 | delta_hz: 6.1694, t90_us: 0.7312, phase_deg: 3.9319 | delta_hz: 5.2065, t90_us: 0.82012, phase_deg: 3.2904 | frequency_hz: -2.7562, t90_us: -0.33016, relative_phase_deg: -1.8275 | True | True | True |

Pokrytie sa vyhodnocuje iba pri dostupnej nezávislej referencii. Detailné polia sú v `comparison.csv` a `results.json`.

## Pilot

Stav: **FROZEN_VALIDATED**; podrobnosti sú v `results.json` a `plan.json`.

## Zmrazený plán

Stav: **FROZEN**; rozpočet a tolerancie sú v `plan.json`.

## Súbory

`results.zip` obsahuje kompletný zachovaný export vrátane pôvodných FID NPZ, sanitizovaného denníka udalostí, vendor_reference, modelov, pulzov, grafov a snímky spusteného zdrojového kódu. Pri rozdelenom uploade sú diely a návod vo výsledkovej Git vetve.
