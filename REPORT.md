# Gemini Lab: Bayesovská kalibrácia viacerých parametrov

Stav: **PILOT_FAILED**. Dokončené fyzické úlohy: **3**. Porovnávacie riadky: **0**.
Exportovaný komplexný FID nie je potvrdený RAW ADC. FID vzorky nie sú nezávislé qubitové shots; serverové FFT môže ďalej bežať.
Referencie a kontrolné rotácie sú oddelené od výberu meraní. Chýbajúce referencie sa nenahrádzajú pilotným odhadom.

## Porovnanie po blokoch

| Metóda | Baseline | Blok | Akvizície | Akvizícia (s) | Fit (s) | Inferencia (s) | Návrh (s) | Koniec–koniec (s) | Δf (Hz) | Δt90 (µs) | Δφ (°) | Kontrolná chyba | Stav | Dôvod |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|

Zatiaľ nevzniklo porovnanie. Pilot alebo bezpečnostná podmienka zatiaľ nedovolili ďalšie merania.

## Neistota a pokrytie

| Metóda | Blok | Neistota referencie | Neistota odhadu | Kalibračný bias | Pokrytie f | Pokrytie t90 | Pokrytie φ |
|---|---:|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

Pokrytie sa vyhodnocuje iba pri dostupnej nezávislej referencii. Detailné polia sú v `comparison.csv` a `results.json`.

## Chyby a obmedzenia

- reason: SignalIdentificationError: Pilot FID coherence window is too short for six phase features, traceback: Traceback (most recent call last):   File "C:\Users\kuruc\spinq_lab\spinq_pokus\experiments\bayes_calibration.py", line 1598, in execute     self.run_pilot()     ~~~~~~~~~~~~~~^^   File "C:\Users\kuruc\spinq_lab\spinq_pokus\experiments\bayes_calibration.py", line 640, in run_pilot     self.pilot = identify_pilot_multiplet(repeats)                  ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^   File "C:\Users\kuruc\spinq_lab\spinq_pokus\experiments\signal_01.py", line 504, in identify_pilot_multiplet     features_windows = _feature_windows(mean_fid - fit["baseline"], covariance,                                         sample_hz, fit["frequencies_hz"], primary)   File "C:\Users\kuruc\spinq_lab\spinq_pokus\experiments\signal_01.py", line 434, in _feature_windows     raise SignalIdentificationError("Pilot FID coherence window is too short for six phase features") experiments.signal_01.SignalIdentificationError: Pilot FID coherence window is too short for six phase features 

## Súbory

`results.zip` obsahuje kompletný zachovaný export vrátane pôvodných FID NPZ, sanitizovaného denníka udalostí, vendor_reference, modelov, pulzov, grafov a snímky spusteného zdrojového kódu. Pri rozdelenom uploade sú diely a návod vo výsledkovej Git vetve.
