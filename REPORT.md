# Gemini Lab: Bayesovská kalibrácia viacerých parametrov

Stav: **PILOT_FAILED**. Dokončené fyzické úlohy: **13**. Porovnávacie riadky: **0**.
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

## Pilot

Stav: **RECORDED**; podrobnosti sú v `results.json` a `plan.json`.

## Chyby a obmedzenia

- reason: ValueError: Pilot multiplet fit misses the measured coherent FID window, traceback: Traceback (most recent call last):   File "C:\Users\kuruc\spinq_lab\spinq_pokus\experiments\bayes_calibration.py", line 1612, in execute     self.run_pilot()     ~~~~~~~~~~~~~~^^   File "C:\Users\kuruc\spinq_lab\spinq_pokus\experiments\bayes_calibration.py", line 700, in run_pilot     model, validation = _pilot_response_model(self.pilot, rabi.to_dict(), train,                         ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                heldout)                                                ^^^^^^^^   File "C:\Users\kuruc\spinq_lab\spinq_pokus\experiments\bayes_calibration.py", line 422, in _pilot_response_model     raise ValueError("Pilot multiplet fit misses the measured coherent FID window") ValueError: Pilot multiplet fit misses the measured coherent FID window 

## Súbory

`results.zip` obsahuje kompletný zachovaný export vrátane pôvodných FID NPZ, sanitizovaného denníka udalostí, vendor_reference, modelov, pulzov, grafov a snímky spusteného zdrojového kódu. Pri rozdelenom uploade sú diely a návod vo výsledkovej Git vetve.
