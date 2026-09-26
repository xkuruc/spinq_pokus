# Gemini Lab: Bayesovská kalibrácia viacerých parametrov

Stav: **STOPPED_UNCERTAIN**. Dokončené fyzické úlohy: **0**. Porovnávacie riadky: **0**.
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

- HardwareUncertain: Queue unavailable and exclusive use not confirmed

## Súbory

`results.zip` obsahuje kompletný zachovaný export vrátane pôvodných FID NPZ, sanitizovaného denníka udalostí, vendor_reference, modelov, pulzov, grafov a snímky spusteného zdrojového kódu. Pri rozdelenom uploade sú diely a návod vo výsledkovej Git vetve.
