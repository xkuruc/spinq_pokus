# SpinQ Gemini Lab — lokálny low-level benchmark

Stav: **COMPLETED_WITH_EXPLICIT_LIMITATIONS**; skutočné hardvérové údaje: **áno**.
Plný ZIP archív: **NOT_CREATED_THIS_RUN**; merané FID ostáva lokálne v `raw/` a `data/`.
Všetky FFT, fity, modely a porovnania v tomto behu počíta lokálny Windows proces.
Prijímaný FID môže byť predspracovaný serverom; surový ADC nebol potvrdený.
Tri pilotné bloky nie sú silný dôkaz zlepšenia. Neznáme interné opakovania sú UNKNOWN.

## Moduly

- **A**: REFERENCE_INADEQUATE — Physical Rabi/phase design and five estimators ran; independent joint parameter reference and Ramsey timing remain unverified
- **B**: METHOD_FAILED — Physical FID-length arms measured; echo T2 requires independently validated echo/refocus and adequate TE range; frozen component frequency changed with FID length
- **C**: UNVERIFIED_TIMING — Multiple H segments failed the live timing contract; no GRAPE task submitted
- **D**: DEPENDENCY_FAILED — ValueError: Effective D acquisition-frequency fit hit frozen band edge
- **E**: DEPENDENCY_FAILED — RuntimeError: CPU PyTorch import failed; neural C/D/E stages unavailable
- **F**: REFERENCE_INADEQUATE — Insufficient independent reference blocks or neural model to establish F superiority
- **G**: UNVERIFIED_TIMING — Compiler/control ablation needs verified H segment order
- **H**: REFERENCE_INADEQUATE — Effective x/y map and bounded 2D command search measured; independent x/y/z Bloch reconstruction and repetition cost unavailable

## Porovnanie

| Modul | Metóda | Referencia | Úloha | Blok | Akvizície | Čas (s) | Chyba | CI | Tolerancia | Stav |
|---|---|---|---|---:|---:|---:|---:|---|---:|---|
| A | fixed_fit | fixed_fit | A_b00_fixed_fit | 0 | 1 | 15.868 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | coarse_fine_fit | fixed_fit | A_b00_coarse_fine_fit | 0 | 1 | 15.865 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | posterior_fixed | fixed_fit | A_b00_posterior_fixed | 0 | 1 | 16.258 | — | [—, —] | — | REFERENCE_INADEQUATE |
| B | diagnostic_8000 | fixed_16000 | B_b00_diagnostic_8000 | 0 | 1 | 14.854 | 138.19 | [—, —] | 3 | METHOD_FAILED |
| B | fixed_16000 | fixed_16000 | B_b00_fixed_16000 | 0 | 1 | 15.053 | 14.143 | [—, —] | 3 | METHOD_FAILED |
| H | coarse | coarse | H_b00_coarse_0 | 0 | 1 | 16.255 | 0.37608 | [—, —] | — | REFERENCE_INADEQUATE |
| H | sequential | coarse | H_b00_sequential_0 | 0 | 1 | 15.073 | 0.38771 | [—, —] | — | REFERENCE_INADEQUATE |
| A | coarse_fine_fit | fixed_fit | A_b01_coarse_fine_fit | 1 | 2 | 16.062 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | posterior_fixed | fixed_fit | A_b01_posterior_fixed | 1 | 2 | 14.68 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_variance | fixed_fit | A_b01_bayes_variance | 1 | 1 | 15.473 | — | [—, —] | — | REFERENCE_INADEQUATE |
| B | fixed_16000 | fixed_16000 | B_b01_fixed_16000 | 1 | 1 | 15.668 | 16.693 | [—, —] | 3 | METHOD_FAILED |
| B | diagnostic_8000 | fixed_16000 | B_b01_diagnostic_8000 | 1 | 1 | 14.65 | 119.14 | [—, —] | 3 | METHOD_FAILED |
| H | nelder_mead | coarse | H_b01_nelder_mead_0 | 1 | 1 | 15.109 | 0.41375 | [—, —] | — | REFERENCE_INADEQUATE |
| H | gp | coarse | H_b01_gp_0 | 1 | 1 | 15.264 | 0.45157 | [—, —] | — | REFERENCE_INADEQUATE |
| A | posterior_fixed | fixed_fit | A_b02_posterior_fixed | 2 | 3 | 15.082 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_variance | fixed_fit | A_b02_bayes_variance | 2 | 2 | 15.058 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_thresholded | fixed_fit | A_b02_bayes_thresholded | 2 | 1 | 14.851 | — | [—, —] | — | REFERENCE_INADEQUATE |
| B | diagnostic_8000 | fixed_16000 | B_b02_diagnostic_8000 | 2 | 1 | 13.849 | 154.9 | [—, —] | 3 | METHOD_FAILED |
| B | fixed_16000 | fixed_16000 | B_b02_fixed_16000 | 2 | 1 | 15.06 | 7.4656 | [—, —] | 3 | METHOD_FAILED |
| H | coarse | coarse | H_b02_coarse_1 | 2 | 2 | 15.057 | 0.033248 | [—, —] | — | REFERENCE_INADEQUATE |
| H | sequential | coarse | H_b02_sequential_1 | 2 | 2 | 14.257 | 0.17141 | [—, —] | — | REFERENCE_INADEQUATE |
| H | nelder_mead | coarse | H_b02_nelder_mead_1 | 2 | 2 | 14.261 | 0.44059 | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_variance | fixed_fit | A_b03_bayes_variance | 3 | 3 | 13.855 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_thresholded | fixed_fit | A_b03_bayes_thresholded | 3 | 2 | 15.283 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | fixed_fit | fixed_fit | A_b03_fixed_fit | 3 | 2 | 13.252 | — | [—, —] | — | REFERENCE_INADEQUATE |
| B | fixed_16000 | fixed_16000 | B_b03_fixed_16000 | 3 | 1 | 15.059 | 10.781 | [—, —] | 3 | METHOD_FAILED |
| B | diagnostic_8000 | fixed_16000 | B_b03_diagnostic_8000 | 3 | 1 | 13.253 | 58.93 | [—, —] | 3 | METHOD_FAILED |
| H | gp | coarse | H_b03_gp_1 | 3 | 2 | 14.258 | 0.063923 | [—, —] | — | REFERENCE_INADEQUATE |
| H | coarse | coarse | H_b03_coarse_2 | 3 | 3 | 13.855 | 0.082165 | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_thresholded | fixed_fit | A_b04_bayes_thresholded | 4 | 3 | 13.853 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | fixed_fit | fixed_fit | A_b04_fixed_fit | 4 | 3 | 11.849 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | coarse_fine_fit | fixed_fit | A_b04_coarse_fine_fit | 4 | 3 | 13.089 | — | [—, —] | — | REFERENCE_INADEQUATE |
| B | diagnostic_8000 | fixed_16000 | B_b04_diagnostic_8000 | 4 | 1 | 11.852 | 30.95 | [—, —] | 3 | METHOD_FAILED |
| B | fixed_16000 | fixed_16000 | B_b04_fixed_16000 | 4 | 1 | 11.451 | 37.761 | [—, —] | 3 | METHOD_FAILED |
| H | sequential | coarse | H_b04_sequential_2 | 4 | 3 | 12.474 | 0.33874 | [—, —] | — | REFERENCE_INADEQUATE |
| H | nelder_mead | coarse | H_b04_nelder_mead_2 | 4 | 3 | 12.256 | 0.035339 | [—, —] | — | REFERENCE_INADEQUATE |
| H | gp | coarse | H_b04_gp_2 | 4 | 3 | 12.059 | 0.025057 | [—, —] | — | REFERENCE_INADEQUATE |
| A | fixed_fit | fixed_fit | A_b05_fixed_fit | 5 | 4 | 12.651 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | coarse_fine_fit | fixed_fit | A_b05_coarse_fine_fit | 5 | 4 | 12.65 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | posterior_fixed | fixed_fit | A_b05_posterior_fixed | 5 | 4 | 12.656 | — | [—, —] | — | REFERENCE_INADEQUATE |
| B | fixed_16000 | fixed_16000 | B_b05_fixed_16000 | 5 | 1 | 13.071 | 32.414 | [—, —] | 3 | METHOD_FAILED |
| B | diagnostic_8000 | fixed_16000 | B_b05_diagnostic_8000 | 5 | 1 | 12.053 | 7.8444e-11 | [—, —] | 3 | METHOD_FAILED |
| H | coarse | coarse | H_b05_coarse_3 | 5 | 4 | 12.278 | 0.022329 | [—, —] | — | REFERENCE_INADEQUATE |
| H | sequential | coarse | H_b05_sequential_3 | 5 | 4 | 7.6985 | 0.031804 | [—, —] | — | REFERENCE_INADEQUATE |
| A | coarse_fine_fit | fixed_fit | A_b06_coarse_fine_fit | 6 | 5 | 8.0514 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | posterior_fixed | fixed_fit | A_b06_posterior_fixed | 6 | 5 | 12.648 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_variance | fixed_fit | A_b06_bayes_variance | 6 | 4 | 11.05 | — | [—, —] | — | REFERENCE_INADEQUATE |
| B | diagnostic_8000 | fixed_16000 | B_b06_diagnostic_8000 | 6 | 1 | 11.679 | 151.37 | [—, —] | 3 | METHOD_FAILED |
| B | fixed_16000 | fixed_16000 | B_b06_fixed_16000 | 6 | 1 | 12.26 | 19.16 | [—, —] | 3 | METHOD_FAILED |
| H | nelder_mead | coarse | H_b06_nelder_mead_3 | 6 | 4 | 12.29 | 0.022816 | [—, —] | — | REFERENCE_INADEQUATE |
| H | gp | coarse | H_b06_gp_3 | 6 | 4 | 11.47 | 0.42979 | [—, —] | — | REFERENCE_INADEQUATE |
| A | posterior_fixed | fixed_fit | A_b07_posterior_fixed | 7 | 6 | 11.482 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_variance | fixed_fit | A_b07_bayes_variance | 7 | 5 | 11.453 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_thresholded | fixed_fit | A_b07_bayes_thresholded | 7 | 4 | 11.652 | — | [—, —] | — | REFERENCE_INADEQUATE |
| B | fixed_16000 | fixed_16000 | B_b07_fixed_16000 | 7 | 1 | 11.491 | 25.642 | [—, —] | 3 | METHOD_FAILED |
| B | diagnostic_8000 | fixed_16000 | B_b07_diagnostic_8000 | 7 | 1 | 10.449 | 204.33 | [—, —] | 3 | METHOD_FAILED |
| H | coarse | coarse | H_b07_coarse_4 | 7 | 5 | 12.261 | 0.33424 | [—, —] | — | REFERENCE_INADEQUATE |
| H | sequential | coarse | H_b07_sequential_4 | 7 | 5 | 11.852 | 0.3061 | [—, —] | — | REFERENCE_INADEQUATE |
| H | nelder_mead | coarse | H_b07_nelder_mead_4 | 7 | 5 | 10.244 | 0.064934 | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_variance | fixed_fit | A_b08_bayes_variance | 8 | 6 | 12.053 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_thresholded | fixed_fit | A_b08_bayes_thresholded | 8 | 5 | 10.047 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | fixed_fit | fixed_fit | A_b08_fixed_fit | 8 | 5 | 12.255 | — | [—, —] | — | REFERENCE_INADEQUATE |
| B | diagnostic_8000 | fixed_16000 | B_b08_diagnostic_8000 | 8 | 1 | 9.8616 | 160.81 | [—, —] | 3 | METHOD_FAILED |
| B | fixed_16000 | fixed_16000 | B_b08_fixed_16000 | 8 | 1 | 11.25 | 14.687 | [—, —] | 3 | METHOD_FAILED |
| H | gp | coarse | H_b08_gp_4 | 8 | 5 | 11.048 | 0.099054 | [—, —] | — | REFERENCE_INADEQUATE |
| H | coarse | coarse | H_b08_coarse_5 | 8 | 6 | 12.053 | 0.45466 | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_thresholded | fixed_fit | A_b09_bayes_thresholded | 9 | 6 | 8.8449 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | fixed_fit | fixed_fit | A_b09_fixed_fit | 9 | 6 | 11.85 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | coarse_fine_fit | fixed_fit | A_b09_coarse_fine_fit | 9 | 6 | 9.6473 | — | [—, —] | — | REFERENCE_INADEQUATE |
| B | fixed_16000 | fixed_16000 | B_b09_fixed_16000 | 9 | 1 | 10.857 | 10.201 | [—, —] | 3 | METHOD_FAILED |
| B | diagnostic_8000 | fixed_16000 | B_b09_diagnostic_8000 | 9 | 1 | 9.6398 | 81.118 | [—, —] | 3 | METHOD_FAILED |
| H | sequential | coarse | H_b09_sequential_5 | 9 | 6 | 10.25 | 0.11082 | [—, —] | — | REFERENCE_INADEQUATE |
| H | nelder_mead | coarse | H_b09_nelder_mead_5 | 9 | 6 | 10.871 | 0.07648 | [—, —] | — | REFERENCE_INADEQUATE |
| H | gp | coarse | H_b09_gp_5 | 9 | 6 | 9.082 | 0.46897 | [—, —] | — | REFERENCE_INADEQUATE |
| F | unchanged | unchanged | denoise:F_test_120 | block-04 | 4 | — | 44.484 | [—, —] | — | SUCCESS_VALIDATED |
| F | complex_TV | unchanged | denoise:F_test_120 | block-04 | 100 | — | 43.733 | [—, —] | — | SUCCESS_VALIDATED |
| F | randomized_Hankel | unchanged | denoise:F_test_120 | block-04 | 100 | — | 33.038 | [—, —] | — | SUCCESS_VALIDATED |
| F | unchanged | unchanged | denoise:F_test_120 | block-06 | 4 | — | 30.569 | [—, —] | — | SUCCESS_VALIDATED |
| F | complex_TV | unchanged | denoise:F_test_120 | block-06 | 100 | — | 29.515 | [—, —] | — | SUCCESS_VALIDATED |
| F | randomized_Hankel | unchanged | denoise:F_test_120 | block-06 | 100 | — | 24.284 | [—, —] | — | SUCCESS_VALIDATED |
| F | unchanged | unchanged | denoise:F_test_120 | block-08 | 4 | — | 32.127 | [—, —] | — | SUCCESS_VALIDATED |
| F | complex_TV | unchanged | denoise:F_test_120 | block-08 | 100 | — | 31.146 | [—, —] | — | SUCCESS_VALIDATED |
| F | randomized_Hankel | unchanged | denoise:F_test_120 | block-08 | 100 | — | 26.065 | [—, —] | — | SUCCESS_VALIDATED |

## Plán a limity

Rozpočty sú softvérová výskumná obálka odvodená z doteraz dokončených H meraní, nie certifikované hranice výrobcu. Neznáme J, časovanie P a riadenie tretieho spinu neboli dosadené nulou ani vyhlásené za overené.
Echo T2 sa odlišuje od T2* z voľného FID. Simulovaná unitárna zhoda nie je experimentálna fidelita brány. Vendor grafy sú len v vendor_reference/.

```json
{
  "modules": "ABCDEFGH",
  "blocks": 10,
  "seed": 260926,
  "frequency_bands_hz": [
    [
      -1810.673828125,
      -1570.673828125
    ],
    [
      -1449.65087890625,
      -1209.65087890625
    ]
  ],
  "t90_us": 38.0,
  "frequency_tolerance_hz": 3.0,
  "rabi_t90_tolerance_us": 5.0,
  "acquisition_plan": {
    "status": "MODEL_MISMATCH",
    "reason": "Pilot frequency depends on FID length; verify component identity and axis",
    "between_length_spread_hz": 196.08867303307443
  },
  "sample_count_candidates": [
    4000,
    8000,
    16000
  ],
  "rf_amplitude_pct_range": [
    60.0,
    100.0
  ],
  "method_order_rule": "seeded rotation per block; shared pilot charged to all methods",
  "internal_repetitions": "UNKNOWN"
}
```

## Chyby

- Separate vendor FFT replica check unavailable: ValueError: Vendor FFT absent for pilot_40_r0
