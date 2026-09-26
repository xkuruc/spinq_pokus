# SpinQ Gemini Lab — lokálny low-level benchmark

Stav: **COMPLETED_WITH_EXPLICIT_LIMITATIONS**; skutočné hardvérové údaje: **áno**.
Plný ZIP archív: **NOT_CREATED_THIS_RUN**. Merané FID ostáva lokálne v `raw/` a `data/`.
Všetky FFT, fity, modely a porovnania v tomto behu počíta lokálny Windows proces.
Prijímaný FID môže byť predspracovaný serverom; surový ADC nebol potvrdený.
Plán obsahuje 10 meracích blokov. Neznáme interné opakovania zariadenia sú UNKNOWN.

## Stručné zistenia z meraní

Zaúčtovaných fyzických úloh: **129**; riadkov porovnania: **63**. Riadky nie sú nezávislé opakovania jedinej metódy.

- **Rabi pilot:** perióda 151 µs, odhad 90° pulzu 37.8 µs; zhoda fitu nie je vernosťou brány.
- **F:** 3 odložené meracie bloky; medián komplexnej FID RMSE: unchanged 37, complex_TV 36, randomized_Hankel 28. Randomized Hankel mal nižšiu chybu v 3/3 párov. Neurónové porovnanie a interval spoľahlivosti chýbajú; nadradenosť nie je preukázaná.
- **H:** 24 fyzických skúšobných pulzov (coarse 6, gp 6, nelder_mead 6, sequential 6). Chyba je relatívna odchýlka komplexného FID od referenčného pulzu; bez nezávislého Blochovho x/y/z merania nejde o vernosť kvantovej brány.

## Moduly

- **A**: REFERENCE_INADEQUATE — Physical Rabi/phase design and five estimators ran; independent joint parameter reference and Ramsey timing remain unverified
- **B**: REFERENCE_INADEQUATE — Pilot component identity or independent frequency reference unavailable: {}
- **C**: UNVERIFIED_TIMING — Multiple H segments failed the live timing contract; no GRAPE task submitted
- **D**: DEPENDENCY_FAILED — TypeError: float() argument must be a string or a real number, not 'NoneType'
- **E**: DEPENDENCY_FAILED — RuntimeError: CPU PyTorch import failed; neural C/D/E stages unavailable: OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. Error loading "C:\Users\kuruc\spinq_lab\spinq_pokus\.benchmark-venv\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- **F**: REFERENCE_INADEQUATE — Insufficient independent reference blocks or neural model to establish F superiority
- **G**: UNVERIFIED_TIMING — Compiler/control ablation needs verified H segment order
- **H**: REFERENCE_INADEQUATE — Effective x/y map and bounded 2D command search measured; independent x/y/z Bloch reconstruction and repetition cost unavailable

## Porovnanie

| Modul | Metóda | Referencia | Úloha | Blok | Akvizície | Čas (s) | Chyba | CI | Tolerancia | Stav |
|---|---|---|---|---:|---:|---:|---:|---|---:|---|
| A | fixed_fit | fixed_fit | A_b00_fixed_fit | 0 | 1 | 14.455 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | coarse_fine_fit | fixed_fit | A_b00_coarse_fine_fit | 0 | 1 | 16.451 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | posterior_fixed | fixed_fit | A_b00_posterior_fixed | 0 | 1 | 17.06 | — | [—, —] | — | REFERENCE_INADEQUATE |
| H | coarse | coarse | H_b00_coarse_0 | 0 | 1 | 15.656 | 0.098807 | [—, —] | — | REFERENCE_INADEQUATE |
| H | sequential | coarse | H_b00_sequential_0 | 0 | 1 | 17.055 | 0.038912 | [—, —] | — | REFERENCE_INADEQUATE |
| A | coarse_fine_fit | fixed_fit | A_b01_coarse_fine_fit | 1 | 2 | 17.257 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | posterior_fixed | fixed_fit | A_b01_posterior_fixed | 1 | 2 | 12.898 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_variance | fixed_fit | A_b01_bayes_variance | 1 | 1 | 17.456 | — | [—, —] | — | REFERENCE_INADEQUATE |
| H | nelder_mead | coarse | H_b01_nelder_mead_0 | 1 | 1 | 15.855 | 0.031021 | [—, —] | — | REFERENCE_INADEQUATE |
| H | gp | coarse | H_b01_gp_0 | 1 | 1 | 15.868 | 0.030198 | [—, —] | — | REFERENCE_INADEQUATE |
| A | posterior_fixed | fixed_fit | A_b02_posterior_fixed | 2 | 3 | 14.866 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_variance | fixed_fit | A_b02_bayes_variance | 2 | 2 | 14.251 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_thresholded | fixed_fit | A_b02_bayes_thresholded | 2 | 1 | 15.658 | — | [—, —] | — | REFERENCE_INADEQUATE |
| H | coarse | coarse | H_b02_coarse_1 | 2 | 2 | 14.057 | 0.14785 | [—, —] | — | REFERENCE_INADEQUATE |
| H | sequential | coarse | H_b02_sequential_1 | 2 | 2 | 14.854 | 0.40103 | [—, —] | — | REFERENCE_INADEQUATE |
| H | nelder_mead | coarse | H_b02_nelder_mead_1 | 2 | 2 | 15.655 | 0.018681 | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_variance | fixed_fit | A_b03_bayes_variance | 3 | 3 | 16.657 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_thresholded | fixed_fit | A_b03_bayes_thresholded | 3 | 2 | 12.458 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | fixed_fit | fixed_fit | A_b03_fixed_fit | 3 | 2 | 14.455 | — | [—, —] | — | REFERENCE_INADEQUATE |
| H | gp | coarse | H_b03_gp_1 | 3 | 2 | 13.665 | 0.52066 | [—, —] | — | REFERENCE_INADEQUATE |
| H | coarse | coarse | H_b03_coarse_2 | 3 | 3 | 16.062 | 0.13131 | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_thresholded | fixed_fit | A_b04_bayes_thresholded | 4 | 3 | 16.08 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | fixed_fit | fixed_fit | A_b04_fixed_fit | 4 | 3 | 13.871 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | coarse_fine_fit | fixed_fit | A_b04_coarse_fine_fit | 4 | 3 | 14.844 | — | [—, —] | — | REFERENCE_INADEQUATE |
| H | sequential | coarse | H_b04_sequential_2 | 4 | 3 | 14.059 | 0.03347 | [—, —] | — | REFERENCE_INADEQUATE |
| H | nelder_mead | coarse | H_b04_nelder_mead_2 | 4 | 3 | 13.251 | 0.099177 | [—, —] | — | REFERENCE_INADEQUATE |
| H | gp | coarse | H_b04_gp_2 | 4 | 3 | 13.876 | 0.19394 | [—, —] | — | REFERENCE_INADEQUATE |
| A | fixed_fit | fixed_fit | A_b05_fixed_fit | 5 | 4 | 11.85 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | coarse_fine_fit | fixed_fit | A_b05_coarse_fine_fit | 5 | 4 | 13.697 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | posterior_fixed | fixed_fit | A_b05_posterior_fixed | 5 | 4 | 12.877 | — | [—, —] | — | REFERENCE_INADEQUATE |
| H | coarse | coarse | H_b05_coarse_3 | 5 | 4 | 12.066 | 0.47718 | [—, —] | — | REFERENCE_INADEQUATE |
| H | sequential | coarse | H_b05_sequential_3 | 5 | 4 | 14.864 | 0.066847 | [—, —] | — | REFERENCE_INADEQUATE |
| A | coarse_fine_fit | fixed_fit | A_b06_coarse_fine_fit | 6 | 5 | 13.061 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | posterior_fixed | fixed_fit | A_b06_posterior_fixed | 6 | 5 | 13.111 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_variance | fixed_fit | A_b06_bayes_variance | 6 | 4 | 13.052 | — | [—, —] | — | REFERENCE_INADEQUATE |
| H | nelder_mead | coarse | H_b06_nelder_mead_3 | 6 | 4 | 12.888 | 0.11813 | [—, —] | — | REFERENCE_INADEQUATE |
| H | gp | coarse | H_b06_gp_3 | 6 | 4 | 12.454 | 0.10879 | [—, —] | — | REFERENCE_INADEQUATE |
| A | posterior_fixed | fixed_fit | A_b07_posterior_fixed | 7 | 6 | 11.268 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_variance | fixed_fit | A_b07_bayes_variance | 7 | 5 | 13.466 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_thresholded | fixed_fit | A_b07_bayes_thresholded | 7 | 4 | 11.659 | — | [—, —] | — | REFERENCE_INADEQUATE |
| H | coarse | coarse | H_b07_coarse_4 | 7 | 5 | 13.462 | 0.38287 | [—, —] | — | REFERENCE_INADEQUATE |
| H | sequential | coarse | H_b07_sequential_4 | 7 | 5 | 12.054 | 0.42817 | [—, —] | — | REFERENCE_INADEQUATE |
| H | nelder_mead | coarse | H_b07_nelder_mead_4 | 7 | 5 | 11.257 | 0.030193 | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_variance | fixed_fit | A_b08_bayes_variance | 8 | 6 | 11.259 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_thresholded | fixed_fit | A_b08_bayes_thresholded | 8 | 5 | 13.065 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | fixed_fit | fixed_fit | A_b08_fixed_fit | 8 | 5 | 13.853 | — | [—, —] | — | REFERENCE_INADEQUATE |
| H | gp | coarse | H_b08_gp_4 | 8 | 5 | 11.652 | 0.17178 | [—, —] | — | REFERENCE_INADEQUATE |
| H | coarse | coarse | H_b08_coarse_5 | 8 | 6 | 11.904 | 0.063028 | [—, —] | — | REFERENCE_INADEQUATE |
| A | bayes_thresholded | fixed_fit | A_b09_bayes_thresholded | 9 | 6 | 9.8557 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | fixed_fit | fixed_fit | A_b09_fixed_fit | 9 | 6 | 13.256 | — | [—, —] | — | REFERENCE_INADEQUATE |
| A | coarse_fine_fit | fixed_fit | A_b09_coarse_fine_fit | 9 | 6 | 11.26 | — | [—, —] | — | REFERENCE_INADEQUATE |
| H | sequential | coarse | H_b09_sequential_5 | 9 | 6 | 11.28 | 0.054124 | [—, —] | — | REFERENCE_INADEQUATE |
| H | nelder_mead | coarse | H_b09_nelder_mead_5 | 9 | 6 | 11.252 | 0.025246 | [—, —] | — | REFERENCE_INADEQUATE |
| H | gp | coarse | H_b09_gp_5 | 9 | 6 | 10.862 | 0.0048914 | [—, —] | — | REFERENCE_INADEQUATE |
| F | unchanged | unchanged | denoise:F_test_120 | block-04 | 4 | — | 37.697 | [—, —] | — | REFERENCE_INADEQUATE |
| F | complex_TV | unchanged | denoise:F_test_120 | block-04 | 80 | — | 36.811 | [—, —] | — | REFERENCE_INADEQUATE |
| F | randomized_Hankel | unchanged | denoise:F_test_120 | block-04 | 80 | — | 28.179 | [—, —] | — | REFERENCE_INADEQUATE |
| F | unchanged | unchanged | denoise:F_test_120 | block-06 | 4 | — | 36.951 | [—, —] | — | SUCCESS_VALIDATED |
| F | complex_TV | unchanged | denoise:F_test_120 | block-06 | 80 | — | 36.035 | [—, —] | — | SUCCESS_VALIDATED |
| F | randomized_Hankel | unchanged | denoise:F_test_120 | block-06 | 80 | — | 27.979 | [—, —] | — | SUCCESS_VALIDATED |
| F | unchanged | unchanged | denoise:F_test_120 | block-08 | 4 | — | 30.956 | [—, —] | — | SUCCESS_VALIDATED |
| F | complex_TV | unchanged | denoise:F_test_120 | block-08 | 80 | — | 29.938 | [—, —] | — | SUCCESS_VALIDATED |
| F | randomized_Hankel | unchanged | denoise:F_test_120 | block-08 | 80 | — | 26.517 | [—, —] | — | SUCCESS_VALIDATED |

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
      -337.28515625,
      -97.28515625
    ]
  ],
  "t90_us": 37.75,
  "frequency_tolerance_hz": 3.0,
  "rabi_t90_tolerance_us": 5.0,
  "acquisition_plan": {
    "status": "MODEL_MISMATCH",
    "reason": "Local frequency estimates disagree across FID lengths; component identity or fit must be checked before comparison",
    "between_length_spread_hz": 193.2053684340199
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
