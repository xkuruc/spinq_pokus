# SpinQ Gemini Lab — experimentálny benchmark

Začiatok: 2026-09-26T10:41:09.511960+00:00; stav: **PILOT_COMPLETED_WITH_LIMITATIONS**; režim: **pilot**.
Skutočné hardvérové údaje: áno.

Výstupy `data/` sú dekódované grafy SpinQLabLink, nie potvrdený RAW ADC. Interný počet opakovaní a skryté prípravné RF pulzy SDK neoznamuje.
Kladné zlepšenie znamená nižšiu chybu pri rovnakom definovanom rozpočte. Tri bloky sú pilot; intervaly medzi blokmi ostávajú široké.

| Téma | Metóda | Blok | Merania | Čas (s) | Chyba | Neistota | Zlepšenie | Stav |
|---|---|---:|---:|---:|---:|---:|---:|---|
| calibration | coarse_fine_fit | 0 | 13 | 195.8 | — | 326.9 | — | NEPRESVEDČIVÉ |
| calibration | bayesian_complex | 0 | 13 | 196.1 | — | 442 | — | NEPRESVEDČIVÉ |
| calibration | fixed_fit | 0 | 13 | 196 | — | 328.6 | — | NEPRESVEDČIVÉ |
| acquisition | fixed_16k | 0 | 2 | 32.94 | 0.4415 | 3.045 | 0 | PILOT |
| acquisition | adaptive_equal_sample_budget | 0 | 8 | 111.8 | 960.3 | 3.045 | -2174 | NEPRESVEDČIVÉ |
| acquisition | adaptive_early_stop | 0 | 1 | 14.86 | 1292 | 3.045 | -2924 | NEPRESVEDČIVÉ |
| echo_t2 | fixed_vs_adaptive_echo | 0 | — | — | — | — | — | SKIPPED_UNSUPPORTED |
| pulse_tuning | module | 0 | — | — | — | — | — | NEPRESVEDČIVÉ |
| robust_pulse | module | 0 | — | — | — | — | — | NEPRESVEDČIVÉ |
| calibration | module | 1 | — | — | — | — | — | NEPRESVEDČIVÉ |
| acquisition | adaptive_equal_sample_budget | 1 | 8 | 113.8 | 1285 | 1.495 | -1054 | NEPRESVEDČIVÉ |
| acquisition | adaptive_early_stop | 1 | 1 | 13.06 | 1290 | 1.495 | -1057 | NEPRESVEDČIVÉ |
| acquisition | fixed_16k | 1 | 2 | 31.34 | 1.219 | 1.495 | 0 | PILOT |
| echo_t2 | fixed_vs_adaptive_echo | 1 | — | — | — | — | — | SKIPPED_UNSUPPORTED |
| pulse_tuning | module | 1 | — | — | — | — | — | NEPRESVEDČIVÉ |
| robust_pulse | module | 1 | — | — | — | — | — | NEPRESVEDČIVÉ |
| calibration | module | 2 | — | — | — | — | — | NEPRESVEDČIVÉ |
| acquisition | fixed_16k | 2 | 2 | 31.97 | 2.699 | 2.473 | 0 | PILOT |
| acquisition | adaptive_equal_sample_budget | 2 | 8 | 112.8 | 1121 | 2.473 | -414.4 | NEPRESVEDČIVÉ |
| acquisition | adaptive_early_stop | 2 | 1 | 13.97 | 1288 | 2.473 | -476.3 | NEPRESVEDČIVÉ |
| echo_t2 | fixed_vs_adaptive_echo | 2 | — | — | — | — | — | SKIPPED_UNSUPPORTED |
| pulse_tuning | module | 2 | — | — | — | — | — | NEPRESVEDČIVÉ |
| robust_pulse | module | 2 | — | — | — | — | — | NEPRESVEDČIVÉ |

## Súhrn po nezávislých blokoch

- acquisition / adaptive_early_stop: 3 bloky; priemerná chyba 1290; 95 % interval [1285.4267049567359, 1294.1581225281363]; úspechy 0/3 pri vopred stanovenej hranici 1.0918816802286313; PILOT_NEEDS_REPLICATION.
- acquisition / adaptive_equal_sample_budget: 3 bloky; priemerná chyba 1122; 95 % interval [718.5216970965414, 1525.936145179883]; úspechy 0/3 pri vopred stanovenej hranici 1.0918816802286313; PILOT_NEEDS_REPLICATION.
- acquisition / fixed_16k: 3 bloky; priemerná chyba 1.453; 95 % interval [-1.3954044296008967, 4.301409289271598]; úspechy 1/3 pri vopred stanovenej hranici 1.0918816802286313; PILOT_NEEDS_REPLICATION.

## Metodika a obmedzenia

Pri porovnaní kalibrácie sa spoločné pilotné merania započítajú každej metóde ako cena prvého použitia. Cieľové referencie pochádzajú z oddelených meraní a nesmú byť vstupom hodnotenej metódy.
Zložky FID sa sledujú komplexným fitom a vyhodnocujú po blokoch. FID vzorky sa nerátajú ako nezávislé pokusy.
Rotačná chyba z FID je iba experimentálny proxy merateľnej odozvy, nie procesná fidelita.
Optimalizovaný GRAPE pulz používa lokálny jednojadrový model H; simulovaná fidelita je oddelená od merania. Nezmeraná väzba H–P je významné obmedzenie modelu.
Voľný FID poskytuje T2*, nie echo T2. SDK 1.0.2 neodhaľuje jednotlivo nastaviteľné echo časy; adaptívne echo sa preto bez overenej podpory nevykoná.
Hankel je pevná numerická metóda. Noise2Noise sa učí z nezávislých opakovaní; referenčný priemer nie je bezšumová pravda. Ak PyTorch nefunguje, použije sa explicitne označený komplexný lineárny FIR filter naučený bez PyTorch.
Nie je preukázané, že serverová FFT sa dá vypnúť, takže jej prenosový čas môže zostať.

## Témy

### calibration

- Blok 0: NEPRESVEDČIVÉ; calibration not identifiable
- Blok 1: NEPRESVEDČIVÉ; No paired complex FID for cal_b01_shared_4
- Blok 2: NEPRESVEDČIVÉ; No paired complex FID for cal_b02_segment_probe

### acquisition

- Blok 0: PILOT; real short FID acquisitions compared; T2 echo unavailable
- Blok 1: PILOT; real short FID acquisitions compared; T2 echo unavailable
- Blok 2: PILOT; real short FID acquisitions compared; T2 echo unavailable

### pulse_tuning

- Blok 0: NEPRESVEDČIVÉ; Independent H 90-degree working point unidentifiable; pulse tuning skipped
- Blok 1: NEPRESVEDČIVÉ; Independent H 90-degree working point unidentifiable; pulse tuning skipped
- Blok 2: NEPRESVEDČIVÉ; Independent H 90-degree working point unidentifiable; pulse tuning skipped

### robust_pulse

- Blok 0: NEPRESVEDČIVÉ; Calibrated H 90-degree width unavailable
- Blok 1: NEPRESVEDČIVÉ; Calibrated H 90-degree width unavailable
- Blok 2: NEPRESVEDČIVÉ; Calibrated H 90-degree width unavailable

### denoising

- Blok 2: NEPRESVEDČIVÉ; [WinError 1114] A dynamic link library (DLL) initialization routine failed. Error loading "C:\Users\kuruc\spinq_lab\spinq_pokus\.benchmark-venv\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.

## Grafy

- pulse_convergence: SKIPPED_NO_EVALUATIONS
- robustness: SKIPPED_NO_READINGS

## Chyby

- pulse_tuning block 0: ValueError: Independent H 90-degree working point unidentifiable; pulse tuning skipped
- robust_pulse block 0: ValueError: Calibrated H 90-degree width unavailable
- calibration block 1: ValueError: No paired complex FID for cal_b01_shared_4
- pulse_tuning block 1: ValueError: Independent H 90-degree working point unidentifiable; pulse tuning skipped
- robust_pulse block 1: ValueError: Calibrated H 90-degree width unavailable
- calibration block 2: ValueError: No paired complex FID for cal_b02_segment_probe
- pulse_tuning block 2: ValueError: Independent H 90-degree working point unidentifiable; pulse tuning skipped
- robust_pulse block 2: ValueError: Calibrated H 90-degree width unavailable
- denoising: OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed. Error loading "C:\Users\kuruc\spinq_lab\spinq_pokus\.benchmark-venv\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.

