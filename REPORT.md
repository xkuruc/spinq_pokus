# Gemini Lab — 01_bayes_online

Stav: **PILOT_INCONCLUSIVE**; fyzické akvizície: **18**; porovnávacie riadky: **0**.

**Krátky protokol:** iba H kanál, tri scenáre a priamo spárované B (klasický komplexný fit) proti D (adaptívny Bayes). A, C, väzba, PPS a Bell nie sú v tomto behu merané. Jeden 18-FID nominálny anchor je spoločný platený náklad; v každom bloku sú nové referencie, tréningy a kontrolné FID. Tri páry predstavujú predbežné porovnanie, nie univerzálnu štatistickú výhodu.

Primárny vstup je exportovaný komplexný FID. Vzorky FID nie sú nezávislé kvantové shots. Výrobné FFT a skóre nie sú vstupom učenia. Kalibračné H/P FID používajú spoločnú serverovú prípravu (`makePps=True`); samostatná vlastná PPS vetva ju vypína.

## Porovnanie

| Blok | Úloha | Kanál | Metóda | Stav | Akvizície ramena | Samostatný štart¹ | Čas ramena (s) | Samostatný štart¹ (s) | Chyba df (Hz) | Chyba RF | Chyba fázy (°) | Kontrolná chyba |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| — | — | — | — | — | — | — | — | — | — | — | — | — | — |

¹ Samostatný štart = rameno + spoločný pilot + nové referencie. Je to porovnávací náklad jedného ramena, nie skutočný súčet behu. Pilot sa fyzicky meria iba raz; skutočný počet úloh je uvedený na začiatku reportu. Probe a kontroly driftu sa uvádzajú v celkovom počte úloh.

## Párové porovnanie voči klasickému fitu B

Číselné rozdiely zahŕňajú aj neúspešné ramená a sú deskriptívne. Na tvrdenie o výhode sú spôsobilé iba páry z rovnakého bloku/kanála, v ktorých obe metódy splnili zmrazené kontroly a nebol zistený drift.

| Metóda | Zhodné bloky/kanály | Číselné páry | Platné páry | Cieľ metóda/B | Priemer metóda−B z platných párov | Deskriptívny priemer metóda−B zo všetkých číselných párov | Priemerná zmena akvizícií | Priemerná zmena času (s) | Záver |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D_adaptive_bayes | 0 | 0 | 0 | 0/0 | — | — | — | — | INSUFFICIENT_VALID_PAIRED_EVIDENCE |

### Jednotlivé predbežné páry B verzus D

Záporný rozdiel znamená nižšiu kontrolnú chybu D. Neplatný pár zostáva viditeľný, ale nesmie podporiť tvrdenie o výhode.

| Blok | B chyba | D chyba | D − B | B stav | D stav | Platný pár |
|---:|---:|---:|---:|---|---|---|

## Schopnosti a obmedzenia

- Chyba: {'block': 0, 'channel': 'H', 'stage': 'pilot', 'error': 'PILOT_INCONCLUSIVE: no connected early coherent FID', 'trace': 'Traceback (most recent call last):\n  File "C:\\Users\\kuruc\\spinq_lab\\spinq_pokus\\bayes_online_core\\runner.py", line 782, in run_calibration\n    anchors[channel] = self._pilot(block, channel)\n                       ~~~~~~~~~~~^^^^^^^^^^^^^^^^\n  File "C:\\Users\\kuruc\\spinq_lab\\spinq_pokus\\bayes_online_core\\runner.py", line 452, in _pilot\n    readout = estimate_anchor(np.asarray(fids), base, np.asarray(actual_widths), amplitude)\n  File "C:\\Users\\kuruc\\spinq_lab\\spinq_pokus\\bayes_online_core\\inference.py", line 212, in estimate_anchor\n    coherence = detect_coherent_window(data, axis)\n  File "C:\\Users\\kuruc\\spinq_lab\\spinq_pokus\\bayes_online_core\\inference.py", line 161, in detect_coherent_window\n    raise ValueError("PILOT_INCONCLUSIVE: no connected early coherent FID")\nValueError: PILOT_INCONCLUSIVE: no connected early coherent FID\n'}

Samotný úspech fitu nie je dôkaz výhody. Výhoda vyžaduje rovnakú nezávisle overenú kvalitu a nižší počet akvizícií alebo celý čas naprieč blokmi. Efektívna NMR matica nie je dôkaz prepletenia úplného tepelného súboru.
