# Gemini Lab — 01_bayes_online

Stav: **FAILED**; fyzické akvizície: **0**; porovnávacie riadky: **0**.

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
| — | 0 | 0 | 0 | 0/0 | — | — | — | — | Zatiaľ bez párov |

### Jednotlivé predbežné páry B verzus D

Záporný rozdiel znamená nižšiu kontrolnú chybu D. Neplatný pár zostáva viditeľný, ale nesmie podporiť tvrdenie o výhode.

| Blok | B chyba | D chyba | D − B | B stav | D stav | Platný pár |
|---:|---:|---:|---:|---|---|---|

## Schopnosti a obmedzenia

- Chyba: {'stage': 'run', 'error': 'TransportError: SpinQLabLink login did not complete', 'trace': 'Traceback (most recent call last):\n  File "C:\\Users\\kuruc\\spinq_lab\\spinq_pokus\\bayes_online_core\\runner.py", line 1153, in execute\n    with PhysicalTransport(host=self.config["host"], port=self.config["port"],\n         ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n            exclusive_use_confirmed=self.exclusive_use_confirmed,\n            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n            timeout_s=self.config["timeout_seconds"]) as transport:\n            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "C:\\Users\\kuruc\\spinq_lab\\spinq_pokus\\bayes_online_core\\transport.py", line 304, in __enter__\n    self.connect()\n    ~~~~~~~~~~~~^^\n  File "C:\\Users\\kuruc\\spinq_lab\\spinq_pokus\\bayes_online_core\\transport.py", line 333, in connect\n    raise TransportError("SpinQLabLink login did not complete")\nbayes_online_core.transport.TransportError: SpinQLabLink login did not complete\n'}

Samotný úspech fitu nie je dôkaz výhody. Výhoda vyžaduje rovnakú nezávisle overenú kvalitu a nižší počet akvizícií alebo celý čas naprieč blokmi. Efektívna NMR matica nie je dôkaz prepletenia úplného tepelného súboru.
