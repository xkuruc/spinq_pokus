# SpinQ Gemini Lab — lokálny low-level benchmark

Stav: **COMPLETED_WITH_EXPLICIT_LIMITATIONS**; skutočné hardvérové údaje: **nie**.
Všetky FFT, fity, modely a porovnania v tomto behu počíta lokálny Windows proces.
Prijímaný FID môže byť predspracovaný serverom; surový ADC nebol potvrdený.
Tri pilotné bloky nie sú silný dôkaz zlepšenia. Neznáme interné opakovania sú UNKNOWN.

## Moduly

- **A**: DEPENDENCY_FAILED — IncompleteFID: Exported axis inconsistent with uniform requested sampling
- **B**: DEPENDENCY_FAILED — Primary pilot failed; no valid frozen physical model
- **C**: DEPENDENCY_FAILED — Primary pilot failed; no valid frozen physical model
- **D**: DEPENDENCY_FAILED — Primary pilot failed; no valid frozen physical model
- **E**: DEPENDENCY_FAILED — Primary pilot failed; no valid frozen physical model
- **F**: DEPENDENCY_FAILED — Primary pilot failed; no valid frozen physical model
- **G**: DEPENDENCY_FAILED — Primary pilot failed; no valid frozen physical model
- **H**: DEPENDENCY_FAILED — Primary pilot failed; no valid frozen physical model

## Porovnanie

| Modul | Metóda | Referencia | Úloha | Blok | Akvizície | Čas (s) | Chyba | CI | Tolerancia | Stav |
|---|---|---|---|---:|---:|---:|---:|---|---:|---|

## Plán a limity

Rozpočty sú softvérová výskumná obálka odvodená z doteraz dokončených H meraní, nie certifikované hranice výrobcu. Neznáme J, časovanie P a riadenie tretieho spinu neboli dosadené nulou ani vyhlásené za overené.
Echo T2 sa odlišuje od T2* z voľného FID. Simulovaná unitárna zhoda nie je experimentálna fidelita brány. Vendor grafy sú len v vendor_reference/.

```json
{}
```

## Chyby

- A: IncompleteFID: Exported axis inconsistent with uniform requested sampling
