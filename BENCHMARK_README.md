# Gemini Lab experimentálny benchmark

Tento program spúšťa skutočné merania iba na Windows počítači s prístupom k tabletu `172.19.20.100:8181`. Na Macu sa žiadny experiment neodoslal. Predvolený beh je **pilot**, nie potvrdenie zlepšenia.

## Spustenie na Windowse

V PowerShelli v priečinku repozitára:

```powershell
git pull
.\run_windows.cmd
```

Skript vytvorí samostatné `.benchmark-venv`, nainštaluje oficiálny `spinqlablink==1.0.2`, NumPy, SciPy, scikit-learn, matplotlib a PyTorch a **neupraví existujúcu `.venv`**. Ak firemný počítač zablokuje inštaláciu, zastaví sa pred meraním. Inštalácia Torch môže byť veľká. GPU netreba.

Výsledky sú v `results/<UTC_run_id>/results.zip`, `REPORT.md`, `comparison.csv`, `results.json`, `data/`, `models/` a `plots/`. Program po skončení skúsi neinteraktívne odoslanie ZIP do vetvy `benchmark/<UTC_run_id>`. Použije existujúce Git prihlasovanie alebo `GH_TOKEN` / `GITHUB_TOKEN` v prostredí. Token sa nezapisuje do súborov. Ak push zlyhá, ZIP ostáva lokálne a v správe bude `UPLOAD_FAILED`.

Pokračovanie po prerušení:

```powershell
.\run_windows.cmd --resume .\results\<UTC_run_id>
```

Po dokončení prvých troch blokov možno pridať ďalšie meracie bloky:

```powershell
.\run_windows.cmd --resume .\results\<UTC_run_id> --blocks 6
```

Predvolený plán má tri bloky, najviac 140 požiadaviek a 10 000 µs požadovaného RF na blok, najviac 200 µs v jednej úlohe. Predvolený súčet je 420 požiadaviek a 30 000 µs. Sú to **softvérové rozpočty štúdie**, nie potvrdené bezpečnostné limity výrobcu. Skrytú prípravu stavu a interné opakovania SDK neoznamuje. Každá požiadavka kontroluje čerstvý lock, stav, telemetriu a frontu; pri nejasnom stave sa ďalšia neposiela. Očakávané trvanie môže byť niekoľko desiatok minút.

Používajú sa len H pulzy fyzikálnej vrstvy s pôvodným `makePps=true`, pôvodným numerickým relaxation parametrom 15 a bez gradientov či zápisu kalibrácie. Frekvenčný sken mení `Pulse.detuning` vysielača; `h_freDemo` prijímača zostáva 0. P kanál sa nepoužíva.

## Čo je možné a čo sa musí označiť ako nevykonané

- Kalibrácia: pevný sken, hrubý/jemný sken a priebežný Bayesovský výber ďalšieho bodu. Všetky používajú spoločné pilotné merania a oddelenú referenciu. Komplexný FID sa fituje po meraniach; časové body sa nepovažujú za nezávislé pokusy.
- Rozpočet akvizície: reálne kratšie `sampleCount` oproti pevnému 16k, čas a presnosť proti samostatným 16k referenciám. Program tiež obsahuje D-optimal výber echo času, no nainštalované SDK 1.0.2 nevystavuje jednotlivo nastaviteľné echo časy. T2 echo preto zostane `SKIPPED_UNSUPPORTED`; voľný FID je T2*.
- RF doladenie: tradičný pracovný bod, postupné skenovanie, Nelder–Mead a GP s očakávaným zlepšením. Cieľ a kontroly používajú viac fázových čítaní. Skóre je proxy správnosti odozvy a nesmie sa nazvať procesnou fidelitou.
- Robustný pulz: lokálna optimalizácia segmentov s modelom jednej H zložky, porovnanie na testovacích rozladeniach a prípravných sekvenciách. Nezmeraná H–P väzba obmedzuje platnosť modelu. BB1 sa vypočíta, ale pri typickom 40 µs pracovnom bode by presiahol pozorovanú hranicu 200 µs na úlohu, a preto sa fyzický BB1 test preskočí s dôvodom.
- Denoising: nezávislé opakovania, delenie podľa blokov a fázových rodín, Hann inicializácia + fyzikálny fit, Hankel rank 2, reálne trénovaná PyTorch sieť. Samostatný priemer troch meraní je cena 3 akvizícií, nie bezšumová pravda. Prvý pilot má len jednu úplne odloženú testovaciu rodinu, takže zovšeobecnenie ostane `NEPRESVEDČIVÉ`.

Zdrojové API: [SpinQLabLink physical-layer dokumentácia](https://doc.spinq.cn/doc/SpinQLAB_Link/en/experiment/Research_Experiment.html), [oficiálny repozitár](https://github.com/SpinQTech/spinqlablink). Metódy: [qutip-qtrl control guide](https://qutip.readthedocs.io/projects/qutip-qtrl/en/latest/guide/guide-control.html), [Wimperis BB1](https://eprints.gla.ac.uk/52429/), [Noise2Noise](https://proceedings.mlr.press/v80/lehtinen18a.html).
