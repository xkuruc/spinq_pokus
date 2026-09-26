# 01 — Bayesovská kalibrácia Gemini Lab

Na Windows počítači s existujúcou funkčnou `.venv` a SpinQLabLink 1.0.2:

```powershell
git pull
.\run_01_bayes_kalibracia.cmd
```

Spúšťač vytvorí samostatné `.bayes-venv`, skontroluje nainštalované SDK a numerické knižnice, urobí spoločný pilot a potom tri randomizované bloky A–E. Používa len pripojenie z [config-01-bayes.json](config-01-bayes.json), založené na predchádzajúcom úspešnom spojení s tabletom. Nezapisuje trvalú kalibráciu ani nemení pôvodnú `.venv`.

Ak server po prihlásení nepošle stav frontu, meranie sa zastaví ešte pred prvým pulzom. SpinQLabLink 1.0.2 nemá zdokumentovaný príkaz na jeho vyžiadanie. Keď máš zariadenie výhradne pre seba a na tablete ani z iného klienta nebeží experiment, použi pre **nový** beh:

```powershell
.\run_01_bayes_kalibracia.cmd --exclusive-use-confirmed
```

Prepínač je výslovné potvrdenie pre tento jeden beh; neukladá sa do konfigurácie. Čerstvá správa o obsadenej fronte, strata spojenia/locku alebo nejasný stav úlohy stále zastaví nové merania. Ak prebehla len kontrola frontu a žiadna úloha, výsledok má stav `PAUSED_QUEUE`, zostane lokálne v ZIPe a neposiela sa ako prázdny výsledok na GitHub.

Výsledky priebežne zapisuje do `results/01_bayes_kalibracia/<run_id>/`: `REPORT.md`, `comparison.csv`, `results.json`, `plan.json`, `calibration.json`, kompletný exportovaný komplexný FID v `raw/`, pôvodné sanitizované udalosti v `data/`, oddelené grafy servera vo `vendor_reference/` a celý `results.zip`. Po skončení sa pokúsi výsledky poslať do samostatnej vetvy `benchmark/01_bayes_kalibracia/<run_id>` cez už fungujúce prihlasovanie Gitu. Ak odoslanie zlyhá, lokálny ZIP zostane zachovaný a dôvod bude vo výstupe.

Pri prerušení sa dá pokračovať bez opakovania už dokončených úloh:

```powershell
.\run_01_bayes_kalibracia.cmd --resume results\01_bayes_kalibracia\<run_id>
```

Opätovná analýza uloženého behu neodosiela príkazy prístroju:

```powershell
.\run_01_bayes_kalibracia.cmd --reanalyze results\01_bayes_kalibracia\<run_id>
```

Počas jedného behu musí mať spojenie a prístroj jediného vlastníka. Nejasný stav úlohy zastaví ďalšie odosielanie. Výsledky sú z exportovaného komplexného FID; surové ADC ani vypnutie výpočtov tabletu nie sú doložené. Softvérové rozpočty v konfigurácii nie sú bezpečnostným hodnotením výrobcu. Podrobnosti o metodike a jej hraniciach sú v [01_reproduction_scope.md](experiments/01_reproduction_scope.md).
