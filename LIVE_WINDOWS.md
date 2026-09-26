# Pôvodná prieskumná séria Gemini Lab na Windows

Nový benchmark sa spúšťa cez `run_windows.cmd`; pozri [BENCHMARK_README.md](BENCHMARK_README.md).

`spinq_live_suite.py` používa nainštalovaný SpinQLabLink na tvojom Windows
počítači. Tablet zostáva pripojený k prístroju cez USB. Na vývojovom Macu sa
žiadne meranie nespustilo. Skript potrebuje Python 3.11+ a SpinQLabLink 1.0.2
v existujúcom `.venv`; neinštaluje ďalšie balíky.

## Čo znamená pracovný bod a limit

**Pracovný bod** je nastavenie, pri ktorom už zariadenie vrátilo signál.
Tvoj fyzikálny experiment úspešne použil H kanál, 40 µs pulz, 100 % amplitúdu,
90° fázu, nulový detuning a gradient, prípravu zapnutú, 10 000 vzoriek/s,
16 000 bodov a nulové oneskorenie vzorkovania. Číselná hodnota relaxácie
zostáva 15. [Príklad SpinQ](https://doc.spinq.cn/doc/SpinQLAB_Link/en/experiment/Research_Experiment.html)
hovorí o sekundách, [zdroj SDK](https://raw.githubusercontent.com/SpinQTech/spinqlablink/main/spinqlablink/experiment/exp_layer_physical.py)
o mikrosekundách; jednotka zostáva NEOVERENÁ a skript toto číslo nemení.

**Prevádzkový limit** je výrobcom alebo pracoviskom určená hranica pre
konkrétny prístroj, napríklad prípustná RF záťaž alebo teplota. Tieto údaje
nemáme. Rozsahy, ktoré prijíma Python API, nie sú dôkazom bezpečného
prevádzkového rozsahu. Skript ich preto nenazýva prevádzkovými limitmi.

Predvolená konfigurácia používa namiesto toho **obmedzený plán jednej
výskumnej série**: malé zmeny okolo predošlého 40 µs merania, najviac 120
požiadaviek a najviac 6000 µs súčtu *požadovaných* pulzov v tomto spustení.
Sú to softvérové brzdy experimentu, nie záruka bezpečnosti hardvéru; vnútorné
prípravné pulzy a interné opakovania do súčtu nevidíme. Séria kontroluje
pripojenie, lock, čerstvú konečnú teplotu a frontu, ale bez výrobného rozsahu
neoznačuje teplotu za schválenú.

## Ako spustiť ďalšie experimenty

V PowerShelli v priečinku repozitára:

```powershell
git pull
.\.venv\Scripts\python.exe .\spinq_live_suite.py --config .\live_suite.example.toml
```

Predvolený plán postupne skúsi základný NMR signál, fyzikálny FID, 20
samostatných opakovaní, malé zmeny H pulzu s návratom na základný bod,
rozdelenie H pulzu na dva segmenty, frekvenčný a demodulačný posun, zmeny
vzorkovania, fázy 0/90/180/270°, tvarovaný pulz pri potvrdenom type vzorky a päťbodový Rabi sken
40/80/120/160/200 µs, ktorý ti už raz prešiel, s kontrolným meraním vybraného bodu. P kanál sa skúsi
iba pri čerstvých úplných kalibračných hodnotách. Každé meranie má vlastné
dáta a výsledok; serverom vrátený FID sa
spracuje aj lokálne. Ak prehliadka stavu, lock, fronta, SDK alebo experiment
zlyhá, ďalšie požiadavky sa neposielajú naslepo.

Vypnutie prípravy, meranie bez RF, zmenu hodnoty relaxácie,
gradienty a trvalé nastavenia skript preskočí s dôvodom: ich význam alebo
pracovný bod zatiaľ nie je overený. Keďže si už spustil Rabi so šírkami
40–200 µs, tento plán skúša iba malé zmeny okolo 40 µs. Študijné medze sú
uvedené priamo v `spinq_live_suite.py`; zmena konfigurácie mimo nich sa
neodošle.

Heslo nedávaj do súboru. Skript bez otázok použije premennú
`SPINQ_AUDIT_PASSWORD`, alebo demo hodnotu `anyword` z už fungujúceho
`spinq_lab_control.py`. Predvolený
príklad predpokladá, že počas merania prístroj používaš iba ty. Ak ho môže
používať aj niekto iný, nastav v lokálnej kópii konfigurácie
`exclusive_use_confirmed = false`; bez čerstvej prázdnej fronty sa potom
experiment neodošle.
Pred opätovným spustením po prerušení skontroluj na tablete, že stará úloha
skončila; softvérový rozpočet sa medzi spusteniami nesčítava.

Každé spustenie vytvorí nový `results\DATUM_live_suite\results.zip` s
`REPORT.md`, `results.json`, prijatými dekódovanými číselnými dátami,
udalosťami, lokálnou analýzou, náhľadmi a chybami. Výsledky sa ukladajú
priebežne a Git ich ignoruje. Ak zostane lokálny súbor
`.spinq_live_gemini.lock` po násilnom ukončení, odstráň ho až po kontrole
fronty a stavu úlohy na tablete.

Po meraní sa program neinteraktívne pokúsi pushnúť iba výsledky do novej
vetvy `results/DATUM_UTC`. Ak Git autentifikácia na Windowse chýba,
lokálny ZIP zostane a v reporte bude `UPLOAD_FAILED`. Veľký ZIP sa pre Git
rozdelí na číslované časti bez straty lokálneho celku.
