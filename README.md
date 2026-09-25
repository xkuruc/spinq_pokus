# SpinQ Gemini Lab: bezpečná kontrola na Windows

Tento repozitár má oddelené kroky. **Začni diagnostikou USB.** Tá iba
číta informácie z Windows a neposiela zariadeniu žiadny príkaz.

## 1. Zisti, ako Windows rozpozná USB kábel

Otvor PowerShell v tomto priečinku a spusti:

```powershell
.\diagnose_windows.ps1
```

Ak na Gemini Lab nebeží experiment, môžeš porovnať stav pred a po zapojení:

```powershell
.\diagnose_windows.ps1 -Compare
```

Pošli mi tabuľku **Nové zariadenia po zapojení**, **Sériové porty** a
**USB sieťové adaptéry**. Skript nezobrazuje sériové čísla zariadení.
Ak firemná politika blokuje spúšťanie skriptov, neobchádzaj ju; v PowerShelli
môžeš použiť vstavaný príkaz `Get-PnpDevice -PresentOnly` alebo mi poslať
snímku Správcu zariadení po pripojení SpinQ.

## 2. Over priamy prístup k USB čipu FTDI

Tvoje porovnanie pri odpojení a zapojení našlo `USB Serial Converter` s
`VID 0403`, `PID 6014`. Podľa dokumentácie FTDI ide o identifikátory čipu
FT232H. Ak chceš overiť, či ho vidí aj priama knižnica FTDI, spusti:

```powershell
.\ftdi_usb_info.ps1
```

Skript využíva knižnicu `ftd2xx.dll`, **ak je už vo Windows nainštalovaná**.
Nevyžaduje Python, práva správcu ani inštaláciu ovládača. Volá len funkcie
na zistenie počtu a popisu USB zariadení. Neotvorí prístroj a neposiela dáta;
nezobrazuje sériové číslo. Pošli mi výstup, hlavne `Chip`, `Description` a
`OpenedByOtherApp`.

Tento krok ešte nie je ovládanie experimentov. FT232H je USB prevodník a
v dostupných verejných zdrojoch nie je zdokumentovaný príkazový protokol
Gemini Lab cez tento port. Neskúšaj náhodné bajty, zmenu ovládača ani
programovanie EEPROM.

## 3. Voliteľne čítaj stav cez SpinQLabLink

Oficiálny [SpinQLabLink](https://github.com/SpinQTech/spinqlablink) používa
TCP/IP. **Nie je určený na priamu komunikáciu cez zistený USB port FTDI.**
Tento krok má význam, iba ak máš adresu servera SpinQ dostupného cez sieť.

Ak máš Python a na firemnom počítači je povolené inštalovať balíky, vytvor
prostredie iba v priečinku projektu:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install spinqlablink==1.0.2
.\.venv\Scripts\python.exe .\spinqlablink_status.py --host ADRESA_ZARIADENIA --account TVOJ_UCET
```

Heslo sa zadáva až po spustení a nikam sa neukladá. Program sa len prihlási,
počká na stavové správy a odpojí sa. Nespúšťa experimenty ani nemení
parametre. Adresu `192.168.15.4` z príkladov výrobcu nepoužívaj automaticky;
dosadíš skutočnú adresu svojho zariadenia.

## 4. Ovládanie cez pôvodnú obrazovku SpinQ

Pôvodná obrazovka zostane pripojená ku Gemini Lab cez USB kábel. Windows PC
posiela experimenty do jej SpinQLabLink servera cez lokálnu sieť. Názov
zariadenia v aplikácii (napr. `Lab-00`) sa do API nezadáva; rozhodujúca je
adresa z aplikácie.

Najprv na **Windows PC** over spojenie (dosadíš svoju skutočnú IP):

```powershell
Test-NetConnection IP_Z_APLIKACIE -Port 8181
```

`TcpTestSucceeded` musí byť `True`. Na Windows sa Python môže spúšťať
príkazom `python`, aj keď spúšťač `py` chýba. Najprv zisti, čo je dostupné:

```powershell
Get-Command python,python3,py -ErrorAction SilentlyContinue | Select-Object Name,Source
python --version
```

Ak `python --version` zobrazí skutočný Python 3.10 až 3.13, vytvor
prostredie iba v priečinku projektu a nainštaluj oficiálny balík:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install spinqlablink==1.0.2
```

**Python 3.8 nepoužívaj pre SpinQLabLink 1.0.2.** Balík síce uvádza
`python_requires >= 3.8` a jeho závislosti sa nainštalujú, ale import
na Pythone 3.8 zlyhá: pribalený Protobuf kód je verzie 6.31.0, kým
správca balíkov pre Python 3.8 vybral runtime 5.29.6. Overené v
izolovanom prostredí; skript sa v ňom nespustí.

Ak je dostupný `py`, ale nie `python`, over `py -3 --version` a pri verzii
3.10 až 3.13 použi namiesto prvého príkazu `py -3 -m venv .venv`. Ak
funguje iba `python3`, over `python3 --version` a použi
`python3 -m venv .venv`. Ak nefunguje žiadny z týchto príkazov, Python na
tomto PC buď nie je nainštalovaný, alebo
nie je dostupný v PATH. Najprv over jeho inštaláciu podľa pravidiel
firemného PC; [oficiálny návod pre Windows](https://docs.python.org/3/using/windows.html)
opisuje aj inštaláciu pre jedného používateľa. Príkazy ukazujúce cestu
`WindowsApps\python.exe` môžu byť iba zástupné odkazy Windowsu; ak
`python --version` vypíše „Python sa nenašiel“, Python ešte nefunguje.

Ak firemné pravidlá povoľujú inštaláciu pre tvoj účet, stiahni z
[oficiálnej stránky Pythonu 3.13.15](https://www.python.org/downloads/release/python-31315/)
**Windows installer (64-bit)**, zvoľ inštaláciu iba pre seba a nechaj
zapnutý `pip`. Nie je potrebné meniť systémový `PATH` ani inštalovať pre
všetkých používateľov. Potom v novom PowerShelli spusti:

```powershell
$python = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
& $python --version
& $python -m venv .venv
.\.venv\Scripts\python.exe -m pip install spinqlablink==1.0.2
```

Ak sa Python nainštaluje inde, pozri jeho cestu cez
`Get-ChildItem "$env:LOCALAPPDATA\Programs\Python" -Filter python.exe -Recurse`
a nastav `$python` na nájdený súbor. Bez funkčného Pythonu nemôže
vzniknúť `.venv`, takže nasledujúce príkazy zatiaľ nespúšťaj.

Najprv načítaj stav; tento príkaz nespúšťa experiment:

```powershell
.\.venv\Scripts\python.exe .\spinq_lab_control.py status --host IP_Z_APLIKACIE
```

Potom spusti Rabiho meranie (päť šírok pulzu 40–200 µs) alebo jednu
fyzikálnu vrstvu (40 µs pulz, bez gradientu). Program pred experimentom
vyžaduje stav `connected=True` a `lock_state=True`:

```powershell
.\.venv\Scripts\python.exe .\spinq_lab_control.py rabi --host IP_Z_APLIKACIE
.\.venv\Scripts\python.exe .\spinq_lab_control.py physical --host IP_Z_APLIKACIE
```

Príkaz `all` vykoná obe merania po sebe. Predvolené prihlasovacie slová
`anyword` sú z oficiálnych príkladov. Ak tvoje laboratórium používa vlastný
účet, pridaj `--account TVOJ_UCET --ask-password`; heslo sa nikam neukladá.
Výsledky sú v `results/` vo formáte JSON, Rabi aj v CSV. Priečinok
`results/` je v `.gitignore` a neodošle sa na GitHub.

Rabiho meranie má štandardne štyri 10-sekundové prestávky medzi piatimi
pulzmi, podľa oficiálneho príkladu SpinQ. Voliteľné `--rabi-pause 5`
skráti celkový čas približne o 20 sekúnd; kratšia relaxácia však môže
skresliť namerané amplitúdy. Tento prepínač mení iba budúce merania.

Už uložené výsledky môžeš spracovať **lokálne na Windows PC** bez
nového experimentu a bez sieťového pripojenia k prístroju:

```powershell
.\.venv\Scripts\python.exe .\analyze_results.py .\results\NAZOV_RABI_SUBORU.json
```

Program uloží fit Rabiho oscilácie a PNG graf. Ak JSON obsahuje rady
`fidRe` a `fidIm`, spočíta z nich lokálnu FFT do CSV súborov. Používa
časovú os FID grafu, ktorú knižnica SpinQ označuje v milisekundách.
Je to vlastné spracovanie uložených kriviek, nie náhrada merania ani
garancia rovnakého spektra ako FFT v tablete. Prípravu vzorky, pulzy
a zber dát stále musí vykonať prístroj.

Ak chceš zistiť, či tablet posiela FID ešte pred svojím FFT grafom,
spusti **jeden skutočný experiment** fyzikálnej vrstvy s meraním časov:

```powershell
.\.venv\Scripts\python.exe .\spinq_lab_control.py physical --host IP_Z_APLIKACIE --timing
```

Výstup uvedie časy prijatia oboch častí FID (`fidRe`, `fidIm`), prvého
FFT grafu a správy o skončení experimentu. Tento prepínač iba sleduje
prichádzajúce správy; nemení pulzy ani parametre merania. Verejné
SpinQLabLink API nemá zdokumentovanú voľbu na vypnutie FFT na tablete.

Na jednorazové zverejnenie kódu **aj nameraných dát** z `results/` použi
`push_all.ps1`. Skript pridá súbory, vytvorí commit a odošle ho na
`xkuruc/spinq_pokus`. Token je iba argument pri spustení; skript ho
neukladá do repozitára. Namerané dáta budú po odoslaní verejné.

Ak meranie prekročí časový limit, môže ešte bežať na prístroji. Pred
opätovným spustením skontroluj front experimentov v aplikácii SpinQ.
Ak staršia verzia skriptu zlyhala na `get_experiment_status`, najprv
skontroluj na tablete, že odoslaný experiment už skončil, a potom
spusti `git pull`. Opravený skript číta stav priamo z experimentu.

## Zdroj API

- [Oficiálny SpinQLabLink](https://github.com/SpinQTech/spinqlablink)
- [Oficiálny príklad čítania stavu zariadenia](https://github.com/SpinQTech/spinqlablink/blob/main/examples/device_data_example.py)
- [FTDI: predvolené VID/PID čipu FT232H](https://ftdichip.com/wp-content/uploads/2024/09/DS_FT232H.pdf)
- [FTDI: D2XX Programmer's Guide](https://ftdichip.com/wp-content/uploads/2023/09/D2XX_Programmers_Guide.pdf)
- [SpinQLabLink Quick Start](https://doc.spinq.cn/doc/SpinQLAB_Link/en/quickstart.html)
- [SpinQLabLink: fyzikálna vrstva](https://doc.spinq.cn/doc/SpinQLAB_Link/en/experiment/Research_Experiment.html)
- [Oficiálny príklad Rabiho oscilácie](https://github.com/SpinQTech/spinqlablink/blob/main/examples/part1/exp_rabi_example.py)

Žiadne heslá, tokeny ani výstupy diagnostiky nepatria do verejného repozitára.

## Reálne automatické merania na Windows

Nový [stručný návod](LIVE_WINDOWS.md) a `spinq_live_suite.py` spúšťajú
na pripojenom Windows počítači reálnu meraciu sériu. Potrebujú lokálny
`live_suite.toml` s potvrdeným pracovným bodom a prevádzkovými limitmi.
Po jeho vyplnení stačí jeden príkaz; výsledkom je `results.zip`.

## Audit SDK, NMR dát a budúcej kalibrácie

Nový balík `spinq_audit/` je oddelený od funkčného
`spinq_lab_control.py`. Predvolený režim nič nepripája ani neposiela
prístroju. Potrebuje Python 3.11+ a pre offline/replay iba štandardnú
knižnicu. Pri spustení na Windows použi **ten istý** interpreter z `.venv`,
v ktorom funguje SpinQLabLink 1.0.2.

```powershell
.\.venv\Scripts\python.exe -m spinq_audit offline --out audit_offline
```

Otvor `audit_offline\REPORT.html`. Podklady sú v tom istom adresári a v
`audit_bundle.zip`. Audit používa statický inventár nainštalovaného SDK;
ak v danom interpreteri SDK chýba, použije označený referenčný snapshot
oficiálneho balíka 1.0.2. Žiadny SDK balík neinštaluje, nemení ani
neimportuje v offline/replay režime. Porovnanie s GitHub `main` je viazané
na commit `8fe50f65bf87b97bf39dc4e1f8db9363801fd169`; obsah Python
súborov v PyPI wheel 1.0.2 sa pri tomto audite zhodoval po normalizácii
riadkov CRLF/LF. Audit uvádza verziu, pôvod a dostupné súbory SDK;
úplná nemennosť lokálnych súborov tým nie je potvrdená.

Pre pasívny zber vytvor lokálnu konfiguráciu bez hesla:

```powershell
Copy-Item .\audit_config.example.toml .\audit_config.toml
.\.venv\Scripts\python.exe -m spinq_audit passive --config audit_config.toml --duration 60 --out audit_passive
```

Pred druhým príkazom v lokálnom `audit_config.toml` nahraď
`IP_Z_APLIKACIE` adresou z aplikácie SpinQ.

Heslo zadáš do skrytej výzvy alebo cez premennú `SPINQ_AUDIT_PASSWORD`;
nikdy cez argument CLI. Pasívny režim vytvorí vlastné pripojenie iba po
výslovnom spustení, dovolí login a heartbeat, číta push telemetriu a
nespustí experiment. Nezachytáva dáta cudzieho experimentu. Existujúceho
klienta možno pozorovať funkciou `audit_existing_client(client, out=...,
owns_connection=False)`; tá ho neodpojí.

Konkrétny návrh malých meraní bez odoslania vytvoríš takto:

```powershell
Copy-Item .\approved_baseline.example.json .\approved_baseline.json
.\.venv\Scripts\python.exe -m spinq_audit plan --config audit_config.toml --baseline approved_baseline.json --out audit_plan
```

Príklad baseline obsahuje presné parametre predchádzajúceho 40 µs pokusu,
ale je **neschválený**. Plán preto vypíše blokátory: chýbajú potvrdené
jednotky, RF záťaž automatickej prípravy, interný počet akvizícií a
prevádzkové limity. Tieto položky treba doložiť na tomto pracovisku;
hodnoty zo schémy SDK nie sú bezpečnostnými limitmi. `plan.json` obsahuje
presné navrhované payloady a `approval_template.json` úplnú kópiu plánu.

Aktívny režim vyžaduje splnený bezpečnostný plán, `active_enabled=true`
v konfigurácii, schválený baseline, limitný rozpočet a schválenie
**presnej kópie** plánu v samostatnom súbore. Po kontrole plánu skopíruj
`audit_plan\approval_template.json` do lokálneho `approved_plan.json`,
nastav `approved=true`, svoje meno a UTC čas. Zmena plánu alebo
konfigurácie po schválení sa zablokuje. Až potom, pri explicitnom
povolení hardvéru, je dostupný príkaz:

```powershell
.\.venv\Scripts\python.exe -m spinq_audit active --config audit_config.toml --baseline approved_baseline.json --approved-plan approved_plan.json --allow-hardware --max-experiments 3 --out audit_active
```

Číslo `3` je iba príklad rozpočtu pre jeden baseline a dve opakovania;
nie je to bezpečný fyzikálny limit. Aktívny executor navyše vyžaduje
čerstvý stav a prázdnu frontu; ak ich server neposiela, zastaví sa bez
merania. Žiadna chyba nespúšťa slepý retry. Odpojenie klienta nezastavuje
prípadnú už odoslanú hardvérovú úlohu.

Starý uložený JSON alebo nový audit možno vyhodnotiť bez SDK a bez siete:

```powershell
.\.venv\Scripts\python.exe -m spinq_audit replay --input .\results\NAZOV_RABI_SUBORU.json --out audit_replay
.\.venv\Scripts\python.exe -m spinq_audit replay --input audit_passive --out audit_replay_passive
```

Starý `get_result()` už môže mať prepísané priebežné grafy a neobsahuje
všetky hranice prenosu. Replay ho preto výslovne označí ako agregovaný
výsledok, nie originálny transport ani RAW ADC. `audit_*/` je ignorovaný
Gitom: report a namerané dáta sa neposielajú na GitHub automaticky.
