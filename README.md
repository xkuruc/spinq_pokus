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

Ak `python --version` zobrazí skutočný Python 3.10 až 3.12, vytvor
prostredie iba v priečinku projektu a nainštaluj oficiálny balík:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install spinqlablink==1.0.2
```

Ak je dostupný `py`, ale nie `python`, over `py -3 --version` a pri verzii
3.10 až 3.12 použi namiesto prvého príkazu `py -3 -m venv .venv`. Ak
funguje iba `python3`, over `python3 --version` a použi
`python3 -m venv .venv`. Ak nefunguje žiadny z týchto príkazov, Python na
tomto PC buď nie je nainštalovaný, alebo
nie je dostupný v PATH. Najprv over jeho inštaláciu podľa pravidiel
firemného PC; [oficiálny návod pre Windows](https://docs.python.org/3/using/windows.html)
opisuje aj inštaláciu pre jedného používateľa. Bez funkčného Pythonu
nemôže vzniknúť `.venv`, takže nasledujúce príkazy zatiaľ nespúšťaj.

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

Ak meranie prekročí časový limit, môže ešte bežať na prístroji. Pred
opätovným spustením skontroluj front experimentov v aplikácii SpinQ.

## Zdroj API

- [Oficiálny SpinQLabLink](https://github.com/SpinQTech/spinqlablink)
- [Oficiálny príklad čítania stavu zariadenia](https://github.com/SpinQTech/spinqlablink/blob/main/examples/device_data_example.py)
- [FTDI: predvolené VID/PID čipu FT232H](https://ftdichip.com/wp-content/uploads/2024/09/DS_FT232H.pdf)
- [FTDI: D2XX Programmer's Guide](https://ftdichip.com/wp-content/uploads/2023/09/D2XX_Programmers_Guide.pdf)
- [SpinQLabLink Quick Start](https://doc.spinq.cn/doc/SpinQLAB_Link/en/quickstart.html)
- [SpinQLabLink: fyzikálna vrstva](https://doc.spinq.cn/doc/SpinQLAB_Link/en/experiment/Research_Experiment.html)
- [Oficiálny príklad Rabiho oscilácie](https://github.com/SpinQTech/spinqlablink/blob/main/examples/part1/exp_rabi_example.py)

Žiadne heslá, tokeny ani výstupy diagnostiky nepatria do verejného repozitára.
