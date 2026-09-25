# SpinQ Gemini Lab: bezpečná kontrola na Windows

Tento repozitár má dva oddelené kroky. **Začni diagnostikou USB.** Tá iba
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

## 2. Voliteľne čítaj stav cez SpinQLabLink

Oficiálny [SpinQLabLink](https://github.com/SpinQTech/spinqlablink) používa
TCP/IP. Tento krok má význam iba vtedy, keď Windows vidí SpinQ ako sieťové
zariadenie alebo máš IP adresu prístroja zo softvéru SpinQ. Nie je to priama
komunikácia cez COM port.

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

## Zdroj API

- [Oficiálny SpinQLabLink](https://github.com/SpinQTech/spinqlablink)
- [Oficiálny príklad čítania stavu zariadenia](https://github.com/SpinQTech/spinqlablink/blob/main/examples/device_data_example.py)

Žiadne heslá, tokeny ani výstupy diagnostiky nepatria do verejného repozitára.
