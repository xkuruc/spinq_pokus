# Reálna séria meraní Gemini Lab na Windows

Program `spinq_live_suite.py` používa existujúci SpinQLabLink, ktorý sa na
tomto Windows počítači už pripájal k `172.19.20.100:8181`. Tablet zostáva
pripojený k prístroju cez USB. Na vývojovom Macu neprebehlo žiadne meranie.
Potrebuje Python 3.11+ a nainštalovaný SpinQLabLink 1.0.2 v tom istom
`.venv`; ďalšie balíky neinštaluje. Ak nájde inú verziu SDK, uloží report
a zastaví hardvérové požiadavky, kým sa nepotvrdí kompatibilita adaptéra.

## Príprava raz

V PowerShelli v priečinku repozitára:

```powershell
git pull
Copy-Item .\live_suite.example.toml .\live_suite.toml
notepad .\live_suite.toml
```

V lokálnom súbore:

1. Skontroluj host, port, účet a známy 40 µs pracovný bod. Hodnota
   `relaxation_delay_us = 15.0` pochádza zo starého úspešného skriptu;
   jednotka je podľa SDK µs, preto ju treba potvrdiť pre tvoje pracovisko.
2. Doplň všetky hodnoty v `[limits]` z prevádzkových údajov pre tento
   konkrétny prístroj. Komentované `...` sú zámerne prázdne: limity
   Python validátora nie sú bezpečné prevádzkové limity.
3. Po potvrdení pracovného bodu nastav `baseline_verified = true`.
   Tým povoľuješ celú sériu v uvedených limitoch; program sa nepýta pred
   každým experimentom. `max_experiments`, prestávku a timeout môžeš upraviť.
   Ak server neposiela stav fronty, nastav `exclusive_use_confirmed = true`
   len pri potvrdenom výhradnom používaní prístroja.
4. Voliteľné typy meraní v `[features]` zapni len ak poznáš ich význam a
   máš pre ne pracovný bod. Bez nich sa test označí NEOVERENÉ s dôvodom.

Heslo nedávaj do súboru. Program sa naň opýta skrytou výzvou; pri starom
nastavení fungovalo `anyword`.

## Jediný príkaz pre celú sériu

```powershell
.\.venv\Scripts\python.exe .\spinq_live_suite.py --config .\live_suite.toml
```

Vznikne nový `results\DATUM_live_suite\results.zip` s `REPORT.md`,
`results.json`, redigovanými prijatými udalosťami, kompletnými pôvodnými
dekódovanými krivkami, lokálnou analýzou, SVG náhľadmi a chybami.
Priebeh sa ukladá po každom kroku. Výstupný adresár sa nikdy neprepisuje.
`results/` a `live_suite.toml` Git ignoruje. Program nič neposiela do cloudu.

Séria skúsi základný NMR experiment, fyzikálny baseline, predvolene 10
samostatných opakovaní, malé zmeny pulzov a akvizície s návratom na
baseline a krátky Rabi sken s jedným kontrolným bodom. Presný počet závisí
od zapnutých schopností, limitov, stavu prístroja a rozpočtu. P kanál,
vypnutie prípravy, meranie bez RF, frekvenčné/demodulačné posuny a
viacsegmentový pulz sú štandardne vypnuté, pretože ich pracovný bod alebo
význam zatiaľ nie je potvrdený. Gradienty, trvalé shimovanie, lock a
firmvér program nemení.

Ak je baseline alebo limit neúplný, program môže prijať stavové správy,
ale neodošle experiment. Pri timeoute alebo strate locku zastaví nové
požiadavky; odpojenie klienta nie je potvrdenie zastavenia úlohy na prístroji.
Stavy a presné dôvody sú v `REPORT.md`.
Ak po násilnom ukončení zostane lokálny súbor `.spinq_live_gemini.lock`,
odstráň ho až po kontrole fronty a stavu úlohy na tablete.
