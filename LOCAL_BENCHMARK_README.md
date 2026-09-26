# Reálny lokálny benchmark Gemini Lab (A–H)

Tento beh spúšťaj iba na Windows počítači, ktorý má funkčné pripojenie
`172.19.20.100:8181` a existujúce `.venv` s `spinqlablink==1.0.2`.
Pôvodná obrazovka ostáva pripojená k prístroju cez USB. Celé FFT, fitting,
modelovanie, optimalizácia, tréning a vyhodnotenie beží na Windowse.
Tablet/server môže počas merania stále povinne spracúvať signál; prijaté
`fidRe`/`fidIm` sa preto označujú ako exportované komplexné FID.

V PowerShelli v priečinku repozitára:

```powershell
git pull
.\run_windows.cmd
```

Spúšťač vytvorí izolované `.benchmark-venv`, bez aktualizácie funkčného
`.venv`. Pred prvým RF pulzom v samostatnom procese overí SDK, numeriku a
PyTorch vrátane gradientu. Pri chybe základnej kontroly sa nemeria. Chyba
PyTorch iba presne označí závislé neurónové metódy ako nevykonané.

Predvolený hlavný beh používa 10 meracích blokov a pevný softvérový limit
280 úloh / 20 000 µs požadovaného RF trvania. To sú rozpočty tejto štúdie,
nie certifikované prevádzkové limity výrobcu. Každá úloha je H-kanálový
vlastný fyzikálny experiment so zachovaným `makePps=True`, bez gradientov,
shimovania, zápisu kalibrácie alebo aktualizácie firmvéru. Stav locku,
pripojenia a voľného radu sa kontroluje pred odoslaním.

Kratší trojblokový feasibility pilot:

```powershell
.\run_windows.cmd --blocks 3
```

Prerušený beh možno obnoviť bez opakovania dokončených úloh iba vtedy, keď
predchádzajúca úloha nemá neznámy fyzický stav. Pri neznámom stave program
zastaví všetky nové odoslania; najprv skontroluj úlohu na tablete. Cesta sa
dosadzuje podľa skutočného názvu adresára:

```powershell
.\run_windows.cmd --resume .\results\20260926_123456_UTC
```

Ak potrebuješ overiť uložený Rabiho pilot pred ďalšími meraniami, tento
príkaz číta iba súbory na Windowse. Nepripája sa k prístroju a výsledok
vypíše do konzoly:

```powershell
.\.benchmark-venv\Scripts\python.exe .\diagnose_saved_pilot.py .\results\20260926_123456_UTC
```

Živý beh vypisuje krátky stav pilotu a blokov. Chyby vypíše hneď s fázou,
blokom a miestom v kóde; úplný traceback uloží do `results.json`.

Bez kontaktu so zariadením možno znova vytvoriť report a grafy zo
zachovaných výsledkov:

```powershell
.\.benchmark-venv\Scripts\python.exe .\local_benchmark_windows.py --offline-rebuild --resume .\results\20260926_123456_UTC --no-upload
```

`--offline-rebuild` obnoví existujúci report, ale znovu nepočíta modely.
Po oprave kódu môžeš **znova analyzovať všetky uložené FID bez jediného
nového experimentu** takto (nahraď názov adresára skutočným behom):

```powershell
.\reanalyze_windows.cmd .\results\20260926_143851_UTC
```

Vznikne nový podadresár `reanalysis_...` s novým reportom. Pôvodný
`results.json`, pôvodné FID, udalosti a hardvérový žurnál zostanú nedotknuté.
Príkaz znovu preverí lokálny CPU/PyTorch, vypíše stav D/E/F/G a na GitHub
odošle iba nový súhrn; žiadne FID neposiela. Ak žurnál obsahuje neukončenú
úlohu alebo chýba uložený FID, analýzu zastaví s konkrétnym dôvodom.

Plný lokálny ZIP vznikne iba pri výslovnom `--full-archive`. Môžeš ho
vytvoriť aj neskôr cez predchádzajúci offline príkaz s týmto prepínačom.
Starší `results.zip` v obnovenom priečinku ostáva zachovaný, no bez tohto
prepínača sa neaktualizuje.

Každý beh vytvorí `results/<run_id>/REPORT.md`, `comparison.csv`,
`results.json`, `raw/`, `vendor_reference/`, `models/` a `plots/`.
Pôvodné Re/Im a osi sú v komprimovaných NPZ, sanitizované
udalosti v `data/events.jsonl.gz`. Na GitHub sa posiela iba report, tabuľka,
strojovo čitateľné zhrnutie a výsledné grafy vo vetve `benchmark/<run_id>`.
Merané FID ostáva na Windowse. Použije sa existujúce Git prihlásenie alebo
`GH_TOKEN`/`GITHUB_TOKEN` z prostredia, žiadny token sa nezapisuje do kódu.
Pri zlyhaní uploadu ostávajú report a merané dáta lokálne.

Metódy A–H sú implementované, no úspech reálnych pulzových, echo a
viacspinových porovnaní závisí od overeného časovania, dostatočnej
nezávislej referencie a skutočnej kalibrácie čítania. Program tieto vetvy
označí `UNVERIFIED_TIMING`, `REFERENCE_INADEQUATE` alebo
`DEPENDENCY_FAILED`, keď údaje nestačia. Žiadny simulovaný procesný
overlap sa nevydáva za nameranú fidelitu brány.

Podrobnosti o pôvode výpočtov a rozsahu reprodukcie sú v
`computation_map.md`, `reproduction_scope.md` a `sources.md`.
