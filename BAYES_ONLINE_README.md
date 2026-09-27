# `01_bayes_online` — samostatný experiment pre Gemini Lab

Program môže bežať samostatne na Windows počítači s funkčným pripojením k tabletu SpinQ cez `172.19.20.100:8181`. Na tomto Macu je pripravený aj priamy sieťový launcher: celé učenie, FFT, odšumovanie, optimalizácia a vyhodnotenie potom bežia na Macu; SpinQ vykonáva pulzy a exportuje FID. Žiadnu výhodu kalibrácie z toho nepredpokladáme. Experiment nepoužíva kalibrácie, modely ani výsledky iných benchmarkov. Pôvodné funkčné Windows prostredie `.venv` ponecháva na mieste a pre vlastné závislosti používa `.bayes-online-venv`.

## Spustenie na tomto Macu

Keď má Mac priame sieťové spojenie so SpinQLabLink a výhradný prístup k prístroju, krátky fyzický pilot spustíš v priečinku repozitára príkazom:

```sh
./run_01_bayes_online_mac.sh --exclusive-use-confirmed
```

Mac launcher používa už existujúce lokálne prostredie `.bench-test`, predvolene vykoná iba `--quick` a uloží výsledky na Mac bez automatického nahrania. `--full --task all --exclusive-use-confirmed` spustí celý dlhší reťazec; `--full` je prepínač shell launchera. Nespúšťaj Mac a Windows launcher súčasne na tom istom zariadení.

## Spustenie na Windowse

V PowerShelli v priečinku repozitára:

```powershell
git fetch origin codex/01-bayes-online
git switch --detach FETCH_HEAD
.\run_01_bayes_online.cmd --task all --exclusive-use-confirmed
```

Aktuálna implementácia je vo vetve `codex/01-bayes-online`; predvolená vetva `main` zostáva bez tejto zmeny do jej prijatia.

### Kratšie porovnanie na jednom kanáli

Ak chceš najprv porovnať kalibráciu bez hodinového behu, spusti samostatný rýchly profil:

```powershell
.\run_01_bayes_online.cmd --quick --exclusive-use-confirmed
```

Profil meria len kanál H a spoločný pilot Rabi/frekvencie. Má tri samostatné scenáre: bez skrytej chyby, s kladnou a so zápornou vratnou chybou povelov. Porovná B (klasický fit) a D (adaptívny Bayes). Jeden spoločný anchor stojí najviac 18 fyzických FID. Po prvých šiestich rôznych pulzoch program preverí, či je vôbec viditeľný skorý koherentný signál; ak nie, archivuje ich a zastaví meranie s `PILOT_INCONCLUSIVE`. V každom platnom bloku sa zvlášť zmerajú dve nominálne referencie, dve nové finálne kontrolné FID pre každú metódu a kontrola driftu pred ramenami, medzi nimi a po nich; v dvoch chybných blokoch pribudne po jednom kontrolnom probe. B a D majú rovnaký strop 10 tréningových FID. Po šiestom FID obe dostanú rovnaké pravidlo skorého zastavenia a dve osobitné kontrolné akvizície; tie sa nikdy nepoužijú pri finálnom skórovaní. Najhorší plán je **119 fyzických úloh** s limitom 120. Po 50 minútach aktívneho behu program nespustí ďalšiu fyzickú úlohu; práve bežiaca úloha a uloženie výsledkov môžu trvať dlhšie. Pri doterajších približne 15–20 sekundách na úlohu počítaj orientačne s **30–40 minútami** pri dobrom signáli, pri zlyhaní skorého pilotu asi s dvoma minútami. Nie je to záruka času: prístroj a tablet môžu odpovedať pomalšie.

Rovnaká chyba povelov, spoločný anchor a nezávislé kontrolné FID umožňujú párové porovnanie B/D v každom bloku. Tri bloky a skrátený tréning však dávajú len predbežný výsledok pre túto zostavu. Profil **nezahŕňa A (bez učenia), C (pevný Bayes), P kanál, väzbu, PPS ani Bella**. Preto z neho nemožno pripísať rozdiel D oproti B výlučne adaptívnemu výberu bodov; líši sa aj inferenčná metóda. Ak pilot nepotvrdí identifikovateľný model, report to označí a nepremení nejasné merania na tvrdenie o výhode.

Pri `--quick` sa na výsledkovú vetvu GitHubu nahrá aj úplný `results.zip` so surovými FID; kópia ostáva v zobrazenom výsledkovom adresári. Ak je archív veľký, rozdelí sa na očíslované časti. Zlyhanie nahrávania nemení lokálne výsledky.

`--exclusive-use-confirmed` znamená, že máš pre tento beh výhradné používanie prístroja a na tablete nebeží iná úloha. Ak server nahlási obsadený rad, program nové meranie neodošle. Spúšťaj vždy len jeden proces launchera. Pred prvou fyzickou úlohou prebehne kontrola verzie SpinQLabLink **1.0.2**, importov a numerickej algebry. Konfigurácia je v `config-bayes-online.json`; prednastavené hranice šírky, amplitúdy, počtu úloh a RF času sú **softvérové obálky tohto protokolu**, nie certifikované bezpečné limity výrobcu. Program nemení magnet, lock, shim, firmvér ani trvalú kalibráciu.

Príkaz podporuje `--task frequency`, `rabi`, `coupling`, `pps`, `bell` alebo `all`. `frequency` a `rabi` spúšťajú spoločný trojparametrový kalibračný protokol, pretože frekvenčný posun, RF škála a fáza sa na tých istých komplexných FID navzájom ovplyvňujú. `all` postupuje po závislostiach; samostatné `coupling`, `pps` a `bell` vykonajú len potrebný H/P pilot a kvalifikáciu, bez troch plných A/B/C/D kalibračných blokov. Frekvencia a Rabi sú vlastné merania fyzikálnej vrstvy, po jednom exportovanom komplexnom FID na úlohu. `coupling`, `pps` a `bell` vyžadujú dodatočne kvalifikované časovanie, väzbu, spoločnú H/P os a čítací model. Ak niektorý primitív nie je potvrdený, výstup musí uviesť `UNSUPPORTED_REQUIRED_PRIMITIVE`, `NONIDENTIFIABLE` alebo inú konkrétnu príčinu; numerická matica ani vendor skóre ho nenahrádzajú. `all` zachová platné jednorozmerné výsledky aj pri takomto zastavení dvojqubitovej vetvy.

Samostatné `--task bell` plánuje dve Bellove matice (`Phi+`, `Psi+`); `--task all` plánuje všetky štyri. Táto vetva sa spustí iba po fyzickom potvrdení potrebných primitívov a readout modelu.

Program zavádza len vratné chyby **odoslaných povelov** v úlohe DUT: posun RF, škálovanie amplitúdy a fázu. Rovnako platia pre porovnávané ramená jedného bloku, vrátane nulového scenára. Neznamenajú zmenu prirodzenej rezonancie magnetu. Skutočný upravený payload a pravda perturbácie sú uložené v `evaluator_only/`; learner dostáva len logický povel a povolený komplexný FID. Samotná digitálna demodulácia nie je náhradou opravy fyzického pulzu.

## Čo porovnáva

| Metóda | Úloha |
|---|---|
| A `prior_only` | Nominálny anchor bez vlastných tréningových akvizícií. |
| B klasika | Hrubý/jemný plán a komplexný ohraničený fit z vlastných meraní. |
| C pevný Bayes | Časticový model a pevný informatívny plán. |
| D adaptívny Bayes | Ten istý typ časticového modelu s online výberom ďalšieho bodu. |

Anchor a oddelené referenčné FID sú platený náklad všetkých ramien. Kalibračné FID používajú rovnakú serverovú prípravu `makePps=True`; vlastná PPS je oddelená vetva s `makePps=False`. Každé rameno začína s vlastným novým stavom; finálne kontroly sú nové fyzické akvizície po zmrazení modelu. Program porovnáva ich komplexnú chybu, odhad parametrov, počet akvizícií a celý čas. Počet bodov jedného FID nie je počet nezávislých experimentov. `TARGET_REACHED` znamená splnenie lokálnych kontrolných kritérií; samo osebe nedokazuje všeobecnú výhodu oproti klasike.

V dvojqubitovej vetve je **vlastná PPS** možná iba s overeným neunitárnym postupom. Pri časovom priemerovaní ide o `TEMPORAL_AVERAGED_EFFECTIVE_PPS` z viacerých kompletných behov. Bellova rekonštrukcia sa musí robiť bez znalosti ideálneho Bellovho cieľa; ten patrí až do oddeleného hodnotenia. Minimálna nezávislá tomografia používa štyri predvolené konfigurácie čítania, pri ideálnom modeli má hodnosť 15/15 a číslo podmienenosti 2. To znižuje cenu trojvetvovej PPS zo 48 na 24 FID a všetkých štyroch Bellových stavov zo 192 na 96 FID; pri skutočnom meraní sa hodnosť a podmienenosť znovu preveria. Aj perfektné offline identity H/CNOT a ideálna hodnosť sú len matematické kontroly, kým nie je potvrdená reálna H/P časová os, znamienko efektívnej väzby a rozlíšenie čítacích zložiek. Súčasná automatická kvalifikácia vie overiť len nadväznosť pulzov a voľný koherentný úsek na jednotlivých kanáloch. **Spoločné H/P zarovnanie, znamienko J, vlastný CNOT a nezávislý čítací model nevie týmto pripojením potvrdiť.** Preto `pps` a `bell` pri tomto stave skončia s konkrétnym `UNSUPPORTED_REQUIRED_PRIMITIVE`; párové PPS/bránové a R0/R1 porovnanie sa nesmie vydávať za hotové meranie. Implementované cesty pre PPS a Bell možno použiť až po doplnení skutočnej kvalifikácie týchto vlastností.

## Výstupy a čítanie výsledku

Každý beh dostane nový priečinok `results/01_bayes_online/<run_id>/`. `REPORT.md` je zhrnutie, `comparison.csv` a `results.json` sú strojovo čitateľné podklady. `data/*.npz` uchováva **všetky prijaté exportované Re/Im body** a ich časovú os; susedné JSON súbory obsahujú identifikátor úlohy a požadovaný logický povel. `evaluator_only/` obsahuje skryté zásahy a skutočne odoslané nastavenia. `vendor_reference/` oddeľuje serverové FFT a ostatné výstupy výrobcu; nie sú tréningovou pravdou. `models/`, `profiles/`, `sanitized_logs/` a `source_snapshot/` dokumentujú beh. Všetko je aj v `results.zip`.

Launcher sa na konci pokúsi neinteraktívne poslať **iba výsledky tohto behu** do vetvy `benchmark/01_bayes_online/<run_id>` na GitHub. Použije existujúci Git Credential Manager alebo už nastavený `GH_TOKEN`/`GITHUB_TOKEN`; token nezapisuj do príkazu, súboru ani URL. Pri zlyhaní nahratia hľadaj `UPLOAD_FAILED` a lokálny `results.zip` ostáva celý. Veľký archív sa rozdelí na `results.zip.part0001`, `part0002`, … a vo vetve bude `HOW_TO_JOIN.txt`. Žiadny vlastný hash alebo podpis sa nevytvára.

Pri nejasnom stave práve odoslanej fyzickej úlohy sa program zastaví bez ďalších povelov. `Ctrl+C` počas úlohy nemusí znamenať, že hardvér už prestal pracovať; pred ďalším behom treba pozrieť jeho stav na tablete. Zachované čiastkové FID a konkrétne dôvody neúspechu sú hodnotné podklady, ale nie dôkaz dokončenej kalibrácie.

Metódy, zdrojové články a otvorené hardvérové predpoklady sú spísané v [SOURCES.md](bayes_online_core/SOURCES.md).
