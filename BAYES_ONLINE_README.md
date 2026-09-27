# `01_bayes_online` — samostatný experiment pre Gemini Lab

Tento program beží na Windows počítači, ktorý sa už vie pripojiť k tabletu SpinQ cez `172.19.20.100:8181`. Na Macu sa pripravil kód a offline numerické kontroly; **žiadne fyzické meranie sa tam nevykonalo**. Experiment nepoužíva kalibrácie, modely ani výsledky iných benchmarkov. Pôvodné funkčné prostredie `.venv` ponecháva na mieste a pre vlastné závislosti používa `.bayes-online-venv`.

## Spustenie na Windowse

V PowerShelli v priečinku repozitára:

```powershell
git pull origin main
.\run_01_bayes_online.cmd --task all --exclusive-use-confirmed
```

### Kratšie porovnanie na jednom kanáli

Ak chceš najprv porovnať kalibráciu bez hodinového behu, spusti samostatný rýchly profil:

```powershell
.\run_01_bayes_online.cmd --quick --exclusive-use-confirmed
```

Profil meria len kanál H a spoločný pilot Rabi/frekvencie. Má tri samostatné scenáre: bez skrytej chyby, s kladnou a so zápornou vratnou chybou povelov. Porovná B (klasický fit) a D (adaptívny Bayes). Jeden spoločný anchor stojí 18 fyzických FID. V každom bloku sa zvlášť zmerajú dve nominálne referencie, dve nové kontrolné FID pre každú metódu a kontrola driftu pred ramenami, medzi nimi a po nich; v dvoch chybných blokoch pribudne po jednom kontrolnom probe. B a D majú každá po 10 tréningových FID v každom bloku. Plán je **107 fyzických úloh** s limitom 120; skutočný počet môže byť menší, ak sa beh zastaví pri neplatnom pilote alebo chybe. Po 50 minútach aktívneho behu program nespustí ďalšiu fyzickú úlohu; práve bežiaca úloha a uloženie výsledkov môžu trvať dlhšie. Pri doterajších približne 15–20 sekundách na úlohu počítaj orientačne s **30–40 minútami** vrátane prestávok a spracovania. Nie je to záruka času: prístroj a tablet môžu odpovedať pomalšie.

Rovnaká chyba povelov, spoločný anchor a nezávislé kontrolné FID umožňujú párové porovnanie B/D v každom bloku. Tri bloky a skrátený tréning však dávajú len predbežný výsledok pre túto zostavu. Profil **nezahŕňa A (bez učenia), C (pevný Bayes), P kanál, väzbu, PPS ani Bella**. Preto z neho nemožno pripísať rozdiel D oproti B výlučne adaptívnemu výberu bodov; líši sa aj inferenčná metóda. Ak pilot nepotvrdí identifikovateľný model, report to označí a nepremení nejasné merania na tvrdenie o výhode.

Pri `--quick` sa na výsledkovú vetvu GitHubu nahrá iba `REPORT.md`, `comparison.csv` a `results.json`. Surové FID a `results.zip` zostávajú na Windowse v zobrazenom výsledkovom adresári. Každý sieťový krok nahrávania má časový limit 120 sekúnd; jeho zlyhanie nemení lokálne výsledky.

`--exclusive-use-confirmed` znamená, že máš pre tento beh výhradné používanie prístroja a na tablete nebeží iná úloha. Ak server nahlási obsadený rad, program nové meranie neodošle. Spúšťaj vždy len jeden proces launchera. Pred prvou fyzickou úlohou prebehne kontrola verzie SpinQLabLink **1.0.2**, importov a numerickej algebry. Konfigurácia je v `config-bayes-online.json`; prednastavené hranice šírky, amplitúdy, počtu úloh a RF času sú **softvérové obálky tohto protokolu**, nie certifikované bezpečné limity výrobcu. Program nemení magnet, lock, shim, firmvér ani trvalú kalibráciu.

Príkaz podporuje `--task frequency`, `rabi`, `coupling`, `pps`, `bell` alebo `all`. `frequency` a `rabi` spúšťajú spoločný trojparametrový kalibračný protokol, pretože frekvenčný posun, RF škála a fáza sa na tých istých komplexných FID navzájom ovplyvňujú. `all` postupuje po závislostiach; samostatné `coupling`, `pps` a `bell` vykonajú len potrebný H/P pilot a kvalifikáciu, bez troch plných A/B/C/D kalibračných blokov. Frekvencia a Rabi sú vlastné merania fyzikálnej vrstvy, po jednom exportovanom komplexnom FID na úlohu. `coupling`, `pps` a `bell` vyžadujú dodatočne kvalifikované časovanie, väzbu, spoločnú H/P os a čítací model. Ak niektorý primitív nie je potvrdený, výstup musí uviesť `UNSUPPORTED_REQUIRED_PRIMITIVE`, `NONIDENTIFIABLE` alebo inú konkrétnu príčinu; numerická matica ani vendor skóre ho nenahrádzajú. `all` zachová platné jednorozmerné výsledky aj pri takomto zastavení dvojqubitovej vetvy.

Program zavádza len vratné chyby **odoslaných povelov** v úlohe DUT: posun RF, škálovanie amplitúdy a fázu. Rovnako platia pre porovnávané ramená jedného bloku, vrátane nulového scenára. Neznamenajú zmenu prirodzenej rezonancie magnetu. Skutočný upravený payload a pravda perturbácie sú uložené v `evaluator_only/`; learner dostáva len logický povel a povolený komplexný FID. Samotná digitálna demodulácia nie je náhradou opravy fyzického pulzu.

## Čo porovnáva

| Metóda | Úloha |
|---|---|
| A `prior_only` | Nominálny anchor bez vlastných tréningových akvizícií. |
| B klasika | Hrubý/jemný plán a komplexný ohraničený fit z vlastných meraní. |
| C pevný Bayes | Časticový model a pevný informatívny plán. |
| D adaptívny Bayes | Ten istý typ časticového modelu s online výberom ďalšieho bodu. |

Anchor a oddelené referenčné FID sú platený náklad všetkých ramien. Kalibračné FID používajú rovnakú serverovú prípravu `makePps=True`; vlastná PPS je oddelená vetva s `makePps=False`. Každé rameno začína s vlastným novým stavom; finálne kontroly sú nové fyzické akvizície po zmrazení modelu. Program porovnáva ich komplexnú chybu, odhad parametrov, počet akvizícií a celý čas. Počet bodov jedného FID nie je počet nezávislých experimentov. `TARGET_REACHED` znamená splnenie lokálnych kontrolných kritérií; samo osebe nedokazuje všeobecnú výhodu oproti klasike.

V dvojqubitovej vetve je **vlastná PPS** možná iba s overeným neunitárnym postupom. Pri časovom priemerovaní ide o `TEMPORAL_AVERAGED_EFFECTIVE_PPS` z viacerých kompletných behov. Bellova rekonštrukcia sa musí robiť bez znalosti ideálneho Bellovho cieľa; ten patrí až do oddeleného hodnotenia. Aj perfektné offline identity H/CNOT a ideálna hodnosť tomografickej matice sú len matematické kontroly, kým nie je potvrdená reálna H/P časová os, znamienko efektívnej väzby a rozlíšenie čítacích zložiek. Súčasná automatická kvalifikácia vie overiť len nadväznosť pulzov a voľný koherentný úsek na jednotlivých kanáloch. **Spoločné H/P zarovnanie, znamienko J, vlastný CNOT a nezávislý čítací model nevie týmto pripojením potvrdiť.** Preto `pps` a `bell` pri tomto stave skončia s konkrétnym `UNSUPPORTED_REQUIRED_PRIMITIVE`; párové PPS/bránové a R0/R1 porovnanie sa nesmie vydávať za hotové meranie. Implementované cesty pre PPS a Bell možno použiť až po doplnení skutočnej kvalifikácie týchto vlastností.

## Výstupy a čítanie výsledku

Každý beh dostane nový priečinok `results/01_bayes_online/<run_id>/`. `REPORT.md` je zhrnutie, `comparison.csv` a `results.json` sú strojovo čitateľné podklady. `data/*.npz` uchováva **všetky prijaté exportované Re/Im body** a ich časovú os; susedné JSON súbory obsahujú identifikátor úlohy a požadovaný logický povel. `evaluator_only/` obsahuje skryté zásahy a skutočne odoslané nastavenia. `vendor_reference/` oddeľuje serverové FFT a ostatné výstupy výrobcu; nie sú tréningovou pravdou. `models/`, `profiles/`, `sanitized_logs/` a `source_snapshot/` dokumentujú beh. Všetko je aj v `results.zip`.

Launcher sa na konci pokúsi neinteraktívne poslať **iba výsledky tohto behu** do vetvy `benchmark/01_bayes_online/<run_id>` na GitHub. Použije existujúci Git Credential Manager alebo už nastavený `GH_TOKEN`/`GITHUB_TOKEN`; token nezapisuj do príkazu, súboru ani URL. Pri zlyhaní nahratia hľadaj `UPLOAD_FAILED` a lokálny `results.zip` ostáva celý. Veľký archív sa rozdelí na `results.zip.part0001`, `part0002`, … a vo vetve bude `HOW_TO_JOIN.txt`. Žiadny vlastný hash alebo podpis sa nevytvára.

Pri nejasnom stave práve odoslanej fyzickej úlohy sa program zastaví bez ďalších povelov. `Ctrl+C` počas úlohy nemusí znamenať, že hardvér už prestal pracovať; pred ďalším behom treba pozrieť jeho stav na tablete. Zachované čiastkové FID a konkrétne dôvody neúspechu sú hodnotné podklady, ale nie dôkaz dokončenej kalibrácie.

Metódy, zdrojové články a otvorené hardvérové predpoklady sú spísané v [SOURCES.md](bayes_online_core/SOURCES.md).
