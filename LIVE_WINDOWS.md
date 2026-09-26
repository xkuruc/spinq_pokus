# Reálne meranie Gemini Lab na Windows

Program `spinq_live_suite.py` používa existujúci SpinQLabLink, ktorý sa na
tomto Windows počítači už pripájal k `172.19.20.100:8181`. Tablet zostáva
pripojený k prístroju cez USB. Na vývojovom Macu neprebehlo žiadne meranie.
Potrebuje Python 3.11+ a nainštalovaný SpinQLabLink 1.0.2 v tom istom
`.venv`; ďalšie balíky neinštaluje. Ak nájde inú verziu SDK, uloží report
a zastaví hardvérové požiadavky, kým sa nepotvrdí kompatibilita adaptéra.

## Známy pracovný bod

Predvolená konfigurácia zopakuje **najviac jedno** fyzikálne meranie, ktoré
už podľa tvojho výstupu na tomto prístroji skončilo úspešne: vodíkový kanál,
pulz 40 µs, amplitúda 100 %, fáza 90°, detuning 0, bez gradientu, príprava
zapnutá, 10 000 vzoriek/s, 16 000 bodov, oneskorenie vzorkovania 0.
Číselná hodnota `relaxation_delay_value = 15.0` zostáva presne ako pri
starom úspešnom pokuse. [Oficiálny príklad](https://doc.spinq.cn/doc/SpinQLAB_Link/en/experiment/Research_Experiment.html)
ju označuje za sekundy, no zdroj SDK 1.0.2 ju opisuje ako mikrosekundy;
jednotku preto nevyhlasujeme za potvrdenú a hodnotu automaticky nemeníme.

Toto **nie je** bezpečný rozsah parametrov. Verejná dokumentácia neuvádza
potvrdené prevádzkové limity pre opakovania, súčet RF záťaže alebo teplotu
tohto konkrétneho prístroja. Preto predvolené `historical_baseline_only = true`
nedovolí opakovania, Rabi sken ani zmeny nastavení. Program stále vyžaduje
čerstvý stav pripojenia, lock, konečnú teplotu a prázdnu frontu alebo
potvrdené výhradné používanie. Nameranú teplotu uloží, ale bez známeho
intervalu ju nevydáva za schválenú.

## Spustenie známeho bodu

V PowerShelli v priečinku repozitára:

```powershell
git pull
.\.venv\Scripts\python.exe .\spinq_live_suite.py --config .\live_suite.example.toml
```

Konfigurácia už obsahuje adresu tvojho tabletu a známy bod. Pred každým
opätovným spustením skontroluj na tablete, že predchádzajúca úloha skončila.
Ak server neposiela stav fronty, skript sa bezpečne zastaví. Iba keď je
prístroj naozaj vyhradený pre teba, skopíruj príklad do `live_suite.toml`,
nastav tam `exclusive_use_confirmed = true` a spusti ho s týmto lokálnym
súborom.

Heslo nedávaj do súboru. Program sa naň opýta skrytou výzvou; pri starom
nastavení fungovalo `anyword`.

## Až keď získaš prevádzkové limity

Na celú sériu by bolo treba pre tento konkrétny prístroj doložiť povolené
amplitúdy a šírky RF pulzov, RF čas na úlohu aj za sériu, rozsah teploty,
limity odberu a posunov frekvencií a potvrdiť význam času relaxácie.
Hodnoty z Python validátora sú rozsahy vstupov, nie bezpečné prevádzkové
limity. Až po overení môžeš nastaviť `historical_baseline_only = false`,
doplniť `[limits]` a upraviť `repeat_count`, `max_experiments`, prestávku a
jednotlivé `[features]`. Bez týchto údajov širšiu sériu nespúšťaj.

Vznikne nový `results\DATUM_live_suite\results.zip` s `REPORT.md`,
`results.json`, redigovanými prijatými udalosťami, kompletnými pôvodnými
dekódovanými krivkami, lokálnou analýzou, SVG náhľadmi a chybami.
Priebeh sa ukladá po každom kroku. Výstupný adresár sa nikdy neprepisuje.
`results/` a `live_suite.toml` Git ignoruje. Program nič neposiela do cloudu.

Pri potvrdených limitoch možno zapnúť NMR experiment, opakovania, malé zmeny
pulzov a akvizície s návratom na baseline a krátky Rabi sken. Presný počet
závisí od zapnutých schopností, limitov, stavu prístroja a rozpočtu. P kanál,
vypnutie prípravy, meranie bez RF, frekvenčné/demodulačné posuny a
viacsegmentový pulz sú štandardne vypnuté, pretože ich pracovný bod alebo
význam zatiaľ nie je potvrdený. Gradienty, trvalé shimovanie, lock a
firmvér program nemení.

V predvolenom režime sa neúplné limity vzťahujú na všetky ďalšie prípady;
v reporte ich uvidíš ako NEOVERENÉ. Pri timeoute alebo strate locku zastaví nové
požiadavky; odpojenie klienta nie je potvrdenie zastavenia úlohy na prístroji.
Stavy a presné dôvody sú v `REPORT.md`.
Ak po násilnom ukončení zostane lokálny súbor `.spinq_live_gemini.lock`,
odstráň ho až po kontrole fronty a stavu úlohy na tablete.
