# Experiment 01 — rozsah reprodukcie

**Cieľ:** na Gemini Lab porovnať počet fyzických akvizícií a celkový čas potrebný na dosiahnutie rovnakej nezávisle overenej presnosti kalibrácie. Tento dokument určuje implementačný rozsah; neobsahuje namerané výsledky Experimentu 01. Mac nemá hardvér. Živé merania a výskumné výpočty majú prebehnúť na pripojenom Windowse po spustení `run_01_bayes_kalibracia.cmd`.

## Prevzatá metóda verzus nová adaptácia

| Prvok | Publikovaný postup Gerster et al. | Navrhnutá implementácia pre NMR |
| --- | --- | --- |
| Posterior | Vážené častice, Bayesovská aktualizácia, Liu–West resampling a odhad variancií (časť III). | Log-váhy, stabilná normalizácia, ESS, podmienený resampling a jitter v škálovaných súradniciach; periodická fáza sa hodnotí kruhovo. Počet častíc vyberie lokálna konvergenčná kontrola. |
| Likelihood | Diskrétne výstupy/populácie iónovej MS brány. | Reálna a imaginárna časť komplexných koeficientov FID alebo celý FID, s kovarianciou a driftom odhadnutými z nezávislých opakovaní. 16 000 bodov FID nie je 16 000 nezávislých qubitových shots. |
| Parametre | Štyri parametre iónovej entangling brány. | Identifikovateľný `delta` [Hz], jedna RF škála **alebo** `t90` [µs], a iba relatívna fáza [deg] po fixovaní prijímacieho gauge. Neidentifikovateľná fáza nemá zneplatniť samostatný odhad t90. |
| Voľba merania | Očakávaná posteriorná variancia normalizovaná prahmi; alternatívne prahová heuristika (časť IV B). | Monte Carlo integrácia cez **spojité** prediktívne NMR výsledky; D minimalizuje súčet očakávaných variancií/tolerancia², E maximalizuje pokles toho istého skóre na odhadovaný celý čas kandidáta. Časová cena je nová adaptácia. |
| Fyzické merania | Rôzne počty a fázy MS brán na iónoch. | H pulzy cez viac častí Rabiho oscilácie, fázovo citlivé čítanie a Ramsey iba po overení skutočného koherentného delay. Ak delay nie je dostupný, frekvencia z lokálneho modelu FID, s užším tvrdením o identifikovateľnosti. |
| Stop a kvalita | Prahy sú odvodené z tolerovanej chyby iónovej brány. | Tolerancie z nezávislého pilotu a dosiahnuteľnej referencie, zmrazené pred porovnaním; úspech len pri validnom modeli, splnení tolerancií a nezávislej kontrole. Vyčerpanie rozpočtu je samostatný stav. |

## Porovnanie A–E

- **A:** pevný identifikovateľný sken a vážený viacštartový fit.
- **B:** kvalitný hrubý/jemný sken s tým istým fitom.
- **C:** rovnaký particle filter ako D/E, ale pevný plán; oddeľuje výber merania od inferenčného algoritmu.
- **D:** adaptívny particle filter s minimalizáciou toleranciami normalizovanej neistoty.
- **E:** ten istý filter a kandidáti s časovo váženým výberom.

Spoločný pilot určí a zmrazí prior, rozsahy, tolerancie, fyzické podmienky a rozpočet pre každú metódu. Počiatočný návrh je 24 nových akvizícií na metódu a blok plus tri oddelené randomizované porovnávacie bloky; podľa identifikovateľnosti sa návrh môže upraviť **pred** porovnaním a zmena sa zaznamená v `plan.json`. Referencie, návratové kontroly a fyzické kontrolné rotácie zostanú mimo výberu kandidátov. Finálne údaje všetkých metód prejdú tým istým odhadovačom; online posteriory sa vykážu oddelene. Primárne kontrasty sú C–D/E (výber) a B–D/E (celý postup), pri rovnakom rozpočte aj pri rovnakej validovanej presnosti.

## Meracia cesta a neoverené podmienky

Primárna cesta je `PHYSICAL_LAYER_EXPERIMENT` s `compute_type=0` cez existujúce funkčné pripojenie. Z každého skutočného tasku sa uchová odoslaný payload, stav, metadáta a úplné spárované `fidRe + 1j*fidIm`; vendor `fft*`, `fftFit`, matrix/fidelity a serverové frekvenčné odhady ostanú len v oddelenej sekundárnej referencii. Výstup je **exportovaný komplexný FID**, nie preukázane RAW ADC. Zmena vendor fitu pri nezmenenom FID nemá zmeniť lokálny odhad.

Lokálny SDK 1.0.2 dokazuje formát požiadavky a udalostí. **Nedokazuje** skutočné časovanie viacerých pulzných segmentov, funkciu `amplitude=0` ako delay, dostupnosť viacspinového Hamiltoniánu, stabilitu prijímacej fázy ani vypnutie serverového spracovania. Existujúci benchmark hlásil `UNVERIFIED_TIMING` pre viacsegmentové H sekvencie; Experiment 01 preto nemôže označiť Ramsey výsledok za vykonaný bez samostatnej časovej kontroly. Ak táto kontrola neprejde, ostane frekvenčná vetva založená na FID a jej presnosť sa bude testovať len vo svojom deklarovanom rozsahu.

Pred živým porovnaním treba nezávislým pilotom overiť frekvenčné zložky, Rabiho periodu a fázový gauge. Staré fitové hranice ±20/200 Hz a predchádzajúce `t90≈38 µs` sú historické pozorovania, nie fyzikálne konštanty či hotový prior. Príprava stavu, akvizícia, relaxačné intervaly, amplitúdy, lock a čerstvosť základnej kalibrácie musia byť spoločné, zaznamenané a v rámci empiricky overeného režimu. Trvalé nastavenia zariadenia sa nemenia. Počet vzoriek FID a predpokladaný počet interných opakovaní sa nesmú zamieňať za počet fyzických akvizícií.

Ak sa model, fáza, frekvenčná zložka alebo časovanie nedajú nezávisle potvrdiť, report musí uviesť `REFERENCE_INADEQUATE`, `UNVERIFIED_TIMING` alebo konkrétnu chybu, nie výhru metódy. Neistota referencie a neistota odhadu sú samostatné. Záver o úspore akvizícií alebo času vznikne až zo skutočných Windows blokov a kontrolných sekvencií; nie z macOS numerických testov ani z publikovaných iónových výsledkov.
