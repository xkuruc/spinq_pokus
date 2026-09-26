# Primárne zdroje a úloha v projekte

Odkazy boli preverené ako oficiálna dokumentácia, verejný klientsky zdrojový kód alebo pôvodné práce. Ak plný text vydavateľa vyžaduje prihlásenie, používame verejne dostupnú autorskú verziu uvedenú vedľa neho. Odkazy nie sú dôkazom, že konkrétny hardvér podporuje každú navrhnutú sekvenciu.

## SpinQ rozhranie

- **S1:** [SpinQLabLink — fyzikálna vrstva](https://doc.spinq.cn/doc/SpinQLAB_Link/en/experiment/Research_Experiment.html). Definície vlastného merania, parametrov a FID/FFT grafov; správanie sa porovnáva s lokálnym SDK 1.0.2.
- **S2:** [Klient `exp_layer_physical.py`](https://github.com/SpinQTech/spinqlablink/blob/main/spinqlablink/experiment/exp_layer_physical.py). Serializácia `compute_type`, `makePps`, H/P pulzov a výsledkové callbacky.
- **S3:** [Klientsky protokol](https://github.com/SpinQTech/spinqlablink/blob/main/spinqlablink/connection/protocol.py) a [schéma správy](https://github.com/SpinQTech/spinqlablink/blob/main/spinqlablink/connection/message.proto). `task/group/chart/path/qubit/step` a numerické `float` body.
- **S4:** [API SpinQLabLink](https://doc.spinq.cn/doc/SpinQLAB_Link/en/api/spinqlablink.html), [oficiálny repozitár](https://github.com/SpinQTech/spinqlablink) a [PyPI balík](https://pypi.org/project/spinqlablink/). Rozhranie a identifikácia klienta; rozhodujúci je skutočne nainštalovaný zdroj.
- **S5:** [SpinQ: integrovaná kvantová technológia a echo príklady](https://doc.spinq.cn/doc/SpinQLAB_Link/en/experiment/Integrated_Quantum_Technology.html), [klient `exp_spinecho.py`](https://github.com/SpinQTech/spinqlablink/blob/main/spinqlablink/experiment/exp_spinecho.py) a [pomocné funkcie `extension.py`](https://github.com/SpinQTech/spinqlablink/blob/main/spinqlablink/utils/extension.py). Príklad ani pulse-domain FFT nie sú potvrdením časovania vlastnej echo sekvencie alebo serverového FID FFT algoritmu.

## Metódy A–H

- **S6 / A:** Gerster et al., [*Experimental Bayesian calibration of trapped ion entangling operations*](https://arxiv.org/pdf/2112.01411), PRX Quantum 3, 020350 (2022). Časticové váhy, Liu–West resampling, stop podľa neistôt a adaptívny dizajn. Iónový likelihood sa nepoužíva na NMR.
- **S7 / B:** Beracha, Seginer, Tal, [*Adaptive model-based Magnetic Resonance*](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29688), Magnetic Resonance in Medicine 90, 839–851 (2023). Posterior amplitúdy/T₂, TE výber a statický CRLB návrh; MRI časy nie sú Gemini nastavenia.
- **S8 / C:** [*Seedless: on-the-fly pulse calculation for NMR experiments*](https://www.nature.com/articles/s41467-025-61663-8), Nature Communications (2025), [autorská stránka](http://seedless.chem.ox.ac.uk). Phase-only ensemble optimalizácia; nepreberá sa nedostupný autorský Python zdroj ani nekomerčná binárka.
- **S9 / C:** [QuTiP Qtrl: quantum optimal control](https://qutip.readthedocs.io/projects/qutip-qtrl/en/latest/guide/guide-control.html). Otvorená matematická referencia GRAPE, propagátorov a gradientov, nie Q-CTRL cloud.
- **S10 / D:** Youssry et al., [*Experimental graybox quantum system identification and control*](https://www.nature.com/articles/s41534-023-00795-5), npj Quantum Information (2024). Vrstvy fyzikálneho modelu a merania; konkrétna NMR reziduálna sieť je naša adaptácia.
- **S11 / E:** Khaleghian et al., [*Development of Neural Network-Based Optimal Control Pulse Generator for Quantum Logic Gates Using the GRAPE Algorithm in NMR Quantum Computer*](https://arxiv.org/html/2412.05856v3), Quantum Information Processing 25, 131 (2026). GRAPE labely a supervised generátor; bez autorových váh.
- **S12 / F:** Zou et al., [*TVCondNet: A Conditional Denoising Neural Network for NMR Spectroscopy*](https://arxiv.org/html/2405.11064v1) (2024). Real-spectrum vetva a TV kondicionovanie; komplexná fázu zachovávajúca vetva je samostatná adaptácia.
- **S13 / F:** Qiu et al., [*An auto-parameter denoising method for nuclear magnetic resonance spectroscopy based on low-rank Hankel matrix*](https://arxiv.org/abs/2001.11815). Klasická Hankel referencia, nie názov pre ľubovoľný jednoduchý filter.
- **S14 / F:** Lehtinen et al., [*Noise2Noise: Learning Image Restoration without Clean Data*](https://proceedings.mlr.press/v80/lehtinen18a.html), ICML (2018). Podmienky nezávislého šumu a delenia akvizícií.
- **S15 / G:** Mundada et al., [*Experimental Benchmarking of an Automated Deterministic Error-Suppression Workflow for Quantum Algorithms*](https://arxiv.org/pdf/2209.06864), Physical Review Applied 20, 024034 (2023), najmä Appendix C. Opis kompilácie, contextual DD, optimalizácie a korekcie merania; Fire Opal produkčný kód nie je k dispozícii.
- **S16 / H:** [Q-CTRL Boulder Opal: automatizovaná kalibrácia hardvéru](https://docs.q-ctrl.com/boulder-opal/toolkit/automate/automate-closed-loop-optimization/how-to-automate-calibration-of-control-hardware). Verejný notebook s Rabi mapou, `Samp/Srel`, opakovaniami a GP; tu je implementovaný lokálny GP, nie volanie `bo.cloud`.
- **S17 / závislosti:** [Oficiálna inštalácia PyTorch](https://pytorch.org/get-started/locally/). CPU Windows build a kontrola importu i gradientu pred neurónovou akvizíciou.

Poznámka: `message.proto` definuje float x/y a voliteľné `qubit`, čo nie je doklad, že pripojené Gemini Lab hardvérovo sprístupňuje plné ovládanie troch spinov. Miesto serverových FFT/fitu či kompletné vypnutie serverového spracovania verejné zdroje neurčujú; preto ich projekt nepredstiera.
