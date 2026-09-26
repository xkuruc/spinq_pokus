# Windows: pôvodná prieskumná séria

Nový päťtematický benchmark používa `run_windows.cmd`; pozri
[BENCHMARK_README.md](BENCHMARK_README.md). Tento návod ostáva pre starší
`spinq_live_suite.py`.

V PowerShelli v repozitári spusti `git pull` a potom
`.\.venv\Scripts\python.exe .\spinq_live_suite.py --config .\live_suite.example.toml`.
Použije existujúce `.venv`, tablet na `172.19.20.100:8181` a demo
prihlásenie `anyword` z funkčného `spinq_lab_control.py`. Ak používaš iné
heslo, nastav pred spustením premennú `SPINQ_AUDIT_PASSWORD`; skript sa
na nič nepýta. Žiadny Python ani SDK automaticky neaktualizuje.

Každý test sa zaznamená ako vykonaný, preskočený s dôvodom alebo neúspešný.
Neznáme kalibračné sekvencie, gradienty a trvalé zmeny sa neskúšajú.
Na Macu sa nič nemeria. Presný plán a podmienky sú v
`live_suite.example.toml` a `LIVE_WINDOWS.md`.

Po behu nájdeš `results.zip` v novom `results\DATUM_live_suite\`.
Program sa bez otázok pokúsi pushnúť iba výsledný ZIP do novej vetvy
`results/DATUM_UTC`. Ak ZIP presiahne 80 MiB, vo vetve budú jeho
číslované časti; celý ZIP zostane lokálne. Neúspešný push nevymaže dáta a
stav `UPLOAD_FAILED` zapíše do lokálneho `results.json` aj `REPORT.md`.
