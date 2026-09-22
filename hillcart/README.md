# hillcart – voziček s palico v kotanji

Simulator in (kasneje) RL eksperimenti za diplomsko nalogo
*»Učenje problema voziček/palica v kotanji: izvedba in primerjava različnih pristopov učenja«*.

Trenutna faza: **simulator + fizikalna validacija (T1–T12)**. RL metode še niso vključene.

## Namestitev (Windows, PowerShell)

```powershell
cd hillcart
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pip freeze > requirements-lock.txt   # točne verzije za ponovljivost; commitaj
```

## Git – prvi zagon

```powershell
git init -b main
git add -A
git commit -m "Simulator v0.1: model E1/E2, RK4, terminacija, testi T1-T12, beleženje"
# oddaljeni repozitorij (GitHub/GitLab), nato:
git remote add origin <URL>
git push -u origin main
```

Pravila:
- **Vsak eksperiment teče iz commitane kode.** Skripte privzeto zavrnejo zagon, če je koda
  necommitana (`--allow-dirty` samo za razvoj; zastavica se zapiše v `meta.json` in ime mape).
- `runs/` ni v gitu. Vsak zagon v `meta.json` hrani commit hash, zato je rezultat vedno
  povezljiv s točno verzijo kode.
- Mejnike označi z oznako, npr. po uspešni validaciji: `git tag sim-v1.0 -m "validiran simulator"`.
  Rezultate v diplomi navajaj z oznako/hashem.

## Validacija

```powershell
python -m pytest -q                      # T1-T12 + testi pravil epizode
python scripts/run_validation.py         # isto + poročilo v runs/ (red RK4, graf drifta energije)
```

## Eksperimentalni načrt

```powershell
python scripts/make_plan.py     # -> configs/plan_v1.json (+ dnevnik v runs/)
git add configs/plan_v1.json
git commit -m "Načrt v1"
```
`plan_v1.json` je VHOD za vse RL eksperimente (konfiguracije, sile, T_max, meje za tile coding
po konfiguraciji, začetna stanja, semena, faze, metrike). V `generated_by.git` je zapisan commit,
iz katerega je bil načrt ustvarjen. Diagnostične politike (`random`, `pumping`) niso metode diplome.

```powershell
python scripts/animate.py --policy pumping --kappa 0.5 --no-pole-check --out pump.gif
```

## Delovni postopek (vsakič)

1. Spremeni kodo → `python -m pytest -q` → `git add -A` → `git commit -m "..."`.
2. Zaženi skripto (NE z `--allow-dirty`). Rezultat gre v `runs/<datum>_<ime>_<hash>/`.
3. Če skripta ustvari datoteko, ki je vhod za nadaljnje delo (npr. `configs/plan_v1.json`), jo commitaj.
4. `runs/` ni v gitu → varnostno kopiraj (npr. OneDrive/zunanji disk).
5. Mejnik: `git tag plan-v1 -m "..."`, `git push --tags`.

## TensorBoard

```powershell
tensorboard --logdir runs
```
Odpri http://localhost:6006. Vse skalarne metrike so tudi v `runs/<zagon>/scalars.csv`.

## Struktura mape zagona `runs/<datum>_<ime>_<hash>/`

| Datoteka | Vsebina |
|---|---|
| `meta.json` | git commit/branch/dirty, čas, Python, platforma, verzije paketov, seme, ukaz |
| `config.json` | vsi parametri zagona |
| `scalars.csv` | metrike (tag, step, value, wall_time) |
| `events.out.tfevents.*` | TensorBoard |
| `summary.json` | povzetek |

## Parametri, ki še niso dokončni (empirično preveriti)

| Parameter | Trenutna vrednost | Kje |
|---|---|---|
| `t_max` | 40 s v `plan_v1.json` (pravilo 2 x max t_pump); privzeto v `SimConfig` še 10 s | preveri v pilotu |
| meje `x_dot`, `theta_dot` | po konfiguraciji v `plan_v1.json` | preveri v pilotu (delež stanj zunaj meja) |
| šum začetnih stanj | ±0.05 (SI) | `initial_states.TRAIN_NOISE` |
| kotanje, `kappa` | načrt v1 | `experiment_grid.py` |

## Model (povzetek)

Stanje integratorja `(x, v, psi, om)`, opazovanje `(x, x_dot, theta, theta_dot)`, `theta = psi + atan h'(x)`.

```
(E1)  M a x'' + m_p l B psi'' = F - M h' h'' v^2 + m_p l C om^2 - M g h'
(E2)  B x'' + (4/3) l psi''   = (g + h'' v^2) sin(psi)
a = 1 + h'^2,  B = cos psi - h' sin psi,  C = sin psi + h' cos psi
```
Pri `h = 0` se reducira v Florian (2007), en. (23)–(24).
