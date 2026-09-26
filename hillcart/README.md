# hillcart – voziček s palico v kotanji

Simulator in (kasneje) RL eksperimenti za diplomsko nalogo
_»Učenje problema voziček/palica v kotanji: izvedba in primerjava različnih pristopov učenja«_.

Trenutna faza: **simulator + fizikalna validacija (T1–T12)**. RL metode še niso vključene.

## Validacija

```powershell
python -m pytest -q                      # T1-T12 + testi pravil epizode
python scripts/run_validation.py         # isto + poročilo v runs/ (red RK4, graf drifta energije)
```

## RL okolje (faza P0)

`src/hillcart/env.py` je sloj nad validiranim simulatorjem. Fizike ne spreminja.

- Stanje: `(x, x_dot, theta, theta_dot)`, akcije `{0,1,2} -> {-F_max, 0, +F_max}`.
- **SUCCESS:** `|x| >= W` in `|theta| <= 15°` (cilj je dosegljiv na OBEH straneh kotanje).
  Simulator interno še vedno vrne `fail_left`; okolje ga preslika v `success` in stran zapiše
  v `info["exit_side"]`.
- **FAIL_POLE:** `|theta| > 15°`.
- **TIMEOUT:** prekinitev (truncation), `terminated=False, truncated=True`.
- Nagrada R-C: `+1` uspeh, `-1` padec, `0` sicer. Okolje ne pozna `gamma` (gamma = 0.999 je last agenta).
- `td_target(r, q_next, terminated, gamma)`: brez bootstrapa ob terminaciji, z bootstrapom ob prekinitvi.
- `linear_sarsa.py`: čisti posodobitvi (true online Sarsa(lambda) in Sarsa(0)); pri `lambda = 0` sta enaki (test P0-18).

```powershell
python -m pytest -q            # 56 testov: T1-T12 + P0-01..P0-25
python scripts/run_baselines.py   # kontrolni zagon H0 -> runs/
git tag env-v1 -m "P0 okolje"
```

## TensorBoard

```powershell
tensorboard --logdir runs
```

Odpri http://localhost:6006. Vse skalarne metrike so tudi v `runs/<zagon>/scalars.csv`.

## Struktura mape zagona `runs/<datum>_<ime>_<hash>/`

| Datoteka                | Vsebina                                                                      |
| ----------------------- | ---------------------------------------------------------------------------- |
| `meta.json`             | git commit/branch/dirty, čas, Python, platforma, verzije paketov, seme, ukaz |
| `config.json`           | vsi parametri zagona                                                         |
| `scalars.csv`           | metrike (tag, step, value, wall_time)                                        |
| `events.out.tfevents.*` | TensorBoard                                                                  |
| `summary.json`          | povzetek                                                                     |
