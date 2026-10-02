"""Ponovna kalibracija meja za tile coding po fazi A - za konfiguracije, ki krsijo kriterij 5.

ZAKAJ
  Meje v plan_v2.json so bile izmerjene pred ucenjem, z diagnosticnimi hevristikami. Naucena
  politika pri kappa = 1.2 dosega vecje hitrosti, zato je delez korakov ucenja zunaj meja
  0.094 namesto zahtevanih < 0.01 (kriterij 5). Stanja zunaj meja se porezejo na rob, kar
  pomeni, da aproksimacija dveh razlicnih stanj ne loci.

ZAKAJ NE SPREMENIMO plan_v2.json
  plan_v2.json je ZAMRZNJEN in oznacen (git tag phase-a-v1). Spreminjanje bi unicilo
  pre-registracijo. Ta skripta zato zapise NOV nacrt (privzeto plan_v3.json), ki je kopija
  v2 z razsirjenimi mejami samo za navedene konfiguracije, in zabelezi, od kod nova stevilka.
  Prvotni rezultat ostane viden; v diplomi se porocata oba.

KAKO SE MERI
  Kriterij 5 je bil krsen med UCENJEM, ne med vrednotenjem, zato zgolj pozresna politika na
  81 vrednotitvenih stanjih ne zadosca. Merimo oboje in vzamemo vecje:
    (a) pozresno (epsilon = 0) na vseh 81 fiksnih zacetnih stanjih,
    (b) z epsilon = 0.05 (kot med ucenjem) iz nakljucnih ucnih zacetnih stanj, N epizod.
  To je priblizek ucne porazdelitve, ne njena natancna rekonstrukcija - zapisano je v izhodu.

PRAVILO (nasa metodoloska odlocitev)
  nova meja = max(stara meja, izmerjeni |max| * (1 + rezerva)),  rezerva privzeto 0.25.
  Meje se samo SIRIJO, nikoli ozijo. x in theta sta dolocena s pravili epizode in se ne sirita.

UPORABA
  python scripts/recalibrate_phase_a.py --configs H1_tan0.5_p2_kap1.2 --dry-run
  python scripts/recalibrate_phase_a.py --configs H1_tan0.5_p2_kap1.2
  (nato)  python scripts/run_phase_a.py --plan configs/plan_v3.json --configs H1_tan0.5_p2_kap1.2 \
              --no-resume --csv runs/phase_a_v2_summary.csv
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from hillcart.agent import AgentConfig, TrueOnlineSarsaLambda
from hillcart.env import EnvOutcome, ExperimentSpec, HillCartEnv
from hillcart.initial_states import eval_states, sample_train_state
from hillcart.tile_coding import coder_from_bounds
from hillcart.tracking import git_info

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "plan_v2.json"
DEFAULT_OUT = ROOT / "configs" / "plan_v3.json"
KEYS = ("x", "x_dot", "theta", "theta_dot")
WIDEN = ("x_dot", "theta_dot")  # x in theta sta dolocena s pravili epizode


def bounds_of(spec) -> dict:
    """Iste meje, kot jih sestavi run_phase_a.run_one."""
    return {"x": [-spec.valley.W, spec.valley.W],
            "theta": [-spec.sim.theta_max, spec.sim.theta_max],
            "x_dot": list(spec.tile_bounds["x_dot"]),
            "theta_dot": list(spec.tile_bounds["theta_dot"])}


def load_agent(run_dir: Path, coder, epsilon: float, seed: int) -> TrueOnlineSarsaLambda:
    w = np.load(run_dir / "weights_final.npz")["w"]
    if w.size != coder.n_features:
        raise SystemExit(f"{run_dir.name}: utezi ({w.size}) se ne ujemajo s tile codingom "
                         f"({coder.n_features}) - napacen zagon?")
    agent = TrueOnlineSarsaLambda(coder, AgentConfig(epsilon=epsilon), seed=seed)
    agent.w = w
    return agent


def sweep(env, agent, starts, greedy: bool) -> tuple:
    lo = np.full(4, np.inf)
    hi = np.full(4, -np.inf)
    n = 0
    for s0 in starts:
        obs = env.reset(s0)
        lo, hi = np.minimum(lo, obs), np.maximum(hi, obs)
        while env.outcome is EnvOutcome.RUNNING:
            obs, *_ = env.step(agent.act(obs, greedy=greedy))
            lo, hi = np.minimum(lo, obs), np.maximum(hi, obs)
        n += 1
    return lo, hi, n


def recalibrate(config_id: str, plan_path: Path, args) -> dict | None:
    spec = ExperimentSpec.from_plan(plan_path, config_id)
    bounds = bounds_of(spec)
    coder = coder_from_bounds(bounds, n_tilings=args.n_tilings, n_intervals=args.n_intervals)

    pattern = args.runs or str(ROOT / "runs" / f"*phaseA_{config_id}_s*")
    dirs = [Path(d) for d in sorted(glob.glob(pattern))
            if (Path(d) / "weights_final.npz").exists()]
    if not dirs:
        raise SystemExit(f"ni zagonov z utezmi: {pattern}")

    lo = np.full(4, np.inf)
    hi = np.full(4, -np.inf)
    n_greedy = n_explore = 0
    rng = np.random.default_rng(args.seed)
    for d in dirs:
        env = HillCartEnv(spec, check_pole=True, tile_bounds=bounds)

        agent = load_agent(d, coder, epsilon=0.0, seed=args.seed)
        l, h, n = sweep(env, agent, eval_states(), greedy=True)
        lo, hi, n_greedy = np.minimum(lo, l), np.maximum(hi, h), n_greedy + n

        if args.episodes > 0:
            agent = load_agent(d, coder, epsilon=args.epsilon, seed=args.seed)
            starts = [sample_train_state(rng, spec.train_noise) for _ in range(args.episodes)]
            l, h, n = sweep(env, agent, starts, greedy=False)
            lo, hi, n_explore = np.minimum(lo, l), np.maximum(hi, h), n_explore + n

    new_bounds = {k: list(v) for k, v in bounds.items()}
    changed = {}
    for i, k in enumerate(KEYS):
        if k not in WIDEN:
            continue
        seen = float(max(abs(lo[i]), abs(hi[i])))
        need = seen * (1.0 + args.margin)
        old = float(bounds[k][1])
        if need > old:
            new_bounds[k] = [-need, need]
            changed[k] = {"stara": old, "nova": need, "izmerjeni_absmax": seen,
                          "rezerva": args.margin}

    print(f"\n=== {config_id} ===")
    print(f"zagonov: {len(dirs)}   epizod pozresno: {n_greedy}   epizod z raziskovanjem: {n_explore}")
    for i, k in enumerate(KEYS):
        mark = "   <- razsirjeno" if k in changed else ("" if k in WIDEN else "   (fiksno)")
        print(f"  {k:<10} izmerjeno [{lo[i]:+8.3f}, {hi[i]:+8.3f}]   meja "
              f"[{new_bounds[k][0]:+8.3f}, {new_bounds[k][1]:+8.3f}]{mark}")
    if not changed:
        print("  nobene meje ni treba razsiriti.")
        return None
    return {"config_id": config_id, "tile_bounds": new_bounds, "tile_bounds_previous": bounds,
            "changed": changed, "runs": [d.name for d in dirs],
            "episodes": {"greedy": n_greedy, "exploring": n_explore, "epsilon": args.epsilon},
            "measured": {k: [float(lo[i]), float(hi[i])] for i, k in enumerate(KEYS)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, help="id-ji konfiguracij iz nacrta")
    ap.add_argument("--plan", default=str(DEFAULT_PLAN), help="vhodni (zamrznjeni) nacrt")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="izhodni nacrt (nov, ne prepisuj v2)")
    ap.add_argument("--runs", default=None, help="glob za mape zagonov (privzeto runs/*phaseA_<id>_s*)")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--episodes", type=int, default=200,
                    help="epizod z raziskovanjem na zagon (0 = samo pozresno)")
    ap.add_argument("--epsilon", type=float, default=0.05, help="epsilon kot med ucenjem")
    ap.add_argument("--n-tilings", type=int, default=16)
    ap.add_argument("--n-intervals", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan_path = Path(args.plan)
    out_path = Path(args.out)
    if out_path.resolve() == plan_path.resolve():
        raise SystemExit("izhod ne sme biti isti kot vhod: zamrznjenega nacrta ne prepisujemo")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    results = [r for cid in args.configs if (r := recalibrate(cid, plan_path, args)) is not None]

    if not results:
        print("\nnobena konfiguracija ne potrebuje sirsih meja; nov nacrt ni zapisan.")
        return
    if args.dry_run:
        print("\n--dry-run: nov nacrt ni zapisan.")
        return
    if out_path.exists():
        raise SystemExit(f"{out_path} ze obstaja - izbrisi ga ali podaj drug --out")

    by_id = {r["config_id"]: r for r in results}
    for row in plan["configs"]:
        r = by_id.get(row["id"])
        if r is None:
            continue
        # ohrani vse kljuce, kot so v nacrtu (tudi skalarne, npr. "margin");
        # spremeni samo tiste meje, ki so se razsirile
        row["tile_bounds_previous"] = {k: (list(v) if isinstance(v, list) else v)
                                       for k, v in row["tile_bounds"].items()}
        for k in r["changed"]:
            row["tile_bounds"][k] = r["tile_bounds"][k]
        row["tile_bounds"]["recalibration_margin"] = args.margin

    plan["version"] = "plan_v3"
    plan["recalibration"] = {
        "script": "scripts/recalibrate_phase_a.py",
        "razlog": "kriterij 5 (delez korakov zunaj meja < 0.01) je bil krsen pri navedenih konfiguracijah",
        "izhodisce": {"plan": plan_path.name, "oznaka": "phase-a-v1"},
        "pravilo": "nova meja = max(stara, izmerjeni |max| * (1 + rezerva)); samo sirjenje; x in theta fiksna",
        "merjenje": "pozresno na 81 vrednotitvenih stanjih + epsilon = 0.05 iz nakljucnih ucnih "
                    "zacetnih stanj; priblizek ucne porazdelitve, ne njena rekonstrukcija",
        "rezerva": args.margin,
        "git": git_info(ROOT),
        "konfiguracije": {r["config_id"]: {k: r[k] for k in
                                           ("changed", "measured", "episodes", "runs")}
                          for r in results},
    }
    out_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nzapisano: {out_path}   (verzija {plan['version']})")
    print("zamrznjeni plan_v2.json NI spremenjen.")
    print("\nnaslednji korak - ponovi prizadete zagone proti novemu nacrtu:")
    ids = " ".join(r["config_id"] for r in results)
    print(f"  python scripts/run_phase_a.py --plan {out_path.as_posix()} --configs {ids} "
          f"--no-resume --csv runs/phase_a_v2_summary.csv")
    print("  git add -A && git commit -m \"Rekalibracija meja po fazi A (plan_v3)\"")
    print("  git tag phase-a-v2 -m \"Faza A: ponovitev s popravljenimi mejami\"")


if __name__ == "__main__":
    main()
