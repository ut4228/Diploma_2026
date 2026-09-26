"""OQ-17: izmeri meje za tile coding kontrolnih nalog P2 in jih zapise v configs/p2_control.json.

Postopek je enak kot pri glavnem nacrtu (diagnostics.state_ranges + 25 % rezerve), da je
metodologija dosledna. Diagnosticne politike (nakljucna, crpanje) NISO metode diplome in
se ne uporabljajo pri ucenju.

Naloga P2(i)  crpanje brez omejitve palice: meje so prevzete iz configs/plan_v1.json.
Naloga P2(ii) ravni tir z omejitvijo palice: meje se IZMERIJO tukaj.

Uporaba:  python scripts/make_p2_config.py [--episodes 100] [--margin 0.25] [--allow-dirty]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from hillcart.diagnostics import bounds_from_ranges, state_ranges
from hillcart.env import ExperimentSpec
from hillcart.simulator import SimConfig
from hillcart.tracking import Run, git_info
from hillcart.tracks import FlatTrack

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "plan_v1.json"

P2I_CONFIG_ID = "H1_tan0.5_p2_kap0.5"
FLAT_HALF_LENGTH = 10.0
FLAT_F_MAX = 10.0  # Barto, Sutton & Anderson (1983): +-10 N pri m_c = 1.0, m_p = 0.1, l = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "configs" / "p2_control.json"))
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    spec = ExperimentSpec.from_plan(PLAN, P2I_CONFIG_ID)
    prm, sim_cfg = spec.params, spec.sim
    run = Run("make_p2_config", vars(args), base_dir=ROOT / "runs", seed=args.seed,
              require_clean=not args.allow_dirty, repo_dir=ROOT)

    # --- P2(ii): izmeri meje na ravnem tiru
    flat = FlatTrack(half_length=FLAT_HALF_LENGTH)
    ranges = state_ranges(flat, FLAT_F_MAX, sim_cfg, args.episodes, args.seed, prm)

    # ODSTOPANJE OD POSTOPKA GLAVNEGA NACRTA, z razlogom (glej "bounds_source" spodaj):
    # vir "pumping_nopole" je namenjen zajemu hitrosti, potrebnih za CRPANJE ENERGIJE iz kotanje.
    # Na ravnem tiru crpanja ni - hevristika samo enakomerno pospesuje in doseze ~13.8 m/s, kar
    # zahteva trajen polni potisk (theta_eq = 42.9 stopinj) in je pod pogojem |theta| <= 15 stopinj
    # nedosegljivo. Zato se za ravni tir meje dolocijo SAMO iz virov s palico (random, pumping_pole).
    # Odlocitev je fizikalna in sprejeta pred ucenjem, ne na podlagi rezultatov ucenja.
    ax = lambda a: max(abs(a[0]), abs(a[1]))
    with_pole = ("random", "pumping_pole")
    m_xd = max(ax(ranges[s_]["x_dot"]) for s_ in with_pole) * (1 + args.margin)
    m_td = max(ax(ranges[s_]["theta_dot"]) for s_ in with_pole) * (1 + args.margin)
    b_all = bounds_from_ranges(ranges, args.margin)  # zabelezimo tudi neuporabljeno varianto
    flat_bounds = {"x": [-FLAT_HALF_LENGTH, FLAT_HALF_LENGTH],
                   "theta": [-sim_cfg.theta_max, sim_cfg.theta_max],
                   "x_dot": [-m_xd, m_xd], "theta_dot": [-m_td, m_td], "margin": args.margin}

    # --- P2(i): meje prevzete iz nacrta
    valley_bounds = {"x": [-spec.valley.W, spec.valley.W],
                     "theta": [-sim_cfg.theta_max, sim_cfg.theta_max],
                     "x_dot": spec.tile_bounds["x_dot"], "theta_dot": spec.tile_bounds["theta_dot"],
                     "margin": spec.tile_bounds.get("margin")}

    cfg = {
        "version": "p2_control_v1",
        "generated_by": {"script": "scripts/make_p2_config.py", "git": git_info(ROOT), "args": vars(args),
                         "run_dir": run.dir.name},
        "plan_ref": {"path": "configs/plan_v1.json", "sha256": spec.plan_sha256, "version": spec.plan_version},
        "params": prm.describe(),
        "sim": {"dt": sim_cfg.dt, "n_substeps": sim_cfg.n_substeps, "theta_max_deg": sim_cfg.theta_max_deg,
                "t_max": sim_cfg.t_max, "max_steps": sim_cfg.max_steps, "integrator": sim_cfg.integrator},
        "tasks": {
            "P2i_pump_no_pole": {
                "purpose": "ali se agent nauci DOSECI cilj s crpanjem energije",
                "track": {"type": "PowerValley", "config_id": P2I_CONFIG_ID, **spec.valley.describe()},
                "F_max": spec.F_max, "forces": spec.forces, "check_pole": False,
                "tile_bounds": valley_bounds, "bounds_source": "configs/plan_v1.json (nespremenjeno)",
                "known_reference": spec.plan_version,
                "note": ("palica se prosto vrti, zato sta theta in theta_dot v tile codingu degenerirana "
                         "(trajno omejena na robno plosico); kriterij 'delez zunaj meja < 1 %' za to nalogo NE velja"),
            },
            "P2ii_flat_balance": {
                "purpose": "ali se agent nauci OHRANJATI ravnotezje in se hkrati premikati",
                "track": {"type": "FlatTrack", "half_length": FLAT_HALF_LENGTH},
                "F_max": FLAT_F_MAX, "check_pole": True,
                "u_ratio": FLAT_F_MAX / (prm.M * prm.g),
                "theta_eq_deg": math.degrees(math.atan(FLAT_F_MAX / (prm.M * prm.g))),
                "tile_bounds": flat_bounds,
                "bounds_source": ("izmerjeno (OQ-17) SAMO iz virov s palico (random, pumping_pole); "
                                  "vir pumping_nopole je na ravnem tiru nesmiseln - da ~13.8 m/s, kar "
                                  "zahteva trajen polni potisk in je pod |theta| <= 15 stopinj nedosegljivo"),
                "tile_bounds_if_all_sources": {"x_dot": b_all["x_dot"], "theta_dot": b_all["theta_dot"]},
                "state_ranges": ranges,
                "note": ("stalni polni potisk je nemogoc, ker theta_eq >> 15 stopinj; "
                         "F_max = 10 N je sidro na Barto, Sutton & Anderson (1983)"),
            },
        },
        "agent": {"algorithm": "true_online_sarsa_lambda", "alpha0": 0.1, "lambda": 0.9, "gamma": 0.999,
                  "epsilon": 0.05, "w_init": 0.0, "eps_z": 1e-6,
                  "n_tilings": 16, "n_intervals": 8, "offsets": [1, 3, 5, 7]},
        "protocol": {"seeds": [100, 101, 102], "total_steps": 1_000_000, "eval_every": 50_000,
                     "criterion": "koncni delez uspeha >= 0.80 pri vsaj 2 od 3 semen",
                     "fallbacks": ["alpha0 0.1 -> 0.5", "optimisticna inicializacija w_i = 1/16 (q0 = +1)"]},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    run.save_json("p2_control.json", cfg)
    run.close({"flat_bounds": flat_bounds, "valley_bounds": valley_bounds})

    print("P2(ii) ravni tir, izmerjene meje (rezerva %d %%):" % int(args.margin * 100))
    for k in ("x_dot", "theta_dot"):
        print(f"  {k:<10} [{flat_bounds[k][0]:+7.3f}, {flat_bounds[k][1]:+7.3f}]")
    print(f"  u = F_max/(M g) = {cfg['tasks']['P2ii_flat_balance']['u_ratio']:.3f}"
          f"  ->  theta_eq = {cfg['tasks']['P2ii_flat_balance']['theta_eq_deg']:.1f} stopinj (>> 15)")
    print("P2(i)  meje iz nacrta: x_dot [%.2f, %.2f], theta_dot [%.2f, %.2f]"
          % (*valley_bounds["x_dot"], *valley_bounds["theta_dot"]))
    print("konfiguracija:", out)
    print("dnevnik:", run.dir)


if __name__ == "__main__":
    main()
