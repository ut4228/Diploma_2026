"""Staticna slika ene uspesne epizode naucene politike: lega in energija skozi cas.

Namenjeno diplomi, kjer animacija ni uporabna. Pokaze mehanizem resitve: voziceka
se zaganja sem in tja, pri cemer energija narasca po stopnicah, dokler ne zadosca
za izhod. Spodnji trak pokaze, v katero smer motor potiska.

Bere isto kot animate_policy.py -- mapo zagona z weights_final.npz in config.json.

Uporaba:
  python scripts/plot_trajektorija.py --run runs/<mapa> --best
  python scripts/plot_trajektorija.py --run runs/<mapa> --state 0 0 0 0 --out runs/traj.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hillcart.agent import AgentConfig, TrueOnlineSarsaLambda  # noqa: E402
from hillcart.animation import rollout  # noqa: E402
from hillcart.dynamics import energy  # noqa: E402
from hillcart.env import EnvOutcome, ExperimentSpec, HillCartEnv  # noqa: E402
from hillcart.initial_states import eval_states  # noqa: E402
from hillcart.tile_coding import coder_from_bounds  # noqa: E402
from hillcart.tracks import FlatTrack, PowerValley  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LEGA = "#2a78d6"
ENERGIJA = "#d1600f"
NEVTRAL = "#c3c2b7"


def build(run_dir: Path):
    """Sestavi okolje in agenta iz zapisane konfiguracije zagona."""
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    w = np.load(run_dir / "weights_final.npz")["w"]
    bounds = cfg.get("tile_bounds") or cfg["task_spec"]["tile_bounds"]
    ag = cfg["agent"]
    coder = coder_from_bounds(bounds, n_tilings=ag["tile_coding"]["n_tilings"],
                              n_intervals=ag["tile_coding"]["n_intervals"])
    if w.size != coder.n_features:
        raise SystemExit(f"utezi ({w.size}) se ne ujemajo s tile codingom "
                         f"({coder.n_features}) - napacna mapa zagona?")
    agent = TrueOnlineSarsaLambda(coder, AgentConfig(epsilon=0.0), seed=0)
    agent.w = w

    plan = cfg.get("plan", {}).get("path", "plan_v2.json")
    tr = cfg["env"]["track"]
    if "config_id" in cfg:
        spec = ExperimentSpec.from_plan(ROOT / "configs" / plan, cfg["config_id"])
        track = spec.valley
    else:
        spec = ExperimentSpec.from_plan(ROOT / "configs" / "plan_v1.json",
                                        cfg["task_spec"]["track"].get("config_id"))
        track = (FlatTrack(half_length=tr["half_length"]) if tr["type"] == "FlatTrack"
                 else PowerValley(H=tr["H"], tan_phi_max=tr["tan_phi_max"], p=tr["p"]))
    env = HillCartEnv(spec, check_pole=cfg["env"]["check_pole"], track=track,
                      F_max=cfg["env"]["F_max"], tile_bounds=bounds)
    return cfg, env, agent


def izberi_najkrajso(env, agent):
    najboljse = None
    for s0 in eval_states():
        obs = env.reset(s0)
        while env.outcome is EnvOutcome.RUNNING:
            obs, *_ = env.step(agent.act(obs, greedy=True))
        if env.outcome is EnvOutcome.SUCCESS and (najboljse is None
                                                  or env.t_event < najboljse[1]):
            najboljse = (s0, env.t_event)
    return najboljse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--hevristika", action="store_true",
                    help="namesto naucene politike uporabi diagnosticno hevristiko "
                         "za zaganjanje (potisni v smeri hitrosti). NI metoda diplome; "
                         "sluzi preizkusu izrisa in primerjalni sliki")
    ap.add_argument("--state", type=float, nargs=4, default=None,
                    metavar=("X", "XD", "TH", "THD"))
    ap.add_argument("--best", action="store_true",
                    help="izberi zacetno stanje z najkrajso uspesno epizodo")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg, env, agent = build(run_dir)

    if args.hevristika:
        from hillcart.diagnostic_policies import pumping_policy
        pol = pumping_policy(env.F_max)
        agent = type("Hevristika", (), {
            "act": staticmethod(lambda obs, greedy=True:
                                int(np.argmin(np.abs(np.array(env.forces) - pol(obs)))))
        })()

    if args.state:
        s0 = tuple(args.state)
    elif args.best:
        naj = izberi_najkrajso(env, agent)
        if naj is None:
            raise SystemExit("nobena epizoda ni uspesna; izberi drug zagon ali --state")
        s0 = naj[0]
        print(f"najkrajsa uspesna epizoda: s0={s0}, t={naj[1]:.2f} s")
    else:
        s0 = (0.0, 0.0, 0.0, 0.0)

    sim = env._sim
    frames, izid = rollout(sim, lambda o: env.forces[agent.act(o, greedy=True)], s0)

    t = np.array([f[0] for f in frames])
    x = np.array([f[2][0] for f in frames])
    F = np.array([f[3] for f in frames])
    E = np.array([energy(f[1], sim.track, sim.prm) for f in frames])

    track = sim.track
    W = track.x_goal
    M = sim.prm.m_c + sim.prm.m_p
    H = getattr(track, "H", None)
    E0 = energy((0.0, 0.0, 0.0, 0.0), track, sim.prm)
    phi = (E - E0) / (M * sim.prm.g * H) if H else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 5.8), facecolor=SURFACE,
                                   sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1.0],
                                                "hspace": 0.16})

    # ---- zgoraj: lega vozicka
    for s in (-1, 1):
        ax1.axhline(s * W, color=AXIS, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax1.text(t[0], W, "rob kotanje", fontsize=8.5, color=INK_2,
             va="bottom", ha="left")
    ax1.plot(t, x, color=LEGA, lw=2.0, zorder=3)
    ax1.plot([t[-1]], [x[-1]], "o", ms=8, color=LEGA, mec=SURFACE, mew=2, zorder=4)
    ax1.set_ylabel("lega $x$ [m]", fontsize=9, color=INK_2)
    ax1.set_ylim(-W * 1.18, W * 1.18)

    # ---- spodaj: energija (oziroma potencial)
    if phi is not None:
        ax2.axhline(1.0, color=AXIS, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax2.text(t[0], 1.0, "energija, potrebna za izhod", fontsize=8.5, color=INK_2,
                 va="bottom", ha="left")
        ax2.plot(t, phi, color=ENERGIJA, lw=2.0, zorder=3)
        ax2.set_ylabel(r"$\Phi_E = (E - E_0)/MgH$", fontsize=9, color=INK_2)
        ax2.set_ylim(-0.06, max(1.15, float(phi.max()) * 1.08))
    else:
        ax2.plot(t, E, color=ENERGIJA, lw=2.0, zorder=3)
        ax2.set_ylabel("mehanska energija [J]", fontsize=9, color=INK_2)

    # ---- trak s smerjo potiska, pod spodnjo ploscjo
    ymin = ax2.get_ylim()[0]
    vis = (ax2.get_ylim()[1] - ymin) * 0.055
    for znak, barva in ((+1, LEGA), (-1, ENERGIJA), (0, NEVTRAL)):
        m = np.sign(F) == znak
        ax2.fill_between(t, ymin, ymin + vis, where=m, color=barva, alpha=0.55,
                         step="mid", lw=0, zorder=2)
    ax2.text(t[0], ymin + vis * 1.5, "smer potiska", fontsize=8, color=INK_2,
             va="bottom")

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.set_xlim(t[0], t[-1] * 1.04)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}".replace("-", "−").replace(".", ",")))
    ax2.set_xlabel("čas [s]", fontsize=9, color=INK_2)

    oznaka = cfg.get("config_id", cfg.get("task", ""))
    kap = cfg.get("env", {}).get("spec", {}).get("forces", {}).get("kappa")
    kdo = "diagnostične hevristike" if args.hevristika else "naučene politike"
    naslov = f"Potek epizode {kdo}"
    if kap is not None and not args.hevristika:
        naslov += f"  ($\\kappa$ = {kap:g})".replace(".", ",")
    fig.suptitle(naslov, fontsize=12.5, color=INK, x=0.012, ha="left", y=0.975)

    IZID = {"success": "uspeh", "fail_pole": "padec palice", "timeout": "prekinitev"}
    noga = (f"{oznaka} · {'hevristika' if args.hevristika else 'požrešna politika'}"
            f" · izid: {IZID.get(izid.value, izid.value)} pri "
            f"t = {sim.t_event:.2f} s · začetno stanje "
            f"({', '.join(f'{v:g}' for v in s0)})").replace(".", ",")
    fig.text(0.012, 0.02, noga, fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.055, 1, 0.935))

    out = Path(args.out) if args.out else ROOT / "runs" / f"trajektorija_{run_dir.name}.png"
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    print("shranjeno:", out)
    if out.suffix.lower() != ".pdf":
        pdf = out.with_suffix(".pdf")
        fig.savefig(pdf, facecolor=SURFACE)
        print("shranjeno:", pdf, " (za diplomo uporabi tega)")
    print(f"izid: {izid.value}, t = {sim.t_event:.2f} s, "
          f"najvecji |x| = {np.abs(x).max():.2f} m")


if __name__ == "__main__":
    main()
