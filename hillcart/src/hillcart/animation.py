"""Animacija trajektorije (matplotlib). Shrani GIF (Pillow) ali MP4 (če je nameščen ffmpeg)."""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from .simulator import Outcome  # noqa: E402


def rollout(sim, policy, obs0):
    """Vrne seznam (t, stanje, opazovanje, F) in končni izid."""
    obs = sim.reset(obs0)
    frames = [(0.0, sim.state, obs, 0.0)]
    while sim.outcome is Outcome.RUNNING:
        F = policy(obs)
        obs, _ = sim.step(F)
        frames.append((sim.t, sim.state, obs, F))
    return frames, sim.outcome


def animate(sim, frames, outcome, path: str, fps: int = 25, stride: int = 2) -> None:
    track, prm = sim.track, sim.prm
    W = track.x_goal
    L = 2 * prm.l
    xs = np.linspace(-1.1 * W, 1.1 * W, 400)
    ys = np.array([track.h(x) for x in xs])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, "k-", lw=1.5)
    ax.axvline(W, color="g", ls="--", lw=1, label="cilj x = W")
    ax.axvline(-W, color="r", ls="--", lw=1, label="neuspeh x = -W")
    ax.set_aspect("equal")
    ax.set_xlim(-1.15 * W, 1.15 * W)
    ax.set_ylim(min(ys) - 0.3, max(ys) + L + 0.3)
    ax.legend(loc="lower center", fontsize=8)
    cart, = ax.plot([], [], "s", ms=10, color="tab:blue")
    pole, = ax.plot([], [], "-", lw=3, color="tab:orange")
    normal, = ax.plot([], [], ":", lw=1, color="gray")
    txt = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=9, family="monospace")
    sel = frames[::stride]
    if sel[-1] is not frames[-1]:
        sel.append(frames[-1])

    def draw(i):
        t, state, obs, F = sel[i]
        x, _, psi, _ = state
        y = track.h(x)
        cart.set_data([x], [y])
        pole.set_data([x, x + L * math.sin(psi)], [y, y + L * math.cos(psi)])
        phi = math.atan(track.dh(x))
        normal.set_data([x, x - L * math.sin(phi)], [y, y + L * math.cos(phi)])
        end = f"  -> {outcome.value}" if i == len(sel) - 1 else ""
        txt.set_text(f"t={t:5.2f}s  F={F:+6.2f}N  theta={math.degrees(obs[2]):+6.1f}deg{end}")
        return cart, pole, normal, txt

    anim = FuncAnimation(fig, draw, frames=len(sel), interval=1000 / fps, blit=True)
    if path.endswith(".gif"):
        anim.save(path, writer=PillowWriter(fps=fps))
    else:
        anim.save(path, fps=fps)
    plt.close(fig)
