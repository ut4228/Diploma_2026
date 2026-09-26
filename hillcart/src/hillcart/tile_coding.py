"""Tile coding za stanje (x, x_dot, theta, theta_dot) in 3 diskretne akcije.

Specifikacija (S&B 2018, razd. 9.5.4, str. 217-221):
  - n_tilings je potenca stevila 2, vsaj 4k; za k = 4 dimenzij: n = 16,
  - asimetricni zamiki s prvimi lihimi stevili d = (1, 3, 5, 7),
  - 8 intervalov na dimenzijo -> 9 indeksov (zaradi zamika).

Normalizacija in indeksi:
  u_i      = clip((s_i - lo_i) / (hi_i - lo_i), 0, 1)
  offset_j,i = ((j * d_i) mod n_tilings) / n_tilings
  idx_j,i  = floor(u_i * 8 + offset_j,i)  v {0, ..., 8}

Modul n_tilings je nujen: brez njega bi j * d_i preseglo eno plosico in razbilo
zalogo indeksov. Ker je gcd(d_i, 16) = 1 za liha d_i, je pokritost zamikov enakomerna.

Stanje zunaj meja se omeji na rob (clipping) in se zabelezi (out_of_bounds).
Vsaka akcija ima svoj blok utezi; zgoscevanje (hashing) ni potrebno.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_OFFSETS = (1, 3, 5, 7)


@dataclass(frozen=True)
class TileCoderConfig:
    lows: tuple
    highs: tuple
    n_tilings: int = 16
    n_intervals: int = 8
    n_actions: int = 3
    offsets: tuple = DEFAULT_OFFSETS

    def describe(self) -> dict:
        return {"lows": list(self.lows), "highs": list(self.highs), "n_tilings": self.n_tilings,
                "n_intervals": self.n_intervals, "n_actions": self.n_actions, "offsets": list(self.offsets),
                "n_indices_per_dim": self.n_intervals + 1,
                "n_features_per_action": self.n_tilings * (self.n_intervals + 1) ** len(self.lows),
                "n_features_total": self.n_actions * self.n_tilings * (self.n_intervals + 1) ** len(self.lows),
                "out_of_bounds_rule": "clipping + stevec oob"}


class TileCoder:
    def __init__(self, cfg: TileCoderConfig):
        self.cfg = cfg
        self.k = len(cfg.lows)
        if len(cfg.highs) != self.k or len(cfg.offsets) != self.k:
            raise ValueError("lows, highs in offsets morajo imeti enako dolzino")
        self.lows = np.asarray(cfg.lows, dtype=float)
        self.highs = np.asarray(cfg.highs, dtype=float)
        if np.any(self.highs <= self.lows):
            raise ValueError("highs morajo biti vecji od lows")
        self.span = self.highs - self.lows
        self.n_idx = cfg.n_intervals + 1
        # osnove za mesano stevilo (idx_0, ..., idx_{k-1}) -> ena plosica
        self.strides = np.array([self.n_idx**i for i in range(self.k)], dtype=np.int64)
        self.tiles_per_tiling = int(self.n_idx**self.k)
        self.features_per_action = cfg.n_tilings * self.tiles_per_tiling
        self.n_features = cfg.n_actions * self.features_per_action
        j = np.arange(cfg.n_tilings, dtype=np.int64)[:, None]
        d = np.asarray(cfg.offsets, dtype=np.int64)[None, :]
        # (n_tilings, k); vrednosti v [0, 1)
        self._offsets = ((j * d) % cfg.n_tilings) / cfg.n_tilings
        self._tiling_base = (np.arange(cfg.n_tilings, dtype=np.int64) * self.tiles_per_tiling)

    # ------------------------------------------------------------------
    def normalize(self, obs) -> np.ndarray:
        u = (np.asarray(obs, dtype=float) - self.lows) / self.span
        return np.clip(u, 0.0, 1.0)

    def out_of_bounds(self, obs) -> np.ndarray:
        s = np.asarray(obs, dtype=float)
        return (s < self.lows) | (s > self.highs)

    def tiles(self, obs) -> np.ndarray:
        """Indeksi plosic brez akcijskega zamika; dolzina n_tilings."""
        u = self.normalize(obs)
        idx = np.floor(u * self.cfg.n_intervals + self._offsets).astype(np.int64)
        return self._tiling_base + idx @ self.strides

    def features(self, obs, action: int) -> np.ndarray:
        """Indeksi aktivnih znacilk za par (s, a); dolzina n_tilings, vse vrednosti 1.0."""
        a = int(action)
        if not 0 <= a < self.cfg.n_actions:
            raise ValueError(f"akcija izven obsega: {action!r}")
        return self.tiles(obs) + a * self.features_per_action

    def features_all_actions(self, obs) -> np.ndarray:
        """(n_actions, n_tilings) indeksi; en izracun plosic za vse akcije."""
        base = self.tiles(obs)
        return base[None, :] + (np.arange(self.cfg.n_actions, dtype=np.int64) * self.features_per_action)[:, None]

    def dense(self, obs, action: int) -> np.ndarray:
        """Gost binarni vektor (samo za teste in referencne implementacije)."""
        x = np.zeros(self.n_features)
        x[self.features(obs, action)] = 1.0
        return x

    def describe(self) -> dict:
        return self.cfg.describe()


def coder_from_bounds(tile_bounds: dict, n_tilings: int = 16, n_intervals: int = 8,
                      n_actions: int = 3) -> TileCoder:
    """Zgradi TileCoder iz slovarja meja z vrstnim redom (x, x_dot, theta, theta_dot)."""
    keys = ("x", "x_dot", "theta", "theta_dot")
    lows = tuple(float(tile_bounds[k][0]) for k in keys)
    highs = tuple(float(tile_bounds[k][1]) for k in keys)
    return TileCoder(TileCoderConfig(lows=lows, highs=highs, n_tilings=n_tilings,
                                     n_intervals=n_intervals, n_actions=n_actions))
