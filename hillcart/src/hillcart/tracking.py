"""Beleženje eksperimentov (FRI navodila "Izvajanje eksperimentov").

Vsak zagon dobi svojo mapo runs/<datum>_<ime>_<git-hash>/ z:
  meta.json     git commit, branch, 'dirty' zastavica, čas, Python, platforma, verzije paketov, seme, ukaz
  config.json   celotna konfiguracija (parametri sistema, simulatorja, metode)
  scalars.csv   vse skalarne metrike v dolgem formatu (tag, step, value, wall_time)
  events.*      TensorBoard dnevnik (tensorboardX), če je nameščen
  summary.json  povzetek ob koncu

Zagon z necommitano kodo je privzeto zavrnjen (require_clean=True). Za razvoj ga lahko
dovoliš; zastavica 'dirty' ostane zapisana v meta.json in v imenu mape.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

try:
    from tensorboardX import SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover
    SummaryWriter = None

_PACKAGES = ("hillcart", "numpy", "matplotlib", "tensorboardX", "sympy", "pytest")


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info(repo_dir: str | Path = ".") -> dict:
    cwd = Path(repo_dir)
    commit = _git(["rev-parse", "HEAD"], cwd)
    status = _git(["status", "--porcelain"], cwd)
    return {
        "commit": commit,
        "short": commit[:8] if commit else None,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "dirty": bool(status) if status is not None else None,
        "describe": _git(["describe", "--tags", "--always", "--dirty"], cwd),
    }


def _package_versions() -> dict:
    out = {}
    for p in _PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


class Run:
    def __init__(self, name: str, config: dict, base_dir: str | Path = "runs", seed: int | None = None,
                 require_clean: bool = True, repo_dir: str | Path = ".", use_tensorboard: bool = True):
        g = git_info(repo_dir)
        if require_clean and (g["commit"] is None or g["dirty"]):
            raise RuntimeError("Koda ni commitana (git 'dirty' ali ni repozitorija). "
                               "Najprej 'git commit', ali za razvoj uporabi --allow-dirty.")
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = (g["short"] or "nogit") + ("-dirty" if g["dirty"] else "")
        self.dir = Path(base_dir) / f"{stamp}_{name}_{tag}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.meta = {
            "name": name,
            "started": _dt.datetime.now().isoformat(timespec="seconds"),
            "git": g,
            "seed": seed,
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "argv": sys.argv,
        }
        self.save_json("meta.json", self.meta)
        self.save_json("config.json", config)
        self._csv_file = open(self.dir / "scalars.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["tag", "step", "value", "wall_time"])
        self.writer = None
        if use_tensorboard and SummaryWriter is not None:
            self.writer = SummaryWriter(logdir=str(self.dir))
            self.writer.add_text("config", "```\n" + json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n```", 0)
            self.writer.add_text("git", json.dumps(g), 0)

    def scalar(self, tag: str, value: float, step: int) -> None:
        wt = time.time()
        self._csv.writerow([tag, step, float(value), wt])
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), step, walltime=wt)

    def save_json(self, filename: str, obj) -> Path:
        path = self.dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        return path

    def close(self, summary: dict | None = None) -> None:
        self.meta["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
        if summary is not None:
            self.save_json("summary.json", summary)
        self.save_json("meta.json", self.meta)
        self._csv_file.close()
        if self.writer is not None:
            self.writer.close()
