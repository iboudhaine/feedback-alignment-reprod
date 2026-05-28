"""Plot results from `results/` into `plots/`.

Produces six figures:
  linear_fig1.png            paper Fig 1: BP vs FA on the linear task
  linear_fig4.png            paper Fig 4: FA-only sweep over omega
  mnist.png                  paper Fig 2: best BP vs best FA on MNIST
  mnist_sweep_heatmap.png    final test error over the (omega, beta) grid
  nonlinear.png              paper Fig 3: BP / FA / shallow at two depths
  nonlinear_sweep_heatmap.png  4-layer FA final NSE over the (b1, b2) grid
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BP_COLOR = "#222222"
FA_COLOR = "#2ca02c"
SHALLOW_COLOR = "#999999"
GRID_COLOR = "#cccccc"


# ---------- io helpers ----------

@dataclass
class Run:
    path: Path
    df: pd.DataFrame
    final: float


def _save(fig: plt.Figure, out_path: Path, show: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def _smooth(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=window, min_periods=1, center=True).mean()


# ---------- plotters ----------

def plot_linear_fig1(results_dir: Path, out_dir: Path, show: bool) -> None:
    bp_files = list(results_dir.glob("linear_fig1_bp_*.csv"))
    fa_files = list(results_dir.glob("linear_fig1_fa_*.csv"))
    if not bp_files or not fa_files:
        return
    bp = pd.read_csv(bp_files[0])
    fa = pd.read_csv(fa_files[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.5), sharex=True)

    ax1.plot(bp["step"], bp["nse"], color=BP_COLOR, label="Backprop", linewidth=1.0, alpha=0.85)
    ax1.plot(fa["step"], fa["nse"], color=FA_COLOR, label="Feedback alignment", linewidth=1.0, alpha=0.95)
    ax1.set_yscale("log")
    ax1.set_ylabel("NSE")
    ax1.legend(frameon=False, loc="lower left")
    ax1.grid(True, which="both", color=GRID_COLOR, linewidth=0.4)

    ax2.plot(fa["step"], np.degrees(fa["angle_delta_h_fa_bp"]), color=FA_COLOR, linewidth=1.0)
    ax2.axhline(90, color="gray", linestyle="--", linewidth=0.6)
    ax2.set_xlabel("Examples")
    ax2.set_ylabel("FA vs BP angle (deg)")
    ax2.set_ylim(0, 95)
    ax2.grid(True, color=GRID_COLOR, linewidth=0.4)

    fig.tight_layout()
    _save(fig, out_dir / "linear_fig1.png", show)


def plot_linear_fig4(results_dir: Path, out_dir: Path, show: bool, window: int = 20) -> None:
    files = list(results_dir.glob("linear_fig4_seed*.csv"))
    if not files:
        return
    df = pd.read_csv(files[0])

    omegas = sorted(df["omega"].unique())
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=np.log10(min(omegas)), vmax=np.log10(max(omegas)))

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    for omega in omegas:
        sub = df[df["omega"] == omega]
        color = cmap(norm(np.log10(omega)))
        grouped = sub.groupby("step")
        nse_mean = grouped["nse"].mean()
        ang_bp_mean = grouped["angle_delta_h_fa_bp"].mean()
        ang_pbp_mean = grouped["angle_delta_h_fa_pbp"].mean()

        axes[0].plot(nse_mean.index, _smooth(nse_mean, window), color=color, linewidth=1.1)
        axes[1].plot(ang_bp_mean.index, np.degrees(_smooth(ang_bp_mean, window)), color=color, linewidth=1.1)
        axes[2].plot(ang_pbp_mean.index, np.degrees(_smooth(ang_pbp_mean, window)), color=color, linewidth=1.1)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("NSE")
    axes[1].set_ylabel("FA vs BP angle (deg)")
    axes[1].axhline(90, color="gray", linestyle="--", linewidth=0.6)
    axes[1].set_ylim(0, 95)
    axes[2].set_ylabel("FA vs pseudo-BP angle (deg)")
    axes[2].axhline(90, color="gray", linestyle="--", linewidth=0.6)
    axes[2].set_ylim(0, 95)
    axes[2].set_xlabel("Examples")
    for ax in axes:
        ax.grid(True, color=GRID_COLOR, linewidth=0.4)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label="omega (W init scale)", shrink=0.7, pad=0.02)
    cbar.set_ticks(np.log10(omegas))
    cbar.set_ticklabels([f"{o:g}" for o in omegas])

    _save(fig, out_dir / "linear_fig4.png", show)


_MNIST_RX = re.compile(r"mnist_(bp|fa)_seed0(?:_omega([\d.]+))?(?:_beta([\d.]+))?(_sparse50)?\.csv$")


def _load_mnist_runs(results_dir: Path) -> list[dict]:
    runs = []
    for path in results_dir.glob("mnist_*.csv"):
        m = _MNIST_RX.match(path.name)
        if not m:
            continue
        alg, om, be, sp = m.groups()
        df = pd.read_csv(path)
        runs.append({
            "path": path,
            "alg": alg,
            "omega": float(om) if om else None,
            "beta": float(be) if be else None,
            "sparse": bool(sp),
            "df": df,
            "final": float(df["test_error_pct"].iloc[-1]),
        })
    return runs


def plot_mnist(results_dir: Path, out_dir: Path, show: bool) -> None:
    runs = _load_mnist_runs(results_dir)
    bp_runs = [r for r in runs if r["alg"] == "bp"]
    fa_runs = [r for r in runs if r["alg"] == "fa" and not r["sparse"]]
    sparse_runs = [r for r in runs if r["sparse"]]
    if not bp_runs or not fa_runs:
        return

    bp_best = min(bp_runs, key=lambda r: r["final"])
    fa_best = min(fa_runs, key=lambda r: r["final"])
    sparse_best = min(sparse_runs, key=lambda r: r["final"]) if sparse_runs else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6.5), sharex=True)

    # background: faint FA curves to show the spread of the sweep
    for r in fa_runs:
        if r is fa_best:
            continue
        ax1.plot(r["df"]["examples_seen"], r["df"]["test_error_pct"],
                 color=FA_COLOR, alpha=0.15, linewidth=0.8)
        ax2.plot(r["df"]["examples_seen"], np.degrees(r["df"]["mean_angle_delta_h_fa_bp"]),
                 color=FA_COLOR, alpha=0.15, linewidth=0.8)

    ax1.plot(bp_best["df"]["examples_seen"], bp_best["df"]["test_error_pct"],
             color=BP_COLOR, label=f"Backprop  (omega={bp_best['omega']:g})", linewidth=1.6)
    ax1.plot(fa_best["df"]["examples_seen"], fa_best["df"]["test_error_pct"],
             color=FA_COLOR,
             label=f"FA  (omega={fa_best['omega']:g}, beta={fa_best['beta']:g})", linewidth=1.6)
    if sparse_best is not None:
        ax1.plot(sparse_best["df"]["examples_seen"], sparse_best["df"]["test_error_pct"],
                 color=FA_COLOR, linestyle="--",
                 label=f"FA + 50% sparsity  (omega={sparse_best['omega']:g}, beta={sparse_best['beta']:g})",
                 linewidth=1.6)
    ax1.set_yscale("log")
    ax1.set_ylabel("Test error (%)")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, which="both", color=GRID_COLOR, linewidth=0.4)

    ax2.plot(fa_best["df"]["examples_seen"], np.degrees(fa_best["df"]["mean_angle_delta_h_fa_bp"]),
             color=FA_COLOR, linewidth=1.6)
    if sparse_best is not None:
        ax2.plot(sparse_best["df"]["examples_seen"], np.degrees(sparse_best["df"]["mean_angle_delta_h_fa_bp"]),
                 color=FA_COLOR, linestyle="--", linewidth=1.6)
    ax2.axhline(90, color="gray", linestyle="--", linewidth=0.6)
    ax2.set_ylim(0, 95)
    ax2.set_xlabel("Examples")
    ax2.set_ylabel("FA vs BP angle (deg)")
    ax2.grid(True, color=GRID_COLOR, linewidth=0.4)

    fig.tight_layout()
    _save(fig, out_dir / "mnist.png", show)


def plot_mnist_sweep_heatmap(results_dir: Path, out_dir: Path, show: bool) -> None:
    runs = _load_mnist_runs(results_dir)
    fa = {(r["omega"], r["beta"]): r["final"] for r in runs if r["alg"] == "fa" and not r["sparse"]}
    if not fa:
        return
    omegas = sorted({k[0] for k in fa})
    betas = sorted({k[1] for k in fa})
    grid = np.array([[fa.get((o, b), np.nan) for b in betas] for o in omegas])

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f"{b:g}" for b in betas])
    ax.set_yticks(range(len(omegas)))
    ax.set_yticklabels([f"{o:g}" for o in omegas])
    ax.set_xlabel("beta (B feedback scale)")
    ax.set_ylabel("omega (W init scale)")
    ax.set_title("MNIST FA: final test error (%)")
    vmean = np.nanmean(grid)
    for i in range(len(omegas)):
        for j in range(len(betas)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > vmean else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Final test error (%)")
    fig.tight_layout()
    _save(fig, out_dir / "mnist_sweep_heatmap.png", show)


_NL_RX = re.compile(
    r"nonlinear_model(\d+)_(bp|fa|shallow)_seed0_ts([\d.]+)"
    r"(?:_b1([\d.]+))?(?:_b2([\d.]+))?\.csv$"
)


def _load_nonlinear_runs(results_dir: Path) -> list[dict]:
    runs = []
    for path in results_dir.glob("nonlinear_*.csv"):
        m = _NL_RX.match(path.name)
        if not m:
            continue
        df = pd.read_csv(path)
        runs.append({
            "path": path,
            "model": int(m.group(1)),
            "alg": m.group(2),
            "ts": float(m.group(3)),
            "b1": float(m.group(4)) if m.group(4) else None,
            "b2": float(m.group(5)) if m.group(5) else None,
            "df": df,
            "final": float(df["test_nse"].iloc[-1]),
        })
    return runs


def plot_nonlinear(results_dir: Path, out_dir: Path, show: bool) -> None:
    runs = _load_nonlinear_runs(results_dir)
    if not runs:
        return

    def pick(model: int, alg: str, *, best: bool = False):
        subset = [r for r in runs if r["model"] == model and r["alg"] == alg]
        if not subset:
            return None
        return min(subset, key=lambda r: r["final"]) if best else subset[0]

    bp3 = pick(3, "bp")
    bp4 = pick(4, "bp")
    sh3 = pick(3, "shallow")
    sh4 = pick(4, "shallow")
    fa3 = pick(3, "fa", best=True)
    fa4 = pick(4, "fa", best=True)

    fig, ax = plt.subplots(figsize=(7, 4.8))

    def line(r, color, ls, label):
        if r is None:
            return
        ax.plot(r["df"]["examples_seen"], r["df"]["test_nse"],
                color=color, linestyle=ls, linewidth=1.6, label=label)

    line(bp3, BP_COLOR, "-", "3-layer BP")
    line(fa3, FA_COLOR, "-", f"3-layer FA  (b1={fa3['b1']:g})" if fa3 else None)
    line(sh3, SHALLOW_COLOR, "-", "3-layer shallow")
    line(bp4, BP_COLOR, "--", "4-layer BP")
    line(fa4, FA_COLOR, "--",
         f"4-layer FA  (b1={fa4['b1']:g}, b2={fa4['b2']:g})" if fa4 else None)
    line(sh4, SHALLOW_COLOR, "--", "4-layer shallow")

    ax.set_yscale("log")
    ax.set_xlabel("Examples")
    ax.set_ylabel("Test NSE")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.4)
    fig.tight_layout()
    _save(fig, out_dir / "nonlinear.png", show)


def plot_nonlinear_sweep_heatmap(results_dir: Path, out_dir: Path, show: bool) -> None:
    runs = _load_nonlinear_runs(results_dir)
    fa4 = {(r["b1"], r["b2"]): r["final"]
           for r in runs if r["model"] == 4 and r["alg"] == "fa"}
    if not fa4:
        return
    b1s = sorted({k[0] for k in fa4})
    b2s = sorted({k[1] for k in fa4})
    grid = np.array([[fa4.get((a, b), np.nan) for b in b2s] for a in b1s])

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(b2s)))
    ax.set_xticklabels([f"{b:g}" for b in b2s])
    ax.set_yticks(range(len(b1s)))
    ax.set_yticklabels([f"{a:g}" for a in b1s])
    ax.set_xlabel("b2 (B2 feedback scale)")
    ax.set_ylabel("b1 (B1 feedback scale)")
    ax.set_title("4-layer FA: final test NSE")
    vmean = np.nanmean(grid)
    for i in range(len(b1s)):
        for j in range(len(b2s)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v > vmean else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Final test NSE")
    fig.tight_layout()
    _save(fig, out_dir / "nonlinear_sweep_heatmap.png", show)


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out-dir", type=Path, default=Path("plots"))
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_linear_fig1(args.results_dir, args.out_dir, args.show)
    plot_linear_fig4(args.results_dir, args.out_dir, args.show)
    plot_mnist(args.results_dir, args.out_dir, args.show)
    plot_mnist_sweep_heatmap(args.results_dir, args.out_dir, args.show)
    plot_nonlinear(args.results_dir, args.out_dir, args.show)
    plot_nonlinear_sweep_heatmap(args.results_dir, args.out_dir, args.show)


if __name__ == "__main__":
    main()
