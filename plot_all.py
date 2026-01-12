import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _plot_lines(files, xcol, ycol, title, out_path, show):
    if not files:
        return False
    plt.figure()
    plotted = False
    for path in files:
        df = pd.read_csv(path)
        if xcol not in df.columns or ycol not in df.columns:
            continue
        plt.plot(df[xcol], df[ycol], label=path.stem)
        plotted = True
    if not plotted:
        plt.close()
        return False
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()
    return True


def _plot_fig4(csv_path: Path, trial_id: int, out_dir: Path, show: bool):
    df = pd.read_csv(csv_path)
    trial_used = "all"
    if "trial" in df.columns:
        if (df["trial"] == trial_id).any():
            df = df[df["trial"] == trial_id]
            trial_used = str(trial_id)
        else:
            trial_used = str(int(df["trial"].min()))
            df = df[df["trial"] == df["trial"].min()]

    if "omega" not in df.columns or "step" not in df.columns:
        return False

    series = [
        ("nse", "nse", "NSE"),
        ("angle_dh_fa_bp", "angle_bp", "Angle FA vs BP"),
        ("angle_dh_fa_pbp", "angle_pbp", "Angle FA vs PBP"),
    ]
    wrote = False
    for ycol, suffix, ylabel in series:
        if ycol not in df.columns:
            continue
        plt.figure()
        for omega, sub in df.groupby("omega"):
            plt.plot(sub["step"], sub[ycol], label=f"omega={omega}")
        plt.xlabel("step")
        plt.ylabel(ylabel)
        plt.title(f"{csv_path.stem} ({ycol}, trial {trial_used})")
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"{csv_path.stem}_{suffix}_trial{trial_used}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        if show:
            plt.show()
        plt.close()
        wrote = True
    return wrote


def _set_paper_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def _smooth(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def _rolling_std(series, window):
    if window <= 1:
        return None
    return series.rolling(window=window, min_periods=1, center=True).std()


def _algo_label_color(stem):
    stem = stem.lower()
    if "bp" in stem:
        return "Backprop", "black"
    if "fa" in stem:
        return "Feedback alignment", "#2ca02c"
    if "shallow" in stem:
        return "Shallow", "#bdbdbd"
    if "reinforce" in stem or "rl" in stem:
        return "Reinforcement", "#7f7f7f"
    return stem, "#1f77b4"


def _paper_fig1_linear(results_dir: Path, out_dir: Path, show: bool, window: int):
    files = list(results_dir.glob("linear_fig1_*_seed*.csv"))
    if not files:
        return False

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
    fa_angle = None
    steps = None

    for path in files:
        df = pd.read_csv(path)
        if "step" not in df.columns:
            continue
        label, color = _algo_label_color(path.stem)
        ax1.plot(df["step"], df["nse"], color=color, label=label, linewidth=1.3)
        if "angle_dh_fa_bp" in df.columns and "fa" in path.stem:
            steps = df["step"]
            fa_angle = np.degrees(df["angle_dh_fa_bp"])

    ax1.set_yscale("log")
    ax1.set_ylabel("Error (NSE)")
    ax1.legend(frameon=False)

    if fa_angle is not None:
        mean = _smooth(pd.Series(fa_angle), window)
        std = _rolling_std(pd.Series(fa_angle), window)
        ax2.plot(steps, mean, color="#2ca02c", linewidth=1.3)
        if std is not None:
            ax2.fill_between(steps, mean - std, mean + std, color="#2ca02c", alpha=0.25, linewidth=0)
    ax2.axhline(90, color="#7f7f7f", linestyle="--", linewidth=1.0)
    ax2.set_ylim(0, 90)
    ax2.set_ylabel("FA vs BP angle (deg)")
    ax2.set_xlabel("No. Examples")

    fig.tight_layout()
    out_path = out_dir / "fig1_linear.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)
    return True


def _paper_fig2_mnist(results_dir: Path, out_dir: Path, show: bool):
    files = list(results_dir.glob("mnist_*_seed*.csv"))
    if not files:
        return False

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
    plotted = False

    for path in files:
        df = pd.read_csv(path)
        if "examples_seen" not in df.columns or "test_error_pct" not in df.columns:
            continue
        label, color = _algo_label_color(path.stem)
        ax1.plot(df["examples_seen"], df["test_error_pct"], color=color, label=label, linewidth=1.3)
        plotted = True

        if "fa" in path.stem and "mean_angle_dh_fa_bp" in df.columns:
            angles = np.degrees(df["mean_angle_dh_fa_bp"])
            ax2.plot(df["examples_seen"], angles, color="#2ca02c", linewidth=1.3)

    if not plotted:
        plt.close(fig)
        return False

    ax1.set_yscale("log")
    ax1.set_ylabel("% Error on Test Set")
    ax1.legend(frameon=False)

    ax2.axhline(90, color="#7f7f7f", linestyle="--", linewidth=1.0)
    ax2.set_ylim(0, 90)
    ax2.set_ylabel("FA vs BP angle (deg)")
    ax2.set_xlabel("No. Examples")

    fig.tight_layout()
    out_path = out_dir / "fig2_mnist.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)
    return True


def _paper_fig2_nonlinear(results_dir: Path, out_dir: Path, show: bool):
    files = list(results_dir.glob("nonlinear_model*_seed*.csv"))
    if not files:
        return False

    color_map = {
        ("3", "bp"): "black",
        ("3", "fa"): "#2ca02c",
        ("3", "shallow"): "#bdbdbd",
        ("4", "bp"): "#e377c2",
        ("4", "fa"): "#1f77b4",
        ("4", "shallow"): "#bdbdbd",
    }

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plotted = False

    for path in files:
        m = re.search(r"nonlinear_model(\\d+)_(\\w+)_", path.stem)
        if not m:
            continue
        model = m.group(1)
        alg = m.group(2)
        df = pd.read_csv(path)
        if "examples_seen" not in df.columns or "test_nse" not in df.columns:
            continue
        color = color_map.get((model, alg), "#1f77b4")
        label = f"{model}-layer {alg.upper()}"
        ax.plot(df["examples_seen"], df["test_nse"], color=color, label=label, linewidth=1.3)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_yscale("log")
    ax.set_ylabel("Error (NSE)")
    ax.set_xlabel("No. Examples")
    ax.legend(frameon=False)

    fig.tight_layout()
    out_path = out_dir / "fig2_nonlinear.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)
    return True


def _paper_fig4_linear(results_dir: Path, out_dir: Path, show: bool, trial_id: int, window: int):
    files = list(results_dir.glob("linear_fig4_fa_sweep_seed*.csv"))
    if not files:
        return False

    wrote = False
    for csv_path in files:
        df = pd.read_csv(csv_path)
        if "trial" in df.columns:
            if (df["trial"] == trial_id).any():
                df = df[df["trial"] == trial_id]
                trial_used = str(trial_id)
            else:
                trial_used = str(int(df["trial"].min()))
                df = df[df["trial"] == df["trial"].min()]
        else:
            trial_used = "all"

        if "omega" not in df.columns or "step" not in df.columns:
            continue

        omegas = sorted(df["omega"].unique())
        cmap = plt.cm.coolwarm
        colors = {omega: cmap(i / max(1, len(omegas) - 1)) for i, omega in enumerate(omegas)}

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(5, 8))

        for omega, sub in df.groupby("omega"):
            color = colors.get(omega, "#1f77b4")
            axes[0].plot(sub["step"], _smooth(sub["nse"], window), color=color, linewidth=1.0)
            if "angle_dh_fa_bp" in sub.columns:
                angles = np.degrees(sub["angle_dh_fa_bp"])
                axes[1].plot(sub["step"], _smooth(pd.Series(angles), window), color=color, linewidth=1.0)
            if "angle_dh_fa_pbp" in sub.columns:
                angles = np.degrees(sub["angle_dh_fa_pbp"])
                axes[2].plot(sub["step"], _smooth(pd.Series(angles), window), color=color, linewidth=1.0)

        axes[0].set_yscale("log")
        axes[0].set_ylabel("Error (NSE)")
        axes[1].set_ylabel("FA vs BP angle (deg)")
        axes[2].set_ylabel("FA vs PBP angle (deg)")
        axes[2].set_xlabel("No. Examples")
        for ax in axes[1:]:
            ax.axhline(90, color="#7f7f7f", linestyle="--", linewidth=1.0)
            ax.set_ylim(0, 90)

        fig.tight_layout()
        out_path = out_dir / f"{csv_path.stem}_fig4_trial{trial_used}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        if show:
            plt.show()
        plt.close(fig)
        wrote = True
    return wrote


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--out_dir", type=str, default="plots")
    p.add_argument("--fig4_trial", type=int, default=0)
    p.add_argument("--paper", action="store_true")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    if args.paper:
        _set_paper_style()
        ok = False
        ok |= _paper_fig1_linear(results_dir, out_dir, args.show, args.window)
        ok |= _paper_fig2_mnist(results_dir, out_dir, args.show)
        ok |= _paper_fig2_nonlinear(results_dir, out_dir, args.show)
        ok |= _paper_fig4_linear(results_dir, out_dir, args.show, args.fig4_trial, args.window)
        if not ok:
            print("No matching CSVs found for paper-style plots.")
        return

    # Default plots
    linear_fig1 = list(results_dir.glob("linear_fig1_*_seed*.csv"))
    _plot_lines(linear_fig1, "step", "nse", "Linear Fig.1 NSE", out_dir / "linear_fig1_nse.png", args.show)
    _plot_lines(linear_fig1, "step", "angle_dh_fa_bp", "Linear Fig.1 Angle (FA vs BP)", out_dir / "linear_fig1_angle.png", args.show)

    for path in results_dir.glob("linear_fig4_fa_sweep_seed*.csv"):
        _plot_fig4(path, args.fig4_trial, out_dir, args.show)

    mnist_files = list(results_dir.glob("mnist_*_seed*.csv"))
    _plot_lines(mnist_files, "examples_seen", "test_error_pct", "MNIST Test Error", out_dir / "mnist_error.png", args.show)
    _plot_lines(mnist_files, "examples_seen", "mean_angle_dh_fa_bp", "MNIST Angle (FA vs BP)", out_dir / "mnist_angle.png", args.show)

    nonlinear_files = list(results_dir.glob("nonlinear_model*_seed*.csv"))
    _plot_lines(nonlinear_files, "examples_seen", "test_nse", "Nonlinear Test NSE", out_dir / "nonlinear_nse.png", args.show)


if __name__ == "__main__":
    main()
