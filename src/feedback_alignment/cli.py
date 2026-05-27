"""Console-script entry point for the feedback-alignment reproduction.

After `pip install -e .` the command `feedback-alignment` is available on $PATH
with five subcommands:

    feedback-alignment linear --figure 1 --alg bp
    feedback-alignment linear --figure 4
    feedback-alignment mnist --alg fa --omega 0.1 --beta 0.1
    feedback-alignment mnist-sweep --omega 0.05 0.1 0.2 --beta 0.05 0.1 0.2
    feedback-alignment nonlinear --model 4 --alg fa --target-scale 1.0 \\
        --b1-scale 0.1 --b2-scale 0.1
    feedback-alignment nonlinear-sweep --target-scale 0.5 1.0 2.0 \\
        --b1 0.05 0.1 0.2 --b2 0.05 0.1 0.2
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .linear import LinearConfig, run_fig1, run_fig4
from .mnist import MNISTConfig, run_mnist, run_mnist_sweep
from .nonlinear import NonlinearConfig, nonlinear_sweep, run_nonlinear

_DEFAULT_OMEGA_LIST_FIG4 = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.125, 0.15, 0.2, 0.25]


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--quiet", action="store_true", help="suppress progress bars")


def _scalar_or_list(values: Sequence[float]) -> float | list[float]:
    return float(values[0]) if len(values) == 1 else [float(v) for v in values]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="feedback-alignment",
        description="Reproduction of Lillicrap et al. 2014 (random feedback weights).",
    )
    sp = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    # --- linear (Task 1)
    p_lin = sp.add_parser("linear", help="Task 1: linear function approximation (Fig 1 / Fig 4)")
    p_lin.add_argument("--figure", type=int, choices=[1, 4], required=True)
    p_lin.add_argument(
        "--alg",
        choices=["bp", "fa"],
        help="(figure 1 only; figure 4 is FA-only by construction)",
    )
    p_lin.add_argument("--eta", type=float, default=1e-3)
    p_lin.add_argument("--omega", type=float, default=0.01, help="(figure 1 only)")
    p_lin.add_argument("--n-trials", type=int, default=20, help="(figure 4 only)")
    p_lin.add_argument("--n-examples", type=int, default=2000)
    _add_common(p_lin)

    # --- mnist (Task 2)
    p_m = sp.add_parser("mnist", help="Task 2: MNIST classification")
    p_m.add_argument("--alg", choices=["bp", "fa", "shallow"], required=True)
    p_m.add_argument("--omega", type=float, required=True)
    p_m.add_argument("--beta", type=float, help="required for FA")
    p_m.add_argument("--eta", type=float, default=1e-3)
    p_m.add_argument("--alpha", type=float, default=1e-6)
    p_m.add_argument("--max-examples", type=int, default=1_500_000)
    p_m.add_argument("--eval-every", type=int, default=50_000)
    p_m.add_argument("--sparse50", action="store_true")
    p_m.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    _add_common(p_m)

    # --- mnist-sweep
    p_ms = sp.add_parser("mnist-sweep", help="Sweep (omega, beta) for MNIST (paper's manual search)")
    p_ms.add_argument("--omega", type=float, nargs="+", required=True)
    p_ms.add_argument("--beta", type=float, nargs="+", required=True)
    p_ms.add_argument("--eta", type=float, default=1e-3)
    p_ms.add_argument("--alpha", type=float, default=1e-6)
    p_ms.add_argument("--max-examples", type=int, default=1_500_000)
    p_ms.add_argument("--eval-every", type=int, default=50_000)
    p_ms.add_argument(
        "--no-bp-baseline",
        action="store_true",
        help="skip the BP-only baseline row at each omega",
    )
    p_ms.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    _add_common(p_ms)

    # --- nonlinear (Task 3)
    p_nl = sp.add_parser("nonlinear", help="Task 3: nonlinear function approximation")
    p_nl.add_argument("--model", type=int, choices=[3, 4], required=True)
    p_nl.add_argument("--alg", choices=["bp", "fa", "shallow"], required=True)
    p_nl.add_argument("--steps", type=int, default=1_500_000)
    p_nl.add_argument("--eta", type=float, default=1e-3)
    p_nl.add_argument(
        "--target-scale",
        type=float,
        nargs="+",
        required=True,
        help="single value (broadcast) or 3 values for per-layer teacher scales",
    )
    p_nl.add_argument(
        "--teacher-bias-scale",
        type=float,
        default=0.0,
        help="0.0 means no teacher biases (paper default); >0 draws biases uniform[-this, this]",
    )
    p_nl.add_argument("--b1-scale", type=float)
    p_nl.add_argument("--b2-scale", type=float)
    p_nl.add_argument("--eval-every", type=int, default=50_000)
    p_nl.add_argument("--test-size", type=int, default=5000)
    _add_common(p_nl)

    # --- nonlinear-sweep
    p_nls = sp.add_parser(
        "nonlinear-sweep",
        help="Sweep target/feedback scales for Task 3 (paper's manual search)",
    )
    p_nls.add_argument("--target-scale", type=float, nargs="+", required=True)
    p_nls.add_argument("--b1", type=float, nargs="+", required=True)
    p_nls.add_argument("--b2", type=float, nargs="+", required=True)
    p_nls.add_argument("--steps", type=int, default=1_500_000)
    p_nls.add_argument("--eta", type=float, default=1e-3)
    p_nls.add_argument("--eval-every", type=int, default=50_000)
    p_nls.add_argument("--test-size", type=int, default=5000)
    p_nls.add_argument("--teacher-bias-scale", type=float, default=0.0)
    _add_common(p_nls)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    verbose = not args.quiet
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "linear":
        _run_linear(args, results_dir)
    elif args.cmd == "mnist":
        _run_mnist(args, results_dir, verbose)
    elif args.cmd == "mnist-sweep":
        _run_mnist_sweep(args, results_dir, verbose)
    elif args.cmd == "nonlinear":
        _run_nonlinear(args, results_dir, verbose)
    elif args.cmd == "nonlinear-sweep":
        _run_nonlinear_sweep(args, results_dir, verbose)
    else:  # pragma: no cover; argparse should have rejected this
        raise SystemExit(f"unknown command: {args.cmd}")


def _run_linear(args: argparse.Namespace, results_dir: Path) -> None:
    cfg = LinearConfig(
        eta=args.eta,
        omega=args.omega,
        seed=args.seed,
        n_examples=args.n_examples,
    )
    if args.figure == 1:
        if args.alg is None:
            raise SystemExit("linear --figure 1 requires --alg {bp,fa}")
        out = results_dir / f"linear_fig1_{args.alg}_{cfg.slug()}.csv"
        run_fig1(cfg, args.alg, out)
        print(out)
    else:
        if args.alg is not None:
            raise SystemExit("linear --figure 4 is FA-only; do not pass --alg")
        out = results_dir / f"linear_fig4_seed{args.seed}.csv"
        run_fig4(
            _DEFAULT_OMEGA_LIST_FIG4,
            n_trials=args.n_trials,
            eta=args.eta,
            seed=args.seed,
            out_csv=out,
            n_examples=args.n_examples,
        )
        print(out)


def _run_mnist(args: argparse.Namespace, results_dir: Path, verbose: bool) -> None:
    cfg = MNISTConfig(
        eta=args.eta,
        alpha=args.alpha,
        omega=args.omega,
        beta=args.beta,
        max_examples=args.max_examples,
        eval_every=args.eval_every,
        seed=args.seed,
        sparse50=args.sparse50,
    )
    out = results_dir / f"mnist_{args.alg}_{cfg.slug()}.csv"
    run_mnist(cfg, args.alg, out, args.data_dir, verbose=verbose)
    print(out)


def _run_mnist_sweep(args: argparse.Namespace, results_dir: Path, verbose: bool) -> None:
    cfg = MNISTConfig(
        eta=args.eta,
        alpha=args.alpha,
        max_examples=args.max_examples,
        eval_every=args.eval_every,
        seed=args.seed,
    )
    out = results_dir / f"mnist_sweep_seed{args.seed}.csv"
    run_mnist_sweep(
        cfg,
        args.omega,
        args.beta,
        out,
        args.data_dir,
        include_bp_baseline=not args.no_bp_baseline,
        verbose=verbose,
    )
    print(out)


def _run_nonlinear(args: argparse.Namespace, results_dir: Path, verbose: bool) -> None:
    cfg = NonlinearConfig(
        steps=args.steps,
        eta=args.eta,
        seed=args.seed,
        test_size=args.test_size,
        eval_every=args.eval_every,
        target_scale=_scalar_or_list(args.target_scale),
        teacher_bias_scale=args.teacher_bias_scale,
        b1_scale=args.b1_scale,
        b2_scale=args.b2_scale,
    )
    out = results_dir / f"nonlinear_model{args.model}_{args.alg}_{cfg.slug()}.csv"
    run_nonlinear(cfg, args.model, args.alg, out, verbose=verbose)
    print(out)


def _run_nonlinear_sweep(args: argparse.Namespace, results_dir: Path, verbose: bool) -> None:
    cfg = NonlinearConfig(
        steps=args.steps,
        eta=args.eta,
        seed=args.seed,
        test_size=args.test_size,
        eval_every=args.eval_every,
        teacher_bias_scale=args.teacher_bias_scale,
    )
    out = results_dir / f"nonlinear_sweep_seed{args.seed}.csv"
    nonlinear_sweep(cfg, args.b1, args.b2, args.target_scale, out, verbose=verbose)
    print(out)


if __name__ == "__main__":
    main()
