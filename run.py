import argparse
from pathlib import Path

from src.linear_task import LinearTaskConfig, run_fig1, run_fig4
from src.mnist_task import MNISTConfig, run_mnist, run_mnist_sweep
from src.nonlinear_task import NonlinearConfig, run_nonlinear, nonlinear_sweep

def _parse_list_floats(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    # Linear
    p_lin = sp.add_parser("linear")
    p_lin.add_argument("--figure", type=int, choices=[1,4], required=True)
    p_lin.add_argument("--alg", type=str, choices=["bp","fa"], required=True)
    p_lin.add_argument("--seed", type=int, default=0)
    p_lin.add_argument("--eta", type=float, default=None)
    p_lin.add_argument("--n_trials", type=int, default=20)

    # MNIST
    p_m = sp.add_parser("mnist")
    p_m.add_argument("--alg", type=str, choices=["bp","fa","shallow"], required=True)
    p_m.add_argument("--omega", type=float, required=True)
    p_m.add_argument("--beta", type=float, default=None)
    p_m.add_argument("--eta", type=float, default=1e-3)
    p_m.add_argument("--alpha", type=float, default=1e-6)
    p_m.add_argument("--max_examples", type=int, default=1500000)
    p_m.add_argument("--eval_every", type=int, default=50000)
    p_m.add_argument("--seed", type=int, default=0)
    p_m.add_argument("--sparse50", action="store_true")

    p_ms = sp.add_parser("mnist_sweep")
    p_ms.add_argument("--omega_list", type=str, required=True)
    p_ms.add_argument("--beta_list", type=str, required=True)
    p_ms.add_argument("--eta", type=float, default=1e-3)
    p_ms.add_argument("--alpha", type=float, default=1e-6)
    p_ms.add_argument("--max_examples", type=int, default=1500000)
    p_ms.add_argument("--eval_every", type=int, default=50000)
    p_ms.add_argument("--seed", type=int, default=0)

    # Nonlinear
    p_nl = sp.add_parser("nonlinear")
    p_nl.add_argument("--model", type=int, choices=[3,4], required=True)
    p_nl.add_argument("--alg", type=str, choices=["bp","fa","shallow"], required=True)
    p_nl.add_argument("--steps", type=int, default=1500000)
    p_nl.add_argument("--eta", type=float, default=1e-3)
    p_nl.add_argument("--seed", type=int, default=0)
    p_nl.add_argument("--target_scale", type=float, required=True)
    p_nl.add_argument("--b1_scale", type=float, default=None)
    p_nl.add_argument("--b2_scale", type=float, default=None)

    p_nls = sp.add_parser("nonlinear_sweep")
    p_nls.add_argument("--steps", type=int, default=1500000)
    p_nls.add_argument("--eta", type=float, default=1e-3)
    p_nls.add_argument("--seed", type=int, default=0)
    p_nls.add_argument("--b1_list", type=str, required=True)
    p_nls.add_argument("--b2_list", type=str, required=True)
    p_nls.add_argument("--target_scale_list", type=str, required=True)

    args = p.parse_args()
    results_dir = Path("results")
    data_dir = Path("data") / "mnist"

    if args.cmd == "linear":
        if args.figure == 1:
            eta = args.eta if args.eta is not None else 1e-3
            cfg = LinearTaskConfig(seed=args.seed, eta=eta)
            out = results_dir / f"linear_fig1_{args.alg}_seed{args.seed}.csv"
            run_fig1(cfg, args.alg, out)
            print(out)
        else:
            eta = args.eta if args.eta is not None else 1e-3
            omega_list = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.125, 0.15, 0.2, 0.25]
            out = results_dir / f"linear_fig4_fa_sweep_seed{args.seed}.csv"
            run_fig4(omega_list, n_trials=args.n_trials, eta=eta, seed=args.seed, out_csv=out)
            print(out)

    elif args.cmd == "mnist":
        cfg = MNISTConfig(
            eta=args.eta, alpha=args.alpha, omega=args.omega, beta=args.beta,
            max_examples=args.max_examples, eval_every=args.eval_every,
            seed=args.seed, sparse50=args.sparse50
        )
        out = results_dir / f"mnist_{args.alg}_omega{args.omega}_beta{args.beta}_seed{args.seed}{'_sparse50' if args.sparse50 else ''}.csv"
        run_mnist(cfg, args.alg, out, data_dir)
        print(out)

    elif args.cmd == "mnist_sweep":
        cfg = MNISTConfig(
            eta=args.eta, alpha=args.alpha, omega=None, beta=None,
            max_examples=args.max_examples, eval_every=args.eval_every, seed=args.seed
        )
        omega_list = _parse_list_floats(args.omega_list)
        beta_list = _parse_list_floats(args.beta_list)
        out = results_dir / f"mnist_sweep_seed{args.seed}.csv"
        run_mnist_sweep(cfg, omega_list, beta_list, out, data_dir)
        print(out)

    elif args.cmd == "nonlinear":
        cfg = NonlinearConfig(
            steps=args.steps, eta=args.eta, seed=args.seed,
            target_scale=args.target_scale, b1_scale=args.b1_scale, b2_scale=args.b2_scale
        )
        out = results_dir / f"nonlinear_model{args.model}_{args.alg}_ts{args.target_scale}_b1{args.b1_scale}_b2{args.b2_scale}_seed{args.seed}.csv"
        run_nonlinear(cfg, args.model, args.alg, out)
        print(out)

    elif args.cmd == "nonlinear_sweep":
        cfg = NonlinearConfig(steps=args.steps, eta=args.eta, seed=args.seed, target_scale=None)
        b1_list = _parse_list_floats(args.b1_list)
        b2_list = _parse_list_floats(args.b2_list)
        ts_list = _parse_list_floats(args.target_scale_list)
        out = results_dir / f"nonlinear_sweep_seed{args.seed}.csv"
        nonlinear_sweep(cfg, b1_list, b2_list, ts_list, out)
        print(out)

if __name__ == "__main__":
    main()
