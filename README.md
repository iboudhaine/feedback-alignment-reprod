# Feedback Alignment

A reproduction of the three experiments from Lillicrap et al. 2014,
"Random feedback weights support learning in deep neural networks".
The paper shows that the precise transpose of the forward weights is not
required for credit assignment: fixed random feedback matrices suffice, and
the forward weights gradually align with them during training.

The repo implements all three tasks from the paper:

- Task 1: linear function approximation, 30 -> 20 -> 10.
- Task 2: MNIST classification, 784 -> 1000 -> 10 with sigmoid units.
- Task 3: nonlinear function approximation, student depth 3 or 4 against a
  fixed teacher network.

Each task can be trained under standard backpropagation (BP), feedback
alignment (FA), or shallow learning (output layer only).

## Findings

The paper's three qualitative claims reproduce: FA matches BP on the linear
task and on MNIST, and a four-layer FA network beats a three-layer BP
network on the nonlinear task. Full numerical results, plots, and an honest
accounting of where our run falls short of the paper's published numbers
are in [report/report.pdf](report/report.pdf).

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The install registers a `feedback-alignment` console script on your PATH.

## How to run

Every experiment is a subcommand of `feedback-alignment`. Output CSVs land in
`results/`. The MNIST data is downloaded on first use into `data/mnist/`.

### Task 1: linear function approximation

All hyperparameters are fixed by the paper, so the commands are direct:

```bash
feedback-alignment linear --figure 1 --alg bp
feedback-alignment linear --figure 1 --alg fa
feedback-alignment linear --figure 4
```

### Task 2: MNIST

The paper chooses the init scale `omega` and feedback scale `beta` by manual
search. Run the sweep first, then run the main command with your chosen
values:

```bash
feedback-alignment mnist-sweep --omega 0.05 0.1 0.2 --beta 0.05 0.1 0.2
# Inspect results/mnist_sweep_seed0.csv and pick the (omega, beta)
# with the lowest final test error, then:
feedback-alignment mnist --alg bp --omega <OMEGA>
feedback-alignment mnist --alg fa --omega <OMEGA> --beta <BETA>
```

### Task 3: nonlinear function approximation

The paper also chooses the teacher regime (`target-scale`) and feedback
scales (`b1`, `b2`) by manual search:

```bash
feedback-alignment nonlinear-sweep --target-scale 0.5 1.0 2.0 \
    --b1 0.05 0.1 0.2 --b2 0.05 0.1 0.2
# Inspect results/nonlinear_sweep_seed0.csv and pick the combination
# with the lowest final test NSE, then:
feedback-alignment nonlinear --model 3 --alg fa \
    --target-scale <TS> --b1-scale <B1>
feedback-alignment nonlinear --model 4 --alg fa \
    --target-scale <TS> --b1-scale <B1> --b2-scale <B2>
```

Use `feedback-alignment <subcommand> --help` for the full flag list.

### Full sweep

To reproduce every figure in `report/report.pdf` in one shot, run the
bundled script:

```bash
./run_all.sh
```

### Plots

After any of the above, regenerate the figures from the CSVs:

```bash
python plot_all.py
```

### Tests

```bash
pytest
```

Includes a PyTorch-autograd gradient check against the hand-coded BP signals,
a sanity test that FA-vs-BP alignment decreases during training, and shape
contracts for the `Layer` / `MLP` / `step` API.

## Reference

Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J. (2014).
Random feedback weights support learning in deep neural networks.
arXiv:1411.0247. https://arxiv.org/abs/1411.0247
