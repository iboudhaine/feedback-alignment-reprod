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

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The install registers a `feedback-alignment` console script on your PATH.

## How to run

Every experiment is a subcommand of `feedback-alignment`. Output CSVs land in
`results/`. The MNIST data is downloaded on first use into `data/mnist/`.

Examples (short smoke runs):

```bash
# Task 1, Fig 1 style: BP vs FA learning curves
feedback-alignment linear --figure 1 --alg bp
feedback-alignment linear --figure 1 --alg fa

# Task 1, Fig 4 style: FA-only sweep over the init scale
feedback-alignment linear --figure 4

# Task 2: MNIST
feedback-alignment mnist --alg bp --omega 0.1
feedback-alignment mnist --alg fa --omega 0.1 --beta 0.1

# Task 2 sweep over (omega, beta)
feedback-alignment mnist-sweep --omega 0.05 0.1 0.2 --beta 0.05 0.1 0.2

# Task 3: nonlinear
feedback-alignment nonlinear --model 3 --alg fa --target-scale 1.0 --b1-scale 0.1
feedback-alignment nonlinear --model 4 --alg fa --target-scale 1.0 --b1-scale 0.1 --b2-scale 0.1

# Task 3 sweep
feedback-alignment nonlinear-sweep --target-scale 0.5 1.0 2.0 --b1 0.05 0.1 0.2 --b2 0.05 0.1 0.2
```

Use `feedback-alignment <subcommand> --help` for the full flag list.

To generate plots from the CSVs in `results/`:

```bash
python plot_all.py            # default plots
python plot_all.py --paper    # paper-style figures
```

To run the test suite:

```bash
pytest
```

## Reference

Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J. (2014).
Random feedback weights support learning in deep neural networks.
arXiv:1411.0247. https://arxiv.org/abs/1411.0247
