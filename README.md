# Feedback Alignment (Lillicrap et al., 2014) — Reproduction Scaffold

This repository implements **the algorithms and experimental setups exactly as specified in the paper’s “Full Methods”** for:
- **Task (1)** Linear function approximation (30–20–10)
- **Task (2)** MNIST classification (784–1000–10 logistic units)
- **Task (3)** Nonlinear function approximation (30–20–10 and 30–20–10–10 vs a 30–20–10–10 target)

## What the paper specifies vs. what it leaves to manual search

The paper explicitly specifies:
- Task (1): target `T ~ Uniform[-1,1]`, `x ~ N(0,I)`, `W0,W ~ Uniform[-0.01,0.01]`, `B ~ Uniform[-0.5,0.5]`, dataset fixed across algorithms.
- Fig. 4: same as Fig. 1 but **η = 1e-3** and `ω ∈ {0.0001, 0.001, 0.01, 0.05, 0.1, 0.125, 0.15, 0.2, 0.25}`.
- Task (2): logistic hidden+output units, biases, 1-hot labels, MNIST 60k/10k, **η = 1e-3**, **weight decay α = 1e-6**, `W0,W ~ Uniform[-ω,ω]` and `B ~ Uniform[-β,β]` with **ω and β chosen by manual search**, and (optionally) 50% sparsity in `W` and `B` by masking at init.
- Task (3): tanh hidden, linear output, biases, target-network formula; `x ~ N(0,I)`, dataset fixed, 5000 test points, `B1,B2` scales selected manually; termination chosen by observing negligible gains.

Because ω/β (Task 2) and B-scales/target regime (Task 3) are explicitly chosen by **manual search** in the paper, this repo provides **sweep commands** and writes results to CSV so you can replicate the paper’s selection procedure without inventing hyperparameters.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce Task (1) / Fig. 1-style curves (linear 30–20–10)

```bash
python run.py linear --figure 1 --alg bp --seed 0
python run.py linear --figure 1 --alg fa --seed 0
```

Outputs: `results/linear_fig1_*.csv` (per-step NSE and angles).

## Reproduce Fig. 4 sweep over ω (linear 30–20–10)

```bash
python run.py linear --figure 4 --alg fa --n_trials 20 --seed 0
```

Outputs: `results/linear_fig4_*.csv` (per-step NSE and angles per omega and trial).

## Reproduce Task (2) MNIST

### 1) Hyperparameter sweep to select ω and β (paper: manual search)
```bash
python run.py mnist_sweep --omega_list 0.01,0.05,0.1,0.2 --beta_list 0.05,0.1,0.2 --seed 0 --max_examples 1500000
```

### 2) Run backprop and feedback alignment with your selected ω,β
```bash
python run.py mnist --alg bp --omega 0.1 --seed 0 --max_examples 1500000
python run.py mnist --alg fa --omega 0.1 --beta 0.1 --seed 0 --max_examples 1500000
```

### 3) 50% sparsity experiment (mask W and B at init and after each update)
```bash
python run.py mnist --alg fa --omega 0.1 --beta 0.1 --seed 0 --sparse50 --max_examples 1500000
```

Outputs: `results/mnist_*.csv` (test error and ∠(Δh_FA, Δh_BP) over time).

## Reproduce Task (3) Nonlinear function approximation

Because the paper states that the **target network regime** and **B1/B2 scales** are chosen manually, you should sweep.

```bash
python run.py nonlinear_sweep --seed 0 --steps 1500000 \
  --b1_list 0.05,0.1,0.2 --b2_list 0.05,0.1,0.2 --target_scale_list 0.5,1.0,2.0
```

Then run:
```bash
python run.py nonlinear --model 3 --alg bp --seed 0 --steps 1500000 --target_scale 1.0
python run.py nonlinear --model 3 --alg fa --seed 0 --steps 1500000 --target_scale 1.0 --b1_scale 0.1 --b2_scale 0.1
python run.py nonlinear --model 4 --alg bp --seed 0 --steps 1500000 --target_scale 1.0
python run.py nonlinear --model 4 --alg fa --seed 0 --steps 1500000 --target_scale 1.0 --b1_scale 0.1 --b2_scale 0.1
```

Outputs: `results/nonlinear_*.csv`

## Plotting

Plot utilities are in `src/plotting.py` (matplotlib) and can read the CSVs.

