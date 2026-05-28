# CLI reference

After `pip install -e ".[dev]"`, the `feedback-alignment` console script is on
your PATH. This document covers the higher-level patterns and conventions.
For per-flag descriptions, run `feedback-alignment <subcommand> --help`.

## Subcommands at a glance

| Subcommand          | Task         | Purpose                                                                     |
| ------------------- | ------------ | --------------------------------------------------------------------------- |
| `linear`            | 1            | One linear-task run. Reproduces paper Fig 1 (single setting) or Fig 4 (FA sweep over omega). |
| `mnist`             | 2            | One MNIST run under BP, FA, or shallow learning at chosen `omega` and `beta`. |
| `mnist-sweep`       | 2            | Grid over `(omega, beta)` for MNIST. Writes one row per cell with final test error. |
| `nonlinear`         | 3            | One nonlinear teacher-student run at a chosen depth, target scale, and feedback scales. |
| `nonlinear-sweep`   | 3            | Grid over `(target_scale, b1, b2)` for the nonlinear task.                  |

## Workflows

### Reproduce a paper figure directly (Task 1)

Task 1 hyperparameters are pinned by the paper, so the runs are direct:

```bash
feedback-alignment linear --figure 1 --alg bp
feedback-alignment linear --figure 1 --alg fa
feedback-alignment linear --figure 4
```

### Manual search, then a main run (Tasks 2 and 3)

The paper picks the init and feedback scales by manual search. Reproduce that
flow with the `*-sweep` subcommand:

```bash
# Task 2
feedback-alignment mnist-sweep --omega 0.05 0.1 0.2 --beta 0.05 0.1 0.2
# Pick the row in results/mnist_sweep_seed0.csv with the lowest final test
# error, then:
feedback-alignment mnist --alg fa --omega <BEST_OMEGA> --beta <BEST_BETA>

# Task 3
feedback-alignment nonlinear-sweep --target-scale 0.5 1.0 2.0 \
    --b1 0.05 0.1 0.2 --b2 0.05 0.1 0.2
feedback-alignment nonlinear --model 4 --alg fa \
    --target-scale <BEST_TS> --b1-scale <BEST_B1> --b2-scale <BEST_B2>
```

### Reproduce every figure in the report

The bundled script runs the exact sweep that produced `report/report.pdf`:

```bash
./run_all.sh
```

It writes CSVs to `results/`, then regenerates `plots/`.

### Custom experiment

All flags have sensible defaults matching the paper. Override only what you
care about:

```bash
feedback-alignment mnist --alg fa --omega 0.1 --beta 0.2 \
    --max-examples 500000 --eval-every 25000 --seed 42
```

## Output filename schema

CSV outputs are written to `--results-dir` (default `results/`) with names
that encode the run configuration. The scheme:

| Subcommand        | Filename pattern                                                                        |
| ----------------- | --------------------------------------------------------------------------------------- |
| `linear`          | `linear_fig1_{alg}_seed{S}_eta{ETA}_omega{OMEGA}_b{B}.csv`                              |
| `linear --figure 4` | `linear_fig4_seed{S}.csv`                                                             |
| `mnist`           | `mnist_{alg}_seed{S}_omega{OMEGA}[_beta{BETA}][_sparse50].csv`                          |
| `mnist-sweep`     | `mnist_sweep_seed{S}.csv`                                                               |
| `nonlinear`       | `nonlinear_model{M}_{alg}_seed{S}_ts{TS}[_b1{B1}][_b2{B2}].csv`                         |
| `nonlinear-sweep` | `nonlinear_sweep_seed{S}.csv`                                                           |

`None`-valued slug fields (e.g. `beta` on a BP run) are skipped, so you never
get `betaNone` in a filename.

## CSV column schemas

### `linear_fig1_*.csv`

| Column                | Meaning                                              |
| --------------------- | ---------------------------------------------------- |
| `step`                | Training step (0-indexed).                           |
| `nse`                 | Normalized squared error on the current example.     |
| `angle_delta_h_fa_bp` | Angle (radians) between the FA and BP hidden signals. |

### `linear_fig4_*.csv`

| Column                  | Meaning                                                                        |
| ----------------------- | ------------------------------------------------------------------------------ |
| `omega`                 | Init scale for this row.                                                       |
| `trial`                 | Trial index (0 to `n_trials-1`).                                               |
| `step`                  | Training step.                                                                 |
| `nse`                   | NSE on the current example.                                                    |
| `angle_delta_h_fa_bp`   | Angle (rad) between FA and BP hidden signals.                                  |
| `angle_delta_h_fa_pbp`  | Angle (rad) between FA and pseudo-backprop (`pinv(W) e`) hidden signals.       |

### `mnist_*.csv`

| Column                       | Meaning                                                                  |
| ---------------------------- | ------------------------------------------------------------------------ |
| `examples_seen`              | Total online examples processed when this row was logged.                |
| `test_error_pct`             | Test error percentage on the 10k MNIST test set at this point.           |
| `mean_angle_delta_h_fa_bp`   | Running mean of the FA-vs-BP hidden-signal angle up to this point (rad). `NaN` on BP runs that did not pass a feedback matrix. |

### `mnist_sweep_*.csv`

| Column                  | Meaning                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| `alg`                   | `bp` (baseline row) or `fa`.                                                              |
| `omega`                 | W init scale for this row.                                                                |
| `beta`                  | B feedback scale for this row (empty on BP rows).                                         |
| `final_test_error_pct`  | Final test error after `--max-examples` examples.                                         |

### `nonlinear_*.csv`

| Column           | Meaning                                                              |
| ---------------- | -------------------------------------------------------------------- |
| `examples_seen`  | Total online examples processed when this row was logged.            |
| `test_nse`       | Mean per-example NSE on the held-out test set at this point.         |

### `nonlinear_sweep_*.csv`

| Column            | Meaning                                                       |
| ----------------- | ------------------------------------------------------------- |
| `model`           | Student depth (3 or 4).                                       |
| `alg`             | `bp`, `fa`, or `shallow`.                                     |
| `target_scale`    | Teacher weight scale for this row.                            |
| `b1_scale`        | B1 feedback scale (empty on non-FA rows).                     |
| `b2_scale`        | B2 feedback scale (empty on non-FA rows or 3-layer students). |
| `final_test_nse`  | Final test NSE.                                               |

## Notes

- Every subcommand prints the path of the written CSV to stdout. Pipe it into
  scripts if you want to chain runs.
- `--seed` propagates into independent streams for data construction, weight
  initialisation, feedback construction, test-set construction, and example
  ordering. Two runs at the same seed share student initialisation exactly,
  which is useful when comparing algorithms.
- `--quiet` suppresses tqdm progress bars but does not silence the final
  output path or any error messages.
- The MNIST data is downloaded on first use into `--data-dir` (default
  `data/mnist`). When launching many MNIST runs in parallel, prime the cache
  with one serial run first or you will hit a download race.
