#!/usr/bin/env bash
set -e
mkdir -p results

echo "=== Task 1: linear ==="
feedback-alignment linear --figure 1 --alg bp --quiet
feedback-alignment linear --figure 1 --alg fa --quiet
feedback-alignment linear --figure 4 --quiet

echo "=== Task 2: MNIST grid (13 runs in parallel, 7 at a time) ==="
parallel -j 7 --line-buffer ::: \
    "feedback-alignment mnist --alg bp --omega 0.05 --quiet" \
    "feedback-alignment mnist --alg bp --omega 0.1  --quiet" \
    "feedback-alignment mnist --alg bp --omega 0.2  --quiet" \
    "feedback-alignment mnist --alg fa --omega 0.05 --beta 0.05 --quiet" \
    "feedback-alignment mnist --alg fa --omega 0.05 --beta 0.1  --quiet" \
    "feedback-alignment mnist --alg fa --omega 0.05 --beta 0.2  --quiet" \
    "feedback-alignment mnist --alg fa --omega 0.1  --beta 0.05 --quiet" \
    "feedback-alignment mnist --alg fa --omega 0.1  --beta 0.1  --quiet" \
    "feedback-alignment mnist --alg fa --omega 0.1  --beta 0.2  --quiet" \
    "feedback-alignment mnist --alg fa --omega 0.2  --beta 0.05 --quiet" \
    "feedback-alignment mnist --alg fa --omega 0.2  --beta 0.1  --quiet" \
    "feedback-alignment mnist --alg fa --omega 0.2  --beta 0.2  --quiet" \
    "feedback-alignment mnist --alg fa --omega 0.1  --beta 0.1  --sparse50 --quiet"

echo "=== Task 3: nonlinear grid (14 runs in parallel, 7 at a time) ==="
parallel -j 7 --line-buffer ::: \
    "feedback-alignment nonlinear --model 3 --alg bp      --target-scale 1.0 --quiet" \
    "feedback-alignment nonlinear --model 3 --alg shallow --target-scale 1.0 --quiet" \
    "feedback-alignment nonlinear --model 4 --alg bp      --target-scale 1.0 --quiet" \
    "feedback-alignment nonlinear --model 4 --alg shallow --target-scale 1.0 --quiet" \
    "feedback-alignment nonlinear --model 3 --alg fa --target-scale 1.0 --b1-scale 0.05 --quiet" \
    "feedback-alignment nonlinear --model 3 --alg fa --target-scale 1.0 --b1-scale 0.1  --quiet" \
    "feedback-alignment nonlinear --model 3 --alg fa --target-scale 1.0 --b1-scale 0.2  --quiet" \
    "feedback-alignment nonlinear --model 4 --alg fa --target-scale 1.0 --b1-scale 0.05 --b2-scale 0.05 --quiet" \
    "feedback-alignment nonlinear --model 4 --alg fa --target-scale 1.0 --b1-scale 0.05 --b2-scale 0.1  --quiet" \
    "feedback-alignment nonlinear --model 4 --alg fa --target-scale 1.0 --b1-scale 0.1  --b2-scale 0.05 --quiet" \
    "feedback-alignment nonlinear --model 4 --alg fa --target-scale 1.0 --b1-scale 0.1  --b2-scale 0.1  --quiet" \
    "feedback-alignment nonlinear --model 4 --alg fa --target-scale 1.0 --b1-scale 0.1  --b2-scale 0.2  --quiet" \
    "feedback-alignment nonlinear --model 4 --alg fa --target-scale 1.0 --b1-scale 0.2  --b2-scale 0.1  --quiet" \
    "feedback-alignment nonlinear --model 4 --alg fa --target-scale 1.0 --b1-scale 0.2  --b2-scale 0.2  --quiet"

echo "=== Plots ==="
python plot_all.py
python plot_all.py --paper

echo "=== Done ==="
