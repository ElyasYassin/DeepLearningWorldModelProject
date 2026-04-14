# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Find My Pen" — a robotic arm object-tracking system that learns purely from visual observations using a world model. The robot predicts future states in latent space rather than reacting frame-by-frame.

**Team:** Elyas Larfi, Sadiqah Al-Masyabi, Amrit Prajapati — UCD Deep Learning, Spring 2026

**Status (as of Apr 2026):** Phase 3 — controller training and baseline benchmarking. Most modules are stubs (`raise NotImplementedError`); implementation is the active work.

---

## Running training scripts

All training entry points accept a YAML config path as the first argument:

```bash
python training/train_encoder.py configs/default.yaml
python training/train_dynamics.py configs/default.yaml
python training/train_controller.py configs/default.yaml

# Baselines (model-free, no world model)
python baselines/ppo_baseline.py configs/default.yaml
python baselines/sac_baseline.py configs/default.yaml
```

---

## Architecture

The pipeline is a three-stage world model (inspired by Dreamer / World Models):

```
RGB frame (64×64) → VAE encoder → latent z (dim 128)
(z_t, a_t) sequence → Transformer dynamics model → predicted z_{t+1}
z → MLP policy → action
```

**Training order matters:** encoder must be trained first (produces latent data), then dynamics model (needs latent sequences), then controller (needs the dynamics model for imagined rollouts).

### Key design choice
The dynamics model uses a **transformer** (not an RNN like the original Dreamer), with a context window of 64 steps. The hypothesis is better long-range temporal dependency capture.

### Component locations
| Component | File | Class |
|-----------|------|-------|
| VAE encoder | `models/encoder/vae.py` | `VAE` (uses `Encoder`/`Decoder` sub-modules) |
| Latent dynamics | `models/dynamics/transformer.py` | `LatentDynamicsModel` |
| RL policy | `models/controller/policy.py` | `Policy` + `ValueFunction` (actor-critic) |
| Sim wrapper | `sim/env.py` | `RoboticArmEnv` (Gym-compatible interface) |
| Evaluation | `evaluation/metrics.py` | Three functions: `tracking_accuracy`, `response_time`, `fluidity` |
| Logger | `evaluation/logger.py` | `Logger` |

### Simulation engine
MuJoCo is the only supported engine (`env.engine: mujoco`). `RoboticArmEnv` wraps robosuite's `TargetTracking` environment behind a `gymnasium.Env`-compatible interface (`reset`, `step`, `render`, `close`).

**Robosuite source:** `git@github.com:ElyasYassin/robosuite.git`

Install with:
```bash
pip install git+ssh://git@github.com/ElyasYassin/robosuite.git
```

The task uses `moving_target=False` (static pen position) for the initial training phase. Switch to `moving_target=True` for the dynamic tracking phase.

### Controller training loop
Follows the **Dreamer-style** approach: the policy is trained entirely in latent space by unrolling `imagination_horizon` steps (default 15) using the frozen dynamics model, without interacting with the real environment during policy updates. The reward signal is Euclidean distance between the end-effector and the pen target.

### Experiment roadmap

Three progressive stages on the static TargetTracking task:

| Stage | Obs into RL | Dynamics | Real env interaction during policy training |
|-------|-------------|----------|---------------------------------------------|
| 1 — Pixel PPO | Raw 64×64 RGB | None | Every step |
| 2 — Latent PPO | Pretrained encoder features `z` | None | Every step |
| 3 — World model | Pretrained encoder `z` | Transformer (`z_{t+1}` prediction) | Data collection only |

**Stage 2** requires: a pretrained vision encoder (loaded from external weights, not trained in this repo) + `sim/latent_env.py` wrapper that replaces the image obs with the encoder's output features. Policy uses `MlpPolicy`.

**Stage 3 pipeline:**
```
RGB frame → pretrained encoder (frozen) → z_t
(z_t, a_t) sequences → LatentDynamicsModel (transformer) → z_{t+1}
Policy trained entirely in imagination for imagination_horizon=15 steps (Dreamer-style)
```

**Stage 3 training order:**
1. Collect rollouts in real env → encode frames offline → `(z_t, a_t)` sequences
2. Train `LatentDynamicsModel` on `(z_t, a_t) → z_{t+1}` supervised loss (`train_dynamics.py`)
3. Freeze encoder + dynamics; train MLP policy via imagined rollouts (`train_controller.py`)

**Hypothesis:** Stage 3 is more sample-efficient — policy training happens in imagination, not the slow real simulator.

### Baselines
PPO and SAC via `stable-baselines3`, trained on raw observations (no world model). These serve as comparison targets for the three evaluation metrics: tracking accuracy, response time, and fluidity (jerk minimization).

---

## Config

`configs/default.yaml` is the single source of truth for all hyperparameters. Create experiment-specific YAML files that override only the relevant keys; pass the experiment config to the training script.

Key defaults: `latent_dim=128`, `d_model=256`, `nhead=4`, `num_layers=4`, `max_seq_len=64`, `imagination_horizon=15`, `log_dir=logs/`.
