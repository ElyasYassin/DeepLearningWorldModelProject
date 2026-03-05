# Find My Pen: Robotic Arm Object Tracking with Learned World Models

**Elyas Larfi · Sadiqah Al-Masyabi · Amrit Prajapati**  
University of Colorado Denver — Deep Learning Project, Spring 2026

---

## Overview

This project trains a robotic arm to track a moving object (a pen) in real time using reinforcement learning and a learned world model. The system learns purely from visual observations in a simulated environment, with the long-term goal of transferring the learned behavior to a real robotic arm.

Rather than reacting to each frame independently, the robot learns to *predict* future states and plan actions accordingly — enabling smoother, more robust tracking behavior.

---

## Architecture

The system is composed of three learned components:

```
RGB Image → [ Vision Encoder ] → latent z
latent z + action → [ Dynamics Model ] → predicted future latent z'
latent z → [ RL Controller ] → action
```

### 1. Vision Encoder
A Variational Autoencoder (VAE) that compresses high-dimensional RGB camera frames into a compact latent vector `z`. This latent space captures the essential visual information needed for tracking (e.g., pen position, end-effector position).

### 2. Latent Dynamics Model
A sequence model trained to predict future latent states given the current latent state and the action taken. We explore **transformer-style sequence models** as an alternative to the RNN used in the original World Models and Dreamer papers, hypothesizing that transformers better capture long-range temporal dependencies in state-action sequences.

### 3. RL Controller
An RL policy trained entirely in latent space to select actions that maximize cumulative reward. The reward function is designed to minimize the Euclidean distance between the robotic arm's end effector and the moving target.

---

## Simulation Environment

The digital twin is built in one of the following simulation engines (to be finalized in Phase 1):

| Engine | Notes |
|--------|-------|
| [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) | Photorealistic, strong ROS2 support |
| [MuJoCo](https://mujoco.org/) | Lightweight, widely used in RL research |
| [Gazebo](https://gazebosim.org/) | Open-source, strong ROS ecosystem |

---

## Baselines

Performance is compared against two model-free RL baselines trained with [`stable-baselines3`](https://github.com/DLR-RM/stable-baselines3):

- **PPO** — Proximal Policy Optimization
- **SAC** — Soft Actor-Critic

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Tracking Accuracy | Average end-effector distance to target over an episode |
| Response Time | Latency between target movement and arm correction |
| Fluidity | Smoothness of joint trajectories (e.g., jerk minimization) |

Testing is conducted under varying conditions including different object sizes and movement speeds to assess robustness.

---

## Project Timeline

| Phase | Dates | Goal |
|-------|-------|------|
| Phase 1 | Feb 14 – Mar 14 | Literature review + digital twin setup; realistic robotic arm simulation in a photorealistic environment |
| Phase 2 | Mar 14 – Apr 11 | Vision encoder + dynamics model integration; visual pipeline outputting pen-centric latent features |
| Phase 3 | Apr 11 – May 15 | Controller training; benchmark against baselines on all evaluation metrics |

---

## Related Work

- [Continuous Control with Deep RL](https://arxiv.org/abs/1509.02971)
- [Survey of Deep RL for Motor Planning](https://arxiv.org/abs/2105.14218)
- [Deep Visual Foresight Planning](https://arxiv.org/abs/1610.00696)
- [World Models](https://arxiv.org/abs/1803.10122)
- [Dreamer: RL with Latent World Models](https://arxiv.org/abs/1912.01603)
- [Hand-Eye Coordination for Robotic Grasping](https://arxiv.org/abs/1603.02199)
- [End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/abs/1504.00702)
- [Domain Randomization](https://arxiv.org/abs/1703.06907)
- [Progressive Neural Networks](https://arxiv.org/abs/1606.04671)

---

## Repository Structure

```
DeepLearningWorldModelProject/
├── sim/                  # Simulation environment setup and wrappers
├── models/
│   ├── encoder/          # VAE vision encoder
│   ├── dynamics/         # Latent dynamics model (transformer)
│   └── controller/       # RL policy
├── training/             # Training scripts for each component
├── baselines/            # PPO and SAC baseline training
├── evaluation/           # Metric logging and analysis
├── configs/              # Hyperparameter and experiment configs
└── readme.md
```

---

## Getting Started

> Setup instructions will be added as the simulation environment and dependencies are finalized in Phase 1.

---

## License

This project is for academic purposes at the University of Colorado Denver.
