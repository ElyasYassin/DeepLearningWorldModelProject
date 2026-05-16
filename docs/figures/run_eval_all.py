"""
Runs evaluation on all trained models and writes eval_results.json.
That JSON is consumed by gen_results_figure.py to build the bar chart.

Usage (from project root):
    python docs/figures/run_eval_all.py configs/default.yaml

Edit MODEL_PATHS below to match your actual checkpoint locations.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import yaml

from evaluation.evaluate import evaluate as _evaluate
from evaluation.metrics import tracking_accuracy, fluidity
from sim.env import RoboticArmEnv

OUT = os.path.join(os.path.dirname(__file__), "eval_results.json")

# ── Edit these paths to match your saved checkpoints ─────────────────────────
MODEL_PATHS = {
    "PPO":             "trained_models/ppo_target_tracking",
    "PPO-LSTM":        "trained_models/ppo_lstm_target_tracking",
    "PPO-ResNet18":    "trained_models/ppo_resnet18_target_tracking",
    "PPO-ResNet18-FT": "trained_models/ppo_resnet18_target_tracking_ft",
    # World model uses a custom eval (not SB3), handled separately below
}

N_EPISODES = 20


def eval_sb3_model(config, model_path, n_episodes):
    """Run SB3-based evaluation and return aggregate stats."""
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO

    env = RoboticArmEnv(config)
    try:
        model = RecurrentPPO.load(model_path, env=env)
        is_recurrent = True
    except Exception:
        model = PPO.load(model_path, env=env)
        is_recurrent = False

    ep_rewards, ep_distances, ep_fluidities = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        distances, joint_positions = [], []
        lstm_states = None
        episode_start = True

        while not done:
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs, state=lstm_states,
                    episode_start=episode_start, deterministic=True,
                )
                episode_start = False
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            distances.append(info["eef_to_target"])
            joint_positions.append(info["joint_positions"])

        ep_rewards.append(total_reward)
        ep_distances.append(tracking_accuracy(distances))
        jt = np.stack(joint_positions)
        if jt.shape[0] > 3:
            ep_fluidities.append(fluidity(jt, control_freq=20.0))

    env.close()
    return {
        "ep_return_mean":    float(np.mean(ep_rewards)),
        "ep_return_std":     float(np.std(ep_rewards)),
        "tracking_acc_mean": float(np.mean(ep_distances)),
        "tracking_acc_std":  float(np.std(ep_distances)),
        "fluidity_mean":     float(np.mean(ep_fluidities)) if ep_fluidities else None,
        "fluidity_std":      float(np.std(ep_fluidities))  if ep_fluidities else None,
    }


def eval_world_model(config, n_episodes):
    """Evaluate the world model policy (custom actor-critic, not SB3)."""
    import torch
    from models.encoder.vae import VAE
    from models.controller.policy import Policy
    from sim.env import RoboticArmEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = config["encoder"]["latent_dim"]
    ctrl_cfg   = config["controller"]

    vae = VAE(latent_dim).to(device)
    vae.load_state_dict(torch.load("checkpoints/encoder.pt", map_location=device))
    vae.eval()

    env = RoboticArmEnv(config)
    action_dim = env.action_space.shape[0]

    policy = Policy(latent_dim, action_dim, ctrl_cfg["hidden_dim"]).to(device)
    policy.load_state_dict(torch.load("checkpoints/policy.pt", map_location=device))
    policy.eval()

    ep_rewards, ep_distances, ep_fluidities = [], [], []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            distances, joint_positions = [], []

            while not done:
                img = (
                    torch.from_numpy(obs["image"])
                    .permute(2, 0, 1).float().unsqueeze(0) / 255.0
                ).to(device)
                z = vae.encode(img)
                action = policy(z).squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                distances.append(info["eef_to_target"])
                joint_positions.append(info["joint_positions"])

            ep_rewards.append(total_reward)
            ep_distances.append(tracking_accuracy(distances))
            jt = np.stack(joint_positions)
            if jt.shape[0] > 3:
                ep_fluidities.append(fluidity(jt, control_freq=20.0))

    env.close()
    return {
        "ep_return_mean":    float(np.mean(ep_rewards)),
        "ep_return_std":     float(np.std(ep_rewards)),
        "tracking_acc_mean": float(np.mean(ep_distances)),
        "tracking_acc_std":  float(np.std(ep_distances)),
        "fluidity_mean":     float(np.mean(ep_fluidities)) if ep_fluidities else None,
        "fluidity_std":      float(np.std(ep_fluidities))  if ep_fluidities else None,
    }


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    results = {}

    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path + ".zip") and not os.path.exists(path):
            print(f"Skipping {name} — checkpoint not found: {path}")
            continue
        print(f"\nEvaluating {name} ({N_EPISODES} episodes)...")
        results[name] = eval_sb3_model(config, path, N_EPISODES)
        print(f"  tracking_acc = {results[name]['tracking_acc_mean']:.4f} m")

    if os.path.exists("checkpoints/policy.pt"):
        print(f"\nEvaluating World Model ({N_EPISODES} episodes)...")
        results["World Model"] = eval_world_model(config, N_EPISODES)
        print(f"  tracking_acc = {results['World Model']['tracking_acc_mean']:.4f} m")
    else:
        print("Skipping World Model — checkpoints/policy.pt not found")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved eval results to {OUT}")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(cfg)
