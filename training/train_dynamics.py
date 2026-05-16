"""
Training script for the latent dynamics model.

Uses sequences of (z, a, z') tuples collected from the environment to
train the transformer dynamics model to predict future latent states.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models.dynamics.transformer import LatentDynamicsModel, RewardPredictor
from models.encoder.vae import VAE
from sim.env import RoboticArmEnv


class _SequenceDataset(Dataset):
    def __init__(self, sequences, seq_len: int):
        self.sequences = sequences
        self.seq_len = seq_len
        self.windows = []
        for ep_idx, ep in enumerate(sequences):
            T = ep["latents"].shape[0]
            for t in range(T - seq_len + 1):
                self.windows.append((ep_idx, t))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ep_idx, t = self.windows[idx]
        ep = self.sequences[ep_idx]
        sl = slice(t, t + self.seq_len)
        return {
            "latents": torch.from_numpy(ep["latents"][sl]),
            "actions": torch.from_numpy(ep["actions"][sl]),
            "rewards": torch.from_numpy(ep["rewards"][sl]),
        }


def _collect_latent_sequences(env, vae, device, n_episodes: int):
    sequences = []
    vae.eval()
    with torch.no_grad():
        for ep_idx in range(n_episodes):
            obs, _ = env.reset()
            done = False
            latents, actions, rewards = [], [], []
            while not done:
                img = (
                    torch.from_numpy(obs["image"])
                    .permute(2, 0, 1)
                    .float()
                    .unsqueeze(0)
                    / 255.0
                ).to(device)
                z = vae.encode(img).squeeze(0).cpu().numpy()

                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                latents.append(z)
                actions.append(action.astype(np.float32))
                rewards.append(np.float32(reward))

            if len(latents) > 1:
                sequences.append(
                    {
                        "latents": np.stack(latents),
                        "actions": np.stack(actions),
                        "rewards": np.array(rewards, dtype=np.float32),
                    }
                )
            if (ep_idx + 1) % 50 == 0:
                print(f"  Collected {ep_idx+1}/{n_episodes} episodes")
    return sequences


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_cfg = config["encoder"]
    dyn_cfg = config["dynamics"]
    latent_dim = enc_cfg["latent_dim"]

    # Load frozen encoder
    vae = VAE(latent_dim).to(device)
    vae.load_state_dict(torch.load("checkpoints/encoder.pt", map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    env = RoboticArmEnv(config)
    action_dim = env.action_space.shape[0]

    print("Collecting latent sequences...")
    sequences = _collect_latent_sequences(env, vae, device, n_episodes=500)
    env.close()
    print(f"Collected {len(sequences)} episodes")

    dataset = _SequenceDataset(sequences, seq_len=dyn_cfg["max_seq_len"])
    loader = DataLoader(
        dataset, batch_size=dyn_cfg["batch_size"], shuffle=True, drop_last=True
    )

    dynamics = LatentDynamicsModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        d_model=dyn_cfg["d_model"],
        nhead=dyn_cfg["nhead"],
        num_layers=dyn_cfg["num_layers"],
        max_seq_len=dyn_cfg["max_seq_len"],
    ).to(device)

    reward_pred = RewardPredictor(latent_dim).to(device)

    optimizer = torch.optim.Adam(
        list(dynamics.parameters()) + list(reward_pred.parameters()),
        lr=dyn_cfg["learning_rate"],
    )

    log_dir = os.path.join(config["evaluation"]["log_dir"], "dynamics")
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    for epoch in range(dyn_cfg["epochs"]):
        epoch_loss = 0.0
        for batch in loader:
            z_seq = batch["latents"].to(device)   # (B, T, latent_dim)
            a_seq = batch["actions"].to(device)   # (B, T, action_dim)
            r_seq = batch["rewards"].to(device)   # (B, T)

            # Multi-step teacher forcing: predict z_{t+1} at every position.
            # Causal mask ensures position t only sees steps 0..t-1.
            # This trains the model on short contexts (step 1 from step 0, etc.)
            # which is exactly the regime used during imagination rollouts.
            z_pred = dynamics(z_seq[:, :-1], a_seq[:, :-1])   # (B, T-1, latent_dim)
            dyn_loss = F.mse_loss(z_pred, z_seq[:, 1:])

            # Reward prediction from all but last state
            r_pred = reward_pred(z_seq[:, :-1]).squeeze(-1)   # (B, T-1)
            rew_loss = F.mse_loss(r_pred, r_seq[:, :-1])

            loss = dyn_loss + rew_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(dynamics.parameters(), max_norm=1.0)
            optimizer.step()

            writer.add_scalar("dynamics/train_loss", loss.item(), global_step)
            writer.add_scalar("dynamics/dyn_loss", dyn_loss.item(), global_step)
            writer.add_scalar("dynamics/rew_loss", rew_loss.item(), global_step)
            epoch_loss += loss.item()
            global_step += 1

        print(f"Epoch {epoch+1}/{dyn_cfg['epochs']}  loss={epoch_loss/len(loader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(dynamics.state_dict(), "checkpoints/dynamics.pt")
    torch.save(reward_pred.state_dict(), "checkpoints/reward_predictor.pt")
    writer.close()
    print("Dynamics saved to checkpoints/dynamics.pt")
    print("Reward predictor saved to checkpoints/reward_predictor.pt")


if __name__ == "__main__":
    import sys

    from utils.config import load_config

    config = load_config(sys.argv[1])
    train(config)
