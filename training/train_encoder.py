"""
Training script for the VAE vision encoder.

Collects image observations from the simulation environment and trains the
VAE to reconstruct them, learning a compact latent representation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models.encoder.vae import VAE
from sim.env import RoboticArmEnv


class _ImageDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


def _collect_frames(env, n_episodes: int):
    frames = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            # obs['image'] is (64, 64, 3) uint8
            frame = torch.from_numpy(obs["image"]).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
            done = terminated or truncated
    return frames


def _elbo_loss(recon, x, mu, log_var, beta: float = 1.0):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.shape[0]
    kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()
    return recon_loss + beta * kl_loss


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_cfg = config["encoder"]
    latent_dim = enc_cfg["latent_dim"]

    env = RoboticArmEnv(config)
    print("Collecting frames...")
    frames = _collect_frames(env, n_episodes=200)
    env.close()
    print(f"Collected {len(frames)} frames")

    dataset = _ImageDataset(frames)
    loader = DataLoader(dataset, batch_size=enc_cfg["batch_size"], shuffle=True)

    vae = VAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, vae.parameters()),
        lr=enc_cfg["learning_rate"],
    )

    log_dir = os.path.join(config["evaluation"]["log_dir"], "encoder")
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    for epoch in range(enc_cfg["epochs"]):
        epoch_loss = 0.0
        for batch in loader:
            x = batch.to(device)
            recon, mu, log_var = vae(x)
            loss = _elbo_loss(recon, x, mu, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("encoder/train_loss", loss.item(), global_step)
            epoch_loss += loss.item()
            global_step += 1

        print(f"Epoch {epoch+1}/{enc_cfg['epochs']}  loss={epoch_loss/len(loader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(vae.state_dict(), "checkpoints/encoder.pt")
    writer.close()
    print("Encoder saved to checkpoints/encoder.pt")


if __name__ == "__main__":
    import sys

    import yaml

    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)
    train(config)
