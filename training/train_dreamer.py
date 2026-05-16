"""
DreamerV3 training script.

Trains the full world model (encoder + RSSM + decoders) and actor-critic
jointly. The policy is trained entirely in imagination using RSSM rollouts.

Usage:
    python -m training.train_dreamer configs/dreamer.yaml
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import signal
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.world_model.world_model import WorldModel
from models.controller.actor_critic import DreamerActor, DreamerCritic, ReturnNormalizer
from models.world_model.utils import lambda_returns, symexp
from sim.env import RoboticArmEnv
from utils.config import load_config


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class EpisodeReplayBuffer:
    """Stores full episodes and samples fixed-length sequences."""

    def __init__(self, capacity: int, seq_len: int):
        self.capacity = capacity  # max total steps across all stored episodes
        self.seq_len = seq_len
        self._episodes: list = []
        self._total_steps: int = 0

    def add(self, obs, actions, rewards, dones) -> None:
        """
        obs:     (T, H, W, 3) uint8
        actions: (T, action_dim) float32
        rewards: (T,) float32
        dones:   (T,) bool
        """
        ep = dict(obs=obs, actions=actions, rewards=rewards, dones=dones)
        self._episodes.append(ep)
        self._total_steps += len(rewards)
        # Evict oldest episodes when over capacity
        while self._total_steps > self.capacity and len(self._episodes) > 1:
            removed = self._episodes.pop(0)
            self._total_steps -= len(removed["rewards"])

    def sample(self, batch_size: int, device: torch.device) -> dict:
        valid = [e for e in self._episodes if len(e["rewards"]) >= self.seq_len]
        obs_l, act_l, rew_l, cont_l = [], [], [], []
        for _ in range(batch_size):
            ep = random.choice(valid)
            T = len(ep["rewards"])
            t0 = random.randint(0, T - self.seq_len)
            sl = slice(t0, t0 + self.seq_len)
            obs_l.append(ep["obs"][sl])
            act_l.append(ep["actions"][sl])
            rew_l.append(ep["rewards"][sl])
            # continue = 1 everywhere except the last step if done
            cont = (~ep["dones"][sl]).astype(np.float32)
            cont_l.append(cont)

        # obs: (B, T, H, W, 3) uint8 → (B, T, 3, H, W) float in [0,1]
        obs_t = (
            torch.from_numpy(np.stack(obs_l))
            .permute(0, 1, 4, 2, 3)
            .float()
            .div(255.0)
            .to(device)
        )
        return {
            "obs":     obs_t,
            "actions": torch.from_numpy(np.stack(act_l)).to(device),
            "rewards": torch.from_numpy(np.stack(rew_l)).to(device),
            "cont":    torch.from_numpy(np.stack(cont_l)).to(device),
        }

    @property
    def size(self) -> int:
        return self._total_steps


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect_episode(env, actor, device, world_model, explore: bool = False):
    """Run one episode. If explore=True or actor is None, use random actions."""
    obs_list, act_list, rew_list, done_list = [], [], [], []
    obs_dict, _ = env.reset()
    done = False

    # Seed RSSM state for online action selection (state = h for GRU, ctx for Transformer)
    state, z = world_model.rssm.initial_state(1, device)

    while not done:
        img = (
            torch.from_numpy(obs_dict["image"])
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .div(255.0)
            .to(device)
        )
        embed = world_model.encoder(img)           # (1, embed_dim)
        a_prev = (
            torch.from_numpy(act_list[-1]).float().unsqueeze(0).to(device)
            if act_list
            else torch.zeros(1, env.action_space.shape[0], device=device)
        )
        state, _, post_lg = world_model.rssm.obs_step(state, z, a_prev, embed)

        from models.world_model.utils import straight_through_sample
        z = straight_through_sample(post_lg)

        if explore or actor is None:
            action = env.action_space.sample()
        else:
            feat = world_model.rssm._feat(
                world_model.rssm._transformer_h(state)
                if hasattr(world_model.rssm, "_transformer_h") else state,
                z,
            )
            with torch.no_grad():
                action, _, _ = actor(feat)
            action = action.squeeze(0).cpu().numpy()

        obs_list.append(obs_dict["image"].copy())
        act_list.append(action.astype(np.float32))

        obs_dict, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rew_list.append(np.float32(reward))
        done_list.append(bool(done))

    return (
        np.stack(obs_list),
        np.stack(act_list),
        np.array(rew_list, dtype=np.float32),
        np.array(done_list, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dr_cfg = config["dreamer"]
    eval_cfg = config["evaluation"]

    env = RoboticArmEnv(config)
    action_dim = env.action_space.shape[0]

    # Build models
    wm = WorldModel(
        action_dim=action_dim,
        image_size=config["env"]["image_size"],
        depth=dr_cfg["depth"],
        embed_dim=dr_cfg["embed_dim"],
        h_dim=dr_cfg["h_dim"],
        stoch_size=dr_cfg["stoch_size"],
        stoch_classes=dr_cfg["stoch_classes"],
        hidden_dim=dr_cfg["hidden_dim"],
        kl_scale=dr_cfg["kl_scale"],
        free_bits=dr_cfg["free_bits"],
        kl_balance=dr_cfg["kl_balance"],
        encoder_type=dr_cfg.get("encoder_type", "cnn"),
        pretrained_encoder=dr_cfg.get("pretrained_encoder", False),
        freeze_encoder=dr_cfg.get("freeze_encoder", False),
        dynamics_type=dr_cfg.get("dynamics_type", "gru"),
        d_model=dr_cfg.get("d_model", 512),
        nhead=dr_cfg.get("nhead", 8),
        num_layers=dr_cfg.get("num_layers", 4),
        max_ctx_len=dr_cfg.get("max_ctx_len", 64),
    ).to(device)

    feat_dim = wm.feat_dim
    actor = DreamerActor(feat_dim, action_dim, dr_cfg["hidden_dim"]).to(device)
    critic = DreamerCritic(feat_dim, dr_cfg["hidden_dim"]).to(device)

    wm_opt = torch.optim.Adam(wm.parameters(), lr=dr_cfg["wm_lr"])
    actor_opt = torch.optim.Adam(actor.parameters(), lr=dr_cfg["actor_lr"])
    critic_opt = torch.optim.Adam(critic.parameters(), lr=dr_cfg["critic_lr"])

    replay = EpisodeReplayBuffer(
        capacity=dr_cfg["replay_capacity"],
        seq_len=dr_cfg["seq_len"],
    )

    normalizer = ReturnNormalizer()

    # Auto-number run dirs so TensorBoard runs stay separate
    base = os.path.join(eval_cfg["log_dir"], "dreamer")
    run_num = 1
    while os.path.exists(f"{base}_{run_num}"):
        run_num += 1
    log_dir = f"{base}_{run_num}"
    checkpoint_dir = f"checkpoints/dreamer_{run_num}"
    print(f"Run {run_num} — logs: {log_dir}  checkpoints: {checkpoint_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    # Save on Ctrl+C
    _stop = [False]
    def _handle_interrupt(sig, frame):
        print("\nCtrl+C received — saving and exiting after this episode.")
        _stop[0] = True
    signal.signal(signal.SIGINT, _handle_interrupt)

    # Prefill replay buffer with random episodes
    print(f"Prefilling replay buffer ({dr_cfg['prefill_episodes']} episodes)...")
    for i in range(dr_cfg["prefill_episodes"]):
        ep = _collect_episode(env, actor=None, device=device, world_model=wm, explore=True)
        replay.add(*ep)
    print(f"  Replay buffer: {replay.size} steps")

    H = dr_cfg["imagination_horizon"]
    gamma = dr_cfg["gamma"]
    lam = dr_cfg["lam"]
    ent_coef = dr_cfg["entropy_coef"]
    batch_size = dr_cfg["batch_size"]
    max_env_steps = dr_cfg["max_env_steps"]

    env_steps = 0
    grad_step = 0
    while env_steps < max_env_steps and not _stop[0]:
        # Collect one episode with the current actor
        ep = _collect_episode(
            env, actor, device, wm,
            explore=(env_steps < dr_cfg["explore_steps"]),
        )
        replay.add(*ep)
        env_steps += len(ep[2])

        grad_updates = dr_cfg["grad_steps_per_episode"]
        for g in range(grad_updates):
            # ---- World model update ----
            batch = replay.sample(batch_size, device)
            wm_losses, seed_feats = wm(
                batch["obs"],
                batch["actions"],
                batch["rewards"],
                batch["cont"],
            )
            wm_opt.zero_grad()
            wm_losses["total"].backward()
            nn.utils.clip_grad_norm_(wm.parameters(), max_norm=10.0)
            wm_opt.step()

            # ---- Actor-critic update (imagination) ----
            B_ac = min(batch_size, seed_feats.shape[0] * seed_feats.shape[1])
            flat_feats = seed_feats.flatten(0, 1)
            idx = torch.randperm(flat_feats.shape[0], device=device)[:B_ac]
            init_feat = flat_feats[idx]

            if not torch.isfinite(init_feat).all():
                continue

            imag_feats, imag_actions, terminal_feat = wm.rssm.imagine(init_feat, actor, H)

            with torch.no_grad():
                imag_rewards = wm.reward_dec.predict(imag_feats)
                imag_cont    = wm.cont_dec.predict(imag_feats)

            all_feats = torch.cat([imag_feats, terminal_feat.unsqueeze(1)], dim=1)
            with torch.no_grad():
                values = critic.predict(all_feats)

            targets = lambda_returns(imag_rewards, imag_cont, values, gamma, lam)
            normalizer.update(targets)

            critic_opt.zero_grad()
            critic_loss = critic.loss(imag_feats.detach(), targets.detach())
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
            critic_opt.step()

            # Advantage = λ-return minus baseline (critic value) — reduces REINFORCE variance
            advantages = targets - values[:, :H]
            normalizer.update(advantages)
            normed_adv = normalizer.normalize(advantages)
            # Detach imag_feats to avoid second-order backprop through the 15-step recurrence
            _, log_probs, entropy = actor(imag_feats.detach())
            actor_loss = -(normed_adv * log_probs + ent_coef * entropy).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
            actor_opt.step()

        # Logging — log last gradient step's losses at current env_steps
        writer.add_scalar("wm/loss_total",  wm_losses["total"].item(),  env_steps)
        writer.add_scalar("wm/loss_recon",  wm_losses["recon"].item(),  env_steps)
        writer.add_scalar("wm/loss_kl",     wm_losses["kl"].item(),     env_steps)
        writer.add_scalar("wm/loss_reward", wm_losses["reward"].item(), env_steps)
        writer.add_scalar("wm/loss_cont",   wm_losses["cont"].item(),   env_steps)
        writer.add_scalar("ac/critic_loss", critic_loss.item(),         env_steps)
        writer.add_scalar("ac/actor_loss",  actor_loss.item(),          env_steps)
        writer.add_scalar("ac/mean_return", targets.mean().item(),      env_steps)
        writer.add_scalar("ac/entropy",     entropy.mean().item(),      env_steps)
        writer.add_scalar("env/episode_reward", float(ep[2].sum()),     env_steps)

        grad_step += 1
        if grad_step % 100 == 0:
            ep_reward = float(ep[2].sum())
            print(
                f"Env steps {env_steps}/{max_env_steps} | "
                f"wm={wm_losses['total'].item():.3f} | "
                f"actor={actor_loss.item():.3f} | "
                f"critic={critic_loss.item():.3f} | "
                f"ep_r={ep_reward:.2f} | "
                f"buf={replay.size}"
            )

    env.close()
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(wm.state_dict(),     f"{checkpoint_dir}/world_model.pt")
    torch.save(actor.state_dict(),  f"{checkpoint_dir}/actor.pt")
    torch.save(critic.state_dict(), f"{checkpoint_dir}/critic.pt")
    writer.close()
    print(f"Saved to {checkpoint_dir}/")


if __name__ == "__main__":
    config = load_config(sys.argv[1])
    train(config)
