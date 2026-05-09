"""
Generates Figure 3: Two-panel results figure.
  Left:  Training curves — episode return vs. total real env steps (all 5 models)
  Right: Bar chart — tracking accuracy across all 5 models

Prerequisites:
  - TensorBoard logs in logs/ for baselines and controller
  - eval_results.json produced by running evaluation on all models (see below)

Producing eval_results.json:
    python docs/figures/run_eval_all.py configs/default.yaml

Output: docs/figures/results.pdf

Run from project root:
    python docs/figures/gen_results_figure.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ────────────────────────────────────────────────────────────────────
OUT      = os.path.join(os.path.dirname(__file__), "results.pdf")
EVAL_JSON = os.path.join(os.path.dirname(__file__), "eval_results.json")

# Real-env steps used BEFORE controller training starts (encoder + dynamics)
WORLD_MODEL_PRETRAIN_STEPS = (200 + 500) * 500   # 350,000

MODEL_LABELS = [
    "PPO",
    "PPO-LSTM",
    "PPO-ResNet18",
    "PPO-ResNet18-FT",
    "World Model\n(ours)",
]

# Colour per model — consistent across both panels
COLORS = ["#5B8DB8", "#E07B39", "#5BAD6F", "#9B59B6", "#C0392B"]

# ── TensorBoard reader ────────────────────────────────────────────────────────
def read_tb_scalars(log_dir: str, tag: str):
    """Return (steps, values) arrays from a TensorBoard event file."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError("pip install tensorboard")

    ea = EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        available = ea.Tags().get("scalars", [])
        raise KeyError(f"Tag '{tag}' not in {log_dir}. Available: {available}")
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values, window=15):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ── Build training curves ──────────────────────────────────────────────────────
# Map: model label → (log_dir, scalar_tag, x_offset)
# Adjust log_dir paths to match your actual TensorBoard run names.
TB_CONFIGS = {
    "PPO": (
        "logs/PPO_4",
        "rollout/ep_rew_mean",
        0,
    ),
    "PPO-LSTM": (
        "logs/RecurrentPPO_1",
        "rollout/ep_rew_mean",
        0,
    ),
    "PPO-ResNet18": (
        "logs/resnet18_ppo_1",
        "rollout/ep_rew_mean",
        0,
    ),
    "PPO-ResNet18-FT": (
        "logs/resnet18_ppo_ft_3",
        "rollout/ep_rew_mean",
        0,
    ),
    "World Model\n(ours)": (
        "logs/controller",
        "controller/mean_imagined_reward",
        WORLD_MODEL_PRETRAIN_STEPS,
    ),
}


def load_training_curves():
    curves = {}
    for label, (log_dir, tag, offset) in TB_CONFIGS.items():
        if not os.path.exists(log_dir):
            print(f"  WARNING: log dir not found: {log_dir} — skipping {label}")
            continue
        try:
            steps, values = read_tb_scalars(log_dir, tag)
            curves[label] = (steps + offset, smooth(values))
        except Exception as e:
            print(f"  WARNING: could not read {label} ({log_dir}/{tag}): {e}")
    return curves


# ── Load eval results ─────────────────────────────────────────────────────────
def load_eval_results():
    """
    Expected format of eval_results.json:
    {
      "PPO":             {"tracking_acc_mean": 0.15, "tracking_acc_std": 0.02, "fluidity_mean": ..., ...},
      "PPO-LSTM":        {...},
      "PPO-ResNet18":    {...},
      "PPO-ResNet18-FT": {...},
      "World Model":     {...}
    }
    """
    if not os.path.exists(EVAL_JSON):
        print(f"WARNING: {EVAL_JSON} not found — bar chart will use placeholder zeros.")
        return {label: {"tracking_acc_mean": 0, "tracking_acc_std": 0}
                for label in MODEL_LABELS}
    with open(EVAL_JSON) as f:
        return json.load(f)


# ── Plot ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading training curves from TensorBoard logs...")
    curves = load_training_curves()

    print("Loading evaluation results...")
    eval_data = load_eval_results()

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(10, 3.5),
        gridspec_kw={"wspace": 0.38},
    )

    # ── Left panel: training curves ───────────────────────────────────────────
    for (label, color) in zip(MODEL_LABELS, COLORS):
        short_label = label.replace("\n", " ")
        if label not in curves:
            continue
        steps, vals = curves[label]
        # align lengths after smoothing
        plot_steps = steps[:len(vals)]
        lw = 2.2 if "World" in label else 1.5
        ls = "--" if "World" in label else "-"
        ax_left.plot(plot_steps / 1e6, vals, color=color,
                     lw=lw, linestyle=ls, label=short_label, alpha=0.88)

    ax_left.axvline(1.0, color="#999999", lw=1.0, linestyle=":", alpha=0.7)
    ax_left.text(1.01, ax_left.get_ylim()[0] if ax_left.get_ylim()[0] != 0 else -0.5,
                 "1M baseline\nbudget", fontsize=6.5, color="#888888", va="bottom")

    ax_left.set_xlabel("Total real env steps (×10⁶)", fontsize=9)
    ax_left.set_ylabel("Episode return", fontsize=9)
    ax_left.set_title("(a) Training curves", fontsize=9, pad=4)
    ax_left.legend(fontsize=7, framealpha=0.85, loc="lower right")
    ax_left.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax_left.spines[["top", "right"]].set_visible(False)
    ax_left.tick_params(labelsize=8)

    # ── Right panel: bar chart (tracking accuracy) ─────────────────────────
    short_labels = ["PPO", "PPO\nLSTM", "PPO\nRN18", "PPO\nRN18-FT", "WM\n(ours)"]
    eval_keys    = ["PPO", "PPO-LSTM", "PPO-ResNet18", "PPO-ResNet18-FT", "World Model"]

    means = [eval_data.get(k, {}).get("tracking_acc_mean", 0) for k in eval_keys]
    stds  = [eval_data.get(k, {}).get("tracking_acc_std",  0) for k in eval_keys]

    x = np.arange(len(short_labels))
    bars = ax_right.bar(x, means, yerr=stds, color=COLORS, width=0.6,
                        capsize=4, error_kw={"lw": 1.2}, alpha=0.88, zorder=3)

    # Highlight our model
    bars[-1].set_edgecolor("#222222")
    bars[-1].set_linewidth(1.8)

    ax_right.set_xticks(x)
    ax_right.set_xticklabels(short_labels, fontsize=8)
    ax_right.set_ylabel("Tracking accuracy (m) ↓", fontsize=9)
    ax_right.set_title("(b) Final tracking accuracy", fontsize=9, pad=4)
    ax_right.spines[["top", "right"]].set_visible(False)
    ax_right.tick_params(labelsize=8)
    ax_right.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax_right.set_axisbelow(True)

    plt.savefig(OUT, bbox_inches="tight", dpi=200)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
