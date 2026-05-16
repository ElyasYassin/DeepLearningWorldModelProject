"""
Generates Figure 1: System pipeline diagram.
Output: docs/figures/pipeline.pdf

Run from project root:
    python docs/figures/gen_pipeline.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "pipeline.pdf")

# ── Colour palette ────────────────────────────────────────────────────────────
C_ENCODER   = "#4C8BB5"   # blue  – VAE encoder
C_DYNAMICS  = "#E07B39"   # orange – transformer dynamics
C_CTRL      = "#5BAD6F"   # green  – actor-critic
C_FROZEN    = "#AAAAAA"   # grey   – frozen indicator
C_ARROW     = "#333333"
C_BG        = "#F7F7F7"

# ── Layout constants ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 10, 3.8
BOX_H = 0.9
BOX_W = 2.1
Y_MAIN = 2.2          # y-centre of main pipeline row
Y_LOOP = 0.9          # y-centre of imagination loop row
MARGIN = 0.3

# x-centres of the three component boxes
X_ENC  = 1.7
X_DYN  = 5.0
X_CTRL = 8.3

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")


def rounded_box(ax, cx, cy, w, h, color, label_top, label_bot=None, alpha=0.92):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="white",
        linewidth=1.5, alpha=alpha, zorder=3,
    )
    ax.add_patch(box)
    if label_bot:
        ax.text(cx, cy + 0.12, label_top, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=4)
        ax.text(cx, cy - 0.16, label_bot, ha="center", va="center",
                fontsize=7, color="white", alpha=0.88, zorder=4)
    else:
        ax.text(cx, cy, label_top, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=4)


def arrow(ax, x0, y0, x1, y1, label=None, label_above=True):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.4),
        zorder=2,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dy = 0.18 if label_above else -0.18
        ax.text(mx, my + dy, label, ha="center", va="center",
                fontsize=7, color="#444444",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))


# ── Input frame ───────────────────────────────────────────────────────────────
frame_x, frame_y = 0.45, Y_MAIN
frame_box = FancyBboxPatch(
    (frame_x - 0.35, frame_y - 0.42), 0.70, 0.84,
    boxstyle="round,pad=0.06",
    facecolor="#DDDDDD", edgecolor="#999999", linewidth=1.2, zorder=3,
)
ax.add_patch(frame_box)
ax.text(frame_x, frame_y + 0.15, "RGB", ha="center", va="center",
        fontsize=8, fontweight="bold", color="#333333", zorder=4)
ax.text(frame_x, frame_y - 0.13, "64×64", ha="center", va="center",
        fontsize=7, color="#555555", zorder=4)

# ── Arrow: frame → encoder ────────────────────────────────────────────────────
arrow(ax, frame_x + 0.38, Y_MAIN, X_ENC - BOX_W / 2, Y_MAIN)

# ── Encoder box ───────────────────────────────────────────────────────────────
rounded_box(ax, X_ENC, Y_MAIN, BOX_W, BOX_H, C_ENCODER,
            "VAE Encoder", "ResNet-18 (layer4 FT)\n→ μ, logσ² ∈ ℝ¹²⁸")
# Frozen tag for later stages
ax.text(X_ENC, Y_MAIN - BOX_H / 2 - 0.18, "frozen in stages 2 & 3",
        ha="center", va="top", fontsize=6.5, color=C_FROZEN, style="italic")

# ── Arrow: encoder → z latent ─────────────────────────────────────────────────
arrow(ax, X_ENC + BOX_W / 2, Y_MAIN, X_DYN - BOX_W / 2, Y_MAIN,
      label="z ∈ ℝ¹²⁸")

# ── Dynamics box ──────────────────────────────────────────────────────────────
rounded_box(ax, X_DYN, Y_MAIN, BOX_W, BOX_H, C_DYNAMICS,
            "Transformer Dynamics", "4L · 4H · d=256\nctx = 64 steps")
ax.text(X_DYN, Y_MAIN - BOX_H / 2 - 0.18, "frozen in stage 3",
        ha="center", va="top", fontsize=6.5, color=C_FROZEN, style="italic")

# ── Arrow: dynamics → z_next ──────────────────────────────────────────────────
arrow(ax, X_DYN + BOX_W / 2, Y_MAIN, X_CTRL - BOX_W / 2, Y_MAIN,
      label="ẑₜ₊₁ ∈ ℝ¹²⁸")

# ── Controller box ────────────────────────────────────────────────────────────
rounded_box(ax, X_CTRL, Y_MAIN, BOX_W, BOX_H, C_CTRL,
            "Actor-Critic Policy", "MLP · H=15 steps\nλ-returns (γ=0.99)")

# ── Arrow: controller → action ────────────────────────────────────────────────
ax.annotate(
    "", xy=(FIG_W - MARGIN, Y_MAIN), xytext=(X_CTRL + BOX_W / 2, Y_MAIN),
    arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.4), zorder=2,
)
ax.text(FIG_W - MARGIN + 0.02, Y_MAIN, "action", ha="left", va="center",
        fontsize=8, color="#333333")

# ── Imagination loop annotation ───────────────────────────────────────────────
loop_x0 = X_DYN - BOX_W / 2 + 0.1
loop_x1 = X_CTRL + BOX_W / 2 - 0.1
loop_y  = Y_LOOP + 0.12

# Down arrow from dynamics
ax.annotate("", xy=(X_DYN, loop_y + 0.28), xytext=(X_DYN, Y_MAIN - BOX_H / 2),
            arrowprops=dict(arrowstyle="-", color="#888888", lw=1.1,
                            linestyle="dashed"), zorder=2)
# Down arrow from controller
ax.annotate("", xy=(X_CTRL, loop_y + 0.28), xytext=(X_CTRL, Y_MAIN - BOX_H / 2),
            arrowprops=dict(arrowstyle="-", color="#888888", lw=1.1,
                            linestyle="dashed"), zorder=2)

loop_box = FancyBboxPatch(
    (loop_x0, loop_y - 0.28), loop_x1 - loop_x0, 0.56,
    boxstyle="round,pad=0.07",
    facecolor="#F0F0F0", edgecolor="#AAAAAA",
    linewidth=1.0, linestyle="--", alpha=0.85, zorder=2,
)
ax.add_patch(loop_box)
ax.text((loop_x0 + loop_x1) / 2, loop_y + 0.05,
        "Imagination rollout (H = 15 steps)",
        ha="center", va="center", fontsize=8, color="#555555", style="italic", zorder=3)
ax.text((loop_x0 + loop_x1) / 2, loop_y - 0.14,
        "policy trains in latent space — no real env interaction during updates",
        ha="center", va="center", fontsize=6.8, color="#777777", zorder=3)

# Feedback arrow: ẑ_{t+1} loops back into dynamics input
ax.annotate(
    "", xy=(X_DYN - 0.25, Y_MAIN - BOX_H / 2),
    xytext=(X_DYN - 0.25, loop_y + 0.28),
    arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.1,
                    linestyle="dashed"), zorder=2,
)
ax.text(X_DYN - 0.52, (Y_MAIN - BOX_H / 2 + loop_y + 0.28) / 2,
        "ẑₜ → ctx", ha="center", va="center",
        fontsize=6.5, color="#888888", rotation=90)

# ── Title / caption helper ────────────────────────────────────────────────────
ax.text(FIG_W / 2, FIG_H - 0.20,
        "Find My Pen — Three-Stage World Model Pipeline",
        ha="center", va="top", fontsize=10, fontweight="bold", color="#222222")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=C_ENCODER,  label="Stage 1: Vision Encoder (VAE)"),
    mpatches.Patch(facecolor=C_DYNAMICS, label="Stage 2: Latent Dynamics (Transformer)"),
    mpatches.Patch(facecolor=C_CTRL,     label="Stage 3: Controller (Actor-Critic)"),
]
ax.legend(handles=legend_patches, loc="lower left", fontsize=7,
          framealpha=0.85, edgecolor="#CCCCCC", ncol=3,
          bbox_to_anchor=(0.01, 0.01))

plt.tight_layout(pad=0.3)
plt.savefig(OUT, bbox_inches="tight", dpi=300)
print(f"Saved: {OUT}")
