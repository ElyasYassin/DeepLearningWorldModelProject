# Latent Dynamics Model

**File:** `models/dynamics/transformer.py`  
**Training:** `training/train_dynamics.py`

---

## What it does

The dynamics model is a learned simulator. Given a history of what the robot has seen and done, it predicts what the latent state will look like after the next action — without touching the real environment.

```
(z_0, a_0), (z_1, a_1), ..., (z_t, a_t)  →  z_{t+1}
```

`z` is the 128-dimensional latent vector produced by the VAE encoder. `a` is the robot's action (joint velocities). The model never sees raw pixels — it lives entirely in latent space.

---

## Architecture

### Input projection

Each timestep's `(z_t, a_t)` pair is concatenated and projected into the transformer's working dimension:

```
concat(z_t, a_t)  →  shape (128 + action_dim,)
Linear(128 + action_dim, 256)  →  shape (256,)
```

This is done for every step in the sequence simultaneously, producing a sequence of 256-dimensional tokens.

### Positional encoding

The transformer has no built-in sense of order. A learned embedding table (`nn.Embedding(64, 256)`) maps each timestep index to a 256-dimensional vector that gets added to the token. This tells the model that step 3 comes after step 2, and so on.

Learned embeddings are used instead of sinusoidal because the sequences here are short and fixed-length (max 64 steps), where learned embeddings tend to work better.

### Causal transformer

The core is a 4-layer `nn.TransformerEncoder` with 4 attention heads and a feedforward dimension of 1024 (4 × d_model).

A **causal mask** is applied so that each timestep can only attend to itself and earlier timesteps — never future ones:

```
Timestep 0 can attend to: [0]
Timestep 1 can attend to: [0, 1]
Timestep 2 can attend to: [0, 1, 2]
...
```

The mask is an upper-triangular matrix of `-inf` values added to the attention logits before softmax. Any position receiving `-inf` gets zero weight after softmax, effectively blocking that connection.

```
     t=0   t=1   t=2   t=3
t=0 [  0  -inf  -inf  -inf ]
t=1 [  0    0   -inf  -inf ]
t=2 [  0    0     0   -inf ]
t=3 [  0    0     0     0  ]
```

This is the same masking strategy used in GPT-style language models. Without it, the model could "cheat" during training by looking at future states when predicting the next one.

### Output projection

Only the **last token's** output is used to predict `z_{t+1}`:

```
transformer_output[:, -1, :]  →  shape (256,)
Linear(256, 128)  →  z_{t+1}
```

The last token has attended to the full history (under the causal constraint), so it aggregates all available context. The output projection maps back down to latent space.

### Full shape flow

```
Input latents:  (B, T, 128)
Input actions:  (B, T, action_dim)
After concat:   (B, T, 128 + action_dim)
After proj:     (B, T, 256)
After pos emb:  (B, T, 256)
After transformer: (B, T, 256)
Last token:     (B, 256)
After output proj: (B, 128)  ← predicted z_{t+1}
```

---

## Autoregressive inference: `step()`

During controller training, the policy needs to imagine a sequence of future states one step at a time. The `step()` method handles this:

```python
z_next = dynamics.step(ctx_latents, ctx_actions)
```

It maintains a sliding context window of the last `max_seq_len=64` steps. As new `(z, a)` pairs are appended, old ones are dropped from the front if the window exceeds 64.

This is identical to how GPT generates text token by token — the same model, called repeatedly with a growing context.

---

## Reward predictor

A small companion MLP trained alongside the dynamics model:

```
z_t  →  Linear(128, 64)  →  ReLU  →  Linear(64, 1)  →  r_t
```

This predicts the scalar reward (negative end-effector-to-target distance) from a latent state. It's intentionally small — reward prediction is a simpler function than state transition, and a small predictor is less prone to overfitting.

During imagination rollouts in the controller, the reward predictor provides the reward signal without needing the real environment.

---

## Training

**Script:** `training/train_dynamics.py`

### Data collection

500 episodes are collected from the real environment using a random policy. Each frame is encoded offline by the frozen VAE encoder:

```
obs['image']  →  VAE.encode()  →  z_t
```

The encoder is frozen during dynamics training — its weights do not update. This ensures a stable latent space for the dynamics model to learn in.

Each episode becomes a sequence of `(z_t, a_t, r_t)` tuples.

### Sequence windowing

Episodes are sliced into overlapping windows of length `max_seq_len=64` with stride 1. A 500-step episode produces 437 training windows. This maximizes data use.

```
Episode (500 steps):
  Window 0:  steps [0..63]
  Window 1:  steps [1..64]
  Window 2:  steps [2..65]
  ...
  Window 436: steps [436..499]
```

### Loss

Two losses are computed jointly and summed:

**Dynamics loss** — MSE between the predicted and actual next latent state:
```
z_pred = dynamics(z_seq[:, :-1], a_seq[:, :-1])   # predict from first T-1 steps
loss_dyn = MSE(z_pred, z_seq[:, -1])               # compare to actual T-th state
```

**Reward loss** — MSE between predicted and actual rewards across the sequence:
```
r_pred = reward_predictor(z_seq[:, :-1])           # predict reward at each step
loss_rew = MSE(r_pred, r_seq[:, :-1])
```

Both use teacher forcing — the model always receives ground-truth `z` values as input during training, never its own predictions. This is more stable than feeding predictions back in, especially early in training.

**Gradient clipping** (`max_norm=1.0`) is applied to the dynamics model parameters before each optimizer step. The transformer can produce large gradients when the prediction error is high in early epochs, and clipping prevents these from destabilizing training.

### Optimizer

Adam with `lr=1e-4`, shared across both the dynamics model and reward predictor. Trained for 50 epochs by default.

### Checkpoints

- `checkpoints/dynamics.pt` — `LatentDynamicsModel` weights
- `checkpoints/reward_predictor.pt` — `RewardPredictor` weights

---

## Why a transformer instead of an RNN?

The original Dreamer paper uses an RNN (GRU-based RSSM). This implementation uses a transformer for one reason: **attention over the full context window**.

An RNN compresses the entire history into a fixed-size hidden state at each step. Information from 50 steps ago competes with information from 2 steps ago for space in that state. A transformer can directly attend to any past step with equal access, which is useful when the robot's current situation depends on a specific moment in the past (e.g., the last time the target was clearly visible).

The tradeoff is memory: a transformer over 64 steps with `d_model=256` holds the full attention matrix in memory during training, whereas an RNN's memory cost is constant per step.

---

## Config reference (`configs/default.yaml`)

```yaml
dynamics:
  d_model: 256       # transformer embedding dimension
  nhead: 4           # attention heads
  num_layers: 4      # stacked transformer layers
  max_seq_len: 64    # context window length
  learning_rate: 1.0e-4
  batch_size: 32
  epochs: 50
```
