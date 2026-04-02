---
plan: Fix DDIM Sampling Steps & Device Bug
phase: 1
wave: 1
---

<objective>
Fix the two root-cause bugs blocking animation quality in the Sign-IDD model:
1. `sampling_timesteps` is hard-coded from YAML config (default=5) and cannot be overridden at runtime — so every inference only does 5 denoising steps regardless of what `translate()` is called with.
2. `ACD.py` `ddim_sample()` uses the module-level `device` variable (set at import time) instead of the actual tensor device — causing potential device mismatch bugs.

After fixing these, update the YAML config to use a sensible default (50 steps) and verify the output has measurably higher motion variance.
</objective>

<context>
## Key Files
- `SignIDD_CodeFiles/ACD.py` — The diffusion model. `ddim_sample()` is the sampling loop.
- `backend/sign_bridge_inference.py` — The inference engine. `translate()` calls `ACD.ddim_sample()` directly.
- `backend/model_configs/Sign-IDD.yaml` — YAML config loaded at startup; sets `sampling_timesteps`.

## Root Cause Detail

### Bug 1: Ignored sampling_steps
In `ACD.__init__()` (ACD.py line 58):
```python
sampling_timesteps = args["diffusion"].get('sampling_timesteps', 5)
```
This reads from config ONCE at model initialization. The value is stored as `self.sampling_timesteps`.

In `ddim_sample()` (ACD.py line 190):
```python
total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
```
It always reads `self.sampling_timesteps` — completely ignoring any `sampling_steps` argument that callers might pass.

The inference engine (`sign_bridge_inference.py` line 145) calls:
```python
results = self.model.ACD.ddim_sample(
    encoder_output=encoder_output,
    input_3d=dummy_trg,
    src_mask=src_mask,
    trg_mask=trg_mask
)
```
No `sampling_steps` arg is passed (it would be ignored anyway).

### Bug 2: Module-level device in ddim_sample
In `ACD.py` line 197:
```python
img = torch.randn(shape, device=device)  # BUG: uses module-level `device`
```
And line 204:
```python
time_cond = torch.full((batch,), time, device=device, dtype=torch.long)  # BUG
```
The module-level `device` is set at import: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
On Mac with no CUDA, this is `cpu` — which happens to match our forced `cpu` inference.
But the right fix is to use the actual tensor device for correctness and future MPS compatibility.

## Current YAML Config
Check `backend/model_configs/Sign-IDD.yaml` for `sampling_timesteps` key under `[diffusion]`.
</context>

<tasks>

## Task 1: Fix ACD.py — Device Bug & Runtime sampling_steps

**File:** `SignIDD_CodeFiles/ACD.py`

### 1a. Make `ddim_sample()` accept a `sampling_steps` override
Add a `sampling_steps: int = None` parameter to `ddim_sample()`. If provided, use it; otherwise fall back to `self.sampling_timesteps`.

### 1b. Fix device bug — use tensor device instead of module-level
Replace `device=device` with `device=encoder_output.device` in:
- Line 197: `img = torch.randn(shape, device=device)` → `device=encoder_output.device`
- Line 204: `time_cond = torch.full((batch,), time, device=device, ...)` → `device=encoder_output.device`

### Complete updated `ddim_sample` signature and body:
```python
def ddim_sample(self, encoder_output, input_3d, src_mask, trg_mask, sampling_steps: int = None):
    """
    DDIM sampling with group-wise mixed alphas for Body vs Hands.
    sampling_steps: override self.sampling_timesteps at runtime (optional)
    """
    dev = encoder_output.device  # always use actual tensor device
    batch = encoder_output.shape[0]
    shape = (batch, input_3d.shape[1], 150)
    
    total_timesteps = self.num_timesteps
    _sampling_timesteps = sampling_steps if sampling_steps is not None else self.sampling_timesteps
    eta = self.ddim_sampling_eta

    times = torch.linspace(-1, total_timesteps - 1, steps=_sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))

    img = torch.randn(shape, device=dev)   # FIXED: use dev

    x_start = None
    preds_all = []
    mask_body, mask_hand = make_joint_channel_masks(device=dev)

    for time, time_next in time_pairs:
        time_cond = torch.full((batch,), time, device=dev, dtype=torch.long)  # FIXED: use dev

        preds = self.model_predictions(
            x=img, encoder_output=encoder_output, t=time_cond,
            src_mask=src_mask, trg_mask=trg_mask
        )
        pred_noise, x_start = preds.pred_noise.float(), preds.pred_x_start
        preds_all.append(x_start)

        if time_next < 0:
            img = x_start
            continue

        alpha_B = self.alphas_cumprod_B[time]
        alpha_n_B = self.alphas_cumprod_B[time_next]
        alpha_H = self.alphas_cumprod_H[time]
        alpha_n_H = self.alphas_cumprod_H[time_next]

        sigma_B = eta * ((1 - alpha_B / alpha_n_B) * (1 - alpha_n_B) / (1 - alpha_B)).sqrt()
        sigma_H = eta * ((1 - alpha_H / alpha_n_H) * (1 - alpha_n_H) / (1 - alpha_H)).sqrt()
        c_B = (1 - alpha_n_B - sigma_B ** 2).sqrt()
        c_H = (1 - alpha_n_H - sigma_H ** 2).sqrt()

        alpha_n_mix = alpha_n_B.sqrt() * (~mask_hand) + alpha_n_H.sqrt() * mask_hand
        c_mix = c_B * (~mask_hand) + c_H * mask_hand
        sigma_mix = sigma_B * (~mask_hand) + sigma_H * mask_hand

        noise = torch.randn_like(img)
        img = x_start * alpha_n_mix + c_mix * pred_noise + sigma_mix * noise

    return preds_all
```

## Task 2: Update Sign-IDD.yaml — Set sensible default sampling_timesteps

**File:** `backend/model_configs/Sign-IDD.yaml`

Find the `diffusion:` section and update `sampling_timesteps` to `50`.
If it doesn't exist as a key, add it.

This ensures the model uses 50 DDIM denoising steps by default (instead of 5), which dramatically increases output diversity.

## Task 3: Update sign_bridge_inference.py — Pass sampling_steps through

**File:** `backend/sign_bridge_inference.py`

Update the `translate()` method to pass `sampling_steps` to `ddim_sample()`:

```python
results = self.model.ACD.ddim_sample(
    encoder_output=encoder_output,
    input_3d=dummy_trg,
    src_mask=src_mask,
    trg_mask=trg_mask,
    sampling_steps=sampling_steps  # ADD THIS
)
```

Also update the default in `translate()` signature from `sampling_steps: int = 20` to `sampling_steps: int = 50`.

## Task 4: Run verification test

**File:** `test_cpu.py` (overwrite with the verification test below)

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from backend.sign_bridge_inference import SignBridgeInference
import numpy as np

engine = SignBridgeInference(
    "/Users/harshit/Documents/WEBSITE_EXPLO/sign_idd_model_20260121_171210",
    device='cpu'
)

results = {}
for text in ["HEUTE", "MORGEN", "REGEN", "SONNE"]:
    skels = np.array(engine.translate(text, sampling_steps=50))
    time_var = np.std(skels, axis=0).mean()
    results[text] = {"time_variance": float(time_var), "shape": list(skels.shape)}
    print(f"{text}: time_variance={time_var:.6f}, shape={skels.shape}")

# Cross-input variance (different texts should produce different poses)
skels_a = np.array(engine.translate("HEUTE", sampling_steps=50))
skels_b = np.array(engine.translate("REGEN", sampling_steps=50))
cross_diff = np.abs(skels_a - skels_b).mean()
print(f"\nCross-input difference (HEUTE vs REGEN): {cross_diff:.6f}")
print(f"\nPASS criteria:")
print(f"  time_variance > 0.001: {all(r['time_variance'] > 0.001 for r in results.values())}")
print(f"  cross_diff > 0.001: {cross_diff > 0.001}")
```

Run with: `backend/venv/bin/python3 test_cpu.py`

**Expected:** time_variance > 0.001 for all inputs, cross_diff > 0.001

</tasks>

<verification>
- [ ] ACD.py `ddim_sample()` accepts `sampling_steps` param and uses it
- [ ] ACD.py uses `encoder_output.device` instead of module-level `device`
- [ ] Sign-IDD.yaml has `sampling_timesteps: 50` (or higher)
- [ ] sign_bridge_inference.py passes `sampling_steps` to `ddim_sample()`
- [ ] Verification test runs without errors
- [ ] time_variance > 0.001 for all test inputs
- [ ] cross_diff > 0.001 (different inputs produce different outputs)
</verification>
