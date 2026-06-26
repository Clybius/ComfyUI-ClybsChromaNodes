# ComfyUI-ClybsChromaNodes

A small collection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), designed primarily for use with [Lodestone Rock's Chroma](https://huggingface.co/lodestones/Chroma) model (and compatible flow-matching architectures like FLUX and SD3). The package bundles custom guidance, samplers, schedulers, and an adaptive multi-LoRA loader, all of which integrate as standard ComfyUI nodes.

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory and restart ComfyUI:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/Clybius/ComfyUI-ClybsChromaNodes.git
```

No additional Python dependencies are required beyond a working ComfyUI install. The frontend extension is picked up automatically via `WEB_DIRECTORY = "./js"`.

## Node overview

The package registers **8 nodes**, organized into four groups:

| Group | Nodes |
|---|---|
| Guidance | `ClybGuidance` |
| Samplers | `SamplerClyb_BDF`, `SamplerTaylorFlow`, `SamplerWrapperCFGPP` |
| Schedulers | `InverseSquaredScheduler`, `PrintSigmas` |
| LoRA Loaders | `ClybAdaptiveLoraLoader`, `ClybAdaptiveLoraLoaderModelOnly` |

In addition, the `chroma_NAG.py` module ships a `ChromaNAG` class (Normalized Attention Guidance for Chroma's `DoubleStreamBlock`) that is **not currently registered** in the node mappings — the class is available in code but does not appear in the ComfyUI node browser.

---

## Guidance

### `ClybGuidance`
*File: `clyb_Guidance.py`*
*Category: `sampling/custom_sampling`*

A pre-CFG model patch that rewires how the conditional and unconditional predictions are combined at every sampling step. Stacks the following features on top of standard CFG:

- **Project-and-scale (`eta`)** — the guidance vector is split into a component parallel to the conditional and a component orthogonal to it. The parallel component is scaled by `eta`, the orthogonal component is left alone. `eta = 1.0` recovers default CFG.
- **Norm clamping (`norm_threshold`)** — if the L2 norm of the guided output exceeds the conditional's norm times `norm_threshold`, the guided output is rescaled back down. Disabled at `0.0`.
- **Momentum (`momentum`, `momentum_beta`, `momentum_renorm`)** — adds a fraction of a running-average guidance vector to the current guidance, optionally re-normalized back to its original norm. `momentum = 0` disables it.
- **Scalar projection (`scalar_projection`, `scalar_logsumexp`)** — projects the conditional onto the unconditional as a scalar (`logsumexp` or `sum` reduction), then scales the unconditional by that scalar.
- **STD/var rescale (`rescale_phi`, `var_rescale`)** — blends the guided output toward an output whose standard deviation (or variance) matches the conditional's. `rescale_phi = 0` disables it.
- **Sine-bell schedule (`scale_up_ratio`, `scale_up_shift`)** — animates the effective CFG scale from `1.0` at the start, up to the configured CFG scale at the middle of diffusion, and back down to `1.0` at the end. `scale_up_ratio = 0` disables it. `scale_up_shift < 1.0` shifts the bell later, `> 1.0` earlier.
- **atan2/sin blend (`atan2sin_ratio`)** — blends the unconditional with `uncond.atan().sin() / cond.atan().cos()` (a Chroma-specific twist on the guidance direction).

| Input | Type | Default | Range | Description |
|---|---|---|---|---|
| `model` | MODEL | — | — | Model to patch |
| `eta` | FLOAT | 1.0 | -50, 50 | Parallel guidance scale |
| `norm_threshold` | FLOAT | 0.0 | 0, 50 | Norm clamp (0 = off) |
| `momentum` | FLOAT | 0.0 | -10, 10 | Momentum weight (0 = off) |
| `momentum_beta` | FLOAT | 0.75 | 0, 0.999 | Running-average smoothing |
| `momentum_renorm` | FLOAT | 1.0 | 0, 1 | Renormalize after momentum |
| `scalar_projection` | BOOLEAN | False | — | Scalar projection of cond onto uncond |
| `scalar_logsumexp` | BOOLEAN | False | — | Use `logsumexp` (else `sum`) |
| `rescale_phi` | FLOAT | 0.0 | 0, 1 | STD-rescale blend (0 = off) |
| `var_rescale` | BOOLEAN | False | — | Use `var` (else `std`) for rescale |
| `scale_up_ratio` | FLOAT | 0.0 | 0, 1 | Sine-bell CFG weight (0 = off) |
| `scale_up_shift` | FLOAT | 1.0 | 0.1, 10 | Sine-bell schedule shift |
| `atan2sin_ratio` | FLOAT | 0.0 | -100, 100 | atan2/sin blend (0 = off) |

**Returns:** `MODEL` (patched).

---

## Samplers

All three nodes return a `SAMPLER` object intended to be plugged into the `sampler` input of `KSampler` (or any node that accepts a sampler).

### `SamplerClyb_BDF`
*File: `clyb_Samplers.py`*
*Category: `sampling/custom_sampling/samplers`*

A backward-differentiation-formula-style sampler that takes a single model evaluation at the start of the step, then synthesizes a refined denoised prediction at the `sigma_down` point and combines them with one of three scalar fusions:

- `projection` — projects the half-step prediction back onto the line spanned by the full-step prediction.
- `atan2sin` — uses `atan2(sin(half), cos(full))` to blend the two predictions in angle space.
- `atan2sin+projection` (default) — the atan2/sin blend followed by a projection rescaling, combining both stabilizations.

The sampler detects flow-matching models (FLUX, SD3, Chroma) automatically and switches to the flow-style ancestral update with `alpha_ip1`/`alpha_down`/`renoise_coeff`.

| Input | Type | Default | Range | Description |
|---|---|---|---|---|
| `scalar` | ENUM | `atan2sin+projection` | `projection`, `atan2sin`, `atan2sin+projection` | Scalar fusion mode |
| `eta` | FLOAT | 1.0 | 0, 100 | Ancestral stochasticity |
| `s_noise` | FLOAT | 1.0 | 0, 100 | Noise scaling factor |

### `SamplerTaylorFlow`
*File: `clyb_Samplers.py`*
*Category: `sampling/custom_sampling/samplers`*

A multi-step Taylor-expansion sampler (registered as `taylor_flow` in the ComfyUI sampler list). Implements the algorithm from *"Leveraging Previous Steps: A Training-free Fast Solver for Flow Diffusion"* (Nov 2024):

1. Maintain a rolling history of `(sigma, denoised)` pairs from the previous `order` steps.
2. At each step, perform **one** model evaluation at the current state.
3. Build a Vandermonde matrix from the historical `sigma` values.
4. Solve for Taylor coefficients `B` that predict the latent at `sigma_next`.
5. Apply the Euler step plus a correction term built from the history.
6. Inject ancestral noise as usual.

The four `sigma_calc` modes control how the `sigma_down` / `sigma_up` pair is computed for the ancestral noise injection:

- `clyb` (default) — logarithmic scaling, the original Clyb scheme.
- `taylor-expansion` — exponential factor with a quadratic correction based on the step ratio.
- `ancestral` — standard k-diffusion `get_ancestral_step`.
- `adaptive` — converges toward the standard scheme based on a normalized variance of the denoised history (small history variance ⇒ less noise).

The Vandermonde solve supports two methods internally (iterative two-sided equilibration with Tikhonov regularization, and diagonal-dominant extraction) and falls back to a `lstsq` solve if the regularized system is singular.

| Input | Type | Default | Range | Description |
|---|---|---|---|---|
| `order` | INT | 8 | 1, 16 | Taylor expansion order (history length) |
| `eta` | FLOAT | 1.0 | 0, 1 | Ancestral stochasticity |
| `s_noise` | FLOAT | 1.0 | 0, 2 | Noise scaling factor |
| `sigma_calc` | ENUM | `clyb` | `clyb`, `taylor-expansion`, `ancestral`, `adaptive` | Ancestral sigma calculation method |

### `SamplerWrapperCFGPP`
*File: `clyb_Samplers.py`*
*Category: `sampling/custom_sampling/samplers`*

A *sampler wrapper* — takes any other `SAMPLER` as input and returns a new sampler that runs the inner sampler but with a CFG++-style denoised recomputation. CFG++ replaces the standard CFG blend with a closed-form denoised that uses the unconditional prediction from the *next* sigma:

```
denoised_star = (sigma * alpha_t * denoised_guided
                 - sigma_next * alpha_s * uncond_denoised) / (sigma - sigma_next)
```

where `alpha_s = sigma * exp(lambda(sigma))` and `alpha_t = sigma_next * exp(lambda(sigma_next))`, with `lambda(s) = sigma_to_half_log_snr(s, model_sampling)`.

The wrapper installs a `post_cfg_function` hook on the model to capture `uncond_denoised` from the inner sampler, then uses a `CFGPPProxyModel` to perform the recombination at every step. The wrapper itself is **not** added to the standard KSampler dropdown — it is only reachable through this node (or any node that constructs it via `comfy.samplers.ksampler("cfgpp", {...})`).

| Input | Type | Description |
|---|---|---|
| `sampler` | SAMPLER | Inner sampler to wrap with CFG++ |

---

## Schedulers

### `InverseSquaredScheduler`
*File: `clyb_Schedulers.py`*
*Category: `sampling/custom_sampling/schedulers`*

A sigma scheduler that biases the schedule toward the *end* of diffusion. It uses `(1 - t²)²` (i.e. the inverse of `t` mapped through `(1-t)²`) to pick sigma indices — fine-grained near the end, coarser at the start. The scheduler is also registered into `SCHEDULER_HANDLERS` under the name `inverse_squared`, so it can be used as a string in any node that accepts a scheduler name.

| Input | Type | Default | Range | Description |
|---|---|---|---|---|
| `model` | MODEL | — | — | Model to derive the sigma range from |
| `steps` | INT | 20 | 3, 1000 | Number of steps |
| `denoise` | FLOAT | 1.0 | 0, 1 | Denoise strength (< 1.0 enables img2img-style short schedules) |

**Returns:** `SIGMAS`.

### `PrintSigmas`
*File: `clyb_Schedulers.py`*
*Category: `sampling/custom_sampling/schedulers`*

A debug helper that prints the incoming `SIGMAS` tensor to the console and passes it through unchanged. Useful for inspecting schedules from other nodes without modifying them.

| Input | Type | Description |
|---|---|---|
| `sigmas` | SIGMAS | Sigma tensor to print |

**Returns:** `SIGMAS` (passthrough).

---

## LoRA Loaders

Both loaders share a frontend extension (`js/clyb_adaptive_lora.js`) that dynamically reveals the next `lora_name_N` / `strength_*_N` triplet only after the previous `lora_name_M` is set to a non-`"none"` value (up to a cap of 20 LoRAs). Setting a slot back to `"none"` hides the trailing widgets, and serialization / deserialization are handled correctly so that hidden widget values survive workflow save/load.

### `ClybAdaptiveLoraLoader`
*File: `clyb_ModelLoader.py`*
*Category: `loaders`*

Apply up to 20 LoRAs to a `(MODEL, CLIP)` pair, in order, by cloning the model once and merging all patches into that single clone. The clone-per-call approach is cheaper than the per-LoRA clone done by ComfyUI's built-in chain loader. LoRA file contents are cached in `self.loaded_loras` keyed by slot index — the cache is invalidated when a different LoRA is selected for that slot.

| Input | Type | Default | Range | Description |
|---|---|---|---|---|
| `model` | MODEL | — | — | Diffusion model to patch |
| `clip` | CLIP | — | — | CLIP model to patch |
| `lora_name_1` | ENUM | — | loras list | First LoRA (set to `"none"` to skip) |
| `strength_model_1` | FLOAT | 1.0 | -100, 100 | Diffusion-model strength (negative allowed) |
| `strength_clip_1` | FLOAT | 1.0 | -100, 100 | CLIP strength (negative allowed) |
| `lora_name_2..20` | ENUM | `"none"` | loras list | Additional LoRAs (revealed as you fill slots) |
| `strength_model_2..20` | FLOAT | 1.0 | -100, 100 | Per-slot diffusion-model strength |
| `strength_clip_2..20` | FLOAT | 1.0 | -100, 100 | Per-slot CLIP strength |

**Returns:** `MODEL`, `CLIP`.

### `ClybAdaptiveLoraLoaderModelOnly`
*File: `clyb_ModelLoader.py`*
*Category: `loaders`*

Same as `ClybAdaptiveLoraLoader`, but the `CLIP` input is omitted from the schema and the internal `strength_clip_*` is forced to `0.0` for every slot, leaving only the diffusion-model patches applied. Use this for `MODEL`-only pipelines (e.g. unconditional sampling, flows without a text encoder, or cases where CLIP is wired in separately).

| Input | Type | Default | Range | Description |
|---|---|---|---|---|
| `model` | MODEL | — | — | Diffusion model to patch |
| `lora_name_1` | ENUM | — | loras list | First LoRA (set to `"none"` to skip) |
| `strength_model_1` | FLOAT | 1.0 | -100, 100 | Diffusion-model strength (negative allowed) |
| `lora_name_2..20` | ENUM | `"none"` | loras list | Additional LoRAs (revealed as you fill slots) |
| `strength_model_2..20` | FLOAT | 1.0 | -100, 100 | Per-slot diffusion-model strength |

**Returns:** `MODEL`.

---

## Project layout

```
ComfyUI-ClybsChromaNodes/
├── __init__.py                  # Entry point: imports modules, registers nodes, exposes WEB_DIRECTORY
├── pyproject.toml               # Package metadata (v1.0.5)
├── LICENSE                      # Apache License 2.0
├── js/
│   └── clyb_adaptive_lora.js    # Frontend extension for the dynamic LoRA widget behavior
├── chroma_NAG.py                # ChromaNAG class (currently unregistered)
├── clyb_Guidance.py             # ClybGuidance model patch
├── clyb_Samplers.py             # clyb_bdf, taylor_flow samplers + cfgpp wrapper + 3 node classes
├── clyb_Schedulers.py           # InverseSquaredScheduler, PrintSigmas + scheduler registration
└── clyb_ModelLoader.py          # ClybAdaptiveLoraLoader, ClybAdaptiveLoraLoaderModelOnly
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Repository

[https://github.com/Clybius/ComfyUI-ClybsChromaNodes](https://github.com/Clybius/ComfyUI-ClybsChromaNodes)
