import collections
import math

import torch
from tqdm.auto import trange

import comfy.model_patcher
from comfy.k_diffusion.sampling import (
    default_noise_sampler,
    get_ancestral_step,
    sigma_to_half_log_snr,
)
import comfy.samplers

@torch.no_grad()
def sampler_clyb_bdf(model, x, sigmas, extra_args=None, callback=None, disable=None, scalar="atan2sin+projection", eta=1., s_noise=1., noise_sampler=None, flow=False):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    if len(sigmas) <= 1:
        # If only one sigma value (e.g., start), return initial x
        return x

    prev_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        predictions = []

        sigma_down = (sigmas[i+1]**2 / (1 + math.log(1. + (sigmas[i+1] - sigmas[i]).abs()) * eta))**0.5
        sigma_up = (sigmas[i]**2 - sigma_down**2)**0.5

        alpha_ip1 = None
        alpha_down = None
        renoise_coeff = None
        if flow:
            # If/for flow model
            alpha_ip1 = 1 - sigmas[i+1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i+1]**2 - sigma_down**2*alpha_ip1**2/alpha_down**2)**0.5

        first_denoised = prev_denoised if (prev_denoised is not None and sigma_down > 0) else model(x, sigmas[i] * s_in, **extra_args)

        if sigma_down > 0 and i > 0:
            x_faux = first_denoised.lerp(x, weight=sigma_down/sigmas[i])
            denoised2 = model(x_faux, sigma_down * s_in, **extra_args)
            second_denoised = (first_denoised + denoised2) / 2
            match scalar:
                case "projection":
                    scaling = (denoised2 * second_denoised) / (second_denoised.pow(2).clamp_min(1e-7))
                    denoised_prime = second_denoised * scaling
                case "atan2sin":
                    denoised_prime = denoised2.atan().sin_().div_(second_denoised.atan().cos_())
                case "atan2sin+projection":
                    denoised_prime = denoised2.atan().sin_().div_(second_denoised.atan().cos_())
                    scaling = (denoised2 * denoised_prime) / (denoised_prime.pow(2).clamp_min(1e-7))
                    denoised_prime = denoised_prime * scaling
                case _:
                    scaling = (denoised2 * second_denoised) / (second_denoised.pow(2).clamp_min(1e-7))
                    denoised_prime = second_denoised * scaling
        else:
            denoised_prime = first_denoised

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised_prime})

        # Denoise
        x = denoised_prime.lerp(x, weight=sigma_down/sigmas[i])

        if sigmas[i + 1] > 0 and not flow and eta > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        elif sigmas[i + 1] > 0 and flow and eta > 0:
            x = (alpha_ip1/alpha_down) * x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * renoise_coeff

        prev_denoised = denoised_prime

    return x

@torch.no_grad()
def sample_clyb_bdf(model, x, sigmas, extra_args=None, callback=None, disable=None, scalar="atan2sin+projection", eta=1., s_noise=1., noise_sampler=None):
    flow = False
    if isinstance(model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
        flow = True
    return sampler_clyb_bdf(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, scalar=scalar, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, flow=flow)

# =============================================================================
# TAYLOR FLOW SAMPLER - Multi-step sampler using Taylor expansion on
# previous denoised predictions to approximate higher-order derivatives.
# Based on "Leveraging Previous Steps" (Nov 2024).
# =============================================================================


def _construct_vandermonde_flow(history, sigma_ref, max_order, device, dtype):
    """
    Build Vandermonde matrix R_p for polynomial interpolation.

    R_p[m, i] = (sigma_{n-1-m} - sigma_ref)^i

    Args:
        history: list of (sigma, denoised) tuples (oldest to newest)
        sigma_ref: reference sigma (current timestep t_{n-1})
        max_order: maximum polynomial degree (number of previous steps to use)
        device: torch device
        dtype: torch dtype (float64 for numerical stability)

    Returns:
        R: (k, k) Vandermonde matrix where k = min(len(history), max_order)
    """
    k = min(len(history), max_order)

    # Use most recent k points from history
    recent_history = list(history)[-k:]

    # Build matrix: R[m, i] = (sigma_m - sigma_ref)^i
    R = torch.zeros((k, k), device=device, dtype=dtype)

    for m, (sigma_m, _) in enumerate(reversed(recent_history)):
        h_m = sigma_m - sigma_ref  # Time difference (negative for past points)

        for i in range(k):
            R[m, i] = h_m**i

    return R


def _solve_flow_coefficients(R, h_n, method="equilibration", diag_weight=1.0):
    """
    Solve for B coefficients using either two-sided equilibration or diagonal-dominant regularization.

    Args:
        R: Vandermonde matrix (k, k)
        h_n: step size (sigma_next - sigma_cur)
        method: "equilibration" (default) or "diagonal"
        diag_weight: Weight for diagonal component (0.0 to 1.0, default 1.0 for original behavior)

    Returns:
        B: coefficient vector (k,)
    """
    k = R.shape[0]
    device = R.device
    dtype = R.dtype

    # Handle edge cases
    if k == 0:
        return torch.tensor([], device=device, dtype=dtype)
    if k == 1:
        # Simple case: just use the diagonal element
        return torch.tensor([h_n], device=device, dtype=dtype)

    # Compute C vector: C_i = h_n^{i+1} / (i+1)
    C = torch.zeros(k, device=device, dtype=dtype)
    for i in range(k):
        C[i] = (h_n ** (i + 1)) / (i + 1)

    if method == "equilibration":
        # Iteratively balance row and column norms to equilibrate the matrix
        D = torch.eye(k, device=device, dtype=dtype)
        E = torch.eye(k, device=device, dtype=dtype)
        R_work = R.clone()

        for _ in range(5):  # 5 iterations typically sufficient for convergence
            # Row scaling: normalize rows to unit infinity-norm
            row_norms = torch.norm(R_work, dim=1, p=float('inf'))
            D_scale = torch.diag(1.0 / torch.sqrt(row_norms + 1e-10))
            R_work = D_scale @ R_work
            D = D_scale @ D

            # Column scaling: normalize columns to unit infinity-norm
            col_norms = torch.norm(R_work, dim=0, p=float('inf'))
            E_scale = torch.diag(1.0 / torch.sqrt(col_norms + 1e-10))
            R_work = R_work @ E_scale
            E = E @ E_scale

        # Apply row scaling to C
        C_eq = D @ C

        # Minimal Tikhonov regularization on equilibrated system
        lambda_reg = 0.0001
        R_reg = R_work + lambda_reg * torch.eye(k, device=device, dtype=dtype)

        # Solve and unscale
        try:
            B_eq = torch.linalg.solve(R_reg, C_eq)
            B = E @ B_eq
        except torch.linalg.LinAlgError:
            B_eq = torch.linalg.lstsq(R_reg, C_eq, rcond=1e-10).solution
            B = E @ B_eq

    elif method == "diagonal":
        # Diagonal-Dominant Extraction
        R_diag = torch.diag(torch.diag(R))
        R_weighted = diag_weight * R_diag + (1.0 - diag_weight) * R

        lambda_reg = 1e-16
        R_reg = R_weighted  # + lambda_reg * torch.eye(k, device=device, dtype=dtype)

        try:
            B = torch.linalg.solve(R_reg, C)
        except torch.linalg.LinAlgError:
            B = torch.linalg.lstsq(R_reg, C, rcond=1e-10).solution

    else:
        # Unknown method: fallback to equilibration
        return _solve_flow_coefficients(R, h_n, method="equilibration", diag_weight=diag_weight)

    return B


@torch.no_grad()
def sampler_taylor_flow(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    order=8,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    flow=False,
    sigma_calc="clyb",
):
    """
    Taylor Flow sampler - Multi-step sampler using Taylor expansion on previous
    denoised predictions to approximate higher-order derivatives.

    Based on "Leveraging Previous Steps: A Training-free Fast Solver for
    Flow Diffusion" (Nov 2024). Achieves O(h^p) approximation error with only
    1 function evaluation per step by reusing cached historical predictions.

    Args:
        model: Diffusion model
        x: Initial latent
        sigmas: Sigma schedule
        extra_args: Extra arguments for model
        callback: Progress callback
        disable: Disable progress bar
        order: Taylor expansion order (1-16). Higher = more accurate but uses more history
        eta: Ancestral sampling eta (stochasticity)
        s_noise: Noise scale
        noise_sampler: Noise sampler function
        flow: Whether using flow-based model (FLUX, SD3, Chroma)
        sigma_calc: Ancestral sigma calculation method ("clyb", "taylor-expansion", "ancestral", "adaptive")

    Returns:
        Denoised latent tensor
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = (
        default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    )
    s_in = x.new_ones([x.shape[0]])
    device = x.device

    if len(sigmas) <= 1:
        return x

    # Rolling history buffer for (sigma, denoised) pairs
    history = collections.deque(maxlen=order)

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
        h_n = sigma_next - sigma_cur  # Step size

        # Ancestral sigma calculation - selectable method
        if sigma_calc == "clyb":
            # Original Clyb implementation (logarithmic scaling)
            sigma_down = (
                sigma_next**2 / (1 + math.log(1.0 + abs(sigma_next - sigma_cur)) * eta)
            ) ** 0.5
            sigma_up = (sigma_next**2 - sigma_down**2) ** 0.5
        elif sigma_calc == "taylor-expansion":
            # Taylor-Expansion-Matched: exponential integral with quadratic correction
            step_ratio = abs(h_n) / max(sigma_next, 1e-8)
            taylor_factor = math.exp(-eta * step_ratio)
            quadratic_correction = 1 - eta * 0.5 * step_ratio ** 2
            sigma_down = sigma_next * taylor_factor * quadratic_correction
            sigma_up = sigma_next * max(0.0, 1 - taylor_factor**2) ** 0.5
        elif sigma_calc == "ancestral":
            # Standard k-diffusion ancestral step
            sigma_down, sigma_up = get_ancestral_step(sigma_cur, sigma_next, eta)
        elif sigma_calc == "adaptive":
            window_size = min(order, len(history))
            if window_size >= 2:
                history_list = list(history)
                recent = history_list[-window_size:]
                denoised_list = [d.float() for _, d in recent]
                stacked = torch.stack(denoised_list)
                mean_d = stacked.mean(dim=0)
                var_val = ((stacked - mean_d) ** 2).mean().item()
                norm_val = mean_d.pow(2).mean().item()
                eps = 1e-8
                if math.isfinite(var_val) and math.isfinite(norm_val):
                    normalized_metric = var_val / (var_val + abs(norm_val) + eps)
                    normalized_metric = min(1.0, max(0.0, normalized_metric))
                else:
                    normalized_metric = 0.0
                sigma_down = sigma_next * (1.0 - eta * normalized_metric)
                sigma_down = max(0.0, min(sigma_next, sigma_down))
                sigma_up = math.sqrt(max(0.0, sigma_next**2 - sigma_down**2))
            else:
                sigma_down = (
                    sigma_next**2 / (1 + math.log(1.0 + abs(sigma_next - sigma_cur)) * eta)
                ) ** 0.5
                sigma_up = (sigma_next**2 - sigma_down**2) ** 0.5
        else:
            # Default to clyb if unknown method
            sigma_down = (
                sigma_next**2 / (1 + math.log(1.0 + abs(sigma_next - sigma_cur)) * eta)
            ) ** 0.5
            sigma_up = (sigma_next**2 - sigma_down**2) ** 0.5

        # Flow model coefficients
        alpha_ip1 = None
        alpha_down = None
        renoise_coeff = None
        alpha_ratio = 1.0
        if flow:
            alpha_ip1 = 1 - sigma_next
            alpha_down = 1 - sigma_down
            renoise_coeff = (
                sigma_next**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2
            ) ** 0.5
            alpha_ratio = alpha_ip1 / alpha_down if alpha_down != 0 else 1.0

        # =========================================================================
        # TAYLOR EXPANSION PHASE (main algorithm)
        # =========================================================================
        # 1. Single model evaluation at current state
        denoised_cur = model(x, sigma_cur * s_in, **extra_args)

        # 2. Build Vandermonde matrix from historical timesteps
        R_p = _construct_vandermonde_flow(
            history, sigma_cur, order, device, torch.float64
        )

        # 3. Solve for B coefficients
        B = _solve_flow_coefficients(R_p, h_n)
        B = B.to(dtype=x.dtype)

        # 4. Compute D_m differences: D_m = v_history[m] - v_current
        D_list = []
        for _, denoised_prev in reversed(list(history)[-len(B) :]):
            D_m = denoised_prev - denoised_cur
            D_list.append(D_m)

        # 5. Predictor step: x_pred = Euler + sum(B_m * D_m)
        w_next = 1.0 - sigma_down / sigma_cur
        euler_step = x.lerp(denoised_cur, weight=w_next)

        if len(D_list) > 0:
            correction = sum(B[m] * D_list[m] for m in range(len(D_list)))
        else:
            correction = 0

        x_next = euler_step + correction

        # 6. Update rolling history
        history.append((sigma_cur, denoised_cur))

        x = x_next

        # =========================================================================
        # ANCESTRAL NOISE INJECTION
        # =========================================================================
        if sigma_next > 0 and eta > 0:
            noise = noise_sampler(sigma_cur, sigma_next) * s_noise
            if flow:
                x = alpha_ratio * x + noise * renoise_coeff
            else:
                x = x + noise * sigma_up

        # Callback
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma_cur,
                    "sigma_hat": sigma_cur,
                    "denoised": denoised_cur,
                }
            )

    return x


@torch.no_grad()
def sample_taylor_flow(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    order=8,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    sigma_calc="clyb",
):
    """Wrapper with flow model detection."""
    flow = False
    if isinstance(
        model.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST
    ):
        flow = True
    return sampler_taylor_flow(
        model,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        order=order,
        eta=eta,
        s_noise=s_noise,
        noise_sampler=noise_sampler,
        flow=flow,
        sigma_calc=sigma_calc,
    )


# =============================================================================
# CFG++ SAMPLER WRAPPER - Captures uncond_denoised via post-CFG hook and
# recomputes a CFG++-style denoised using sigma_to_half_log_snr. Wraps any
# inner SAMPLER (KSampler-style).
# =============================================================================


class CFGPPProxyModel:
    def __init__(self, model, sigmas):
        self.model = model
        self.sigmas = sigmas
        self.uncond_denoised = None

    def __call__(self, x, sigma, **kwargs):
        model_options = kwargs.get("model_options", {}).copy()

        def post_cfg_function(args):
            self.uncond_denoised = args["uncond_denoised"]
            return args["denoised"]

        kwargs["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(
            model_options, post_cfg_function, disable_cfg1_optimization=True
        )

        denoised_guided = self.model(x, sigma, **kwargs)

        if self.uncond_denoised is None:
            return denoised_guided

        sigma_val = sigma.flatten()[0].item()
        idx = (self.sigmas - sigma_val).abs().argmin().item()
        sigma_next_val = float(self.sigmas[idx + 1]) if idx + 1 < len(self.sigmas) else 0.0

        if sigma_next_val == 0:
            return denoised_guided

        model_sampling = self.model.inner_model.model_patcher.get_model_object("model_sampling")
        lambda_fn = lambda s: sigma_to_half_log_snr(s, model_sampling)

        alpha_s = sigma_val * lambda_fn(torch.tensor(sigma_val)).exp().item()
        alpha_t = sigma_next_val * lambda_fn(torch.tensor(sigma_next_val)).exp().item()

        denoised_star = (
            sigma_val * alpha_t * denoised_guided
            - sigma_next_val * alpha_s * self.uncond_denoised
        ) / (sigma_val - sigma_next_val)

        return denoised_star

    def __getattr__(self, name):
        return getattr(self.model, name)


@torch.no_grad()
def sample_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None,
                 sampler=None):
    extra_args = {} if extra_args is None else extra_args
    proxy = CFGPPProxyModel(model, sigmas)
    return sampler.sampler_function(
        proxy, x, sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        **sampler.extra_options,
    )


# The following function adds the samplers during initialization, in __init__.py
def add_samplers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    if hasattr(KSampler, "DISCARD_PENULTIMATE_SIGMA_SAMPLERS"):
        KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS |= discard_penultimate_sigma_samplers

    # ---- Top-level samplers: registered into BOTH KSampler.SAMPLERS (dropdown)
    # AND k_diffusion_sampling (function lookup) ----
    added = 0
    for sampler in extra_samplers: #getattr(self, "sample_{}".format(extra_samplers))
        if sampler not in KSampler.SAMPLERS:
            try:
                idx = KSampler.SAMPLERS.index("uni_pc_bh2") # Last item in the samplers list
                KSampler.SAMPLERS.insert(idx+1, sampler) # Add our custom samplers
                setattr(k_diffusion_sampling, "sample_{}".format(sampler), extra_samplers[sampler])
                added += 1
            except ValueError as _err:
                pass
    if added > 0:
        import importlib
        importlib.reload(k_diffusion_sampling)

    # ---- Sampler wrappers: registered into k_diffusion_sampling ONLY (function
    # lookup). They are NOT added to KSampler.SAMPLERS, so they will NOT appear
    # in the standard KSampler node's sampler_name dropdown. They are only
    # accessible through their dedicated wrapper node (e.g. SamplerWrapperCFGPP),
    # which calls comfy.samplers.ksampler("<name>", {"sampler": inner_sampler}).
    # comfy.samplers.ksampler() resolves the function via
    # getattr(k_diffusion_sampling, "sample_<name>"), which is what we set here.
    for name, func in extra_sampler_wrappers.items():
        if not hasattr(k_diffusion_sampling, "sample_{}".format(name)):
            setattr(k_diffusion_sampling, "sample_{}".format(name), func)

extra_samplers = {
    "clyb_bdf": sample_clyb_bdf,
    "taylor_flow": sample_taylor_flow,
}

# Wrappers are NOT in the standard sampler dropdown. They are only reachable
# via dedicated wrapper nodes (e.g. SamplerWrapperCFGPP).
extra_sampler_wrappers = {
    "cfgpp": sample_cfgpp,
}

discard_penultimate_sigma_samplers = set(())

class SamplerClyb_BDF:
    @classmethod
    def INPUT_TYPES(s):
        NOISE_SAMPLER_NAMES=("projection", "atan2sin", "atan2sin+projection")
        return {"required":
                    {"scalar": (NOISE_SAMPLER_NAMES, {"default": NOISE_SAMPLER_NAMES[2]}),
                     "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, scalar, eta, s_noise):
        sampler = comfy.samplers.ksampler("clyb_bdf", {"scalar": scalar, "eta": eta, "s_noise": s_noise})
        return (sampler, )


class SamplerTaylorFlow:
    """
    Taylor Flow sampler - Multi-step sampler using Taylor expansion on previous
    denoised predictions to approximate higher-order derivatives.

    Based on "Leveraging Previous Steps: A Training-free Fast Solver for
    Flow Diffusion" (Nov 2024). Achieves O(h^p) approximation error with only
    1 function evaluation per step by reusing cached historical predictions.

    Key features:
    - Leverages previous steps via rolling history buffer
    - Polynomial interpolation via Vandermonde matrix
    - Compatible with both flow models (FLUX, SD3, Chroma) and diffusion models
    - Multiple ancestral sigma calculation methods

    Parameters:
    - order (1-16): Taylor expansion order. Higher = more accurate but uses more history
    - eta: Ancestral sampling stochasticity (0=deterministic, 1=full stochastic)
    - s_noise: Noise scaling factor
    - sigma_calc: Ancestral sigma calculation method (clyb, taylor-expansion, ancestral, adaptive)

    Recommended for:
    - High-quality generation with fewer steps
    - Flow-based models (FLUX, SD3, Chroma) with ancestral sampling
    - Balancing speed (fewer NFEs) and quality (higher-order accuracy)
    - Experimenting with different ancestral noise schedules
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "order": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Taylor expansion order (1-16). Higher = more accurate but uses more history",
                    },
                ),
                "eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Ancestral sampling stochasticity (0=deterministic, 1=full stochastic)",
                    },
                ),
                "s_noise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Noise scaling factor",
                    },
                ),
                "sigma_calc": (
                    ["clyb", "taylor-expansion", "ancestral", "adaptive"],
                    {
                        "default": "clyb",
                        "tooltip": "Ancestral sigma calculation method: clyb (original log-based), taylor-expansion (exponential+quadratic), ancestral (standard k-diffusion), adaptive (history-based convergence-aware)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, order, eta, s_noise, sigma_calc):
        sampler = comfy.samplers.ksampler(
            "taylor_flow",
            {
                "order": order,
                "eta": eta,
                "s_noise": s_noise,
                "sigma_calc": sigma_calc,
            },
        )
        return (sampler,)


class SamplerWrapperCFGPP:
    """
    CFG++ Sampler Wrapper.

    Wraps an inner SAMPLER and intercepts its model call via a proxy that
    captures the uncond_denoised output through a post-CFG hook, then
    recomputes a CFG++-style denoised. Implementation follows the standard
    CFG++ paper formulation: the uncond output is taken from a model sampling
    at the next sigma, and the final denoised_star is computed as
        (sigma * alpha_t * denoised_guided
         - sigma_next * alpha_s * uncond_denoised) / (sigma - sigma_next).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler": ("SAMPLER",),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, sampler):
        sampler = comfy.samplers.ksampler(
            "cfgpp",
            {"sampler": sampler},
        )
        return (sampler,)