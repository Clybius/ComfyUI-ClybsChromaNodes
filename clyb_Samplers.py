import math

import torch
from tqdm.auto import trange

from comfy.k_diffusion.sampling import default_noise_sampler
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

# The following function adds the samplers during initialization, in __init__.py
def add_samplers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    if hasattr(KSampler, "DISCARD_PENULTIMATE_SIGMA_SAMPLERS"):
        KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS |= discard_penultimate_sigma_samplers
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

extra_samplers = {
    "clyb_bdf": sample_clyb_bdf,
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