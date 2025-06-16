import torch
import math
import comfy
import re

def project(v0, v1):
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel, v0_orthogonal

class ClybGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "eta": ("FLOAT", {"default": 1.0, "min": -50.0, "max": 50.0, "step": 0.01, "tooltip": "Controls the scale of the parallel guidance vector. Default CFG behavior at a setting of 1."}),
                "norm_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1, "tooltip": "Normalize guidance vector to this value, normalization disable at a setting of 0."}),
                "momentum": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip":"Controls the amount of momentum applied to the latent, disabled at a setting of 0."}),
                "momentum_beta": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 0.999, "step": 0.01, "tooltip":"Controls a running average of guidance during diffusion."}),
                "momentum_renorm": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":"Re-normalizes your latent after applying momentum, back to its norm before momentum."}),
                "scalar_projection": ("BOOLEAN", {"default": False, "tooltip":"Applies scalar projection of cond -> uncond onto the uncond."}),
                "scalar_logsumexp": ("BOOLEAN", {"default": False, "tooltip":"Whether we use torch.logsumexp (true) or torch.sum (false) for scalar projection."}),
                "rescale_phi": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":"Applies standard deviation renormalization of CFG to cond at this rate."}),
                "var_rescale": ("BOOLEAN", {"default": False, "tooltip":"Whether we use torch.var (true) or torch.std (false) for rescaling."}),
                "scale_up_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":"Weight of: Initiating CFG at guidance scale 1, increasing to your guidance scale in the middle of diffusion, and lower back to 1."}),
                "scale_up_shift": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01, "tooltip":"Whether to shift to your CFG scale later (lower than 1.0) or earlier (higher than 1.0) in the schedule."}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "sampling/custom_sampling"

    def patch(self, model, eta, norm_threshold, momentum, momentum_beta, momentum_renorm, scalar_projection, scalar_logsumexp, rescale_phi, var_rescale, scale_up_ratio, scale_up_shift):
        running_avg = 0
        prev_sigma = None

        def pre_cfg_function(args):
            nonlocal running_avg, prev_sigma

            if len(args["conds_out"]) == 1: return args["conds_out"]

            cond = args["conds_out"][0]
            uncond = args["conds_out"][1]
            sigma = args["sigma"][0]
            cond_scale = args["cond_scale"]
            model_sampling = model.get_model_object("model_sampling")
            flow = False
            if isinstance(model_sampling, comfy.model_sampling.CONST):
                flow = True

            if prev_sigma is not None and args["timestep"] > prev_sigma:
                running_avg = 0
            prev_sigma = args["timestep"]

            # Lerp from static 1.0 to a sine interp of [0, 1.0, 0]
            if hasattr(model_sampling, "num_timesteps"):
                num_timesteps = model_sampling.num_timesteps
            else:
                num_timesteps = None

            if flow or num_timesteps is None:
                timestep_ratio = args["timestep"].float()
            else:
                timestep_ratio = model_sampling.timestep(args["timestep"]).float() / float(num_timesteps - 1) # Ratio scaling from 0 to 1 as diffusion goes on.

            guidance_multiplier = torch.lerp(torch.ones_like(args["timestep"]), torch.sin((1. - timestep_ratio**scale_up_shift) * math.pi), weight=scale_up_ratio) # Lerp from static 1.0 scale to bell-curve (sine wave) scale
            cfg_scalar = 1 / cond_scale + guidance_multiplier * ((cond_scale - 1) / cond_scale) # The guidance scale ought to be at least 1 (guidance is multiplied by cond scale, so ensure a min of 1/cond_scale)

            if scalar_projection:
                cond_flat, uncond_flat = cond.view(cond.shape[0], -1), uncond.view(uncond.shape[0], -1)

                dot_product = torch.logsumexp(cond * uncond, dim=1, keepdim=True) if scalar_logsumexp else torch.sum(cond_flat * uncond_flat, dim=1, keepdim=True)

                squared_norm = torch.logsumexp(uncond**2, dim=1, keepdim=True) if scalar_logsumexp else torch.sum(uncond_flat**2, dim=1, keepdim=True)

                alpha = dot_product / squared_norm.clamp_min(1e-7)

                uncond = uncond * alpha

            guidance = ((cond - uncond) * cfg_scalar) if scale_up_ratio != 0 else (cond - uncond) # Guidance is equivalent to (uncond -> cond) 

            if momentum != 0:
                if not torch.is_tensor(running_avg):
                    running_avg = guidance
                else:
                    running_avg = running_avg.lerp(guidance, weight=1. - momentum_beta)#running_avg.lerp(guidance, weight=1. - abs(momentum))# Update running average
                    #running_avg = running_avg * (guidance.pow(2).mean().sqrt_() / running_avg.pow(2).mean().sqrt_().clamp_min_(1e-8)) # Normalize running average to guidance
                momentumized_guidance = guidance.add(running_avg, alpha=momentum)
                momentumized_guidance_flat, guidance_flat = momentumized_guidance.view(momentumized_guidance.shape[0], -1), guidance.view(guidance.shape[0], -1)
                guidance = momentumized_guidance.lerp(momentumized_guidance * (guidance_flat.norm(1, dim=1, keepdim=True) / momentumized_guidance_flat.norm(1, dim=1, keepdim=True).clamp_min(1e-7)), weight=momentum_renorm)

            if norm_threshold > 0:
                guidance_norm = guidance.view(guidance.shape[0], -1).norm(p=2, dim=1, keepdim=True)
                scale = torch.minimum(
                    torch.ones_like(guidance_norm),
                    norm_threshold / guidance_norm
                )
                guidance = guidance * scale

            guidance_parallel, guidance_orthogonal = project(guidance, cond)
            modified_guidance = guidance_orthogonal + eta * guidance_parallel

            modified_cond = (uncond + modified_guidance)
            if rescale_phi != 0:
                # Formulate CFG
                x_cfg = uncond + modified_guidance * cond_scale

                # STD Renorm
                rescale_func = torch.std if not var_rescale else torch.var
                modified_cond_flat, x_cfg_flat = modified_cond.view(modified_cond.shape[0], -1), x_cfg.view(x_cfg.shape[0], -1)
                ro_pos = torch.std(modified_cond_flat, dim=1, keepdim=True) if not var_rescale else torch.var(modified_cond_flat, dim=1, keepdim=True)
                ro_cfg = torch.std(x_cfg_flat, dim=1, keepdim=True) if not var_rescale else torch.var(x_cfg_flat, dim=1, keepdim=True)

                x_rescaled = x_cfg * (ro_pos / ro_cfg.clamp_min(1e-7))

                # Deformulate CFG
                rescaled_guidance = (x_cfg.lerp(x_rescaled, weight=rescale_phi) - uncond) / cond_scale

                modified_cond = (uncond + rescaled_guidance)# + (cond - uncond) / cond_scale

            return [modified_cond, uncond] + args["conds_out"][2:]

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_function)
        return (m,)

#args = {"conds":conds, "conds_out": out, "cond_scale": self.cfg, "timestep": timestep,
#                    "input": x, "sigma": timestep, "model": self.inner_model, "model_options": model_options}
#out  = fn(args)

def create_number_range(range_str: str) -> list[int] | None:
    """
    Creates a list of numbers from a string in "start-end" format.

    Args:
        range_str: The input string (e.g., "1-5", "10-20", " 5 - 10 ").

    Returns:
        A list of integers representing the range (inclusive),
        or None if the string format is invalid or start > end.
    """
    # The core 're' module line to define and apply the pattern:
    # 1. r"..." denotes a raw string to avoid issues with backslashes.
    # 2. (\d+) is a capturing group for one or more digits (the start number).
    # 3. \s* matches zero or more whitespace characters (optional spaces around the hyphen).
    # 4. - matches the literal hyphen.
    # 5. \s* matches zero or more whitespace characters again.
    # 6. (\d+) is another capturing group for the end number.
    # re.match() attempts to match the pattern from the beginning of the string.
    match = re.match(r"(\d+)\s*-\s*(\d+)", range_str.strip())

    if match:
        # Extract the captured groups and convert them to integers
        start_str, end_str = match.groups()
        start = int(start_str)
        end = int(end_str)

        # Ensure the start is not greater than the end for a valid range
        if start <= end:
            return list(range(start, end + 1))
        else:
            # Handle cases like "5-1" if they should not produce a range
            print(f"Warning: Start ({start}) is greater than end ({end}) for '{range_str}'")
            return None # Or [] if an empty list is preferred for invalid ranges
    else:
        return None # String does not match the expected format

class ClybLayerGuidanceDiT:
    '''
    Enhance guidance towards detailed dtructure by having another set of CFG negative with skipped layers.
    Inspired by Perturbed Attention Guidance (https://arxiv.org/abs/2403.17377)
    Original experimental implementation for SD3 by Dango233@StabilityAI.
    '''
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                             "double_layers": ("STRING", {"default": "7, 8, 9", "multiline": False}),
                             "single_layers": ("STRING", {"default": "7, 8, 9", "multiline": False}),
                             "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                             "start_percent": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "rescaling_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "attn_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                                }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "skip_guidance"
    EXPERIMENTAL = True

    DESCRIPTION = "Generic version of ClybLayerGuidance node that can be used on every DiT model."

    CATEGORY = "advanced/guidance"

    def skip_guidance(self, model, scale, start_percent, end_percent, double_layers="", single_layers="", rescaling_scale=0, attn_scale=1.0):
        # check if layer is comma separated integers
        def skip(args, extra_args):
            print(f"ARGS: {args.items()}", "\n\n\n", f"EXTRA_ARGS: {extra_args.items()}", "\n\n\n")
            for x in args:
                if 'vec' in x:
                    for y in x:
                        if 'scale' in y:
                            args[x][y] = args[x][y] * attn_scale
                    #args[x] = y * attn_scale
                #print(x, y)
            return args
            #for x, y in args.items():
            #    if 'img' in x:
            #        return x

        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        #double_layers = re.findall(r'\d+', double_layers)
        #double_layers = [int(i) for i in double_layers]
        double_layers = create_number_range(double_layers)

        #single_layers = re.findall(r'\d+', single_layers)
        #single_layers = [int(i) for i in single_layers]
        single_layers = create_number_range(single_layers)

        if len(double_layers) == 0 and len(single_layers) == 0:
            return (model, )

        def post_cfg_function(args):
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            x = args["input"]
            model_options = args["model_options"].copy()
            #print(model_options)
            for layer in double_layers:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, skip, "dit", "double_block", layer)

            for layer in single_layers:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, skip, "dit", "single_block", layer)

            model_sampling.percent_to_sigma(start_percent)

            sigma_ = sigma[0].item()
            if scale > 0 and sigma_ >= sigma_end and sigma_ <= sigma_start:
                (slg,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)
                cfg_result = cfg_result + (cond_pred - slg) * scale
                if rescaling_scale != 0:
                    factor = cond_pred.std() / cfg_result.std()
                    factor = rescaling_scale * factor + (1 - rescaling_scale)
                    cfg_result *= factor

            return cfg_result

        m = model.clone()
        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m, )

#NODE_CLASS_MAPPINGS = {
#    "ClybGuidance": ClybGuidance,
#    "ClybLayerGuidanceDiT": ClybLayerGuidanceDiT,
#}

#NODE_DISPLAY_NAME_MAPPINGS = {
#    "ClybGuidance": "ClybGuidance",
#    "ClybLayerGuidanceDiT": "ClybLayerGuidanceDiT",
#}
