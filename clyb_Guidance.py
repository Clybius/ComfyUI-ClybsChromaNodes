import torch
import torch.nn.functional as F
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
                "atan2sin_ratio": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip":"Applies standard deviation renormalization of CFG to cond at this rate."}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "sampling/custom_sampling"

    def patch(self, model, eta, norm_threshold, momentum, momentum_beta, momentum_renorm, scalar_projection, scalar_logsumexp, rescale_phi, var_rescale, scale_up_ratio, scale_up_shift, atan2sin_ratio):
        running_avg = 0
        prev_sigma = None

        m = model.clone()
        model_sampling = m.get_model_object("model_sampling")

        def pre_cfg_function(args):
            nonlocal running_avg, prev_sigma

            if len(args["conds_out"]) == 1: return args["conds_out"]

            cond = args["conds_out"][0]
            uncond = args["conds_out"][1]
            sigma = args["sigma"][0]
            cond_scale = args["cond_scale"]
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

            if atan2sin_ratio != 0:
                uncond = uncond.lerp(cond.atan().sin_().div_(uncond.atan().cos_()), weight=atan2sin_ratio)

            if scalar_projection:
                cond_flat, uncond_flat = cond.view(cond.shape[0], -1), uncond.view(uncond.shape[0], -1)

                dot_product = torch.logsumexp(cond_flat * uncond_flat, dim=1, keepdim=True) if scalar_logsumexp else torch.sum(cond_flat * uncond_flat, dim=1, keepdim=True)

                squared_norm = torch.logsumexp(uncond_flat**2, dim=1, keepdim=True) if scalar_logsumexp else torch.sum(uncond_flat**2, dim=1, keepdim=True)

                alpha = dot_product / squared_norm.clamp_min(1e-7)

                uncond = uncond * alpha

            guidance_multiplier = torch.lerp(torch.ones_like(args["timestep"]), torch.sin((1. - timestep_ratio**scale_up_shift) * math.pi), weight=scale_up_ratio) # Lerp from static 1.0 scale to bell-curve (sine wave) scale
            cfg_scalar = 1 / cond_scale + guidance_multiplier * ((cond_scale - 1) / cond_scale) # The guidance scale ought to be at least 1 (guidance is multiplied by cond scale, so ensure a min of 1/cond_scale)
            guidance = ((cond - uncond) * cfg_scalar) if scale_up_ratio != 0 else (cond - uncond) # Guidance is equivalent to (uncond -> cond) 

            if momentum != 0:
                if not torch.is_tensor(running_avg):
                    running_avg = guidance
                else:
                    running_avg = running_avg.lerp(guidance, weight=1. - momentum_beta)#running_avg.lerp(guidance, weight=1. - abs(momentum))# Update running average
                momentumized_guidance = guidance.add(running_avg, alpha=momentum)
                momentumized_guidance_flat, guidance_flat = momentumized_guidance.view(momentumized_guidance.shape[0], -1), guidance.view(guidance.shape[0], -1)
                guidance = momentumized_guidance.lerp(momentumized_guidance * (guidance_flat.norm(2, dim=1, keepdim=True) / momentumized_guidance_flat.norm(2, dim=1, keepdim=True).clamp_min(1e-7)), weight=momentum_renorm)

            guidance_parallel, guidance_orthogonal = project(guidance, cond)
            modified_guidance = guidance_orthogonal + eta * guidance_parallel

            if norm_threshold > 0:
                cond_norm = cond.norm(p=2, dim=tuple(range(1, len(cond.shape))), keepdim=True) * norm_threshold
                guidance_norm = (uncond + modified_guidance * cond_scale).norm(p=2, dim=tuple(range(1, len(cond.shape))), keepdim=True)
                if guidance_norm >= cond_norm:
                    modified_guidance = modified_guidance * (cond_norm / guidance_norm)

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

        """
        TODO: Rework these and add a selector
        def magnitude_guidance(args):
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            x = args['input']
            out = args["denoised"]

            # Flatten cond and uncond, ensure cond scale is positive, utilize double precision
            device = cond.device
            b, c, h, w = cond.shape

            # 1. Create the 2D Hann window kernel for convolution
            hann_1d = torch.signal.windows.hann(63, device=device)
            #hann_2d = torch.outer(hann_1d, hann_1d)
            # Normalize the kernel so that the sum of its elements is 1
            hann_1d /= hann_1d.sum()
            
            # Reshape kernel for depthwise convolution: (out_channels, in_channels/groups, kH, kW)
            # We use groups=c to apply the same 2D filter to each channel independently.
            kernel = hann_1d.unsqueeze(0).unsqueeze(0)#.repeat(cond.shape[1], 1, 1, 1)

            # 2. Calculate the local average magnitude of the `cond` tensor
            # We use the absolute value to measure magnitude, not the raw value.
            # 'same' padding ensures the output has the same HxW dimensions as the input.
            view_shape = (cond.shape[0], -1)
            cond_flat = cond.view(view_shape)
            uncond_flat = uncond.view(view_shape)
            local_avg_magnitude = F.conv1d((cond_flat - uncond_flat), kernel, padding='same')

            # 3. Normalize the magnitude map for each image in the batch to the [0, 1] range
            # This makes the `strength` parameter behave consistently across different images.
            batch_mins = torch.min(local_avg_magnitude.view(view_shape), dim=-1)[0]#.view(cond.shape[0], 1, 1, 1)
            batch_maxs = torch.max(local_avg_magnitude.view(view_shape), dim=-1)[0]#.view(cond.shape[0], 1, 1, 1)
            
            normalized_magnitude = (local_avg_magnitude - batch_mins) / (batch_maxs - batch_mins).clamp_min_(1e-16)

            # 4. Create the local scale multiplier
            # The dampening is proportional to the normalized local magnitude and the `strength` param.
            # We clamp to ensure the multiplier stays within a reasonable [0, 1] range.
            dampening = torch.clamp(normalized_magnitude, 0, 1.0)
            
            # Invert the dampening: high magnitude -> low multiplier, low magnitude -> high multiplier
            scale_multiplier = 1.0 - dampening
            
            cond_normed = cond_flat / torch.linalg.norm(cond_flat,dim=1,keepdim=True)
            uncond_normed = uncond_flat / torch.linalg.norm(uncond_flat,dim=1,keepdim=True)
            dot_product = torch.sum(cond_normed*uncond_normed,dim=1,keepdim=True)
            # The final local scale is the base scale modulated by our multiplier
            local_scale = cond_scale * scale_multiplier * dot_product

            #print(local_scale)

            # 5. Apply the standard CFG formula, but with our dynamic local scale
            # uncond + local_scale * (cond - uncond)
            guided_tensor = cond + (local_scale * (cond_flat - uncond_flat)).view(cond.shape)

            output = out.lerp(guided_tensor.view(cond.shape).to(cond.dtype), weight=atan2sin_ratio)

            if norm_threshold > 0:
                guidance_norm = output.norm(p=2, dim=1, keepdim=True)
                output = torch.where(
                    guidance_norm > norm_threshold,
                    output * (norm_threshold / guidance_norm),
                    output
                )
            return output

        def frequency_guidance(args):
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            x = args['input']
            out = args["denoised"]

            # 1. Move to Frequency domain using 2D Fast Fourier Transform
            # We use norm='ortho' to ensure the transform is unitary and preserves energy.
            fft_cond = torch.fft.fftshift(torch.fft.fftn(cond.to(torch.float64), norm='ortho'))
            fft_uncond = torch.fft.fftshift(torch.fft.fftn(uncond.to(torch.float64), norm='ortho'))
            fft_out = torch.fft.fftshift(torch.fft.fftn(out.to(torch.float64), norm='ortho'))

            # 1. Create the 2D Hann window kernel for convolution
            hann_1d = torch.signal.windows.hann(5, device=cond.device)
            #hann_2d = torch.outer(hann_1d, hann_1d)
            # Normalize the kernel so that the sum of its elements is 1
            hann_1d /= hann_1d.sum()

            kernel = hann_1d.unsqueeze(0).unsqueeze(0)#.repeat(cond.shape[1], 1, 1, 1)

            # 2. Calculate the local average magnitude of the `cond` tensor
            # We use the absolute value to measure magnitude, not the raw value.
            # 'same' padding ensures the output has the same HxW dimensions as the input.
            view_shape = (fft_cond.shape[0], -1)

            fft_cond_flat = fft_cond.view(view_shape)
            fft_uncond_flat = fft_uncond.view(view_shape)

            fft_cond_real = fft_cond_flat.real
            fft_uncond_real = fft_uncond_flat.real
            #guidance_direction = (cond - uncond)
            local_avg_magnitude = F.conv1d(fft_cond_real, kernel.to(torch.float64), padding='same')

            # 3. Normalize the magnitude map for each image in the batch to the [0, 1] range
            # This makes the `strength` parameter behave consistently across different images.
            #view_shape = (cond.shape[0], -1)
            batch_mins = torch.min(local_avg_magnitude.view(view_shape), dim=-1)[0]#.view(cond.shape[0], 1, 1, 1)
            batch_maxs = torch.max(local_avg_magnitude.view(view_shape), dim=-1)[0]#.view(cond.shape[0], 1, 1, 1)
            
            normalized_magnitude = (local_avg_magnitude - batch_mins) / (batch_maxs - batch_mins).clamp_min_(1e-16)

            dampening = torch.clamp(normalized_magnitude, 0, 1.0)
            
            # Invert the dampening: high magnitude -> low multiplier, low magnitude -> high multiplier
            scale_multiplier = 1.0 - dampening
            
            # The final local scale is the base scale modulated by our multiplier and dot product
            cond_normed = fft_cond_real / torch.linalg.norm(fft_cond_real,dim=1,keepdim=True).clamp_min_(1e-16)
            uncond_normed = fft_uncond_real / torch.linalg.norm(fft_uncond_real,dim=1,keepdim=True).clamp_min_(1e-16)
            dot_product = torch.sum(cond_normed*uncond_normed,dim=1,keepdim=True)

            local_scale = cond_scale * scale_multiplier * dot_product

            print(local_scale)

            guided_tensor = fft_cond + (local_scale.to(torch.cdouble) * (fft_cond_flat - fft_uncond_flat)).view(fft_cond.shape)

            guided_tensor = torch.fft.ifftshift(guided_tensor)
            guided_tensor = torch.fft.ifftn(guided_tensor, norm='ortho').real

            output = out.lerp(guided_tensor.view(cond.shape).to(cond.dtype), weight=atan2sin_ratio)

            if norm_threshold > 0:
                guidance_norm = output.norm(p=2, dim=1, keepdim=True)
                output = torch.where(
                    guidance_norm > norm_threshold,
                    output * (norm_threshold / guidance_norm),
                    output
                )
            return output

        def kron_guidance(args):
            cond = args["cond_denoised"]
            uncond = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            x = args['input']
            out = args["denoised"]

            guidance = cond - uncond

            device = guidance.device

            # 2. To Frequency Domain
            # Apply 2D FFT to the spatial dimensions (H, W)
            guidance_fft = torch.fft.fftn(guidance)

            # Shift the zero-frequency component to the center for easier mask creation
            guidance_fft_shifted = torch.fft.fftshift(guidance_fft).reshape(guidance_fft.shape[0], -1)

            U, S, Vh = torch.linalg.svd(guidance_fft_shifted, full_matrices=False)

            # 3. Truncate the SVD components to the desired rank
            U_k = U[:, :1]
            S_k = S[:1]
            Vh_k = Vh[:1, :]
            #S_k_sqrt = S_k.sqrt()

            # 4. Apply the Mask
            # Multiply the shifted FFT of the guidance by the scale mask
            scaled_guidance_fft_shifted = (U_k @ (torch.diag(S_k).to(torch.cfloat) @ Vh_k)).reshape(guidance_fft.shape)

            # 5. Back to Latent Domain
            # Inverse shift to move the zero-frequency back to the corner
            scaled_guidance_fft = torch.fft.ifftshift(scaled_guidance_fft_shifted)

            # Inverse 2D FFT to get back to the spatial (latent) domain
            # The result of ifft2 will be complex; we take the real part. The imaginary
            # part should be negligible for real inputs.
            modified_guidance = torch.fft.ifftn(scaled_guidance_fft).real

            # 6. Apply Guidance
            # Add the modified guidance to the unconditional prediction
            output = out + (guidance - modified_guidance) * atan2sin_ratio

            #output = out.lerp(guided_tensor.view(cond.shape).to(cond.dtype), weight=atan2sin_ratio)

            if norm_threshold > 0:
                guidance_norm = output.norm(p=2, dim=1, keepdim=True)
                output = torch.where(
                    guidance_norm > norm_threshold,
                    output * (norm_threshold / guidance_norm),
                    output
                )
            return output
        """

        m.set_model_sampler_pre_cfg_function(pre_cfg_function)

        return (m,)