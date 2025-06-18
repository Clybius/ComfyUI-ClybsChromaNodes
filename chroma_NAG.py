import torch
import torch.nn.functional as F
import math
import types
from typing import Optional

import comfy.model_management as mm
from comfy.ldm.flux.math import attention
from comfy.ldm.flux.layers import SelfAttention # Assuming SelfAttention is where qkv and norm are defined

# Initial implementation of ComfyUI NAG by kijai https://github.com/kijai/ComfyUI-KJNodes/commit/d584c711a374e8267496dc5241ff879588212360

def match_seq_len(tensor_to_resize, reference_tensor):
    """
    Resizes a tensor to match the sequence length of a reference tensor.
    It either pads with zeros or truncates the tensor along dimension 2.

    Args:
        tensor_to_resize (torch.Tensor): The tensor to be resized. Shape [B, H, S1, D]
        reference_tensor (torch.Tensor): The tensor with the target sequence length. Shape [B, H, S2, D]

    Returns:
        torch.Tensor: The resized tensor. Shape [B, H, S2, D]
    """
    target_len = reference_tensor.shape[2]
    current_len = tensor_to_resize.shape[2]

    if current_len == target_len:
        return tensor_to_resize

    if current_len > target_len:
        # Truncate the tensor
        return tensor_to_resize[:, :, :target_len, ...]
    else:
        # Pad the tensor
        # The padding format is (pad_left, pad_right, pad_top, pad_bottom, ...)
        # for the last, second-to-last, etc. dimensions.
        # We only want to pad the 2nd dimension (sequence length).
        diff = target_len - current_len
        # Assuming shape is (B, Heads, Seq_Len, Dim), we pad the 3rd dimension from the end.
        # F.pad expects padding for (..., last_dim, 2nd_to_last_dim, ...)
        # So we specify (0,0) for the last dim (D) and (0, diff) for the 2nd-to-last dim (S).
        padding = (0, 0, 0, diff) 
        return F.pad(tensor_to_resize, padding, "constant", 0)

# This new forward function will contain the NAG logic for Chroma's DoubleStreamBlock.
def chroma_doublestream_forward_nag(self, img, txt, pe, vec=None, attn_mask=None, distill_vec=None):
    """
    A patched forward function for a DoubleStreamBlock that incorporates Normalized Attention Guidance.
    
    Original DoubleStreamBlock logic involves a single, unified attention call for both
    image and text tokens. This patched version splits the process for Classifier-Free
    Guidance:
    1. The unconditional part is processed normally using the original forward method.
    2. The conditional part is where NAG is applied:
        a. An "positive" attention output is calculated using the image tokens and the positive text context.
        b. A "synthetic negative" attention output is calculated using the image tokens and the `nag_context`.
        c. The NAG formula (scaling, norm-based clamping, and blending) is applied to these two outputs
           to create a final, guided attention result for the image tokens.
        d. The rest of the block's operations (MLP, etc.) proceed with this guided result.
    """
    if distill_vec is not None:
        vec = distill_vec
    # Deconstruct modulation vectors
    (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

    
    # Get Q, K, V for conditional image tokens
    img_modulated = torch.addcmul(img_mod1.shift, 1 + img_mod1.scale, self.img_norm1(img))
    img_qkv_c = self.img_attn.qkv(img_modulated)
    img_q_c, img_k_c, img_v_c = img_qkv_c.view(img_qkv_c.shape[0], img_qkv_c.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    img_q_c, img_k_c = self.img_attn.norm(img_q_c, img_k_c, img_v_c)

    # Get Q, K, V for positive text tokens
    txt_modulated = torch.addcmul(txt_mod1.shift, 1 + txt_mod1.scale, self.txt_norm1(txt))
    txt_qkv_c = self.txt_attn.qkv(txt_modulated)
    txt_q_c, txt_k_c, txt_v_c = txt_qkv_c.view(txt_qkv_c.shape[0], txt_qkv_c.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    txt_q_c, txt_k_c = self.txt_attn.norm(txt_q_c, txt_k_c, txt_v_c)

    # Perform unified attention for the positive case
    attn_pos = attention(
        torch.cat((txt_q_c, img_q_c), dim=2),
        torch.cat((txt_k_c, img_k_c), dim=2),
        torch.cat((txt_v_c, img_v_c), dim=2),
        pe=pe, mask=attn_mask
    )
    txt_attn_pos, img_attn_pos = attn_pos[:, : txt.shape[1]], attn_pos[:, txt.shape[1] :]
    x_positive = self.img_attn.proj(img_attn_pos)
    x_txt_positive = self.txt_attn.proj(txt_attn_pos)

    # B. Calculate "Synthetic Negative" Attention Output
    
    # Get K, V for the NAG context (reusing positive text modulation)
    nag_context_modulated = torch.addcmul(txt_mod1.shift, 1 + txt_mod1.scale, self.txt_norm1(self.nag_context))
    nag_qkv = self.txt_attn.qkv(nag_context_modulated)
    nag_q, nag_k, nag_v = nag_qkv.view(nag_qkv.shape[0], nag_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    nag_q, nag_k = self.txt_attn.norm(nag_q, nag_k, nag_v)

    # Perform unified attention for the synthetic negative case
    # Match the sequence length of nag_* tensors to their img_* counterparts
    nag_q_resized = match_seq_len(nag_q, txt_q_c)
    nag_k_resized = match_seq_len(nag_k, txt_k_c)
    nag_v_resized = match_seq_len(nag_v, txt_v_c)
    attn_neg = attention(
        torch.cat((nag_q_resized, img_q_c), dim=2),
        torch.cat((nag_k_resized, img_k_c), dim=2),
        torch.cat((nag_v_resized, img_v_c), dim=2),
        pe=pe, mask=attn_mask
    )
    txt_attn_neg, img_attn_neg = attn_neg[:, : txt.shape[1]], attn_neg[:, txt.shape[1] :]
    x_negative = self.img_attn.proj(img_attn_neg)
    x_txt_negative = self.txt_attn.proj(txt_attn_neg)
    
    # C. Apply NAG Guidance
    
    nag_guidance = x_positive * self.nag_scale - x_negative * (self.nag_scale - 1)
    nag_txt_guidance = x_txt_positive * self.nag_scale - x_txt_negative * (self.nag_scale - 1)
    
    norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
    norm_txt_positive = torch.norm(x_txt_positive, p=1, dim=-1, keepdim=True)
    norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True)
    norm_txt_guidance = torch.norm(nag_txt_guidance, p=1, dim=-1, keepdim=True)
    
    # Add epsilon to prevent division by zero
    scale = norm_guidance / (norm_positive + 1e-7)
    scale = torch.nan_to_num(scale, nan=10.0)
    scale_txt = norm_txt_guidance / (norm_txt_positive + 1e-7)
    scale_txt = torch.nan_to_num(scale_txt, nan=10.0)
    
    mask = scale > self.nag_tau
    mask_txt = scale_txt > self.nag_tau
    adjustment = (norm_positive * self.nag_tau) / (norm_guidance + 1e-7)
    adjustment_txt = (norm_txt_positive * self.nag_tau) / (norm_txt_guidance + 1e-7)
    
    # Use torch.where for conditional adjustment
    nag_guidance = torch.where(mask, nag_guidance * adjustment, nag_guidance)
    nag_txt_guidance = torch.where(mask_txt, nag_txt_guidance * adjustment_txt, nag_txt_guidance)
    
    # Blend the original positive output with the guided output
    guided_img_attn_output = nag_guidance * self.nag_alpha + x_positive * (1 - self.nag_alpha)
    guided_txt_attn_output = nag_txt_guidance * self.nag_alpha + x_txt_positive * (1 - self.nag_alpha)

    # --- 3. Complete the rest of the block for the conditional part ---
    
    # Image stream uses the guided attention output
    img_out_c = img.clone()
    img_out_c.addcmul_(img_mod1.gate, guided_img_attn_output)
    img_out_c.addcmul_(img_mod2.gate, self.img_mlp(torch.addcmul(img_mod2.shift, 1 + img_mod2.scale, self.img_norm2(img_out_c))))

    # Text stream uses the original positive attention output
    txt_out_c = txt.clone()
    txt_out_c.addcmul_(txt_mod1.gate, guided_txt_attn_output)
    txt_out_c.addcmul_(txt_mod2.gate, self.txt_mlp(torch.addcmul(txt_mod2.shift, 1 + txt_mod2.scale, self.txt_norm2(txt_out_c))))

    # Final dtype check from original implementation
    if txt_out_c.dtype == torch.float16:
        txt_out_c = torch.nan_to_num(txt_out_c, nan=0.0, posinf=65504, neginf=-65504)
        
    return img_out_c, txt_out_c

class ChromaNAGPatch:
    def __init__(self, nag_context, nag_scale, nag_alpha, nag_tau):
        self.nag_context = nag_context
        self.nag_scale = nag_scale
        self.nag_alpha = nag_alpha
        self.nag_tau = nag_tau

    def __get__(self, obj, objtype=None):
        # obj is the DoubleStreamBlock instance
        # Save the original forward method if it hasn't been saved already
        if not hasattr(obj, 'original_forward'):
            obj.original_forward = obj.forward
        
        # Create a new bound method with the NAG logic
        def wrapped_forward(self_module, *args, **kwargs):
            # Attach NAG parameters to the module instance for access within the forward pass
            self_module.nag_context = self.nag_context
            self_module.nag_scale = self.nag_scale
            self_module.nag_alpha = self.nag_alpha
            self_module.nag_tau = self.nag_tau
            return chroma_doublestream_forward_nag(self_module, *args, **kwargs)
        
        return types.MethodType(wrapped_forward, obj)

class ChromaNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "nag_scale": ("FLOAT", {"default": 5.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.01}),
           }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    
    def patch(self, model, conditioning, nag_scale, nag_alpha, nag_tau):
        if nag_scale == 0:
            return (model,)
        
        device = mm.get_torch_device()
        #dtype = mm.unet_dtype()
        
        # The NAG context is derived from the negative prompt's embeddings.
        # For FLUX/Chroma, the conditioning input is already embedded.
        # Shape: [1, sequence_length, embedding_dim]
        nag_context = conditioning[0][0].clone()
        
        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")
        diffusion_model.txt_in.to(device)
        # Get the actual dtype of the txt_in layer weights
        target_dtype = next(diffusion_model.txt_in.parameters()).dtype
        txt = diffusion_model.txt_in(nag_context.to(device, target_dtype))

        # Chroma models have `double_blocks` where image and text tokens interact.
        # This is the equivalent of a cross-attention stage in other models.
        for i, block in enumerate(diffusion_model.double_blocks):
            # Create the patch object which will handle saving the original method and returning the new one.
            patcher = ChromaNAGPatch(txt, nag_scale, nag_alpha, nag_tau)
            
            # Call __get__ explicitly to get the bound method and patch it to the model
            patched_forward = patcher.__get__(block, block.__class__)
            model_clone.add_object_patch(f"diffusion_model.double_blocks.{i}.forward", patched_forward)
            
        return (model_clone,)

#NODE_CLASS_MAPPINGS = {
#    "ChromaNAG": ChromaNAG,
#}

#NODE_DISPLAY_NAME_MAPPINGS = {
#    "ChromaNAG": "ChromaNAG",
#}
