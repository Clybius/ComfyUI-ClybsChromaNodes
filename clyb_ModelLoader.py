import torch
import logging
import comfy.sd
import folder_paths
import comfy.utils
import comfy.lora
import comfy.lora_convert


class ClybAdaptiveLoraLoader:
    def __init__(self):
        self.loaded_loras = {}

    @classmethod
    def INPUT_TYPES(s):
        file_list = folder_paths.get_filename_list("loras")
        file_list.insert(0, "none")

        inputs = {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name_1": (file_list, {"tooltip": "The name of the LoRA."}),
                "strength_model_1": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip_1": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            },
            "optional": {}
        }

        for i in range(2, 21):
            inputs["optional"][f"lora_name_{i}"] = (file_list, {"default": "none"})
            inputs["optional"][f"strength_model_{i}"] = ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})
            inputs["optional"][f"strength_clip_{i}"] = ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})

        return inputs

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"
    DESCRIPTION = "Apply multiple LoRAs adaptively by merging patches into a single model clone."
    EXPERIMENTAL = True

    def load_lora(self, model, clip, **kwargs):
        lora_keys = [k for k in kwargs.keys() if k.startswith("lora_name_")]
        try:
            lora_keys.sort(key=lambda x: int(x.split("_")[-1]))
        except:
            pass

        model_lora = model.clone() if model is not None else None
        clip_lora = clip.clone() if clip is not None else None

        for k in lora_keys:
            lora_name = kwargs[k]
            if lora_name == "none":
                continue

            idx = k.split("_")[-1]
            strength_model = kwargs.get(f"strength_model_{idx}", 1.0)
            strength_clip = kwargs.get(f"strength_clip_{idx}", 1.0)

            if strength_model == 0 and strength_clip == 0:
                continue

            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            lora = None
            if idx in self.loaded_loras:
                if self.loaded_loras[idx][0] == lora_path:
                    lora = self.loaded_loras[idx][1]
                else:
                    self.loaded_loras.pop(idx, None)

            if lora is None:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_loras[idx] = (lora_path, lora)

            key_map = {}
            if model_lora is not None:
                key_map = comfy.lora.model_lora_keys_unet(model_lora.model, key_map)
            if clip_lora is not None:
                key_map = comfy.lora.model_lora_keys_clip(clip_lora.cond_stage_model, key_map)

            lora_converted = comfy.lora_convert.convert_lora(lora)
            loaded = comfy.lora.load_lora(lora_converted, key_map)

            if model_lora is not None:
                model_lora.add_patches(loaded, strength_model)
            if clip_lora is not None:
                clip_lora.add_patches(loaded, strength_clip)

        return (model_lora, clip_lora)


class ClybAdaptiveLoraLoaderModelOnly(ClybAdaptiveLoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        file_list = folder_paths.get_filename_list("loras")
        file_list.insert(0, "none")
        inputs = {
            "required": {
                "model": ("MODEL",),
                "lora_name_1": (file_list, ),
                "strength_model_1": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {}
        }
        for i in range(2, 21):
            inputs["optional"][f"lora_name_{i}"] = (file_list, {"default": "none"})
            inputs["optional"][f"strength_model_{i}"] = ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})

        return inputs

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model_only"

    def load_lora_model_only(self, model, **kwargs):
        new_kwargs = kwargs.copy()
        lora_keys = [k for k in kwargs.keys() if k.startswith("lora_name_")]
        for k in lora_keys:
            idx = k.split("_")[-1]
            new_kwargs[f"strength_clip_{idx}"] = 0.0

        return (self.load_lora(model, None, **new_kwargs)[0],)
