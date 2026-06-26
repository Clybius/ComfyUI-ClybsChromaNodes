from . import chroma_NAG
from . import clyb_Guidance
from . import clyb_Samplers
from . import clyb_Schedulers
from . import clyb_ModelLoader

clyb_Samplers.add_samplers()

NODE_CLASS_MAPPINGS = {
    # Guidance
    "ClybGuidance": clyb_Guidance.ClybGuidance,
    # Samplers
    "SamplerClyb_BDF": clyb_Samplers.SamplerClyb_BDF,
    "SamplerTaylorFlow": clyb_Samplers.SamplerTaylorFlow,
    "SamplerWrapperCFGPP": clyb_Samplers.SamplerWrapperCFGPP,
    # Schedulers
    "InverseSquaredScheduler": clyb_Schedulers.InverseSquaredScheduler,
    "PrintSigmas": clyb_Schedulers.PrintSigmas,
    # LoraLoaders
    "ClybAdaptiveLoraLoader": clyb_ModelLoader.ClybAdaptiveLoraLoader,
    "ClybAdaptiveLoraLoaderModelOnly": clyb_ModelLoader.ClybAdaptiveLoraLoaderModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Guidance
    "ClybGuidance": "ClybGuidance",
    # Samplers
    "SamplerClyb_BDF": "SamplerClyb_BDF",
    "SamplerTaylorFlow": "SamplerTaylorFlow",
    "SamplerWrapperCFGPP": "SamplerWrapperCFGPP",
    # Schedulers
    "InverseSquaredScheduler": "InverseSquaredScheduler",
    "PrintSigmas": "PrintSigmas",
    # LoraLoaders
    "ClybAdaptiveLoraLoader": "ClybAdaptiveLoraLoader",
    "ClybAdaptiveLoraLoaderModelOnly": "ClybAdaptiveLoraLoaderModelOnly",
}

WEB_DIRECTORY = "./js"
