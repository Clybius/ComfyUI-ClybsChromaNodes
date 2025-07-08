from . import chroma_NAG
from . import clyb_Guidance
from . import clyb_Samplers
from . import clyb_Schedulers

clyb_Samplers.add_samplers()

NODE_CLASS_MAPPINGS = {
    "ClybGuidance": clyb_Guidance.ClybGuidance,
    "SamplerClyb_BDF": clyb_Samplers.SamplerClyb_BDF,
    "InverseSquaredScheduler": clyb_Schedulers.InverseSquaredScheduler,
    "PrintSigmas": clyb_Schedulers.PrintSigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClybGuidance": "ClybGuidance",
    "SamplerClyb_BDF": "SamplerClyb_BDF",
    "InverseSquaredScheduler": "InverseSquaredScheduler",
    "PrintSigmas": "PrintSigmas",
}