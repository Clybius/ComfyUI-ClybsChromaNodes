import numpy
import torch
from comfy.samplers import SchedulerHandler, SCHEDULER_HANDLERS, SCHEDULER_NAMES

def inverse_squared_scheduler(model_sampling, steps):
    total_timesteps = (len(model_sampling.sigmas) - 1)
    ts = (1 - numpy.linspace(0, 1, steps, endpoint=False)**2)**2
    ts = numpy.rint(ts * total_timesteps)

    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs += [float(model_sampling.sigmas[int(t)])]
        last_t = t
    sigs += [0.0]
    return torch.FloatTensor(sigs)

class InverseSquaredScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "steps": ("INT", {"default": 20, "min": 3, "max": 1000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            total_steps = int(steps/denoise)

        sigmas = inverse_squared_scheduler(model.get_model_object("model_sampling"), total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]

        return (sigmas, )

class PrintSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS",),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "PrintSigmas"

    def PrintSigmas(self, sigmas):
        print(sigmas)
        return (sigmas, )

scheduler_name = "inverse_squared"
if scheduler_name not in SCHEDULER_HANDLERS:
    scheduler_handler = SchedulerHandler(handler=inverse_squared_scheduler, use_ms=True)
    SCHEDULER_HANDLERS[scheduler_name] = scheduler_handler
    if scheduler_name not in SCHEDULER_NAMES:
        SCHEDULER_NAMES.append(scheduler_name)

extra_schedulers = {
    "inverse_squared": inverse_squared_scheduler,
}