import contextlib

import torch
from .quantize import fp8_forward
from comfy.ops import cast_bias_weight, CastWeightBiasOp


class Linear(torch.nn.Linear, CastWeightBiasOp):
    def reset_parameters(self):
        return None

    def forward_comfy_cast_weights(self, input):
        if hasattr(self, "scale"):
            return fp8_forward(self, input)
        weight, bias = cast_bias_weight(self, input)
        return torch.nn.functional.linear(input, weight, bias)

    def forward(self, *args, **kwargs):
        if self.comfy_cast_weights:
            return self.forward_comfy_cast_weights(*args, **kwargs)
        else:
            if hasattr(self, "scale"):
                return fp8_forward(self, args[0])
            return super().forward(*args, **kwargs)


@contextlib.contextmanager
def bizyair_enhancer_ctx(is_bypass=False):
    if is_bypass:
        try:
            print(f"bizyair_enhancer_ctx starts with no enhancment")
            yield
        finally:
            pass
    else:
        from comfy.ops import disable_weight_init

        old_linear = disable_weight_init.Linear
        disable_weight_init.Linear = Linear
        print("bizyairenhancer:  hijack the Linear class")
        try:
            yield
        finally:
            disable_weight_init.Linear = old_linear
            print("bizyairenhancer:  revert the Linear class")
