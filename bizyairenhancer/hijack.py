import contextlib

import torch
import torch.nn as nn
from .quantize import fp8_forward
from comfy.ops import cast_bias_weight, CastWeightBiasOp, manual_cast


class Linear(torch.nn.Linear, CastWeightBiasOp):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=torch.float8_e4m3fn,
        device=torch.device("cuda", 0),
    ):
        super().__init__(in_features, out_features, bias, dtype=dtype, device=device)
        self.register_buffer("scale", torch.tensor(1, device=device, dtype=torch.float))
        self.weight = nn.Parameter(
            torch.empty(
                out_features, in_features, dtype=torch.float8_e4m3fn, device=device
            )
        )

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


class ManualCastLinear(Linear):
    comfy_cast_weights = True


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

        old_cast_linear = manual_cast.Linear
        manual_cast.Linear = ManualCastLinear
        print("bizyairenhancer:  hijack the Linear class")
        try:
            yield
        finally:
            disable_weight_init.Linear = old_linear
            manual_cast.Linear = old_cast_linear
            print("bizyairenhancer:  revert the Linear class")
