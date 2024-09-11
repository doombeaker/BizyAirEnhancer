import torch
import torch.nn as nn
from .quantize import fp8_forward, fp8_quantize
from comfy.ops import disable_weight_init, cast_bias_weight


class fp8_quantize_ops(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            dtype=torch.float8_e4m3fn,
            device=torch.device("cuda", 0),
        ):
            super().__init__(
                in_features, out_features, bias, dtype=dtype, device=device
            )
            self.weight = nn.Parameter(
                torch.empty(
                    out_features, in_features, dtype=torch.float8_e4m3fn, device=device
                )
            )

        def forward_comfy_cast_weights(self, input):
            if hasattr(self, "scale") or hasattr(self, "origin_scale"):
                return fp8_forward(self, input)
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                if hasattr(self, "scale") or hasattr(self, "origin_scale"):
                    return fp8_forward(self, args[0])
                return super().forward(*args, **kwargs)
