import contextlib

import torch
import torch.nn as nn
from .quantize import fp8_forward
from comfy.ops import cast_bias_weight, CastWeightBiasOp, manual_cast
