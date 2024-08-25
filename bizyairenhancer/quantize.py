import torch

__all__ = ["sd_quantize_model", "clip_quantize_model"]


def fp8_forward(self, input):
    qinput, scale, _ = fp8_quantize(input)
    input_shape = qinput.shape
    try:
        output, _ = torch._scaled_mm(
            qinput.reshape(-1, input_shape[-1]),
            self.weight.t(),
            out_dtype=input.dtype,
            scale_a=scale,
            scale_b=self.scale,
            bias=self.bias,
        )
    except Exception as e:
        print(f"{input.shape=}, {qinput.shape=}, {self.weight.shape=}")
        raise e
    return output.reshape(*input_shape[:-1], output.shape[-1])


def fp8_quantize(weight, qdtype=torch.float8_e4m3fn):
    device = weight.device
    finfo = torch.finfo(qdtype)

    scale = finfo.max / weight.abs().max().clamp(min=1e-12)

    qweight = (weight * scale).clamp(min=finfo.min, max=finfo.max)

    qweight = qweight.to(qdtype)
    scale = scale.float().reciprocal()
    return qweight, scale, device


def sd_quantize_model(model: torch.nn.Module, new_state_dict):
    for name, module in model.diffusion_model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = new_state_dict[f"{name}.weight"]
        if module.bias is not None:
            module.bias.data = module.bias.data.to(torch.bfloat16)
        qweight, scale, device = fp8_quantize(weight)
        module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
        module.weight.data.copy_(qweight.data)
        module.register_buffer("scale", scale)
        new_state_dict.pop(f"{name}.weight")


def clip_quantize_model(model: torch.nn.Module, new_state_dict):
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = new_state_dict[f"{name}.weight"]
        if module.bias is not None:
            module.bias.data = module.bias.data.to(torch.float16)
        qweight, scale, device = fp8_quantize(weight)
        module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
        module.weight.data.copy_(qweight.data)
        module.register_buffer("scale", scale.to(device))
        new_state_dict.pop(f"{name}.weight")
