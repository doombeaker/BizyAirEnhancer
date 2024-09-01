import torch

__all__ = [
    "fp8_quantize_model",
    "fp8_dequantize",
    "fp8_quantize",
]


def fp8_forward(self, input):
    input_mode = "per_tensor"
    weight_mode = "per_tensor"
    qinput, scale, _ = fp8_quantize(input, mode=input_mode)
    input_shape = qinput.shape
    try:
        if weight_mode == "per_tensor":
            if self.bias is not None:
                if self.bias.dtype == torch.float8_e4m3fn:
                    self.bias = torch.nn.Parameter(self.bias.to(dtype=torch.bfloat16))
            output, _ = torch._scaled_mm(
                qinput.reshape(-1, input_shape[-1]),
                self.weight.t(),
                out_dtype=input.dtype,
                scale_a=scale,
                scale_b=self.scale.to(qinput.device),
                bias=self.bias,
            )
        elif weight_mode == "per_token":
            scale_fake = torch.tensor(1, device=input.device, dtype=torch.float)
            # mm_out = torch.matmul(qinput.reshape(-1, input_shape[-1]), self.weight.t())
            output, _ = torch._scaled_mm(
                qinput.reshape(-1, input_shape[-1]),
                self.weight.t(),
                out_dtype=input.dtype,
                scale_a=scale_fake,
                scale_b=scale_fake,
                bias=None,
            )
            output = output * scale.view(-1, 1)
            output = output * self.scale.view(1, -1)
            output = output.to(input.dtype)
            if self.bias is not None:
                output = output + self.bias

    except Exception as e:
        print(
            f"{input.shape=}, {qinput.shape=}, {self.weight.shape=}, {self.scale.dtype=}, {self.scale.shape=}, {scale.shape=}, {scale.dtype=}"
        )
        raise e
    return output.reshape(*input_shape[:-1], output.shape[-1])


def fp8_quantize(tensor, qdtype=torch.float8_e4m3fn, mode: str = "per_tensor"):
    tensor_size = tensor.size()
    tensor = tensor.view(-1, tensor_size[-1])
    if mode == "per_tensor":
        device = tensor.device
        finfo = torch.finfo(qdtype)

        scale = finfo.max / tensor.abs().max().clamp(min=1e-12)

        qtensor = (tensor * scale).clamp(min=finfo.min, max=finfo.max)

        qtensor = qtensor.to(qdtype)
        scale = scale.float().reciprocal()
    elif mode == "per_token":
        device = tensor.device
        try:
            finfo = torch.finfo(qdtype)

            scale = finfo.max / tensor.abs().amax(dim=1).clamp(min=1e-12)

            qtensor = (tensor * scale.view(-1, 1)).clamp(min=finfo.min, max=finfo.max)

            qtensor = qtensor.to(qdtype)
            scale = scale.float().reciprocal()
        except Exception as e:
            print(f"debug==== {scale.shape=}, {tensor.shape=}")
            raise e
    elif mode == "per_channel":
        pass
    return qtensor.view(tensor_size), scale, device


def fp8_quantize_model(model: torch.nn.Module, new_state_dict):
    module_type_name = type(model).__name__
    print(f"Quantize model: {module_type_name}")
    qdtype = torch.float8_e4m3fn
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if module_type_name == "Flux":
            if not (
                name.startswith("double_blocks") or name.startswith("single_blocks")
            ):
                continue
        weight = new_state_dict.pop(f"{name}.weight")
        if module.bias is not None:
            if module_type_name == "Flux":
                module.bias.data = module.bias.data.to(torch.bfloat16)
            elif module_type_name == "T5":
                module.bias.data = module.bias.data.to(torch.float16)
        qweight, scale, device = fp8_quantize(weight, mode="per_tensor")
        module.weight.data = module.weight.data.to(qdtype)
        module.weight.data.copy_(qweight.data)
        module.register_buffer("scale", scale.to(device))


def fp8_dequantize(qtensor, scale, device):
    scale = scale.to(device)
    qtensor = qtensor.to(torch.float16)
    tensor = qtensor * scale

    return tensor
