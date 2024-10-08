import gc

import torch

__all__ = [
    "fp8_quantize_model",
    "fp8_prepare_model",
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
    qdtype = torch.float8_e4m3fn
    print(f"Quantize model: {module_type_name} to {qdtype}")
    from bizyairenhancer import fp8_quantize_ops

    if module_type_name == "Flux":
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight_key = f"{name}.weight"
            bias_key = f"{name}.bias"
            weight = new_state_dict.get(weight_key)
            if bias_key in new_state_dict:
                module.bias = torch.nn.Parameter(
                    torch.empty(
                        new_state_dict.get(bias_key).shape, dtype=torch.bfloat16
                    )
                )
            if not (
                name.startswith("double_blocks") or name.startswith("single_blocks")
            ):
                module.weight.data = module.weight.data.to(torch.bfloat16)
            else:
                qweight, scale, device = fp8_quantize(weight, mode="per_tensor")
                new_state_dict[weight_key] = qweight
                del weight

                scale_key = f"{name}.scale"
                new_state_dict[scale_key] = scale.to(device)
                module.scale = torch.nn.Parameter(scale)
    elif module_type_name == "T5":
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                dtype = torch.float8_e4m3fn
                device = module.weight.device
                fp8linear = fp8_quantize_ops.Linear(
                    in_features, out_features, bias=False, dtype=dtype, device=device
                )

                parent_module = model
                module_names = name.split(".")
                for sub_name in module_names[:-1]:
                    parent_module = getattr(parent_module, sub_name)
                setattr(parent_module, module_names[-1], fp8linear)
                del module

                weight_key = f"{name}.weight"
                weight = new_state_dict.get(weight_key)
                qweight, scale, device = fp8_quantize(weight, mode="per_tensor")
                new_state_dict[weight_key] = qweight
                del weight

                scale_key = f"{name}.scale"
                new_state_dict[scale_key] = scale.to(device)
                fp8linear.scale = torch.nn.Parameter(scale)
            else:
                module = module.to(dtype=torch.float16)
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print(f"==== unmatched {module_type_name}")


def fp8_prepare_model(model: torch.nn.Module, new_state_dict):
    module_type_name = type(model).__name__
    qdtype = torch.float8_e4m3fn
    print(f"Prepare model: {module_type_name} to {qdtype}")
    from bizyairenhancer import fp8_quantize_ops

    if module_type_name == "Flux":
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue

            bias_key = f"{name}.bias"
            if bias_key in new_state_dict:
                module.bias = torch.nn.Parameter(
                    torch.empty(
                        new_state_dict.get(bias_key).shape, dtype=torch.bfloat16
                    )
                )
            if not (
                name.startswith("double_blocks") or name.startswith("single_blocks")
            ):
                module.weight.data = module.weight.data.to(torch.bfloat16)
            else:
                scale_key = f"{name}.scale"
                if scale_key in new_state_dict:
                    module.scale = torch.nn.Parameter(new_state_dict[scale_key])
    elif module_type_name == "T5":
        # import pdb;pdb.set_trace()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                dtype = module.weight.dtype
                device = module.weight.device
                fp8linear = fp8_quantize_ops.Linear(
                    in_features, out_features, bias=False, dtype=dtype, device=device
                )

                parent_module = model
                module_names = name.split(".")
                for sub_name in module_names[:-1]:
                    parent_module = getattr(parent_module, sub_name)
                setattr(parent_module, module_names[-1], fp8linear)
                del module

                scale_key = f"t5xxl.transformer.{name}.scale"
                if scale_key in new_state_dict:
                    fp8linear.scale = torch.nn.Parameter(new_state_dict[scale_key])
                else:
                    print(f"not found: {scale_key}")
            else:
                module = module.to(dtype=torch.float8_e4m3fn)

        # import pdb;pdb.set_trace()
        torch.cuda.empty_cache()
        gc.collect()


def fp8_dequantize(qtensor, scale, device):
    scale = scale.to(device)
    qtensor = qtensor.to(torch.float16)
    qtensor.mul_(scale)

    return qtensor
