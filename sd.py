import torch

import comfy
import comfy.model_detection as model_detection
import comfy.model_management as model_management
import folder_paths


def load_diffusion_model_state_dict(
    sd, model_options={}, is_online_quantize=True
):  # load unet in diffusers or regular format
    dtype = model_options.get("dtype", None)

    # Allow loading unets from checkpoint files
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(
        sd, {diffusion_model_prefix: ""}, filter_keys=True
    )
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = comfy.utils.calculate_parameters(sd)
    load_device = model_management.get_torch_device()
    model_config = model_detection.model_config_from_unet(sd, "")

    if model_config is not None:
        new_sd = sd
    else:
        new_sd = model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None:  # diffusers mmdit
            model_config = model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                return None
        else:  # diffusers unet
            model_config = model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                return None

            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)

            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
                else:
                    logging.warning("{} {}".format(diffusers_keys[k], k))

    offload_device = model_management.unet_offload_device()
    if dtype is None:
        unet_dtype = model_management.unet_dtype(
            model_params=parameters,
            supported_dtypes=model_config.supported_inference_dtypes,
        )
    else:
        unet_dtype = dtype

    manual_cast_dtype = model_management.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", None)
    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    if is_online_quantize:
        from bizyairenhancer import fp8_quantize_model

        fp8_quantize_model(model.diffusion_model, new_sd)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in unet: {}".format(left_over))
    return comfy.model_patcher.ModelPatcher(
        model, load_device=load_device, offload_device=offload_device
    )


def load_diffusion_model(unet_path, model_options={}, is_online_quantize=True):
    sd = comfy.utils.load_torch_file(unet_path)
    model = load_diffusion_model_state_dict(
        sd, model_options=model_options, is_online_quantize=is_online_quantize
    )
    if model is None:
        logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
        raise RuntimeError(
            "ERROR: Could not detect model type of: {}".format(unet_path)
        )
    return model
