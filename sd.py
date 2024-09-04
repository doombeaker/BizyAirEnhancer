import logging

import torch

import comfy
import comfy.model_detection as model_detection
import comfy.model_management as model_management
from comfy.sd import CLIP
import folder_paths

from bizyairenhancer import fp8_quantize_model, fp8_prepare_model


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
        fp8_quantize_model(model.diffusion_model, new_sd)
    else:
        fp8_prepare_model(model.diffusion_model, new_sd)
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


from comfy.sd import CLIPType


def load_clip(
    ckpt_paths, embedding_directory=None, clip_type=CLIPType.FLUX, model_options={}
):
    clip_data = []
    for p in ckpt_paths:
        clip_data.append(comfy.utils.load_torch_file(p, safe_load=True))
    return load_text_encoder_state_dicts(
        clip_data,
        embedding_directory=embedding_directory,
        clip_type=clip_type,
        model_options=model_options,
    )


def load_text_encoder_state_dicts(
    state_dicts=[],
    embedding_directory=None,
    clip_type=CLIPType.STABLE_DIFFUSION,
    model_options={},
):
    clip_data = state_dicts

    class EmptyClass:
        pass

    for i in range(len(clip_data)):
        if "transformer.resblocks.0.ln_1.weight" in clip_data[i]:
            clip_data[i] = comfy.utils.clip_text_transformers_convert(
                clip_data[i], "", ""
            )
        else:
            if "text_projection" in clip_data[i]:
                clip_data[i]["text_projection.weight"] = clip_data[i][
                    "text_projection"
                ].transpose(
                    0, 1
                )  # old models saved with the CLIPSave node

    clip_target = EmptyClass()
    clip_target.params = {}

    weight_name = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
    weight = clip_data[0].get(weight_name, clip_data[1].get(weight_name, None))

    # use fp8 for T5
    dtype_t5 = torch.float16
    clip_target.clip = comfy.text_encoders.flux.flux_clip(dtype_t5=dtype_t5)
    clip_target.tokenizer = comfy.text_encoders.flux.FluxTokenizer

    parameters = 0
    for c in clip_data:
        parameters += comfy.utils.calculate_parameters(c)

    clip = CLIP(
        clip_target,
        embedding_directory=embedding_directory,
        parameters=parameters,
        model_options=model_options,
    )
    for c in clip_data:
        if not "text_model.encoder.layers.1.mlp.fc1.weight" in c:
            if model_options["is_online"]:
                fp8_quantize_model(clip.cond_stage_model.t5xxl.transformer, c)
            else:
                fp8_prepare_model(clip.cond_stage_model.t5xxl.transformer, c)
                c = {
                    key.replace("t5xxl.transformer.", ""): value
                    for key, value in c.items()
                }
        m, u = clip.load_sd(c)
        if len(m) > 0:
            logging.warning("clip missing: {}".format(m))

        if len(u) > 0:
            logging.debug("clip unexpected: {}".format(u))
    return clip
