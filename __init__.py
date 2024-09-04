import torch

from .sd import folder_paths, load_diffusion_model, load_clip
from bizyairenhancer import fp8_quantize_ops
from .sd import CLIPType


class BizyAirFluxLoaderOnline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (
                    [
                        "default",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "advanced/loaders"

    def load_unet(self, unet_name, weight_dtype):
        model_options = {}
        model_options["custom_operations"] = fp8_quantize_ops
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        model = load_diffusion_model(unet_path, model_options=model_options)
        return (model,)


class BizyAirFluxLoaderOffline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (
                    [
                        "fp8_e4m3fn",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "advanced/loaders"

    def load_unet(self, unet_name, weight_dtype):
        model_options = {}

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        model_options["custom_operations"] = fp8_quantize_ops
        model = load_diffusion_model(
            unet_path, model_options=model_options, is_online_quantize=False
        )
        return (model,)


class QuantizeDualCLIPLoaderOnline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("clip"),),
                "clip_name2": (folder_paths.get_filename_list("clip"),),
                "type": (["flux"],),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        model_options = {}
        model_options["is_online"] = True
        clip = load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=CLIPType.FLUX,
            model_options=model_options,
        )
        return (clip,)


class QuantizeDualCLIPLoaderOffline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("clip"),),
                "clip_name2": (folder_paths.get_filename_list("clip"),),
                "type": (["flux"],),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        model_options = {}
        model_options = {"is_online": False}
        clip = load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=CLIPType.FLUX,
            model_options=model_options,
        )
        return (clip,)


NODE_CLASS_MAPPINGS = {
    "BizyAirFluxLoaderOffLine": BizyAirFluxLoaderOffline,
    "BizyAirFluxLoaderOnline": BizyAirFluxLoaderOnline,
    "QuantizeDualCLIPLoaderOnline": QuantizeDualCLIPLoaderOnline,
    "QuantizeDualCLIPLoaderOffline": QuantizeDualCLIPLoaderOffline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BizyAirFluxLoaderOffLine": "BizyAir Flux Loader - Quantize Offline",
    "BizyAirFluxLoaderOnline": "BizyAir Flux Loader - Quantize Online",
    "QuantizeDualCLIPLoaderOnline": "BizyAir Flux CLIP Loader - Quantize Online",
    "QuantizeDualCLIPLoaderOffline": "BizyAir Flux CLIP Loader - Quantize Offline",
}
