import torch

from .sd import folder_paths, load_diffusion_model
from bizyairenhancer import online_quantize_ops


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
        model_options["custom_operations"] = online_quantize_ops
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
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        from bizyairenhancer import bizyair_enhancer_ctx

        with bizyair_enhancer_ctx():
            model = load_diffusion_model(
                unet_path, model_options=model_options, is_online_quantize=False
            )
            return (model,)


NODE_CLASS_MAPPINGS = {
    "BizyAirFluxLoaderOffLine": BizyAirFluxLoaderOffline,
    "BizyAirFluxLoaderOnline": BizyAirFluxLoaderOnline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BizyAirFluxLoaderOffLine": "BizyAir Flux Loader - Quantize Offline",
    "BizyAirFluxLoaderOnline": "BizyAir Flux Loader - Quantize Online",
}
