from PIL import Image, ImageOps
import torch
import comfy
from .image_utils import *
from .utils import *


class ImageBatchLoader:
    def __init__(self):
        self.cur_index = 0
        self.folder_path = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "folder_path" : ("STRING", {"default":""}),
                "images_to_load" : ("INT", {"default":0, "min":0}),
                "mode" : (["all", "incremental"], {"default":"all"})
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images","file name")
    FUNCTION = "execute"
    CATEGORY = PACK_NAME

    def execute(self, folder_path, images_to_load, mode) -> torch.Tensor:
        if self.folder_path != folder_path:
            self.folder_path = folder_path
            self.cur_index = 0
        folder_path = Path(folder_path)
        # check path validity
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Path error : {folder_path} is either non-existent or isn't a directory!!")

        image_paths = []
        for path in folder_path.iterdir():
            if path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                image_paths.append(path)

        display_text = []
        display_text.append(f"There are {len(image_paths)} images in this directory.")
        images = []
        file_list = []
        if images_to_load == 0:
            images_to_load = None

        if mode == "all":
            for path in image_paths[:images_to_load]:
                file_list.append(path.name)
                i = Image.open(path)
                i = ImageOps.exif_transpose(i)
                i = pil2tensor(i)
                if len(images)>0:
                    if images[0].shape[1:] != i.shape[1:]:
                        i = comfy.utils.common_upscale(i.movedim(-1,1), images[0].shape[2], images[0].shape[1], "bilinear", "center").movedim(1,-1) # rescale image to fit the tensor array
                images.append(i)
            file_name = str(file_list)
        else:
            file_name = image_paths[self.cur_index].name
            i = Image.open(image_paths[self.cur_index])
            i = ImageOps.exif_transpose(i)
            i = pil2tensor(i)
            images.append(i)
            self.cur_index += 1
            if self.cur_index >= len(image_paths):
                self.cur_index = 0

        return {"ui": {"text":display_text},"result":(torch.cat(images, dim=0), file_name,)}

    @classmethod
    def IS_CHANGED(s, **kwargs):
        if kwargs["mode"] == "incremental":
            return float("NaN")
        
NODE_CLASS_MAPPINGS = {"ImageBatchLoader": ImageBatchLoader,}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageBatchLoader": PACK_NAME + " ImageBatchLoader",}