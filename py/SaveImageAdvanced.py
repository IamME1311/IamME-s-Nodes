import torch
from PIL import Image
from .utils import *
class SaveImageAdvanced:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "images" : ("IMAGE",),
                "parent_folder" : ("STRING", {"default":""}),
                "subfolder_name" : ("STRING",{"default":""}),
                "overwrite" : ("BOOLEAN", {"default":False}),
                "format" : (["png", "jpg", "jpeg"], {"default":"jpg"}),
                "quality" : ("INT", {"default":75, "min":0, "max":100}),
                "dpi" : ("INT", {"default":300, "min":1, "max":2400}),
                "file_name_suffix" : ("STRING", {"default":"", "multiline": True})
            }
        }
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("opt",)
    CATEGORY = PACK_NAME
    FUNCTION = "execute"
    OUTPUT_NODE =True

    def execute(self,
                    images:torch.Tensor,
                    parent_folder:str,
                    subfolder_name:str,
                    overwrite:bool,
                    file_name_suffix:str,
                    format:str,
                    quality:int,
                    dpi:int
                    ) -> tuple:
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            parent_path = Path(parent_folder)
            if not parent_path.is_dir():
                raise OSError("Provided parent_path is not a directory!!")
            subfolder_path = parent_path.joinpath(subfolder_name)

            subfolder_path.mkdir(exist_ok=True)
            file_name = f"{subfolder_name}_{file_name_suffix}.{format}"

            save_path = subfolder_path.joinpath(file_name)
            if overwrite:
                if format in ["jpg", "jpeg"]:
                    img.save(save_path,  quality=quality, dpi=(dpi, dpi))
                else: #png case
                    img.save(save_path, dpi=(dpi, dpi))
            else:
                if save_path.exists():
                    raise ValueError("This filename already exsists in the provided folder!!")
                else:
                    if format in ["jpg", "jpeg"]:
                        img.save(save_path,  quality=quality, dpi=(dpi, dpi))
                    else: #png case
                        img.save(save_path, dpi=(dpi, dpi))
        return (file_name,)
    
NODE_CLASS_MAPPINGS = {"SaveImageAdvanced": SaveImageAdvanced,}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveImageAdvanced": PACK_NAME + " SaveImageAdvanced",}