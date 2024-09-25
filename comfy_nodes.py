import comfy
import math

MAX_RESOLUTION = 16384

class AspectEmptyLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "width": ("INT", {"default":1024, "min":16, "max":MAX_RESOLUTION, "step":8}),
                "height": ("INT", {"default":1024, "min":16, "max":MAX_RESOLUTION, "step":8}),
                "aspect_control" : ("BOOLEAN", {"default":False, "tooltip":"Set to 'True' if you want to select size based on a particular aspect ratio."}),
                "model_type":(["SD1.5", "SDXL"],),
                "aspect_width_ratio" : ("INT",{"default":9, "min":0}),
                "aspect_height_ratio" : ("INT",{"default":16, "min":0}),
                "width_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            }
        }
    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("samples","width","height",)
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)
    FUNCTION = "aspect_latent_gen"

    CATEGORY = "IamME"
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."

    def aspect_latent_gen(self, width, height, model_type, aspect_control, aspect_width_ratio, aspect_height_ratio, width_override, batch_size=1):
        if aspect_control==True:
            if width_override > 0:
                width = width_override - (width_override%8)
                height = int((aspect_height_ratio * width) / aspect_width_ratio)
                height = height - (height%8)
            else:
                total_pixels = {
                    "SD1.5" : 512 * 512,
                    "SDXL" : 1024 * 1024
                }
                pixels = total_pixels.get(model_type, 0)

                aspect_ratio_value = aspect_width_ratio / aspect_height_ratio

                width = int(math.sqrt(pixels * aspect_ratio_value))
                height = int(pixels / width)

        else:
            width = int(width - (width%8))
            height = int(height - (height%8))
    
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return {"ui":{
                    "text": [f"{height}x{width}"]
                }, 
                "result":
                    ({"samples":latent}, width, height)
            }


class LiveTextEditor:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                    "text":("STRING", {"forceInput": True}),
                    "modify_text":("STRING", {"multiline":True, "default":""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "TextEditor"
    OUTPUT_NODE = True
    CATEGORY = "IamME"

    def TextEditor(self, text, modify_text="", unique_id=None, extra_pnginfo=None):
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [text]


        if modify_text[0]:
            out_text = modify_text[0]
        else:
            out_text = text[0]
        
        # print("out_text", type(out_text), out_text)
        # encoded_text = LiveTextEditor.TextEncoder(clip, out_text)

        return {"ui": {"text": text}, "result": (out_text,)}

    # def TextEncoder(clip, text_to_encode):
        
    #     print(text_to_encode, " ",type(text_to_encode))
    #     # encoding
    #     tokens = clip.tokenize(text_to_encode)
    #     output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    #     cond = output.pop("cond")
    #     return ([[cond, output]], )


# class ImageLivePreview:

#     @classmethod
#     def INPUT_TYPES(self):
#         return {
#             "required": {
#                 "image": ("IMAGE", ),
#             },
#             "hidden": {
#                 "unique_id": "UNIQUE_ID",
#                 "extra_pnginfo": "EXTRA_PNGINFO",
#             },
#         }

#     RETURN_TYPES = ()
#     OUTPUT_NODE = True
#     FUNCTION = 'blend_preview'
#     CATEGORY = 'IamME'

#     def blend_preview(self, image, unique_id=None, extra_pnginfo=None):
        
#         return {"ui":{"image":image},
#                 "result" : image}















NODE_CLASS_MAPPINGS = {
    "AspectEmptyLatentImage" : AspectEmptyLatentImage,
    "LiveTextEditor": LiveTextEditor, 
    # "ImageLivePreview":ImageLivePreview
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "AspectEmptyLatentImage" : "AspectEmptyLatentImage",
    "LiveTextEditor" : "LiveTextEditor",
    # "ImageLivePreview":"ImageLivePreview"
}