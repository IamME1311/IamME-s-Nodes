from xml.dom import ValidationErr
import comfy
import math
import torch
import json
import os
import random
import numpy as np

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


#Helper functions
def json_loader(file_name:str) -> dict:
    cwd_name = os.path.dirname(__file__)
    path_to_asset_file = os.path.join(cwd_name, f"assets/{file_name}.json")
    with open(path_to_asset_file, "r") as f:
        asset_data = json.load(f)
    return asset_data

def apply_attention(text:str, weight:float) -> str:
    weight = float(np.round(weight, 2))
    return f"({text}:{weight})"


random_opt = "Randomize 🎲"
option_dict = json_loader("FacePromptMaker")

class FacePromptMaker:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "opt_append_this" : ("STRING", {"forceInput": True}),
            },
            "required" : {

                "seed" : ("INT", {"forceInput": True}),
                "Activate" : ("BOOLEAN", {"default": True}),
                "Gender" : ([random_opt] + option_dict["gender"], {"default": random_opt}),
                "Age" : ([random_opt] + [str(age) for age in range(20, 31)], {"default": random_opt}),
                "nationality_to_mix" : (["None"] + [random_opt] + option_dict["nationality_to_mix"], {"default": "None"}),
                "body_type" : (["None"] + [random_opt] + option_dict["body_type"], {"default": "None"}),
                "body_type_weight" : ("FLOAT", {"default": 1, "min":0, "max":2, "step":0.01, "display":"slider",}),
                "Skin_Tone" : (["None"] + [random_opt] + option_dict["Skin_Tone"], {"default": "None"}), 
                "Face_Shape" : (["None"] + [random_opt] + option_dict["Face_Shape"], {"default": "None"}),
                "Forehead" : (["None"] + [random_opt] + option_dict["Forehead"], {"default": "None"}),
                "Hair_Color" : (["None"] + [random_opt] + option_dict["hair_color"], {"default": "None"}),
                "Hair_Style" : (["None"] + [random_opt] + option_dict["hair_style"], {"default": "None"}),
                "Hair_Length" : (["None"] + [random_opt] + option_dict["Hair_Length"], {"default": "None"}),
                "General_weight" : ("FLOAT", {"default": 1.05, "min":0.05, "max":2, "step":0.01, "display":"slider",}),
                "Eye_Color" : (["None"] + [random_opt] + option_dict["Eye_Color"], {"default": "None"}),
                "Eye_Shape" : (["None"] + [random_opt] + option_dict["Eye_Shape"], {"default": "None"}),
                "Eyebrows" : (["None"] +[random_opt] + option_dict["Eyebrows"], {"default": "None"}),
                "Nose_Shape" : (["None"] + [random_opt] + option_dict["Nose_Shape"], {"default": "None"}),
                "Lip_Color" : (["None"] + [random_opt] + option_dict["Lip_Color"], {"default": "None"}),
                "Expression" : (["None"] + [random_opt] + option_dict["Expression"], {"default": "None"}),
                "Facial_Hair" : (["None"] + [random_opt] + option_dict["Facial_Hair"], {"default": "None"}),
                "Cheekbones" : (["None"] + [random_opt] + option_dict["Cheekbones"], {"default": "None"}),
                "Chin_Shape" : (["None"] + [random_opt] + option_dict["Chin_Shape"], {"default": "None"}),
                "beard" : (["None"] + [random_opt] + option_dict["beard"], {"default": "None"}),
                "beard_color" : (["None"] + [random_opt] + option_dict["beard_color"], {"default": "None"}),

            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Prompt",)
    CATEGORY = "IamME"
    FUNCTION = "PromptMaker"

    def PromptMaker(self, seed, 
                    Activate, 
                    Gender, 
                    Age, 
                    nationality_to_mix, 
                    body_type, 
                    body_type_weight, 
                    Skin_Tone,
                    Face_Shape,
                    Forehead, 
                    Hair_Color, 
                    Hair_Style,
                    Hair_Length,
                    General_weight,
                    Eye_Color,
                    Eye_Shape,
                    Eyebrows,
                    Nose_Shape,
                    Lip_Color,
                    Expression,
                    Facial_Hair,
                    Cheekbones,
                    Chin_Shape,
                    beard,
                    beard_color, 
                    opt_append_this=""
            ):


        if Activate==True:
            prompt_list = []

            if Gender==random_opt:
                Gender=random.choice(option_dict["gender"])
            
            if body_type!="None":
                if body_type==random_opt:
                    body_type = random.choice(option_dict["body_type_prompt"])
                else:
                    index = option_dict["body_type"].index(body_type)
                    body_type = option_dict["body_type_prompt"][index]
                body_type = Gender + " " + body_type
                prompt_list.append(apply_attention(body_type, body_type_weight))

            if Age==random_opt:
                Age=random.choice([str(age) for age in range(20, 31)])
            Age = Age +"-year old"
            prompt_list.append(apply_attention(Age, General_weight))

            if nationality_to_mix==random_opt:                
                nationality_to_mix=random.choice(option_dict["nationality_to_mix"])
            elif nationality_to_mix=="None":
                nationality_to_mix=""
            nationality = f"[Indian:{nationality_to_mix}]"
            prompt_list.append(apply_attention(nationality, General_weight))

            

            if Skin_Tone!="None":
                if Skin_Tone==random_opt:
                    Skin_Tone = random.choice(option_dict["Skin_Tone"])

                prompt_list.append(apply_attention(Skin_Tone, General_weight))

            if Face_Shape!="None":
                if Face_Shape==random_opt:
                    Face_Shape = random.choice(option_dict["Face_Shape"])

                prompt_list.append(apply_attention(Face_Shape, General_weight))

            if Forehead!="None":
                if Forehead==random_opt:
                    Forehead = random.choice(option_dict["Forehead"])

                prompt_list.append(apply_attention(Forehead, General_weight))

            #hair
            if Hair_Color!="None":
                if Hair_Color==random_opt:
                    Hair_Color = random.choice(option_dict["Hair_Color"])

            if Hair_Style!="None":
                if Hair_Style==random_opt:
                    Hair_Style = random.choice(option_dict["Hair_Style"])

            if Hair_Length!="None":
                if Hair_Length==random_opt:
                    Hair_Length = random.choice(option_dict["Hair_Length"])

            if Hair_Length!="None" and Hair_Style!="None" and Hair_Color!="None":
                hair = f"{Hair_Color} {Hair_Style} {Hair_Length} hair"  
                prompt_list.append(apply_attention(hair, General_weight))
            
            #eyes
            if Eye_Color!="None":
                if Eye_Color==random_opt:
                    Eye_Color = random.choice(option_dict["Eye_Color"])

            if Eye_Shape!="None":
                if Eye_Shape==random_opt:
                    Eye_Shape = random.choice(option_dict["Eye_Shape"])

            if Eyebrows!="None":
                if Eyebrows==random_opt:
                    Eyebrows = random.choice(option_dict["Eyebrows"])

            if Eye_Color!="None" and Eye_Shape!="None" and Eyebrows!="None":
                eyes = f"{Eye_Color} {Eye_Shape} eyes with {Eyebrows} eyebrows"  
                prompt_list.append(apply_attention(eyes, General_weight))


            if Nose_Shape!="None":
                if Nose_Shape==random_opt:
                    Nose_Shape = random.choice(option_dict["Nose_Shape"])

            if Lip_Color!="None":
                if Lip_Color==random_opt:
                    Lip_Color = random.choice(option_dict["Lip_Color"])

            if Nose_Shape!="None" and Lip_Color!="None":
                nose_lip = f"{Nose_Shape} nose with {Lip_Color} colored lips"  
                prompt_list.append(apply_attention(nose_lip, General_weight))

            
            if Expression!="None":
                if Expression==random_opt:
                    Expression = random.choice(option_dict["Expression"])

                prompt_list.append(apply_attention(Expression, General_weight))
            
            #jaw structure
            if Cheekbones!="None":
                if Cheekbones==random_opt:
                    Cheekbones = random.choice(option_dict["Cheekbones"])

            if Chin_Shape!="None":
                if Chin_Shape==random_opt:
                    Chin_Shape = random.choice(option_dict["Chin_Shape"])

            if Chin_Shape!="None" and Cheekbones!="None":
                jaw = f"{Chin_Shape} chin with {Cheekbones} cheekbones"  
                prompt_list.append(apply_attention(jaw, General_weight))

            #facial hair
            if Gender=="Male":
                if Facial_Hair!="None":
                    if Facial_Hair==random_opt:
                        Facial_Hair = random.choice(option_dict["Facial_Hair"])

                if beard!="None":
                    if beard==random_opt:
                        beard = random.choice(option_dict["beard"])

                if beard_color!="None":
                    if beard_color==random_opt:
                        beard_color = random.choice(option_dict["beard_color"])

                if Facial_Hair!="None" and beard!="None" and beard_color!="None":
                    facial_hair = f"{Facial_Hair} facial hair, {beard_color} {beard} beard"  
                    prompt_list.append(apply_attention(facial_hair, General_weight))


            if len(prompt_list) > 0:
                if "__faceprompt__" not in opt_append_this:
                    raise ValidationErr("trigger word __faceprompt__ not found!!")
                else:
                    prompt = ", ".join(prompt_list)
                    prompt = prompt.lower()
                    prompt = opt_append_this.replace("__faceprompt__", prompt)
                    return (prompt, )
            else:
                return("",)
        else:
            return(opt_append_this,)
        

class TriggerWordProcessor:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"forceInput": True}),
                "text_in": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = 'TextProcessor'
    CATEGORY = 'IamME'
    OUTPUT_NODE = True

    def TextProcessor(self, seed, text_in):
        options = json_loader("TriggerWords")
        bg_options = options["background"]

        #background
        if "__background__" not in text_in:
            raise ValidationErr("trigger word __background__ not found!!")
        else:
            bg_choice = random.choice(bg_options)
            text_out = text_in.replace("__background__", bg_choice)
            text = [text_out]

        return {"ui": {"text": text}, "result": (text_out,)}









NODE_CLASS_MAPPINGS = {
    "AspectEmptyLatentImage" : AspectEmptyLatentImage,
    "LiveTextEditor": LiveTextEditor,
    "FacePromptMaker" : FacePromptMaker,
    "TriggerWordProcessor" : TriggerWordProcessor   
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "AspectEmptyLatentImage" : "AspectEmptyLatentImage",
    "LiveTextEditor" : "LiveTextEditor",
    "FacePromptMaker": "FacePromptMaker",
    "TriggerWordProcessor" : "TriggerWordProcessor"    
}