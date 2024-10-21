import comfy
import math
import random
import google.generativeai as genai
from PIL import ImageOps
import yaml
from pathlib import Path
import math
import numpy as np
from .utils import (
    MAX_RESOLUTION, ASPECT_CHOICES, IMAGE_DATA, BUS_DATA,
    random_opt, option_dict, any_type,
    json_loader,
    apply_attention,
    tensor_to_image,
    image_to_tensor,
    parser,
    torch, Image
)




class AspectEmptyLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "width": ("INT", {"default":1024, "min":16, "max":MAX_RESOLUTION, "step":8}),
                "height": ("INT", {"default":1024, "min":16, "max":MAX_RESOLUTION, "step":8}),
                "aspect_ratio": (ASPECT_CHOICES, {"default": "None"}),
                "model_type":(["SD1.5", "SDXL"],),
                "aspect_width_override" : ("INT",{"default":0, "min":0}),
                "aspect_height_override" : ("INT",{"default":0, "min":0}),
                "width_override": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            }
        }
    RETURN_TYPES = ("LATENT", "INT", "INT", IMAGE_DATA["type"])
    RETURN_NAMES = ("samples","width","height", IMAGE_DATA["name"])
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)
    FUNCTION = "aspect_latent_gen"

    CATEGORY = "IamME"
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."

    def aspect_latent_gen(self, width, height, model_type, aspect_ratio, aspect_width_override, aspect_height_override, width_override, batch_size=1):
        if aspect_ratio!="None":
            if aspect_ratio in ASPECT_CHOICES[2:]:
                aspect_width_override, aspect_height_override = parser(aspect_ratio)
            if width_override > 0:
                width = width_override - (width_override%8)
                height = int((aspect_height_override * width) / aspect_width_override)
                height = height - (height%8)
            else:
                total_pixels = {
                    "SD1.5" : 512 * 512,
                    "SDXL" : 1024 * 1024
                }
                pixels = total_pixels.get(model_type, 0)

                aspect_ratio_value = aspect_width_override / aspect_height_override

                width = int(math.sqrt(pixels * aspect_ratio_value))
                height = int(pixels / width)


        else: # normal empty latent
            width = int(width - (width%8))
            height = int(height - (height%8))
    
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        image_data = {"width":width, "height":height}
        return {"ui":{
                    "text": [f"{height}x{width}"]
                }, 
                "result":
                    ({"samples":latent}, width, height, image_data)
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

        return {"ui": {"text": text}, "result": (out_text,)}


class TextTransformer:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput":True}),
                "prepend" : ("STRING", {"multiline":True, "default":""}), 
                "append" : ("STRING", {"multiline":True, "default":""}),
                "replace" : ("BOOLEAN", {"default" : False})
            },
            "optional": {
                "to_replace" : ("STRING", {"default":""}),
                "replace_with" : ("STRING", {"default":""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = 'texttransformer'
    CATEGORY = 'IamME'

    def texttransformer(self, text, replace, prepend="", append="", to_replace="", replace_with=""):
        text = prepend + " " + text
        text = text + " " + append

        if replace:
            text = text.replace(to_replace, replace_with)

        
        return (text,)


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
                "body_type" : (["None"] + [random_opt] + option_dict["body_type"], {"default": random_opt}),
                "body_type_weight" : ("FLOAT", {"default": 1, "min":0, "max":2, "step":0.01, "display":"slider",}),
                "Skin_Tone" : (["None"] + [random_opt] + option_dict["Skin_Tone"], {"default": random_opt}), 
                "Face_Shape" : (["None"] + [random_opt] + option_dict["Face_Shape"], {"default": random_opt}),
                "Forehead" : (["None"] + [random_opt] + option_dict["Forehead"], {"default": random_opt}),
                "Hair_Color" : (["None"] + [random_opt] + option_dict["Hair_Color"], {"default": random_opt}),
                "Hair_Style" : (["None"] + [random_opt] + option_dict["Hair_Style"], {"default": random_opt}),
                "Hair_Length" : (["None"] + [random_opt] + option_dict["Hair_Length"], {"default": random_opt}),
                "General_weight" : ("FLOAT", {"default": 1.05, "min":0.05, "max":2, "step":0.01, "display":"slider",}),
                "Eye_Color" : (["None"] + [random_opt] + option_dict["Eye_Color"], {"default": random_opt}),
                "Eye_Shape" : (["None"] + [random_opt] + option_dict["Eye_Shape"], {"default": random_opt}),
                "Eyebrows" : (["None"] +[random_opt] + option_dict["Eyebrows"], {"default": random_opt}),
                "Nose_Shape" : (["None"] + [random_opt] + option_dict["Nose_Shape"], {"default": random_opt}),
                "Lip_Color" : (["None"] + [random_opt] + option_dict["Lip_Color"], {"default": random_opt}),
                "Expression" : (["None"] + [random_opt] + option_dict["Expression"], {"default": random_opt}),
                "Facial_Hair" : (["None"] + [random_opt] + option_dict["Facial_Hair"], {"default": "None"}),
                "Cheekbones" : (["None"] + [random_opt] + option_dict["Cheekbones"], {"default": random_opt}),
                "Chin_Shape" : (["None"] + [random_opt] + option_dict["Chin_Shape"], {"default": random_opt}),
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
            prompt_list_final = []
            prompt_list = []

            #gender, age and body type
            if Gender==random_opt:
                Gender=random.choice(option_dict["gender"])
            
            if Age==random_opt:
                Age=random.choice([str(age) for age in range(20, 31)])
            Age = Age +"-year old "

            if body_type!="None":
                if body_type==random_opt:
                    body_type = random.choice(option_dict["body_type_prompt"])
                else:
                    index = option_dict["body_type"].index(body_type)
                    body_type = option_dict["body_type_prompt"][index]
                    if Gender=="Male":
                        body_type = body_type.replace("beautiful", "handsome").replace("petite", "lean")
                body_type = Age + Gender + ", " + body_type
                prompt_list_final.append(apply_attention(body_type, body_type_weight))

            if nationality_to_mix==random_opt:                
                nationality_to_mix=random.choice(option_dict["nationality_to_mix"])
            elif nationality_to_mix=="None":
                nationality_to_mix=""
            nationality = f"[Indian:{nationality_to_mix}]"
            prompt_list.append(nationality)

            

            if Skin_Tone!="None":
                if Skin_Tone==random_opt:
                    Skin_Tone = random.choice(option_dict["Skin_Tone"])
                    Skin_Tone = f"{Skin_Tone} skin"

                prompt_list.append(Skin_Tone)

            if Face_Shape!="None":
                if Face_Shape==random_opt:
                    Face_Shape = random.choice(option_dict["Face_Shape"])
                    Face_Shape = f"{Face_Shape} shaped face"

                prompt_list.append(Face_Shape)

            if Forehead!="None":
                if Forehead==random_opt:
                    Forehead = random.choice(option_dict["Forehead"])
                    Forehead = f"{Forehead} forehead,"

                prompt_list.append(Forehead)

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
                hair = f"{Hair_Color} {Hair_Length} {Hair_Style} hair"  
                prompt_list.append(hair)
            
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
                prompt_list.append(eyes)


            if Nose_Shape!="None":
                if Nose_Shape==random_opt:
                    Nose_Shape = random.choice(option_dict["Nose_Shape"])

            if Lip_Color!="None":
                if Lip_Color==random_opt:
                    Lip_Color = random.choice(option_dict["Lip_Color"])

            if Nose_Shape!="None" and Lip_Color!="None":
                nose_lip = f"{Nose_Shape} nose with {Lip_Color} colored lips"  
                prompt_list.append(nose_lip)

            
            if Expression!="None":
                if Expression==random_opt:
                    Expression = random.choice(option_dict["Expression"])
                    Expression = f"{Expression} expression"

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
                prompt_list.append(jaw)

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
                    prompt_list.append(facial_hair)

            if len(prompt_list_final) > 0:
                prompt = ", ".join(prompt_list)
                prompt = prompt_list_final[0] + ", " + apply_attention(prompt, General_weight)

            if prompt:
                if not opt_append_this:
                    prompt = prompt.lower()
                    return (prompt, )
                
                elif "__faceprompt__" not in opt_append_this:
                    raise ValueError("trigger word __faceprompt__ not found!!")
                else:
                    prompt = prompt.lower()
                    prompt = opt_append_this.replace("__faceprompt__", prompt)
                    return (prompt, )
            else: # if all options are none, return blank string
                return(opt_append_this,)
        else: #if activate is false no need to do anything..
            return(opt_append_this,)
        

class TriggerWordProcessor:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_in": ("STRING", {"forceInput": True}),
                "gender" : (["male", "female"], {"default":"female"}),
                "seed": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = 'TextProcessor'
    CATEGORY = 'IamME'
    OUTPUT_NODE = True

    def TextProcessor(self, seed, text_in, gender):
        options = json_loader("TriggerWords")
        bg_options = options["background"]
        pose_options = options["pose"]
        whimsical_pose_options = options["whimsical_pose"]
        color_options = options["color"]
        topwear_options = options["topwear"]
        bottomwear_options = options["bottomwear"]
        region_options = options["regionality"]
        body_type_options = options["body_type"]

        trigger_words = ["__background__", "__topwear__", "__bottomwear__", "__pose__", "__region__", "__bodytype__"]

        for word in trigger_words:
            if word not in text_in:
                print(f"{word} not found in input text, skipping!!")
                continue
            
            #background
            if word == "__background__":
                bg_choice = random.choice(bg_options)
                text_in = text_in.replace(word, bg_choice)

            #topwear
            elif word == "__topwear__":          
                if gender=="male":
                    topwear_choice = random.choice(topwear_options["male"])
                else:
                    topwear_choice = random.choice(topwear_options["female"])
                color_choice = random.choice(color_options)
                clothing = f"{color_choice} {topwear_choice} "
                text_in = text_in.replace(word, clothing)
            
            #bottomwear
            elif word == "__bottomwear__":          
                if gender=="male":
                    bottomwear_choice = random.choice(bottomwear_options["male"])
                else:
                    bottomwear_choice = random.choice(bottomwear_options["female"])
                color_choice = random.choice(color_options)
                clothing = f"{color_choice} {bottomwear_choice} "
                text_in = text_in.replace(word, clothing)
            
            #pose
            elif word == "__pose__":
                pose_choice = random.choice(pose_options)
                text_in = text_in.replace(word, pose_choice)
            
            #pose
            elif word == "__whimsical_pose__":
                whimsical_pose_choice = random.choice(whimsical_pose_options)
                text_in = text_in.replace(word, whimsical_pose_choice)
            
            #region
            elif word == "__region__":
                region_choice = random.choice(region_options)
                text_in = text_in.replace(word, region_choice)

            elif word == "__bodytype__":
                if gender=="male":
                    body_type_choice = random.choice(body_type_options["male"])
                else:
                    body_type_choice = random.choice(body_type_options["female"])
                body_type = f"{body_type_choice} "
                text_in = text_in.replace(word, body_type)
            
        text = [text_in]

        return {"ui": {"text": text}, "result": (text_in,)}


class GeminiVision:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "image" : ("IMAGE",),
                "seed" : ("INT", {"forceInput":True}),
                "randomness" : ("FLOAT", {"default":0.7, "min":0, "max": 1, "step":0.1, "display":"slider"}),
                "output_tokens" : ("INT", {"min":0, "max":500, "step":1}),
                "prompt" : ("STRING", {"default":"Describe the image", "multiline":True})
            }
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = 'IamME'
    FUNCTION = 'gen_gemini'

    def gen_gemini(self, image, seed, randomness, prompt, output_tokens=None):
        cwd = Path(__file__).parent

        with open(f"{cwd}\config.yaml", "r") as f:
            config = yaml.safe_load(f)
        pil_image = tensor_to_image(image)
        genai.configure(api_key=config["GEMINI_API_KEY"])
        llm = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=genai.GenerationConfig(temperature=randomness, max_output_tokens=output_tokens,))
        response = llm.generate_content([prompt, pil_image])
        return (response.text,)


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
    FUNCTION = "ImageLoader"
    CATEGORY = 'IamME'

    def ImageLoader(self, folder_path, images_to_load, mode):
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
                i = image_to_tensor(i)
                if len(images)>0:
                    if images[0].shape[1:] != i.shape[1:]:
                        i = comfy.utils.common_upscale(i.movedim(-1,1), images[0].shape[2], images[0].shape[1], "bilinear", "center").movedim(1,-1) # rescale image to fit the tensor array
                images.append(i)
            file_name = str(file_list)
        else:
            file_name = image_paths[self.cur_index].name
            i = Image.open(image_paths[self.cur_index])
            i = ImageOps.exif_transpose(i)
            i = image_to_tensor(i)
            images.append(i)
            self.cur_index += 1
            if self.cur_index >= len(image_paths):
                self.cur_index = 0
                

        return {"ui": {"text":display_text},"result":(torch.cat(images, dim=0), file_name,)}
    
    @classmethod
    def IS_CHANGED(s, **kwargs):
        if kwargs["mode"] == "incremental":
            return float("NaN")


class GetImageData:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "Image" : ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", IMAGE_DATA["type"])
    RETURN_NAMES = ("Image", "Width", "Height", "Aspect Ratio", IMAGE_DATA["name"])
    FUNCTION = "getData"
    CATEGORY = "IamME"

    def getData(self, Image:torch.tensor):
        width = Image.shape[2]
        height = Image.shape[1]
        aspect_ratio_str = f"{int(width / math.gcd(width, height))}:{int(height / math.gcd(width, height))}"

        if width > height:
            orientation = "Landscape"
        elif height > width:
            orientation = "Portrait"
        else:
            orientation = "Square"
        image_data = {"width":width, "height":height, "aspect_ratio_str":aspect_ratio_str, "orientation":orientation}
        return {
            "ui" : {
                "text" : [f"{aspect_ratio_str}"],
            },
            "result" : 
            (Image, width, height, aspect_ratio_str, image_data),
        }


class ConnectionBus:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {},
            "optional" : {
                "bus" : (BUS_DATA["type"],),
                "value_1" : (any_type,),
                "value_2" : (any_type,),
                "value_3" : (any_type,),
                "value_4" : (any_type,),
                "value_5" : (any_type,),
                "value_6" : (any_type,),
                "value_7" : (any_type,),
                "value_8" : (any_type,),
                "value_9" : (any_type,),
                "value_10" : (any_type,),
            }
        }

    RETURN_TYPES = (BUS_DATA["type"], any_type, any_type, any_type, any_type, any_type, any_type, any_type, any_type, any_type, any_type,)
    RETURN_NAMES = (BUS_DATA["name"], "value_1", "value_2", "value_3", "value_4", "value_5", "value_6", "value_7", "value_8", "value_9", "value_10",)
    CATEGORY = "IamME"
    FUNCTION = "HandleBus"
# value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8, value_9, value_10
    def HandleBus(self, bus:list=None, value_1=None, value_2=None, value_3=None, value_4=None, value_5=None, value_6=None, value_7=None, value_8=None, value_9=None, value_10=None):
        
        #Initializing original values
        NoneList = [None, None, None, None, None, None, None, None, None, None]
        org_value_1, org_value_2, org_value_3, org_value_4, org_value_5, org_value_6, org_value_7, org_value_8, org_value_9, org_value_10 = NoneList

        # putting
        if bus is not None:
            org_value_1, org_value_2, org_value_3, org_value_4, org_value_5, org_value_6, org_value_7, org_value_8, org_value_9, org_value_10 = bus
        
        new_bus = []

        for i in range(1,11):
            exec(f"new_bus.append((value_{i} if value_{i} is not None else org_value_{i}))")

        

        return (new_bus, *new_bus)

class AspectRatioCalculator:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "width" : ("FLOAT",),
                "height" : ("FLOAT",),
                "aspect_width" : ("INT",),
                "aspect_height" : ("INT",),
            }
        }
    

class SaveImageAdvanced:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "image" : ("IMAGE",),
                "parent_folder" : ("STRING", {"default":""}),
                "subfolder_name" : ("STRING",{"default":""}),
                "file_name" : ("STRING", {"default":""}),
                "extension" : (["png", "jpg", "jpeg"])
            }
        }
    
    RETURN_TYPES = ()
    CATEGORY = "IamME"
    FUNCTION = "save_image"

    def save_image(self, image:torch.tensor, parent_folder, subfolder_name, file_name):
        img = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))

        parent_path = Path(parent_folder)
        subfolder_path = parent_path.joinpath(subfolder_name)

        subfolder_path.mkdir(exist_ok=True)

        img.save(subfolder_path.joinpath(file_name), format="")
        return True

NODE_CLASS_MAPPINGS = {
    "AspectEmptyLatentImage" : AspectEmptyLatentImage,
    "LiveTextEditor": LiveTextEditor,
    "FacePromptMaker" : FacePromptMaker,
    "TriggerWordProcessor" : TriggerWordProcessor,
    "BasicTextEditor" : TextTransformer,
    "GeminiVision": GeminiVision,
    "ImageBatchLoader":ImageBatchLoader,
    "GetImageData":GetImageData,
    "ConnectionBus":ConnectionBus
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "AspectEmptyLatentImage" : "AspectEmptyLatentImage",
    "LiveTextEditor" : "LiveTextEditor",
    "FacePromptMaker": "FacePromptMaker",
    "TriggerWordProcessor" : "TriggerWordProcessor",
    "BasicTextEditor" : "BasicTextEditor",
    "GeminiVision":"GeminiVision",
    "ImageBatchLoader":"ImageBatchLoader",
    "GetImageData":"GetImageData",
    "ConnectionBus":"ConnectionBus"
}