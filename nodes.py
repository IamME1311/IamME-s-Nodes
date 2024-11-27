#_________server related imports_________
import comfy
from server import PromptServer
from aiohttp import web
import requests

#_________LLM related imports_________
import google.generativeai as genai
from langchain_community.llms.ollama import Ollama

#_________file operations_________
from PIL import ImageOps
from pathlib import Path


#_________custom self imports from this folder_________
from .utils import *
from .image_utils import *

#_________miscellaneous imports_________
from tqdm import tqdm
import re
import time
from datetime import datetime
import math
import random

class IamME_Database:
    def __init__(self) -> None:
        self.main_folder_path = Path(r"\\sridhar\Myntra_Backup\Prompt Database")

        if not self.main_folder_path.exists() and not self.main_folder_path.is_dir(): # if main path doesn't exist, go for temporary approach
            self.temp_folder_path = Path(__file__).cwd().joinpath(".temp")
            self.temp_folder_path.mkdir(exist_ok=True) # make temp folder in current folder

            self.temp_db_path = self.temp_folder_path.joinpath("temp_db.json")
            self.temp_db_path.touch(exist_ok=True) # make temp db file
            self.temp_db_path.write_text("[]")

            self.type = "temp"
        else:
            self.main_db_path = self.main_folder_path.joinpath("database.json")
            self.main_db_path.touch(exist_ok=True) # make main db file if it doesn't exist
            self.type = "main"

    def update_db(self, input_prompt:str, output_prompt:str) -> None:
        """ Update the current database with input_prompt and output_prompt"""
        input_prompt = input_prompt.replace("\"", "")
        if self.type == "main":
            with open(self.main_db_path, "r") as f:
                data:list = json.load(f)
            f.close()
            if not data.__class__.__name__ == "list":
                raise TypeError(f"#{PACK_NAME}'s Nodes : database file is not in correct format")

            with open(self.main_db_path, "w") as f:
                data.append({"datetime":datetime.now().strftime("%B %d, %Y %I:%M %p"), "input_prompt":input_prompt, "output_prompt":output_prompt})
                f.write(json.dumps(data)) # save to file
            f.close()

        elif self.type == "temp":
            with open(self.temp_db_path, "r") as f:
                data:list = json.load(f)
            f.close()
            if not data.__class__.__name__ == "list":
                raise TypeError(f"#{PACK_NAME}'s Nodes : database file is not in correct format")

            with open(self.temp_db_path, "w") as f:
                data.append({"datetime":datetime.now().strftime("%B %d, %Y %I:%M %p"), "input_prompt":input_prompt, "output_prompt":output_prompt})
                f.write(json.dumps(data)) # save to file
            f.close()

        log_to_console(f"{self.type} database updated")

    def delete_temp_db(db_path:Path) -> None:
        """ Delete temporary database"""
        if db_path.exists():
            for item in db_path.parent.iterdir():
                if item.is_file():
                    item.unlink(missing_ok=True)
        
            db_path.parent.rmdir()
            log_to_console("Successfully removed temp database")
        else:
            log_to_console("Temp database not found!!")

    def merge_DB(self) -> None:
        """Merge temp and main database"""
        temp_db_path = Path(__file__).cwd().joinpath(".temp").joinpath("temp_db.json")
        if temp_db_path.exists() and self.main_folder_path.exists():
            log_to_console("merging databases")
            with open(temp_db_path, "r") as f:
                temp_data = json.load(f)
            f.close()
            
            with open(self.main_db_path, "r") as m:
                main_data = json.load(m)
            m.close()
            
            if main_data.__class__.__name__ != "list" or temp_data.__class__.__name__ !="list":
                raise TypeError(f"#{PACK_NAME}'s Nodes : both database datatypes mismatch!!")
            
            main_data = main_data + temp_data

            with open(self.main_db_path, "w") as f:
                f.write(json.dumps(main_data))
            f.close()
            
            log_to_console("Databases merged")
            
            self.delete_temp_db(temp_db_path)
        else:
            log_to_console("No temp DB to merge with, aborting!!")
            
#____________NODES_____________
class AspectEmptyLatentImage:
    def __init__(self) -> None:
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s) -> dict[str:tuple[str, dict]]:
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
    FUNCTION = "execute"

    CATEGORY = PACK_NAME
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."

    def execute(self, 
                    width:int, 
                    height:int, 
                    model_type:str, 
                    aspect_ratio:int, 
                    aspect_width_override:int, 
                    aspect_height_override:int, 
                    width_override:int, 
                    batch_size:int=1
                    ) -> dict:
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


#___________WIP__________
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


class ColorCorrect:

    @classmethod
    def INPUT_TYPES(s) -> dict[str:tuple[str, dict]]:
        return {
            "required" : {
                "images" : ("IMAGE",),
                "gamma": (
                    "FLOAT",
                    {"default": 1, "min": 0.1, "max": 10, "step": 0.01},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 1, "min": 0.0, "max": 3, "step": 0.01},
                ),
                "exposure": (
                    "INT",
                    {"default": 0, "min": -100, "max": 100, "step": 1},
                ),
                "temperature": (
                    "FLOAT", 
                    {"default": 0, "min": -100, "max": 100, "step": 1},
                ),
                "hue": (
                    "INT",
                    {"default": 0, "min": -255, "max": 255, "step": 1},
                ),
                "saturation": (
                    "INT",
                    {"default": 0, "min": -255, "max": 255, "step": 1},
                ),
                "value": (
                    "INT",
                    {"default": 0, "min": -255, "max": 255, "step": 1},
                ),
                "cyan_red": (
                    "FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001}
                ),
                "magenta_green": (
                    "FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001}
                ),
                "yellow_blue": (
                    "FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001}
                ),
            },
            "optional": {
                "mask": ("MASK",)
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    CATEGORY = PACK_NAME
    FUNCTION = "execute"

    def execute(self,
                    images : torch.Tensor,
                    gamma : float=1,
                    contrast : float=1,
                    exposure : int=0,
                    hue : int=0,
                    saturation : int=0,
                    value : int=0,
                    cyan_red : float=0,
                    magenta_green : float=0,
                    yellow_blue : float=0,
                    temperature : float=0,
                    mask : torch.Tensor | None = None
                ) -> tuple[torch.Tensor]:
        

        l_images = []
        l_masks = []
        return_images = []

        for l in images:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))

        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = []
            for m in mask:
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        if len(l_images) != len(l_masks):
            raise ValueError("Number of images and masks is not the same, aborting!!")

        for  index, image in enumerate(images):
            log_to_console(f"index:{index}, no of images:{len(images)}")
            mask = l_masks[index]
            # image = l_images[index]
            # preserving original image for later comparisions
            original_image = tensor2pil(image)

            # apply temperature [takes in tensor][gives out tensor]
            if temperature != 0:
                log_to_console("in temperature") # for debugging
                temperature /= -100
                # result = torch.zeros_like(i)
                
                tensor_image = image.numpy()
                modified_image = Image.fromarray((tensor_image * 255).astype(np.uint8))
                modified_image = np.array(modified_image).astype(np.float32)
                if temperature > 0:
                    modified_image[:, :, 0] *= 1 + temperature
                    modified_image[:, :, 1] *= 1 + temperature * 0.4
                elif temperature < 0:
                    modified_image[:, :, 0] *= 1 + temperature * 0.2
                    modified_image[:, :, 2] *= 1 - temperature

                modified_image = np.clip(modified_image, 0, 255)
                modified_image = modified_image.astype(np.uint8)
                modified_image = modified_image / 255
                image = torch.from_numpy(modified_image).unsqueeze(0)
            
            # apply HSV, [takes in tensor][gives out PIL]
            _h, _s, _v = tensor2pil(image).convert('HSV').split()
            if hue != 0 :
                log_to_console("in hue")
                _h = image_hue_offset(_h, hue)
            if saturation != 0 :
                log_to_console("in saturation")
                _s = image_gray_offset(_s, saturation)
            if value != 0 :
                log_to_console("in value")
                _v = image_gray_offset(_v, value)
            return_image = image_channel_merge((_h, _s, _v), 'HSV')

           # apply color balance [takes in PIL][gives out PIL]
            return_image = color_balance(return_image,
                    [cyan_red, magenta_green, yellow_blue],
                    [cyan_red, magenta_green, yellow_blue],
                    [cyan_red, magenta_green, yellow_blue],
                            shadow_center=0.15,
                            midtone_center=0.5,
                            midtone_max=1,
                            preserve_luminosity=True)

            #apply gamma [takes in tensor][gives out PIL image]
            if (type(return_image) == Image.Image):
                log_to_console("gamma trigger")
                return_image = gamma_trans(return_image, gamma)
            else:
                return_image = gamma_trans(tensor2pil(return_image), gamma)

            #apply contrast [takes and gives out PIL]
            if (contrast != 1):
                log_to_console("in contrast")
                contrast_image = ImageEnhance.Contrast(return_image)
                return_image = contrast_image.enhance(factor=contrast)

            # apply exposure [takes in tensor][gives out PIL]
            if exposure:
                log_to_console("in exposure")
                return_image = pil2tensor(return_image)
                temp = return_image.detach().clone().cpu().numpy().astype(np.float32)
                more = temp[:, :, :, :3] > 0
                temp[:, :, :, :3][more] *= pow(2, exposure / 32)
                if exposure < 0:
                    bp = -exposure / 250
                    scale = 1 / (1 - bp)
                    temp = np.clip((temp - bp) * scale, 0.0, 1.0)
                return_image = tensor2pil(torch.from_numpy(temp))
            

            return_image = chop_image_v2(original_image, return_image, blend_mode="normal", opacity=100)
            return_image.paste(original_image, mask=ImageChops.invert(mask))
            if original_image.mode == 'RGBA':
                return_image = RGB2RGBA(return_image, original_image.split()[-1])

            return_images.append(pil2tensor(return_image))

        return (torch.cat(return_images, dim=0),)


class ConnectionBus:
    def __init__(self) -> None:
        self.default_len = 10

    @classmethod
    def INPUT_TYPES(s) -> dict[str:tuple[str, dict]]:
        return {
            "required" : {},
            "optional" : {
                "bus" : (BUS_DATA["type"],),
                "value_1" : (any_type, {"default":None,}),
                "value_2" : (any_type,{"default":None,}),
                "value_3" : (any_type,{"default":None,}),
                "value_4" : (any_type,{"default":None,}),
                "value_5" : (any_type,{"default":None,}),
                "value_6" : (any_type,{"default":None,}),
                "value_7" : (any_type,{"default":None,}),
                "value_8" : (any_type,{"default":None,}),
                "value_9" : (any_type,{"default":None,}),
                "value_10" : (any_type,{"default":None,}),
            }
        }

    RETURN_TYPES = (BUS_DATA["type"], any_type, any_type, any_type, any_type, any_type, any_type, any_type, any_type, any_type, any_type,)
    RETURN_NAMES = (BUS_DATA["name"], "value_1", "value_2", "value_3", "value_4", "value_5", "value_6", "value_7", "value_8", "value_9", "value_10",)
    CATEGORY = PACK_NAME
    FUNCTION = "execute"
    # value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8, value_9, value_10
    def execute(self, 
                  bus:list=None, 
                  value_1=None, value_2=None, value_3=None, value_4=None, value_5=None, value_6=None, value_7=None, value_8=None, value_9=None, value_10=None, # static inputs
                  **kwargs # for dynamic inputs
                  ) -> dict:
        
        #Initializing original values
        org_values = [None for i in range(self.default_len)]

        # initializing original values with bus values
        if bus is not None:
            org_values = bus
        
        # log_to_console(f"org_values = {org_values}")
        new_bus = []
        message_to_js = []

        # log_to_console(f"IamME's ConnectionBus : No. of arguments passed is, {len(kwargs)}, kwargs are {kwargs}")

        counter = 10

        if len(kwargs) > 0:
            counter+=len(kwargs)

        for  i in range(counter):
            exec(f"new_bus.append(value_{i+1} if value_{i+1} is not None else org_values[i])")
                
        

        return {"ui" : {"message":message_to_js}, "result":(new_bus, *new_bus)}


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
    CATEGORY = PACK_NAME
    FUNCTION = "execute"

    def execute(self, seed:int, 
                    Activate:bool, 
                    Gender:str, 
                    Age:str, 
                    nationality_to_mix:str, 
                    body_type:str, 
                    body_type_weight:float, 
                    Skin_Tone:str,
                    Face_Shape:str,
                    Forehead:str, 
                    Hair_Color:str, 
                    Hair_Style:str,
                    Hair_Length:str,
                    General_weight:float,
                    Eye_Color:str,
                    Eye_Shape:str,
                    Eyebrows:str,
                    Nose_Shape:str,
                    Lip_Color:str,
                    Expression:str,
                    Facial_Hair:str,
                    Cheekbones:str,
                    Chin_Shape:str,
                    beard:str,
                    beard_color:str, 
                    opt_append_this:str=""
            ) -> tuple:


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


class GeminiVision:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "image" : ("IMAGE",),
                "seed" : ("INT", {"forceInput":True}),
                "clip" : ("CLIP",),
                "randomness" : ("FLOAT", {"default":0.7, "min":0, "max": 1, "step":0.1, "display":"slider"}),
                "api_key" : ("STRING", {"default":""}),
                "prompt" : ("STRING", {"default":"Describe the image", "multiline":True})
            }
        }

    RETURN_TYPES = ("STRING", "CONDITIONING", )
    CATEGORY = PACK_NAME
    FUNCTION = 'execute'

    def execute(self, 
                   image:torch.Tensor, 
                   seed:int,
                   clip:object, 
                   randomness:float, 
                   prompt:str, 
                   api_key:str,
                   ) -> tuple[str, list]:
        
        pil_image = tensor2pil(image)

        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            raise (f"Error configuring gemini model : {e}")
        llm = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=genai.GenerationConfig(temperature=randomness))
        response = llm.generate_content([pil_image, prompt])
        tokens = clip.tokenize(response.text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")

        db_obj = IamME_Database()
        db_obj.update_db(input_prompt=prompt, output_prompt=response.text)
        db_obj.merge_DB()

        return (response.text, [[cond, output]],)


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
    FUNCTION = "execute"
    CATEGORY = PACK_NAME

    def execute(self, Image:torch.Tensor) -> dict:
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
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = PACK_NAME

    def execute(self, text:str, modify_text:str="", unique_id=None, extra_pnginfo=None) -> dict:
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                log_to_console("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                log_to_console("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
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


class ModelManager:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        return{
            "required" : {
                "civitAI_model_link" : ("STRING", {"default":"", "multiline":True})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("data", "dl_link")
    CATEGORY = PACK_NAME
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @PromptServer.instance.routes.post("/execute")
    async def handle_event(request):
        log_to_console("inside handle event")
        data = await request.json()
        ModelManager.download_model_with_progress(download_link=data["download_Url"], model_name=data["model_Name"])
        return web.json_response({"message":"successfully executed"}, status=200)

    def download_model_with_progress(download_link: str, model_name: str) -> None:
        this_file_path = Path(__file__)
        checkpoints_path = this_file_path.parent.parent.parent.joinpath("models/checkpoints")

        if not checkpoints_path.exists() and checkpoints_path.is_dir():
            raise ValueError("Checkpoints folder path not found!!")
        
        if model_name in [model.name for model in checkpoints_path.iterdir()]:
            log_to_console("Model already present in checkpoints folder, aborting download!!")
        
        else:
            log_to_console(f"Starting download for {model_name}..")
            civitai_auth_token = "dde477748946d47366ce09db94b81584"
            headers = {
                "Authorization": f"Bearer {civitai_auth_token}"
            }
            
            if not model_name.endswith('.safetensors'):
                model_name = f"{model_name}.safetensors"
            
            try:
                response = requests.get(
                    download_link,
                    headers=headers,
                    allow_redirects=True,
                    stream=True
                )
                response.raise_for_status()
                
                # Get the file size from headers
                total_size = int(response.headers.get('content-length', 0))
                
                save_path = checkpoints_path.joinpath(model_name)

                # Create progress bar and save model
                with open(save_path, 'wb') as f, tqdm(
                    desc=model_name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
                        
                log_to_console(f"Successfully downloaded {model_name} and saved to checkpoints directory.")
                
            except requests.exceptions.RequestException as e:
                log_to_console(f"Error downloading file: {e}")
                raise

    def link_processor(self, link:str)->str:
        pattern = r'https://civitai.com/models/(\d+)(?:/|(?:\?modelVersionId=\d+))'
        match = re.search(pattern, link)
        if match:
            model_id = match.group(1)
        else:
            logger.log(level=30, msg=f"[{PACK_NAME}'s Nodes] : model_id not found in given URL")
            raise

        return model_id

    def execute(self, civitAI_model_link:str) -> dict:
        model_names = []

        try:
            url = f"https://civitai.com/api/v1/models/{self.link_processor(civitAI_model_link)}"
            response = requests.get(url, headers={"Accept":"application/json"})
            response.raise_for_status()
            
            response_json = response.json()

            for value in response_json["modelVersions"]:
                model_info = {
                    "id": value["id"],
                    "name": value["files"][0]["name"],
                    "type": value["files"][0]["type"],
                    "metadata": value["files"][0]["metadata"],
                    "dl_link": value["files"][0]["downloadUrl"]
                }
                if all(key in value["files"][0]["metadata"] for key in ("format", "size", "fp")):
                    model_info["download_url"] = value["files"][0]["downloadUrl"] + f"?type={value['files'][0]['type']}" + f"&format={value['files'][0]['metadata']['format']}" + f"&size={value['files'][0]['metadata']['size']}" + f"&fp={value['files'][0]['metadata']['fp']}"
                else:
                    continue
                model_names.append(model_info)    

        except Exception as e:
            log_to_console(f"Error in model_downloader: {str(e)}")  # Debug log
            raise
        return {
            "ui" : {
                "names" : model_names,
            },
            "result" : (str(model_names), str([m["download_url"] for m in model_names]))
        }
   
    
class OllamaVision:

    @classmethod
    def INPUT_TYPES(s)-> dict:
        return {
            "required" : {
                "image" : ("IMAGE",),
                "seed" : ("INT", {"forceInput":True}),
                "clip" : ("CLIP",),
                "randomness" : ("FLOAT", {"default":0.7, "min":0, "max": 1, "step":0.1, "display":"slider"}),
                "prompt" : ("STRING", {"default":"Describe the image", "multiline":True})
            }, 
            "optional" : {
                "opt_model_name" : ("STRING", {"default":"llama3.2-vision:latest", "tooltip":"an optional input, write the name of ollama model you wanna use!!"})
            }
        }

    RETURN_TYPES = ("STRING", "CONDITIONING", )
    CATEGORY = PACK_NAME
    FUNCTION = 'execute'

    def execute(self, 
                    image:torch.Tensor, 
                    seed:int,
                    clip:object, 
                    prompt:str,
                    randomness:float,
                    opt_model_name:str = "llama3.2-vision:latest"
                    ) -> tuple[str, list]:

        # sys_prompt = ""
        log_to_console(f"model name is {opt_model_name}")
        image_b64:list = tensor2base64(image)
        log_to_console(f"Converted tensor image to base64")
        start_time = time.time()
        llm = Ollama(model=opt_model_name, base_url="http://192.168.0.169:11434", temperature=randomness).bind(images=image_b64)
        response = llm.invoke(prompt)
        end_time = time.time()
        log_to_console(f"generated response, time taken = {end_time-start_time}")
        tokens = clip.tokenize(response)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")

        db_obj = IamME_Database()
        db_obj.update_db(input_prompt=prompt, output_prompt=response)
        db_obj.merge_DB()

        log_to_console("Node executed!!")
        return (response, [[cond, output]],)   


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
    FUNCTION = 'execute'
    CATEGORY = PACK_NAME

    def execute(self, text:str, replace:bool, prepend:str="", append:str="", to_replace:str="", replace_with:str="") -> str:
        text = prepend + " " + text
        text = text + " " + append

        if replace:
            text = text.replace(to_replace, replace_with)

        
        return (text,)
      

class TriggerWordProcessor:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_in": ("STRING", {"multiline": True, "default":""}),
                "gender" : (["male", "female"], {"default":"female"}),
                "seed": ("INT",{"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = 'execute'
    CATEGORY = PACK_NAME
    # OUTPUT_NODE = True

    def execute(self, text_in:str, gender:str, seed:int=None) -> tuple:
        options = json_loader("TriggerWords")
        bg_options = options["background"]
        pose_options = options["pose"]
        whimsical_pose_options = options["whimsical_pose"]
        color_options = options["color"]
        topwear_options = options["topwear"]
        bottomwear_options = options["bottomwear"]
        region_options = options["regionality"]
        body_type_options = options["body_type"]

        if seed is not None:
            random.seed(seed)
        random_num = np.random.default_rng(seed)

        trigger_words = ["__background__", "__topwear__", "__bottomwear__", "__pose__", "__region__", "__bodytype__", "__whimsical_pose__"]

        for word in trigger_words:
            if word not in text_in:
                log_to_console(f"{word} not found in input text, skipping!!")
                continue
            
            #background
            if word == "__background__":
                bg_choice = random_num.choice(bg_options)
                text_in = text_in.replace(word, bg_choice)

            #topwear
            elif word == "__topwear__":          
                if gender=="male":
                    topwear_choice = random_num.choice(topwear_options["male"])
                else:
                    topwear_choice = random_num.choice(topwear_options["female"])
                color_choice = random_num.choice(color_options)
                clothing = f"{color_choice} {topwear_choice} "
                text_in = text_in.replace(word, clothing)
            
            #bottomwear
            elif word == "__bottomwear__":          
                if gender=="male":
                    bottomwear_choice = random_num.choice(bottomwear_options["male"])
                else:
                    bottomwear_choice = random_num.choice(bottomwear_options["female"])
                color_choice = random_num.choice(color_options)
                clothing = f"{color_choice} {bottomwear_choice} "
                text_in = text_in.replace(word, clothing)
            
            #pose
            elif word == "__pose__":
                pose_choice = random_num.choice(pose_options)
                text_in = text_in.replace(word, pose_choice)
            
            #pose
            elif word == "__whimsical_pose__":
                whimsical_pose_choice = random_num.choice(whimsical_pose_options)
                text_in = text_in.replace(word, whimsical_pose_choice)
            
            #region
            elif word == "__region__":
                region_choice = random_num.choice(region_options)
                text_in = text_in.replace(word, region_choice)

            elif word == "__bodytype__":
                if gender=="male":
                    body_type_choice = random_num.choice(body_type_options["male"])
                else:
                    body_type_choice = random_num.choice(body_type_options["female"])
                body_type = f"{body_type_choice} "
                text_in = text_in.replace(word, body_type)
            
        return (text_in,)


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
    


NODE_CLASS_MAPPINGS = {
    "AspectEmptyLatentImage" : AspectEmptyLatentImage,
    "BasicTextEditor" : TextTransformer,
    "ColorCorrect" : ColorCorrect,
    "ConnectionBus": ConnectionBus,
    "FacePromptMaker" : FacePromptMaker,
    "GeminiVision": GeminiVision,
    "GetImageData": GetImageData,
    "ImageBatchLoader": ImageBatchLoader,
    "LiveTextEditor": LiveTextEditor,
    "ModelManager" : ModelManager,
    "OllamaVision": OllamaVision,
    "TriggerWordProcessor" : TriggerWordProcessor,
    "SaveImageAdvanced": SaveImageAdvanced,   
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "AspectEmptyLatentImage" : PACK_NAME + " AspectEmptyLatent",
    "BasicTextEditor" : PACK_NAME + " BasicTextEditor",
    "ColorCorrect": PACK_NAME + " ColorCorrect",
    "ConnectionBus": PACK_NAME + " ConnectionBus",
    "FacePromptMaker": PACK_NAME + " FacePromptMaker",
    "GeminiVision": PACK_NAME + " GeminiVision",
    "GetImageData": PACK_NAME + " GetImageData",
    "ImageBatchLoader": PACK_NAME + " ImageBatchLoader",
    "LiveTextEditor" : PACK_NAME + " LiveTextEditor",
    "ModelManager" : PACK_NAME + " ModelManager",
    "OllamaVision": PACK_NAME + " OllamaVision",
    "TriggerWordProcessor" : PACK_NAME + " TriggerWordProcessor",   
    "SaveImageAdvanced": PACK_NAME + " SaveImageAdvanced",
}