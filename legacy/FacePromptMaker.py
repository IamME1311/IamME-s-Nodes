import random
from ..py.utils import *
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


        if Activate is True:
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
        
NODE_CLASS_MAPPINGS = {"FacePromptMaker" : FacePromptMaker,}
NODE_DISPLAY_NAME_MAPPINGS = {"FacePromptMaker": PACK_NAME + " FacePromptMaker",}