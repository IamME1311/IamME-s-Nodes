import random
from..py.utils import *

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

        trigger_words = ["__background__",
                    "__topwear__",
                    "__bottomwear__",
                    "__pose__",
                    "__region__",
                    "__bodytype__",
                    "__whimsical_pose__"]

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
    
NODE_CLASS_MAPPINGS = {"TriggerWordProcessor" : TriggerWordProcessor,}
NODE_DISPLAY_NAME_MAPPINGS = {"TriggerWordProcessor" : PACK_NAME + " TriggerWordProcessor",}