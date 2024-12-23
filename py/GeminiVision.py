import torch
import google.generativeai as genai
from .utils import *
from .image_utils import *
class GeminiVision:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "image" : ("IMAGE",),
                "seed" : ("INT", {"forceInput":True}),
                "randomness" : ("FLOAT", {"default":0.7, "min":0, "max": 1, "step":0.1, "display":"slider"}),
                "api_key" : ("STRING", {"default":""}),
                "prompt" : ("STRING", {"default":"Describe the image", "multiline":True})
            },
            "optional" : {
                "clip" : ("CLIP",),
            }
        }

    RETURN_TYPES = ("STRING", "CONDITIONING", )
    CATEGORY = PACK_NAME
    FUNCTION = 'execute'

    def execute(self,
                   image:torch.Tensor,
                   seed:int,
                   randomness:float,
                   prompt:str,
                   api_key:str,
                   clip:object|None = None,
                   ) -> tuple[str, list]:

        pil_image = tensor2pil(image)

        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            raise f"Error configuring gemini model : {e}"
        llm = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=genai.GenerationConfig(temperature=randomness))
        response = llm.generate_content([pil_image, prompt])
        if clip is not None:
            tokens = clip.tokenize(response.text)
            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond")
        else:
            cond=output=None
        # db_obj = IamME_Database()
        # db_obj.update_db(input_prompt=prompt, output_prompt=response.text)
        # db_obj.merge_DB()

        return (response.text, [[cond, output]],)

NODE_CLASS_MAPPINGS = {"GeminiVision": GeminiVision,}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiVision": PACK_NAME + " GeminiVision",}
