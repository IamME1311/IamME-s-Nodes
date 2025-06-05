import torch
import google.generativeai as genai
from .utils import *
from .image_utils import *
class GeminiVision:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "mode" : (["Image", "Video"], {"default" : "Image"},),
                "model_name" : (["gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-2.5-flash-preview-05-20"], {"default":"gemini-2.5-flash-preview-05-20"}),
                "seed" : ("INT", {"forceInput":True}),
                "randomness" : ("FLOAT", {"default":0.7, "min":0, "max": 1, "step":0.1, "display":"slider"}),
                "api_key" : ("STRING", {"default":""}),
                "prompt" : ("STRING", {"default":"Describe the image", "multiline":True})
            },
            "optional" : {
                "image" : ("IMAGE",),
                "image2":("IMAGE",),
                "clip" : ("CLIP",),
                "video_path":("STRING", {"default":"", "tooltip":"The path of the video file, only supports LOCAL FILES."})
            }
        }
 
    RETURN_TYPES = ("STRING", "CONDITIONING", )
    CATEGORY = PACK_NAME
    FUNCTION = 'execute'

    def execute(self,
                   mode:str,
                   model_name:str,
                   seed:int,
                   randomness:float,
                   prompt:str,
                   api_key:str,
                   image:torch.Tensor=None,
                   image2:torch.Tensor=None,
                   clip:object|None = None,
                   video_path:str=None
                   ) -> tuple[str, list]:
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            raise f"Error configuring gemini model : {e}"
        llm = genai.GenerativeModel(model_name=model_name, generation_config=genai.GenerationConfig(temperature=randomness))

        if mode.lower() == "image":
            tensor_images = [image, image2]
            llm_input = [*tensor_batch2pil(tensor_images), prompt]
            log_to_console("converted tensor images to list of pil images..", 20)
            response = llm.generate_content(llm_input)

        elif mode.lower() == "video":
            video_path = video_path.replace("\"", "")
            video_path = Path(video_path)
            log_to_console("uploading video file to file server")
            video_file = genai.upload_file(video_path)
            while video_file.state.name == "PROCESSING":
                time.sleep(10)
                video_file = genai.get_file(video_file.name)
            if video_file.state.name=="FAILED":
                log_to_console("video file upload failed", 40)
                raise ValueError(f"video file upload {video_file.state.name}")
            
            response = llm.generate_content([video_file, prompt], request_options={"timeout":600})
            # video_file.delete()
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
