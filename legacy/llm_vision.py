import requests
import os

import folder_paths
from ..py.utils import *



class llm_vision:
    def __init__(self):
        self.input_dir = folder_paths.get_input_directory()

    @staticmethod
    def get_host_IP() -> str:
        client, collection = config_loader()
        host_IP = collection.distinct("sarthak")
        client.close()
        return host_IP
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))] 
        return {
            "required" : {
                "image" : (sorted(files), {"image_upload":True}),
                "seed" : ("INT", {"forceInput":True}),
                "model_type" : (["llama", "gemini", "janus"], {"default":"gemini"}),
                "prompt" : ("STRING", {"defualt":"", "multiline":True})
            }
        }
    
    CATEGORY = PACK_NAME
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("response", "debug",)

    def execute(self, seed, model_type, image, prompt):
        debug  = image # this image is saved to input folder
        image_path = os.path.join(self.input_dir, image)
        host_IP = llm_vision.get_host_IP()
        response = requests.post(f"http://{host_IP}:8000/model/{model_type}/generate", 
                         headers={"accept":"application/json"},
                         files=[("uploaded_images", (image, open(image_path, "rb"), "image/png"))],
                         params={"prompt": prompt}
                         ).json()


        return (response["message"], debug,)
    


NODE_CLASS_MAPPINGS = {"LLM Vision" : llm_vision}
NODE_DISPLAY_NAME_MAPPINGS = {"LLM Vision" : PACK_NAME + " LLM Vision"}

#reference request code
# response = requests.post("http://192.168.0.83:8000/model/gemini/generate", 
#                          headers={"accept":"application/json"},
#                          files=[("uploaded_images", ("flower.png", open("./flower.png", "rb"), "image/png"))],
#                          params={"prompt":"describe the image"}
#                          ).json()
# response