import time
import torch
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from .utils import *
from .image_utils import *
class OllamaVision:

    @classmethod
    def INPUT_TYPES(s)-> dict:
        return {
            "required" : {
                "image" : ("IMAGE",),
                "seed" : ("INT", {"forceInput":True}),
                "clip" : ("CLIP",),
                "randomness" : ("FLOAT", {"default":0.7, "min":0, "max": 1, "step":0.1, "display":"slider"}),
                "host" : (["Sarthak PC", "Sridhar PC", "Krishna PC", "Localhost"],),
                "prompt" : ("STRING", {"default":"Describe the image", "multiline":True})
            },
            "optional" : {
                "clip" : ("CLIP",),
                "opt_model_name" : ("STRING", {"default":"llama3.2-vision:latest", "tooltip":"an optional input, write the name of ollama model you wanna use!!"})
            }
        }

    RETURN_TYPES = ("STRING", "CONDITIONING", )
    CATEGORY = PACK_NAME
    FUNCTION = 'execute'

    def execute(self,
                    image:torch.Tensor,
                    seed:int,
                    prompt:str,
                    randomness:float,
                    host:str,
                    clip:object=None,
                    opt_model_name:str = "llama3.2-vision:latest"
                    ) -> tuple[str, list]:

        # sys_prompt = ""
        log_to_console(f"model name is {opt_model_name}")
        if image.shape[0] > 1:
            log_to_console("OllamaVision will only generate text for 1st image in the batch, ignoring others...")
            image = image[0]
        image_b64:bytes = tensor2base64(image)
        log_to_console("Converted tensor image to base64")

        start_time = time.time()
        client, config_data = config_loader()
        if (host_name := host.split()[0].lower()) == "localhost":
            ip_address = host_name
        else:
            ip_address = config_data.distinct(host_name)[0]

        llm = ChatOllama(model=opt_model_name, base_url=f'http://{ip_address}:11434', temperature=randomness)
        chain = prompt_func | llm | StrOutputParser()
        response = chain.invoke({"text": prompt, "image": image_b64})
        end_time = time.time()

        log_to_console(f"generated response, time taken = {end_time-start_time:.2f} seconds")

        if clip is not None:
            tokens = clip.tokenize(response)
            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond")
        else:
            cond=output=None
        client.close()

        # db_obj = IamME_Database()
        # db_obj.update_db(input_prompt=prompt, output_prompt=response)
        # db_obj.merge_DB()

        log_to_console("Node executed!!")
        return (response, [[cond, output]],)
    
NODE_CLASS_MAPPINGS = {"OllamaVision": OllamaVision,}
NODE_DISPLAY_NAME_MAPPINGS = {"OllamaVision": PACK_NAME + " OllamaVision",}