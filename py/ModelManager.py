import re
from aiohttp import web
from server import PromptServer
from .utils import *

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

    @staticmethod
    @PromptServer.instance.routes.post("/execute")
    async def handle_event(request):
        log_to_console("inside handle event")
        data = await request.json()
        download_model_with_progress(download_link=data["download_Url"], model_name=data["model_Name"])
        return web.json_response({"message":"successfully executed"}, status=200)



    def link_processor(self, link:str)->str:
        pattern = r'https://civitai.com/models/(\d+)(?:/|(?:\?modelVersionId=\d+))'
        match = re.search(pattern, link)
        if match:
            model_id = match.group(1)
            return model_id
        else:
            logger.log(level=30, msg=f"[{PACK_NAME}'s Nodes] : model_id not found in given URL")
            return None

    def execute(self, civitAI_model_link:str) -> dict:
        model_names = []

        try:
            if (link:=self.link_processor(civitAI_model_link)) is not None:
                url = f"https://civitai.com/api/v1/models/{link}"
                response = requests.get(url, headers={"Accept":"application/json"}, timeout=(5, 15))
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
        except requests.exceptions.Timeout:
            log_to_console("Request to CivitAI timed out, please retry...")

        except Exception as e:
            log_to_console(f"Error in model_downloader: {str(e)}")  # Debug log
            raise
        return {
            "ui" : {
                "names" : model_names,
            },
            "result" : (str(model_names), str([m["download_url"] for m in model_names]))
        }

NODE_CLASS_MAPPINGS = {"ModelManager" : ModelManager,}
NODE_DISPLAY_NAME_MAPPINGS = {"ModelManager" : PACK_NAME + " ModelManager",}