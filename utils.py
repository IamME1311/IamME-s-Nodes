import json
from pathlib import Path
import requests
import numpy as np
from colorama import Fore, Style, init
import logging



PACK_NAME = "IamME"
MAX_RESOLUTION = 16384
ASPECT_CHOICES = ["None","custom",
                    "1:1 (Perfect Square)",
                    "2:3 (Classic Portrait)", "3:4 (Golden Ratio)", "3:5 (Elegant Vertical)", "4:5 (Artistic Frame)", "5:7 (Balanced Portrait)", "5:8 (Tall Portrait)",
                    "7:9 (Modern Portrait)", "9:16 (Slim Vertical)", "9:19 (Tall Slim)", "9:21 (Ultra Tall)", "9:32 (Skyline)",
                    "3:2 (Golden Landscape)", "4:3 (Classic Landscape)", "5:3 (Wide Horizon)", "5:4 (Balanced Frame)", "7:5 (Elegant Landscape)", "8:5 (Cinematic View)",
                    "9:7 (Artful Horizon)", "16:9 (Panorama)", "19:9 (Cinematic Ultrawide)", "21:9 (Epic Ultrawide)", "32:9 (Extreme Ultrawide)"
                ]

IMAGE_DATA = {"type":"image_data", "name":"image data"}

BUS_DATA = {"type":"bus", "name":"bus"}

# initialize colorama
init(autoreset=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def json_loader(file_name:str) -> dict:
    cwd_name = Path(__file__).parent
    path_to_asset_file = cwd_name.joinpath(f"assets/{file_name}.json")
    with open(path_to_asset_file, "r") as f:
        asset_data = json.load(f)
    return asset_data

def apply_attention(text:str, weight:float) -> str:
    weight = float(np.round(weight, 2))
    return f"({text}:{weight})"

def parser(aspect : str) -> int:
    aspect = list(map(int,aspect.split()[0].split(":")))
    return aspect

# general log/print function for status messages in CMD
def log_to_console(message:str, level:int=10) -> None:
    logger.log(level, message)
    print(Style.BRIGHT + Fore.CYAN + f"[{PACK_NAME}'s Nodes] : {message}")


random_opt = "Randomize 🎲"
option_dict = json_loader("FacePromptMaker")

def config_loader() -> dict:
    # Raw URL of the file on GitHub (private repo)
    url = "https://raw.githubusercontent.com/IamME1311/IamME-s-Nodes/master/assets/config.json"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()



# thanks to pythongossss..
class AnyType(str):

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")