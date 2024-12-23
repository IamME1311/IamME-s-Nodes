from .utils import *

class Slider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("FLOAT", { "display": "slider", "default": 50.0, "min": 0.0, "max": 100.0, "step": 5.0 }),
            },
        }

    RETURN_TYPES = ("FLOAT","INT",)
    FUNCTION = "execute"
    CATEGORY = PACK_NAME

    def execute(self, value):
        value = float(value)

        return (value, int(value))
    
NODE_CLASS_MAPPINGS = {"IamSlider": Slider,}
NODE_DISPLAY_NAME_MAPPINGS = {"IamSlider": PACK_NAME + " Slider",}