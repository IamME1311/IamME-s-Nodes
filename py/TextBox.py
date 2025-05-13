from .utils import *


class TextBox:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = PACK_NAME
    DESCRIPTION = "Provides a multi-tab interface for managing different versions of text input"

    def __init__(self):
        self.order = 0

    def execute(self, text):
        return (text,)
    

NODE_CLASS_MAPPINGS = {"TextBox" : TextBox,}

NODE_DISPLAY_NAME_MAPPINGS = {"TextBox" : PACK_NAME + " TextBox",}