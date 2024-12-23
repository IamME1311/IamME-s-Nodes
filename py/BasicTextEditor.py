from .utils import *

class TextTransformer:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput":True}),
                "prepend" : ("STRING", {"multiline":True, "default":""}), 
                "append" : ("STRING", {"multiline":True, "default":""}),
                "replace" : ("BOOLEAN", {"default" : False})
            },
            "optional": {
                "to_replace" : ("STRING", {"default":""}),
                "replace_with" : ("STRING", {"default":""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = 'execute'
    CATEGORY = PACK_NAME

    def execute(self,
        text:str,
        replace:bool,
        prepend:str="",
        append:str="",
        to_replace:str="",
        replace_with:str="") -> str:
        text = prepend + " " + text
        text = text + " " + append

        if replace:
            text = text.replace(to_replace, replace_with)


        return (text,)

NODE_CLASS_MAPPINGS = {"BasicTextEditor" : TextTransformer,}
NODE_DISPLAY_NAME_MAPPINGS = {"BasicTextEditor" : PACK_NAME + " BasicTextEditor",}