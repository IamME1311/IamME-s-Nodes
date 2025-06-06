from .utils import PACK_NAME, any_type

class Patch_Type:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required" : {
                "patch_type": (["3:4","1:1", "9:16"], {
                        "default": "3:4",
                    }),
            }
        }
    
    CATEGORY = PACK_NAME
    FUNCTION = "execute"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("patch_type",)

    def execute(self, patch_type:str):

        return (patch_type,)
    


NODE_CLASS_MAPPINGS = {"patch_type" : Patch_Type}
NODE_DISPLAY_NAME_MAPPINGS = {"patch_type" : PACK_NAME + " patch_type"}
