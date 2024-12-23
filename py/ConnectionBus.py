from .utils import *
class ConnectionBus:
    def __init__(self) -> None:
        self.default_len = 10

    @classmethod
    def INPUT_TYPES(s) -> dict[str:tuple[str, dict]]:
        return {
            "required" : {},
            "optional" : {
                "bus" : (BUS_DATA["type"],),
                "value_1" : (any_type, {"default":None,}),
                "value_2" : (any_type,{"default":None,}),
                "value_3" : (any_type,{"default":None,}),
                "value_4" : (any_type,{"default":None,}),
                "value_5" : (any_type,{"default":None,}),
                "value_6" : (any_type,{"default":None,}),
                "value_7" : (any_type,{"default":None,}),
                "value_8" : (any_type,{"default":None,}),
                "value_9" : (any_type,{"default":None,}),
                "value_10" : (any_type,{"default":None,}),
            }
        }

    RETURN_TYPES = (BUS_DATA["type"], any_type, any_type, any_type, any_type, any_type, any_type, any_type, any_type, any_type, any_type,)
    RETURN_NAMES = (BUS_DATA["name"], "value_1", "value_2", "value_3", "value_4", "value_5", "value_6", "value_7", "value_8", "value_9", "value_10",)
    CATEGORY = PACK_NAME
    FUNCTION = "execute"
    # value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8, value_9, value_10
    def execute(self, 
                  bus:list=None, 
                  value_1=None, value_2=None, value_3=None, value_4=None, value_5=None, value_6=None, value_7=None, value_8=None, value_9=None, value_10=None, # static inputs
                  **kwargs # for dynamic inputs
                  ) -> dict:
        
        #Initializing original values
        org_values = [None for i in range(self.default_len)]

        # initializing original values with bus values
        if bus is not None:
            org_values = bus
        
        # log_to_console(f"org_values = {org_values}")
        new_bus = []
        message_to_js = []

        # log_to_console(f"IamME's ConnectionBus : No. of arguments passed is, {len(kwargs)}, kwargs are {kwargs}")

        counter = 10

        if len(kwargs) > 0:
            counter+=len(kwargs)

        for  i in range(counter):
            exec(f"new_bus.append(value_{i+1} if value_{i+1} is not None else org_values[i])")

        return {"ui" : {"message":message_to_js}, "result":(new_bus, *new_bus)}
    
NODE_CLASS_MAPPINGS = {"ConnectionBus": ConnectionBus,}
NODE_DISPLAY_NAME_MAPPINGS = {"ConnectionBus": PACK_NAME + " ConnectionBus",}