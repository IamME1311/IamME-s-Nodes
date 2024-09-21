import os
import torch
import folder_paths
import numpy as np
import nodes


preview_bridge_image_id_map = {}
preview_bridge_image_name_map = {}
preview_bridge_cache = {}

def set_previewbridge_image(node_id, file, item):
    global pb_id_cnt

    if file in preview_bridge_image_name_map:
        pb_id = preview_bridge_image_name_map[node_id, file]
        if pb_id.startswith(f"${node_id}"):
            return pb_id

    pb_id = f"${node_id}-{pb_id_cnt}"
    preview_bridge_image_id_map[pb_id] = (file, item)
    preview_bridge_image_name_map[node_id, file] = (pb_id, item)
    pb_id_cnt += 1

    return pb_id

def empty_pil_tensor(w=64, h=64):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)

def is_execution_model_version_supported():
    try:
        import comfy_execution
        return True
    except:
        return False