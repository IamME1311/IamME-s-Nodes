import torch
from .utils import *
import folder_paths
import torch.nn.functional as F

class ImageConcatenate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "direction": (
                ["right", "left",],
                {"default": "right"}
            ),
            "match_image_size": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "WIDTH_OUT", "HEIGHT_OUT", "X_OFF",)
    FUNCTION = "execute"
    CATEGORY = PACK_NAME
    DESCRIPTION = """
Concatenates the second image to the first image in the specified direction. Optionally resizes the second image to match the dimensions of the first.
    """

    def __init__(self):
        super().__init__()
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    def resize_image(self, image, target_width, target_height):
        # Resize the image using PyTorch's interpolation function
        return F.interpolate(image, size=(target_height, target_width), mode="bilinear", align_corners=False)

    def execute(self, image1, image2, direction, match_image_size):
        # Step 1: Handle batch size mismatch by repeating smaller batch images to match the largest batch size
        batch_size1 = image1.shape[0]
        batch_size2 = image2.shape[0]

        if batch_size1 != batch_size2:
            max_batch_size = max(batch_size1, batch_size2)
            image1 = image1.repeat(max_batch_size // batch_size1, 1, 1, 1)
            image2 = image2.repeat(max_batch_size // batch_size2, 1, 1, 1)

        # Step 2: Resize the second image to match the first image's dimensions if required
        if match_image_size:
            target_shape = image1.shape  # Use the shape of the first image as the target
            original_height = image2.shape[1]
            original_width = image2.shape[2]
            aspect_ratio = original_width / original_height

            # Determine the target width and height based on the concatenation direction
            if direction in ["left", "right"]:
                target_height = target_shape[1]  # Match height for horizontal concatenation
                target_width = int(target_height * aspect_ratio)
            elif direction in ["up", "down"]:
                target_width = target_shape[2]  # Match width for vertical concatenation
                target_height = int(target_width / aspect_ratio)

            # Resize image2 using the resize_image function
            image2_resized = self.resize_image(
                image2.movedim(-1, 1),  # Convert to (B, C, H, W) format
                target_width,
                target_height
            ).movedim(1, -1)  # Convert back to (B, H, W, C) format
        else:
            image2_resized = image2  # If resizing is not required, use the original image

        # Step 3: Concatenate the images in the specified direction
        if direction == "right":
            concatenated_image = torch.cat((image1, image2_resized), dim=2)  # Concatenate along width
        elif direction == "down":
            concatenated_image = torch.cat((image1, image2_resized), dim=1)  # Concatenate along height
        elif direction == "left":
            concatenated_image = torch.cat((image2_resized, image1), dim=2)  # Concatenate along width with image2 on the left
        elif direction == "up":
            concatenated_image = torch.cat((image2_resized, image1), dim=1)  # Concatenate along height with image2 on top
        
        out_width = concatenated_image.shape[2] - image1.shape[2] # with of cropped image
        out_height = concatenated_image.shape[1] # height of second image
        out_x_off = image1.shape[2] # width of first image

        # Step 4: Return the concatenated image
        return (concatenated_image, out_width, out_height, out_x_off,)

NODE_CLASS_MAPPINGS = {
    "ImageConcatenate": ImageConcatenate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageConcatenate": PACK_NAME + " Image Concat"
}