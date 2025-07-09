# ComfyUI Node - uses Loftr to swap dials, RGBA, feature map, final image
import torch
import numpy as np
import cv2
import kornia as K
import kornia.feature as KF
import comfy.model_management
import comfy.utils


from .utils import *
from .image_utils import *



class LoFTRWatchDialSwapperV3_RGBA:
    def __init__(self):
        self.device = comfy.model_management.get_torch_device()
        self.loftr = None
        if self.loftr is None:
            try:
                print(f"[{PACK_NAME}] : Loading LoFTR model (this may download weights on the first run)...")
                pbar = comfy.utils.ProgressBar(1)
                self.loftr = KF.LoFTR(pretrained='outdoor').to(self.device).eval()
                pbar.update(1)
                print(f"[{PACK_NAME}] : LoFTR model loaded successfully to {self.device}.")
            except Exception as e:
                print(f"[{PACK_NAME}] : FATAL: Failed to load LoFTR model: {e}")
                raise e
            
    def detect_dial(self, image_gray:np.ndarray) -> tuple[np.uint16,]:
        # This function is unchanged
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                    param1=100, param2=30, minRadius=50, maxRadius=200)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0, 0]
            return (x, y, r)
        return None

    def extract_and_match_features_loftr(self, img0_gray:np.ndarray, img1_gray:np.ndarray, mask0:np.ndarray|None=None):
        """
        Uses the LoFTR deep learning model to find and match features.
        """
        print(f"[{PACK_NAME}] : Extracting and matching features with LoFTR model...")
        timg0 = K.image_to_tensor(img0_gray, False).float() / 255.
        timg0 = timg0.to(self.device)
        timg1 = K.image_to_tensor(img1_gray, False).float() / 255.
        timg1 = timg1.to(self.device)

        if mask0 is not None:
            tmask0 = torch.from_numpy(mask0).unsqueeze(0).unsqueeze(0).to(self.device)
            timg0[tmask0 == 0] = 0.0

        input_dict = {"image0": timg0, "image1": timg1}
        with torch.no_grad():
            correspondences = self.loftr(input_dict)

        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        confidence = correspondences['confidence'].cpu().numpy()
        
        confidence_threshold = 0.8
        good_indices = np.where(confidence > confidence_threshold)[0]
        
        if len(good_indices) == 0:
            print("-> LoFTR found no confident matches.")
            return None, None

        print(f"-> LoFTR found {len(good_indices)} confident matches.")
        src_pts = mkpts0[good_indices]
        dst_pts = mkpts1[good_indices]
        
        return src_pts, dst_pts

    def draw_loftr_matches(self, src_img:torch.Tensor, dst_img:torch.Tensor, src_pts, dst_pts):
        """
        Draws lines connecting matched points between two images.
        """
        # Create a new image to display the matches
        h1, w1 = src_img.shape[:2]
        h2, w2 = dst_img.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype='uint8')
        vis[:h1, :w1] = src_img
        vis[:h2, w1:] = dst_img

        # Draw the lines
        for p1, p2 in zip(src_pts, dst_pts):
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            # The x-coordinate for the destination point needs to be shifted by the width of the source image
            pt2 = (int(round(p2[0]) + w1), int(round(p2[1])))
            
            # Draw a line and circles at the points
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(vis, pt1, pt2, color, 1)
            cv2.circle(vis, pt1, 3, color, -1)
            cv2.circle(vis, pt2, 3, color, -1)
            
        return vis

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "destination_image": ("IMAGE",),
                "try_lower_confidence": ("BOOLEAN", {"default": True, "label_on": "auto_confidence", "label_off": "manual_confidence"}),
                "confidence_threshold": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "feather_erosion_strength": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
                "feather_blur_size": ("INT", {"default": 15, "min": 1, "max": 99, "step": 2}),
            }
        }

    # UPDATED RETURN TYPES AND NAMES
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("final_blended_image", "feature_match_image", "transformed_dial_with_alpha",)
    FUNCTION = "execute"
    CATEGORY = PACK_NAME

    def execute(    self, 
                    source_image:torch.Tensor, 
                    destination_image:torch.Tensor, 
                    try_lower_confidence:bool, 
                    confidence_threshold:float, 
                    feather_erosion_strength:int, 
                    feather_blur_size:int
                ):
        if source_image is None or destination_image is None:
            print("Error loading images")
        print(f"[{PACK_NAME}] : src img size ", source_image.shape)
        print(f"[{PACK_NAME}] : destn img size ", destination_image.shape)
            
        source_image = tensor_to_cv2(source_image)
        destination_image = tensor_to_cv2(destination_image)
        src_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(destination_image, cv2.COLOR_BGR2GRAY)

        # Detect dials using the original HoughCircles method
        src_circle = self.detect_dial(src_gray)
        dst_circle = self.detect_dial(dst_gray)
        if not src_circle or not dst_circle:
            print("Could not detect circles in one or both images.")
            
        src_x, src_y, src_r = src_circle
        dst_x, dst_y, dst_r = dst_circle

        # Create source mask
        src_mask = np.zeros(src_gray.shape, dtype=np.uint8)
        cv2.circle(src_mask, (src_x, src_y), src_r, 255, -1)

        # Feature Matching with LoFTR
        src_pts, dst_pts = self.extract_and_match_features_loftr(src_gray, dst_gray, src_mask)

        if src_pts is None or len(src_pts) < 10:
            print("Not enough matches found by LoFTR. Cannot proceed.")
            

        # =============================================================================
        # MODIFIED SECTION: SAVE THE VISUALIZATION
        # =============================================================================
        print("Saving feature match visualization to 'feature_matches.jpg'")
        matches_img = self.draw_loftr_matches(source_image, destination_image, src_pts, dst_pts)
        cv2.imwrite("feature_matches.jpg", matches_img)
        # =============================================================================
        # TODO : ISSUE
        # Reshape points for findHomography
        src_pts_reshaped = src_pts.reshape(-1, 1, 2)
        dst_pts_reshaped = dst_pts.reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts_reshaped, dst_pts_reshaped, cv2.RANSAC, 5.0)
        if M is None:
            print("Could not compute a valid homography matrix. Aborting.")
            

        # The rest of the script is unchanged
        h, w = destination_image.shape[:2]
        warped_src = cv2.warpPerspective(source_image, M, (w, h))
        warped_mask = cv2.warpPerspective(src_mask, M, (w, h))

        # Compute centroid of warped source
        moments = cv2.moments(warped_mask)
        if moments["m00"] != 0:
            warped_cx = int(moments["m10"] / moments["m00"])
            warped_cy = int(moments["m01"] / moments["m00"])
        else:
            print("Warped mask has no area.")
            

        dx = dst_x - warped_cx
        dy = dst_y - warped_cy

        # Affine shift to correct final center alignment
        shift_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_src = cv2.warpAffine(warped_src, shift_matrix, (w, h))
        shifted_mask = cv2.warpAffine(warped_mask, shift_matrix, (w, h))

        # Feather mask
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(shifted_mask, kernel, iterations=2)
        feathered = cv2.GaussianBlur(eroded_mask, (15, 15), 0)
        alpha = feathered.astype(float) / 255.0
        alpha_3ch = cv2.merge([alpha] * 3)

        
        foreground = shifted_src.astype(np.float32)
        background = destination_image.astype(np.float32)
        alpha_3ch = alpha_3ch.astype(np.float32)
        blended = cv2.multiply(alpha_3ch, foreground) + cv2.multiply(1 - alpha_3ch, background)
        result = blended.astype(np.uint8)

        # cv2.imwrite(output_filename, result)
        # print(f"Saved final result as {output_filename}")

        # cv2.imshow("Result", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        result_tensor = cv2_to_tensor(result)
        foreground_tensor = cv2_to_tensor(foreground.astype(np.uint8))
        background_tensor = cv2_to_tensor(background.astype(np.uint8))
        return (result_tensor, foreground_tensor, background_tensor,)

NODE_CLASS_MAPPINGS = {
    "LoFTRWatchDialSwapperV3_RGBA": LoFTRWatchDialSwapperV3_RGBA
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoFTRWatchDialSwapperV3_RGBA": PACK_NAME + " Watch Dial Swapper V3"
}