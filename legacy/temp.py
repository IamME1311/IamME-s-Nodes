# --- CORE FUNCTIONS (unchanged) ---
    def _detect_dial(self, image_gray: np.ndarray):
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        max_radius = min(image_gray.shape) // 2
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                   param1=100, param2=30, minRadius=50, maxRadius=max_radius)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            return circles[0, 0]
        return None

    def _draw_loftr_matches(self, src_img, dst_img, src_pts, dst_pts):
        h1, w1 = src_img.shape[:2]; h2, w2 = dst_img.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype='uint8')
        vis[:h1, :w1] = src_img; vis[:h2, w1:] = dst_img
        if src_pts is not None and dst_pts is not None:
            for p1, p2 in zip(src_pts, dst_pts):
                pt1 = (int(round(p1[0])), int(round(p1[1]))); pt2 = (int(round(p2[0]) + w1), int(round(p2[1])))
                cv2.line(vis, pt1, pt2, tuple(np.random.randint(0, 255, 3).tolist()), 1)
        return vis

    def _extract_and_match_features_loftr(self, img0_gray, img1_gray, confidence_threshold, try_lower_confidence, mask0=None):
        print(f"[{self.NODE_NAME}] Running LoFTR feature extraction and matching...")
        timg0 = K.image_to_tensor(img0_gray, False).float().to(self.device) / 255.
        timg1 = K.image_to_tensor(img1_gray, False).float().to(self.device) / 255.
        if mask0 is not None:
            tmask0 = torch.from_numpy(mask0).unsqueeze(0).unsqueeze(0).to(self.device)
            timg0[tmask0 == 0] = 0.0
        with torch.no_grad():
            correspondences = self.loftr({"image0": timg0, "image1": timg1})
        mkpts0, mkpts1, confidence = (d.cpu().numpy() for d in [correspondences['keypoints0'], correspondences['keypoints1'], correspondences['confidence']])
        threshold_list = [0.8, 0.7, 0.6, 0.5] if try_lower_confidence else [confidence_threshold]
        print(f"[{self.NODE_NAME}] -> Using confidence thresholds: {threshold_list}")
        for thresh in threshold_list:
            good_indices = np.where(confidence > thresh)[0]
            if len(good_indices) >= 10:
                print(f"[{self.NODE_NAME}] -> SUCCESS: Found {len(good_indices)} matches at threshold {thresh}.")
                return mkpts0[good_indices], mkpts1[good_indices]
        print(f"[{self.NODE_NAME}] -> FAILURE: Insufficient matches found.")
        return None, None

    def swap_dial(self, source_image, destination_image, try_lower_confidence, confidence_threshold, feather_erosion_strength, feather_blur_size):
        print(f"\n--- [{self.NODE_NAME}] Starting Dial Swap Process ---")
        self._load_model()

        print(f"[{self.NODE_NAME}] Step 1: Converting input tensors to OpenCV images.")
        src_img = tensor_to_cv2(source_image)
        dst_img = tensor_to_cv2(destination_image)
        src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

        h, w = dst_img.shape[:2]
        empty_matches = self._draw_loftr_matches(src_img, dst_img, None, None)
        # UPDATED: Fallback for RGBA image
        empty_rgba = torch.zeros((1, h, w, 4), dtype=torch.float32, device=self.device)
        error_return = (destination_image, cv2_to_tensor(empty_matches), empty_rgba)

        print(f"[{self.NODE_NAME}] Step 2: Detecting dials in images...")
        src_circle = self._detect_dial(src_gray)
        dst_circle = self._detect_dial(dst_gray)
        if src_circle is None:
            print(f"[{self.NODE_NAME}] -> ERROR: Could not detect dial in SOURCE image. Aborting.")
            return error_return
        if dst_circle is None:
            print(f"[{self.NODE_NAME}] -> ERROR: Could not detect dial in DESTINATION image. Aborting.")
            return error_return
        print(f"[{self.NODE_NAME}] -> Source dial found at {src_circle[:2]} with radius {src_circle[2]}.")
        print(f"[{self.NODE_NAME}] -> Destination dial found at {dst_circle[:2]} with radius {dst_circle[2]}.")
        src_x, src_y, src_r = src_circle; dst_x, dst_y, dst_r = dst_circle

        print(f"[{self.NODE_NAME}] Step 3: Creating circular mask for source dial.")
        src_mask = np.zeros(src_gray.shape, dtype=np.uint8)
        cv2.circle(src_mask, (src_x, src_y), src_r, 255, -1)

        print(f"[{self.NODE_NAME}] Step 4: Finding feature matches with LoFTR.")
        src_pts, dst_pts = self._extract_and_match_features_loftr(src_gray, dst_gray, confidence_threshold, try_lower_confidence, src_mask)
        matches_tensor = cv2_to_tensor(self._draw_loftr_matches(src_img, dst_img, src_pts, dst_pts))
        if src_pts is None:
            print(f"[{self.NODE_NAME}] -> ERROR: Not enough matches found. Aborting swap.")
            return (destination_image, matches_tensor, empty_rgba)

        print(f"[{self.NODE_NAME}] Step 5: Calculating homography matrix.")
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            print(f"[{self.NODE_NAME}] -> ERROR: Could not compute a valid homography matrix. Aborting.")
            return (destination_image, matches_tensor, empty_rgba)
        print(f"[{self.NODE_NAME}] -> Homography calculated successfully.")

        print(f"[{self.NODE_NAME}] Step 6: Warping source dial and performing center alignment.")
        warped_src = cv2.warpPerspective(src_img, M, (w, h))
        warped_mask = cv2.warpPerspective(src_mask, M, (w, h))
        moments = cv2.moments(warped_mask)
        if moments["m00"] != 0:
            warped_cx, warped_cy = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
            dx, dy = dst_x - warped_cx, dst_y - warped_cy
            print(f"[{self.NODE_NAME}] -> Aligning center with shift: dx={dx}, dy={dy}")
            shift_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted_src = cv2.warpAffine(warped_src, shift_matrix, (w, h))
            shifted_mask = cv2.warpAffine(warped_mask, shift_matrix, (w, h))
        else:
            print(f"[{self.NODE_NAME}] -> WARNING: Warped mask has no area. Skipping center alignment.")
            shifted_src, shifted_mask = warped_src, warped_mask

        print(f"[{self.NODE_NAME}] Step 7: Feathering mask and blending images.")
        print(f"[{self.NODE_NAME}] -> Feather settings: Erosion={feather_erosion_strength}, Blur Size={feather_blur_size}")
        if feather_erosion_strength > 0:
            eroded_mask = cv2.erode(shifted_mask, np.ones((3, 3), np.uint8), iterations=feather_erosion_strength)
        else:
            eroded_mask = shifted_mask
        ksize = feather_blur_size if feather_blur_size % 2 != 0 else feather_blur_size + 1
        feathered_mask = cv2.GaussianBlur(eroded_mask, (ksize, ksize), 0)
        alpha_3ch = cv2.cvtColor(feathered_mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
        blended = cv2.multiply(alpha_3ch, shifted_src.astype(float)) + cv2.multiply(1 - alpha_3ch, dst_img.astype(float))
        
        # 8. Prepare Final Outputs
        print(f"[{self.NODE_NAME}] Step 8: Finalizing outputs and creating RGBA dial image.")
        final_blended_tensor = cv2_to_tensor(blended.astype(np.uint8))
        
        # --- CREATE RGBA IMAGE OF THE TRANSFORMED DIAL ---
        # Create a 4-channel BGRA image. shifted_src is BGR.
        bgra_dial = cv2.cvtColor(shifted_src, cv2.COLOR_BGR2BGRA)
        # Set the alpha channel from the non-feathered, shifted mask
        bgra_dial[:, :, 3] = shifted_mask
        # Convert from BGRA (OpenCV) to RGBA (ComfyUI)
        rgba_dial = cv2.cvtColor(bgra_dial, cv2.COLOR_BGRA2RGBA)
        # Convert to a ComfyUI tensor (float, 0-1, batch dimension)
        transformed_dial_with_alpha_tensor = torch.from_numpy(rgba_dial.astype(np.float32) / 255.0).unsqueeze(0)

        print(f"--- [{self.NODE_NAME}] Process Complete ---\n")
        return (final_blended_tensor, matches_tensor, transformed_dial_with_alpha_tensor)