import cv2
import numpy as np
import os
import math
import glob

########################################
# --- SIFT DETECTOR CLASS ---
########################################

class SIFTDetector:
    def __init__(self, template_folder="crop_and_perspectived_images_output_color", ratio_thresh=0.75, min_match_count=10):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.ratio_thresh = ratio_thresh
        self.min_match_count = min_match_count
        self.template_folder = template_folder

    @staticmethod
    def compute_bounding_box(projected_corners):
        xs = projected_corners[:, 0, 0]
        ys = projected_corners[:, 0, 1]
        return (np.min(xs), np.min(ys), np.max(xs), np.max(ys))

    @staticmethod
    def iou(boxA, boxB):
        (xA1, yA1, xA2, yA2) = boxA
        (xB1, yB1, xB2, yB2) = boxB
        
        inter_x1 = max(xA1, xB1)
        inter_y1 = max(yA1, yB1)
        inter_x2 = min(xA2, xB2)
        inter_y2 = min(yA2, yB2)
        
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        areaA = (xA2 - xA1) * (yA2 - yA1)
        areaB = (xB2 - xB1) * (yB2 - yB1)
        union_area = areaA + areaB - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area

    def detect_fields(self, scene_img):
        """
        Perform SIFT detection on the provided scene image (BGR) and match
        against all templates in self.template_folder.
        
        Returns:
          scene_with_contours (BGR image)
          detected_fields (list of BGR images corresponding to each extracted field)
        """
        scene_original = scene_img.copy()
        scene_gray = cv2.cvtColor(scene_original, cv2.COLOR_BGR2GRAY)
        keypoints_scene, descriptors_scene = self.sift.detectAndCompute(scene_gray, None)

        candidates = []
        # Loop over all template images in template_folder
        for template_file in glob.glob(os.path.join(self.template_folder, "*")):
            template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"Could not read template file: {template_file}")
                continue
            
            keypoints_template, descriptors_template = self.sift.detectAndCompute(template, None)
            if descriptors_template is None or descriptors_scene is None:
                continue
            
            # KNN match
            knn_matches = self.bf.knnMatch(descriptors_template, descriptors_scene, k=2)
            good_matches = []
            for m, n in knn_matches:
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)
            
            print(f"{template_file}: {len(good_matches)} good matches")
            # If enough good matches, attempt homography
            if len(good_matches) >= self.min_match_count:
                src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    h, w = template.shape
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    projected_corners = cv2.perspectiveTransform(corners, M)
                    new_box = self.compute_bounding_box(projected_corners)
                    M_inv = np.linalg.inv(M)
                    warped_field = cv2.warpPerspective(scene_original, M_inv, (w, h))
                    
                    candidate = {
                        "template_file": template_file,
                        "match_count": len(good_matches),
                        "projected_corners": projected_corners,
                        "bounding_box": new_box,
                        "warped_field": warped_field
                    }
                    candidates.append(candidate)
                else:
                    print(f"Could not compute homography for {template_file}.")
            else:
                print(f"Not enough matches for {template_file} (found {len(good_matches)})")
        
        # Non-maximum suppression by IoU
        candidates = sorted(candidates, key=lambda x: x["match_count"], reverse=True)
        final_candidates = []
        OVERLAP_THRESHOLD = 0
        for candidate in candidates:
            if all(self.iou(candidate["bounding_box"], final["bounding_box"]) <= OVERLAP_THRESHOLD 
                   for final in final_candidates):
                final_candidates.append(candidate)
        
        # Keep up to 4 best
        MAX_FIELDS_TO_SHOW = 4
        final_candidates = final_candidates[:MAX_FIELDS_TO_SHOW]
        
        # Draw bounding boxes
        scene_with_contours = scene_original.copy()
        detected_fields = []
        for cand in final_candidates:
            cv2.polylines(scene_with_contours,
                          [np.int32(cand["projected_corners"])],
                          isClosed=True,
                          color=(0, 255, 0),
                          thickness=6)  # Thicker line for visibility
            detected_fields.append(cand["warped_field"])
            print(f"Selected {cand['template_file']} with {cand['match_count']} good matches.")
        
        return scene_with_contours, detected_fields

########################################
# --- HELPER FUNCTIONS FOR PREPROCESSING ---
########################################

def line_intersection(line1, line2):
    """
    Calculate the intersection (x, y) of two lines given as (x1, y1, x2, y2).
    Returns None if the lines are (nearly) parallel.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return (px, py)

def apply_perspective_correction(img, margin_angle=10):
    """
    Performs perspective correction so that horizontal/vertical lines are properly aligned.
    Returns the corrected image on a new canvas without cropping.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    lines_p = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150,
                              minLineLength=100, maxLineGap=50)
    if lines_p is None:
        print("No lines found for perspective correction, returning original image.")
        return img

    vertical_lines = []
    horizontal_lines = []
    for l in lines_p:
        x1, y1, x2, y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle
        if abs(angle - 90) <= margin_angle:
            vertical_lines.append((x1, y1, x2, y2))
        elif angle <= margin_angle:
            horizontal_lines.append((x1, y1, x2, y2))

    if len(vertical_lines) < 2 or len(horizontal_lines) < 2:
        print("Not enough suitable lines for perspective correction, returning original image.")
        return img

    def avg_x(line):
        x1, _, x2, _ = line
        return (x1 + x2) / 2.0
    vertical_lines_sorted = sorted(vertical_lines, key=avg_x)
    left_line = vertical_lines_sorted[0]
    right_line = vertical_lines_sorted[-1]

    def avg_y(line):
        _, y1, _, y2 = line
        return (y1 + y2) / 2.0
    horizontal_lines_sorted = sorted(horizontal_lines, key=avg_y)
    top_line = horizontal_lines_sorted[0]
    bottom_line = horizontal_lines_sorted[-1]

    tl = line_intersection(left_line, top_line)
    tr = line_intersection(right_line, top_line)
    bl = line_intersection(left_line, bottom_line)
    br = line_intersection(right_line, bottom_line)

    if not all([tl, tr, bl, br]):
        print("Could not determine all corners, returning original image.")
        return img

    src_pts = np.float32([tl, tr, bl, br])
    h_in, w_in = img.shape[:2]
    dst_pts = np.float32([[0, 0], [w_in, 0], [0, h_in], [w_in, h_in]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Find new bounding
    corners_old = np.float32([[0, 0], [w_in, 0], [0, h_in], [w_in, h_in]]).reshape(-1, 1, 2)
    corners_new = cv2.perspectiveTransform(corners_old, M)
    x_coords = corners_new[:, 0, 0]
    y_coords = corners_new[:, 0, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    out_w = int(np.ceil(x_max - x_min))
    out_h = int(np.ceil(y_max - y_min))

    # Shift to avoid negative indices
    shift_matrix = np.array([[1, 0, -x_min],
                             [0, 1, -y_min],
                             [0, 0, 1]], dtype=np.float32)
    M_shifted = shift_matrix @ M

    corrected = cv2.warpPerspective(img, M_shifted, (out_w, out_h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))
    return corrected

def preprocess_denoise_image(img):
    """
    Reduces noise in the input image using a bilateral filter.
    The filter parameters are scaled based on the image size.
    """
    h, w = img.shape[:2]
    longest_side = max(h, w)
    scale_factor = longest_side / 4000.0
    scale_factor = max(0.5, min(scale_factor, 2.0))
    base_d = 4
    base_sigma = 40
    d = int(base_d * scale_factor)
    sigmaColor = base_sigma * scale_factor
    sigmaSpace = base_sigma * scale_factor
    if d < 1:
        d = 1
    denoised_image = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
    return denoised_image

def preprocess_adjust_brightness_and_contrast_histogram(img):
    """
    Adjusts brightness and contrast using histogram equalization in the YCrCb color space.
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    eq_img = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    # Additional brightness step
    eq_img = cv2.convertScaleAbs(eq_img, alpha=1.0, beta=15)
    return eq_img

def find_edges(img):
    """
    Finds edges in the image using Sobel operators.
    Returns a single-channel (grayscale) edge image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    edges = cv2.addWeighted(abs_sobel_x, 0.0, abs_sobel_y, 0.8, 0)
    return edges

def adjust_rotation_image(img, sobel_edges, visualize=False):
    """
    Adjusts the rotation of the image based on detected line angles using the provided Sobel edge map.
    Returns the rotated image.
    """
    ret, edges_bin = cv2.threshold(sobel_edges, 50, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(edges_bin, 1, np.pi/180, threshold=50,
                            minLineLength=50, maxLineGap=5)
    if lines is None:
        print("No lines detected, returning original image.")
        return img
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        angles.append(angle)
    median_angle = np.median(angles)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

########################################
# --- MAIN PREPROCESSING FUNCTION ---
########################################

def preprocess_process_folder(input_folder, output_folder_color, output_folder_edges):
    """
    Iterates over all images in 'input_folder', applies preprocessing steps, 
    and saves the processed images.
    
    For images in the "crop_and_perspectived_images" folder, it applies:
      - Denoising, Histogram/Contrast adjustment, and Edge detection.
      
    For images in the "full_game_areas" folder, it additionally applies:
      - Rotation adjustment and Perspective correction
      - Then runs SIFT detection to find up to 4 playing boards.
      
    The processed color image is saved in 'output_folder_color', and 
    the edge image in 'output_folder_edges'.

    If the image is from full_game_areas, the 4 detected playing boards 
    are also saved to a subfolder in "playingboards_from_fullsize".
    """
    if not os.path.exists(output_folder_color):
        os.makedirs(output_folder_color)
    if not os.path.exists(output_folder_edges):
        os.makedirs(output_folder_edges)
    
    # Folder to store the playing boards extracted from full game areas
    playingboards_base = "playingboards_from_fullsize"
    if not os.path.exists(playingboards_base):
        os.makedirs(playingboards_base)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not open {filename}. Skipping...")
                continue

            # Basic preprocessing
            denoised = preprocess_denoise_image(img)
            histogrammed = preprocess_adjust_brightness_and_contrast_histogram(denoised)
            edges = find_edges(histogrammed)
            
            # Additional for full_game_areas
            if "full_game_areas" in input_folder.lower():
                rotated = adjust_rotation_image(histogrammed, edges, visualize=False)
                perspective_corrected = apply_perspective_correction(rotated, margin_angle=10)
                final_color = perspective_corrected
            else:
                final_color = histogrammed
            

            
            # If the output image is larger than 24,000,000 pixels, resize proportionally
            max_pixels = 12000000
            h_final, w_final = final_color.shape[:2]
            if h_final * w_final > max_pixels:
                scale = math.sqrt(max_pixels / (h_final * w_final))
                new_w = int(w_final * scale)
                new_h = int(h_final * scale)
                final_color = cv2.resize(final_color, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"Resized image to {new_w}x{new_h} to meet the maximum pixel limit.")
            

            # Save the processed color image
            out_path_color = os.path.join(output_folder_color, filename)
            cv2.imwrite(out_path_color, final_color)

            # Save the edge image
            out_path_edges = os.path.join(output_folder_edges, filename)
            cv2.imwrite(out_path_edges, edges)
            
            print(f"Processed and saved color image: {out_path_color}")
            print(f"Processed and saved edges image: {out_path_edges}")

            # If full_game_areas, run SIFT detection and save playing fields
            if "full_game_areas" in input_folder.lower():
                detector = SIFTDetector()
                scene_with_contours, detected_fields = detector.detect_fields(final_color)

                # Create a subfolder named after the file (minus extension)
                file_stem, _ = os.path.splitext(filename)
                subfolder_path = os.path.join(playingboards_base, file_stem)
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                
                # Save the bounding-box scene
                bounding_path = os.path.join(subfolder_path, "scene_with_bounding_boxes.jpg")
                cv2.imwrite(bounding_path, scene_with_contours)
                
                # Save each detected playing field
                for i, field_img in enumerate(detected_fields):
                    field_path = os.path.join(subfolder_path, f"playing_field_{i+1}.jpg")
                    cv2.imwrite(field_path, field_img)
                    print(f"Saved detected playing field {i+1} to: {field_path}")

########################################
# --- MAIN ---
########################################

if __name__ == "__main__":
    # Example usage
    preprocess_process_folder(
        input_folder="crop_and_perspectived_images",
        output_folder_color="crop_and_perspectived_images_output_color",
        output_folder_edges="crop_and_perspectived_images_output_edges"
    )

    preprocess_process_folder(
        input_folder="full_game_areas",
        output_folder_color="full_game_areas_output_color",
        output_folder_edges="full_game_areas_output_edges"
    )
