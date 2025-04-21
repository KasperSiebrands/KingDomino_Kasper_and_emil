import pygame
import os
import sys
import numpy as np
import json
import cv2
from skimage.feature import local_binary_pattern, hog, canny
from skimage.color import rgb2gray
from skimage.transform import resize

# ---------------------------
# CONFIGURATION & PATHS
# ---------------------------
IMAGE_FOLDER = r"C:\Users\kaspe\Desktop\AAU\p2\miniprojekt2\crop_and_perspectived_images_output_color"
JSON_OUTPUT_PATH = r"C:\Users\kaspe\Desktop\AAU\p2\miniprojekt2\V2labels.jsonl"

TILE_TYPE_KEYS = {
    pygame.K_w: "WHEAT",
    pygame.K_g: "GRASSLAND",
    pygame.K_f: "FOREST",
    pygame.K_s: "SWAMP",
    pygame.K_l: "LAKE",
    pygame.K_m: "MINE",
    pygame.K_t: "TABLE",
    pygame.K_h: "HOME"
}

CROWN_KEYS = {
    pygame.K_1: 1,
    pygame.K_2: 2,
    pygame.K_3: 3
}

GRID_ROWS = 5
GRID_COLS = 5
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)

# ---------------------------
# UNIFIED FEATURE EXTRACTOR
# ---------------------------
def extract_tile_features(tile_array):
    def compute_hist_hsv(tile_array):
        hsv = cv2.cvtColor(tile_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv_hist = []
        for i in ['H', 'S', 'V']:
            idx = {'H': 0, 'S': 1, 'V': 2}[i]
            hist, _ = np.histogram(hsv[:, :, idx], bins=256, range=(0, 256))
            hsv_hist.extend(hist.tolist())
        return hsv_hist

    def compute_lbp(tile_array):
        gray = rgb2gray(tile_array)
        gray_uint8 = (gray * 255).astype(np.uint8)
        lbp = local_binary_pattern(gray_uint8, P=8, R=1, method='default')
        max_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=max_bins, range=(0, max_bins), density=True)
        return hist.tolist()

    def compute_hog(tile_array):
        gray = rgb2gray(tile_array)
        resized = resize(gray, (100, 100))
        features = hog(
            resized,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            orientations=9,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        print("[DEBUG] HOG length:", len(features))  # ← zie nu écht de lengte
        return features.tolist()

    def compute_edge_density(tile_array):
        gray = rgb2gray(tile_array)
        edges = canny(gray)
        num_edge_pixels = np.count_nonzero(edges)
        total_pixels = gray.shape[0] * gray.shape[1]
        return num_edge_pixels / float(total_pixels)  # return float, not [float]


    hsv = compute_hist_hsv(tile_array)
    lbp = compute_lbp(tile_array)
    hog_vec = compute_hog(tile_array)
    edge_density = compute_edge_density(tile_array)

    assert isinstance(edge_density, float), "[ASSERT FAIL] edge_density is NOT a float!"
    print("[DEBUG] extract_tile_features called from:", __name__)
    print("[DEBUG] edge_density value:", edge_density)

    print("[DEBUG] feature lengths → HSV:", len(hsv), "LBP:", len(lbp), "HOG:", len(hog_vec), "Edge: 1")

    return hsv + lbp + hog_vec + [edge_density]



# ---------------------------
# DRAWING FUNCTIONS
# ---------------------------
def draw_grid(surface, x, y, width, height, rows, cols, current_row, current_col):
    cell_width = width / cols
    cell_height = height / rows
    for r in range(rows):
        for c in range(cols):
            rect_x = x + c * cell_width
            rect_y = y + r * cell_height
            rect = pygame.Rect(rect_x, rect_y, cell_width, cell_height)
            if r == current_row and c == current_col:
                pygame.draw.rect(surface, RED, rect, 3)
            else:
                pygame.draw.rect(surface, BLACK, rect, 1)

# ---------------------------
# LABELING & IMAGE PROCESSING
# ---------------------------
def label_image(screen, scaled_img, font, img_file):
    tile_types = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    tile_crowns = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    current_row, current_col = 0, 0
    running = True

    while running:
        all_labeled = all(tile_types[r][c] is not None for r in range(GRID_ROWS) for c in range(GRID_COLS))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_col = (current_col - 1) % GRID_COLS
                elif event.key == pygame.K_RIGHT:
                    current_col = (current_col + 1) % GRID_COLS
                elif event.key == pygame.K_UP:
                    current_row = (current_row - 1) % GRID_ROWS
                elif event.key == pygame.K_DOWN:
                    current_row = (current_row + 1) % GRID_ROWS
                elif event.key in TILE_TYPE_KEYS:
                    tile_types[current_row][current_col] = TILE_TYPE_KEYS[event.key]
                elif event.key in CROWN_KEYS:
                    tile_crowns[current_row][current_col] = CROWN_KEYS[event.key]
                elif event.key == pygame.K_RETURN and all_labeled:
                    running = False

        screen.fill(WHITE)
        screen.blit(scaled_img, (0, 0))
        draw_grid(screen, 0, 0, 500, 500, GRID_ROWS, GRID_COLS, current_row, current_col)

        text_y = 20
        def draw_text(line):
            nonlocal text_y
            text_surface = font.render(line, True, BLACK)
            screen.blit(text_surface, (520, text_y))
            text_y += 30

        draw_text(f"File: {img_file}")
        draw_text(f"Current tile: (row={current_row}, col={current_col})")
        draw_text("Arrow keys: navigeer tussen tiles.")
        draw_text("Tile type keys: W=WH, G=GR, F=FO, S=SW, L=LA, M=MI, T=TA, H=HO")
        draw_text("Druk op 1, 2, of 3 voor het aantal kronen.")
        draw_text("Druk op ENTER als alle 25 tiles een label hebben.")

        label_base_x = 520
        label_base_y = 300
        cell_width = 100
        cell_height = 25
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                ttype = tile_types[r][c]
                crown = tile_crowns[r][c]
                if ttype is None:
                    label_str = "_"
                else:
                    label_str = f"{ttype}({crown})" if crown > 0 else ttype
                text_surface = font.render(label_str, True, BLACK)
                screen.blit(text_surface, (label_base_x + c * cell_width, label_base_y + r * cell_height))
        pygame.display.flip()

    return tile_types, tile_crowns

def process_image(img_file, image_folder, screen, font):
    img_path = os.path.join(image_folder, img_file)
    image_surface = pygame.image.load(img_path)
    scaled_img = pygame.transform.scale(image_surface, (500, 500))
    tile_types, tile_crowns = label_image(screen, scaled_img, font, img_file)
    tile_width = 500 // GRID_COLS
    tile_height = 500 // GRID_ROWS
    tile_data = []

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            tile_type = tile_types[r][c]
            crown = tile_crowns[r][c]
            rect = pygame.Rect(c * tile_width, r * tile_height, tile_width, tile_height)
            tile_surface = scaled_img.subsurface(rect).copy()
            tile_array = pygame.surfarray.array3d(tile_surface)
            tile_array = np.transpose(tile_array, (1, 0, 2))
            features = extract_tile_features(tile_array)

            tile_info = {
                "filename": img_file,
                "row": r,
                "col": c,
                "tile_type": tile_type,
                "crown": crown,
                "hist_hsv": None,
                "texture_lbp": None,
                "HOG": None,
                "Edge density": None,
                "features": features
            }
            tile_data.append(tile_info)
    return tile_data

# ---------------------------
# MAIN FUNCTION
# ---------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Tile Labeling Tool (Crown & Features)")
    font = pygame.font.SysFont(None, 24)

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(valid_extensions)]
    image_files.sort()

    processed_images = set()
    if os.path.exists(JSON_OUTPUT_PATH):
        with open(JSON_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_images.add(data["filename"])
                except json.JSONDecodeError:
                    pass

    images_to_process = [img for img in image_files if img not in processed_images]
    total_images = len(images_to_process)
    print(f"Found {len(image_files)} images in total. {len(processed_images)} already processed.")
    print(f"Processing {total_images} remaining images.")

    for idx, img_file in enumerate(images_to_process, start=1):
        tile_data = process_image(img_file, IMAGE_FOLDER, screen, font)
        with open(JSON_OUTPUT_PATH, 'a', encoding='utf-8') as f:
            for tile in tile_data:
                f.write(json.dumps(tile) + "\n")
        print(f"Processed {idx}/{total_images}: Appended data for {img_file} to {JSON_OUTPUT_PATH}")

    pygame.quit()
    print(f"All data appended to {JSON_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
