import cv2
import numpy as np
import joblib
import json
from groundtruth import extract_tile_features

import os

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)
print("[DEBUG] feature_columns loaded from:", os.path.abspath("feature_columns.json"))
print("[DEBUG] Expected length:", len(feature_columns))

# --------------------------
# CONFIG
# --------------------------
image_path = r"C:\Users\kaspe\Desktop\AAU\p2\miniprojekt2\playingboards_from_fullsize\DSC_1279\playing_field_3.jpg"

# --------------------------
# Load model, scaler, PCA, feature names
# --------------------------
model = joblib.load("trained_RF_model.joblib")
scaler = joblib.load("trained_scaler.joblib")
pca = joblib.load("trained_pca.joblib")
with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# --------------------------
# Reverse label map
# --------------------------
reverse_label_map = {
    0: "MINE",
    1: "HOME",
    2: "TABLE",
    3: "WHEAT",
    4: "FOREST",
    5: "GRASSLAND",
    6: "SWAMP",
    7: "LAKE"
}

# --------------------------
# Helper: crop tiles (shared logic)
# --------------------------
def crop_tiles_from_image(image_rgb, rows=5, cols=5):
    height, width = image_rgb.shape[:2]
    tile_h, tile_w = height // rows, width // cols
    tiles = []
    positions = []
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * tile_h, (r + 1) * tile_h
            x1, x2 = c * tile_w, (c + 1) * tile_w
            tile = image_rgb[y1:y2, x1:x2]
            tiles.append(tile)
            positions.append((x1, y1, x2, y2))
    return tiles, positions

# --------------------------
# Load and process image
# --------------------------
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

tiles, tile_positions = crop_tiles_from_image(image_rgb)
tile_features = []

for idx, tile in enumerate(tiles):
    features = extract_tile_features(tile)
    
    print("[DEBUG] Edge density raw:", features[-5:])  # laatste paar waardes
    print("[DEBUG] Feature length:", len(features))

    if idx == 0:
        print("\n[DEBUG] Tile (0,0) Feature Sample:")
        print("First 10 features:", features[:10])
        print("Length:", len(features), "| Expected:", len(feature_columns))

    if len(features) != len(feature_columns):
        raise ValueError(f"Feature mismatch: got {len(features)} but expected {len(feature_columns)}")

    tile_features.append(features)

# --------------------------
# Predict
# --------------------------
X = np.array(tile_features)
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)
preds = model.predict(X_pca)

import matplotlib.pyplot as plt
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA-transformed tiles (inference)")
plt.show()


# --------------------------
# Annotate image
# --------------------------
for i, (x1, y1, x2, y2) in enumerate(tile_positions):
    label_code = int(preds[i])
    print(f"[DEBUG] Tile {i} → label_code: {label_code} → {reverse_label_map.get(label_code)}")

    label_text = reverse_label_map.get(label_code, f"?{label_code}")
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label_text, (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow("Predicted Tiles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
