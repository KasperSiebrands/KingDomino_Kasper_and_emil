import os
import cv2
import numpy as np

# === Instellingen: paden en parameters ===
TEMPLATES_FOLDER = r"C:\Users\kaspe\Downloads\test"
BOARD_IMAGE_PATH = r"C:\Users\kaspe\Desktop\AAU\p2\miniprojekt2\crop_and_perspectived_images_output_color\54.jpg"

TEMPLATE_NAMES = ["normal", "upside down", "L", "R"]
MATCHING_THRESHOLD = 0.6
NMS_DISTANCE_THRESH = 10
IGNORE_CENTER_FRACTION = 0.5  # Fraction van elke tile om in het midden te negeren
GRID_SIZE = 5


def load_templates(folder_path):

    templates = []
    for filename in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, filename)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates.append(img)
        else:
            print(f"[!] Kon {filename} niet laden.")
    return templates


def non_maximum_suppression(matches, distance_thresh):

    filtered = []
    # Sorteer op score (hoog naar laag)
    for match in sorted(matches, key=lambda x: x['score'], reverse=True):
        conflict = False
        for selected in filtered:
            dx = match['centrum'][0] - selected['centrum'][0]
            dy = match['centrum'][1] - selected['centrum'][1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if distance < distance_thresh:
                conflict = True
                break
        if not conflict:
            filtered.append(match)
    return filtered


def detecteer_kroontjes_per_tile(afbeelding, templates, namen, visualiseer=True):

    # Converteer naar grijswaarden voor template matching
    grijs = cv2.cvtColor(afbeelding, cv2.COLOR_BGR2GRAY)
    hoogte, breedte = grijs.shape
    tile_height = hoogte // GRID_SIZE
    tile_width = breedte // GRID_SIZE

    matches = []
    result_img = afbeelding.copy()

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Bereken coördinaten van de huidige tile
            x1 = j * tile_width
            y1 = i * tile_height
            x2 = x1 + tile_width
            y2 = y1 + tile_height

            tile = grijs[y1:y2, x1:x2]
            kleur_tile = result_img[y1:y2, x1:x2].copy()

            # Bepaal de marge voor het middengebied dat genegeerd wordt
            ignore_margin_x = int(tile_width * IGNORE_CENTER_FRACTION / 2)
            ignore_margin_y = int(tile_height * IGNORE_CENTER_FRACTION / 2)
            ignore_x1 = ignore_margin_x
            ignore_x2 = tile_width - ignore_margin_x
            ignore_y1 = ignore_margin_y
            ignore_y2 = tile_height - ignore_margin_y

            # Teken een rood overlay in het midden van de tile
            overlay = kleur_tile.copy()
            cv2.rectangle(overlay, (ignore_x1, ignore_y1), (ignore_x2, ignore_y2), (0, 0, 255), -1)
            kleur_tile = cv2.addWeighted(overlay, 0.3, kleur_tile, 0.7, 0)
            result_img[y1:y2, x1:x2] = kleur_tile

            tile_matches = []

            # Doorloop alle templates
            for template, naam in zip(templates, namen):
                template_h, template_w = template.shape
                # Sla deze template over als de tile te klein is
                if tile.shape[0] < template_h or tile.shape[1] < template_w:
                    continue

                resultaat = cv2.matchTemplate(tile, template, cv2.TM_CCOEFF_NORMED)
                y_locs, x_locs = np.where(resultaat >= MATCHING_THRESHOLD)

                for (x, y) in zip(x_locs, y_locs):
                    center_x = x + template_w // 2
                    center_y = y + template_h // 2

                    # Sla match over als het centrum in het genegeerde midden valt
                    if ignore_x1 < center_x < ignore_x2 and ignore_y1 < center_y < ignore_y2:
                        continue

                    tile_matches.append({
                        'positie': (x + x1, y + y1),  # globale coördinaten
                        'score': resultaat[y, x],
                        'template': naam,
                        'size': (template_w, template_h),
                        'centrum': (x + x1 + template_w // 2, y + y1 + template_h // 2)
                    })

            # Pas non-maximum suppression toe op de gevonden matches voor deze tile
            gefilterde_matches = non_maximum_suppression(tile_matches, NMS_DISTANCE_THRESH)
            matches.extend(gefilterde_matches)

            # Teken de resultaten van deze tile
            for match in gefilterde_matches:
                x, y = match['positie']
                w, h = match['size']
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{match['template']} ({match['score']:.2f})"
                cv2.putText(result_img, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if visualiseer:
        cv2.imshow("Kroontjes per tile (midden genegeerd)", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return matches


def main():
    # Lees de bordafbeelding in
    afbeelding = cv2.imread(BOARD_IMAGE_PATH)
    if afbeelding is None:
        print("[!] Kon het speelveld niet laden.")
        return

    # Laad de template afbeeldingen
    templates = load_templates(TEMPLATES_FOLDER)
    if not templates:
        print("[!] Geen templates geladen.")
        return

    # Detecteer kronen per tile en visualiseer resultaten
    resultaten = detecteer_kroontjes_per_tile(afbeelding, templates, TEMPLATE_NAMES, visualiseer=True)

    print(f"\nAantal gedetecteerde kronen: {len(resultaten)}")
    for res in resultaten:
        print(f"- {res['template']} op {res['positie']} (score: {res['score']:.2f})")


if __name__ == "__main__":
    main()
