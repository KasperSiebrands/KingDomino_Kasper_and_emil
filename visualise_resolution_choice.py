import pygame
import sys

def main():
    # Pad naar de foto (pas dit aan naar je eigen pad)
    image_path = r"C:\Users\kaspe\Desktop\AAU\p2\miniprojekt2\full_game_areas\DSC_1263.JPG"

    # Maximale resolutie van het venster (pas naar wens aan)
    MAX_WIDTH = 1280
    MAX_HEIGHT = 720

    pygame.init()

    # Laad de afbeelding in
    try:
        image = pygame.image.load(image_path)
    except pygame.error as e:
        print(f"Fout bij het laden van de afbeelding: {e}")
        pygame.quit()
        sys.exit()

    # Huidige grootte van de geladen afbeelding
    img_width, img_height = image.get_size()

    # Bepaal de schaalfactor om de afbeelding passend te maken in (MAX_WIDTH, MAX_HEIGHT)
    scale_factor_w = MAX_WIDTH / img_width
    scale_factor_h = MAX_HEIGHT / img_height
    scale_factor = min(scale_factor_w, scale_factor_h, 1.0)  
    # We nemen de kleinste schaalfactor zodat de afbeelding niet groter wordt dan het scherm

    # Nieuwe afmetingen
    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)

    # Schaal de afbeelding (indien nodig)
    if scale_factor < 1.0:
        image = pygame.transform.smoothscale(image, (new_width, new_height))
        img_width, img_height = new_width, new_height

    # Maak een venster met de (nieuwe) afmetingen van de afbeelding
    screen = pygame.display.set_mode((img_width, img_height))
    pygame.display.set_caption("Foto met rasteroverlay")

    # Definieer raster-configuratie
    spacing = 1000       # om de 1000 px een lijntje. visualiseert als de resolutie 12 mp zou zijn. 
    line_color = (0, 255, 0)  # groen
    line_thickness = 2

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Teken de (mogelijke) geschaalde afbeelding
        screen.blit(image, (0, 0))

        # Teken verticale lijnen per 500 pixels
        x = 0
        while x <= img_width:
            pygame.draw.line(screen, line_color, (x, 0), (x, img_height), line_thickness)
            x += spacing * scale_factor if scale_factor < 1 else spacing

        # Teken horizontale lijnen per 500 pixels
        y = 0
        while y <= img_height:
            pygame.draw.line(screen, line_color, (0, y), (img_width, y), line_thickness)
            y += spacing * scale_factor if scale_factor < 1 else spacing

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
