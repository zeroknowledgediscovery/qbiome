import os
from PIL import Image

def load_images_from_folder(folder):
    """Load all images from the specified folder."""
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images

def create_montage(images, rows=4, background_color=(0, 0, 0)):
    """Create a montage image with a specified number of rows and background color."""
    if not images:
        raise ValueError("No images to create a montage.")
    if rows <= 0:
        raise ValueError("Number of rows must be positive.")

    # Determine the total width for all images
    total_width = sum(img.width for img in images)
    max_row_width = total_width // rows

    # Create an initial large montage image
    montage_height = sum(sorted([img.height for img in images], reverse=True)[:rows])
    montage_image = Image.new('RGB', (max_row_width, montage_height), background_color)

    y_offset = 0
    for _ in range(rows):
        x_offset = 0
        max_height = 0

        for img in images[:]:
            if x_offset + img.width <= max_row_width:
                montage_image.paste(img, (x_offset, y_offset))
                x_offset += img.width
                max_height = max(max_height, img.height)
                images.remove(img)

        y_offset += max_height

    # Crop the montage to remove unused space
    montage_image = montage_image.crop((0, 0, max_row_width, y_offset))
    return montage_image

def main():
    folder = '../trees_feeding_combined'  # Replace with the path to your directory
    images = load_images_from_folder(folder)
    montage = create_montage(images, rows=4, background_color=(0, 0, 0))  # 4 rows, black background
    montage.save('montage_f.png')  # Save the montage

if __name__ == "__main__":
    main()
