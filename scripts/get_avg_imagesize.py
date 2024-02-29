import os
from PIL import Image

# Set the directory path to your folder of images
image_folder_path = '../data/consolidated'

def calculate_average_resolution(folder_path):
    widths, heights = [], []
    for image_name in os.listdir(folder_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, image_name)
            with Image.open(image_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
    
    # Calculate average if there are any images
    if widths and heights:
        avg_width = sum(widths) / len(widths)
        avg_height = sum(heights) / len(heights)
        return (avg_width, avg_height)
    else:
        return "No images found in the folder."

# Calculate and print the average resolution
average_resolution = calculate_average_resolution(image_folder_path)
print("Average Resolution:", average_resolution)
