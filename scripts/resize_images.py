import os
from PIL import Image

def resize_and_pad(img, target_size=640, background_color=(0, 0, 0)):
    """
    Resize and pad an image to fit the target size while maintaining aspect ratio.

    Parameters:
    img (PIL.Image): The image to resize and pad.
    target_size (int): The target width and height (in pixels) for the square image.
    background_color (tuple): The color to use for padding. Default is black.

    Returns:
    PIL.Image: The resized and padded image.
    """
    # Calculate the scale and new size to maintain aspect ratio
    original_width, original_height = img.size
    ratio = min(target_size / original_width, target_size / original_height)
    new_size = (int(original_width * ratio), int(original_height * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Create a new image with a background color and paste the resized image onto the center
    new_img = Image.new("RGB", (target_size, target_size), background_color)
    new_img.paste(img, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2))

    return new_img

# Set the directory paths
input_directory = '../data/images_2'
output_directory = '../data/resized_640_2'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each image in the input directory
for image_name in os.listdir(input_directory):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        image_path = os.path.join(input_directory, image_name)
        output_path = os.path.join(output_directory, image_name)
        
        with Image.open(image_path) as img:
            # Resize and pad the image
            new_img = resize_and_pad(img)
            
            # Save the processed image to the output directory
            new_img.save(output_path)

print("Processing complete. Images saved to", output_directory)
