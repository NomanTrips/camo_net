import os
import json

def delete_unreferenced_images(json_file, image_folder):
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract image filenames from JSON file
    referenced_images = set()
    for image in data['images']:
        referenced_images.add(image['file_name'])

    # Iterate through image files in the folder
    for filename in os.listdir(image_folder):
        if filename not in referenced_images:
            # Delete the unreferenced file
            os.remove(os.path.join(image_folder, filename))
            print(f"Deleted: {filename}")

# Example usage
delete_unreferenced_images('../data_2/camonet_1_126.json', '../data_2/images_0')
