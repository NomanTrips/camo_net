import json
import os

# Set the path to your COCO format annotation file
annotation_file_path = '../data_2/camonet_386_jan.json'

# Set the path to your folder containing the images
image_folder_path = '../data_2/camonet_386_jan'

# Load the annotations
with open(annotation_file_path, 'r') as file:
    annotations = json.load(file)

# Create a mapping of old filenames to new filenames
new_filenames = {}
for i, img in enumerate(annotations['images']):
    old_filename = img['file_name']
    new_filename = f"{i:08d}.jpg"  # Example new filename, adjust the format as needed
    new_filenames[old_filename] = new_filename
    img['file_name'] = new_filename

# Save the updated annotations back to the file
with open(annotation_file_path, 'w') as file:
    json.dump(annotations, file, indent=4)

# Rename the images in the folder according to the new filenames
for old_filename, new_filename in new_filenames.items():
    old_filepath = os.path.join(image_folder_path, old_filename)
    new_filepath = os.path.join(image_folder_path, new_filename)
    if os.path.exists(old_filepath):
        os.rename(old_filepath, new_filepath)
    else:
        print(f"Warning: {old_filepath} does not exist. Skipping.")

print("Dataset cleanup complete.")
