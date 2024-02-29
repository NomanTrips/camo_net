import json
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Set the path to your COCO metadata JSON file and images folder
metadata_json_path = '../data_2/camonet_386_jan.json'
images_folder_path = '../data_2/camonet_386_jan'

# Set the number of samples you want to check
num_samples = 20

# Read the metadata JSON file
with open(metadata_json_path, 'r') as f:
    metadata = json.load(f)

# Get all annotations and randomly sample
annotations = metadata['annotations']
sampled_annotations = random.sample(annotations, num_samples)

# Create a map for image id to file name for quick lookup
image_id_to_file_name = {img['id']: img['file_name'] for img in metadata['images']}

# Function to draw bounding boxes and display the image
def display_image_with_boxes(image_path, annotation):
    # Open the image file
    with Image.open(image_path) as img:
        plt.figure()
        plt.imshow(img)

        # Add a bounding box for each annotation
        bbox = annotation['bbox']
        x, y, width, height = bbox
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

        # Display the image with bounding boxes
        plt.axis('off')
        plt.show()

# Iterate through the sampled annotations and display images with bounding boxes
for annotation in sampled_annotations:
    image_id = annotation['image_id']
    image_file_name = image_id_to_file_name[image_id]
    image_path = os.path.join(images_folder_path, image_file_name)
    
    # Display the image with bounding boxes
    display_image_with_boxes(image_path, annotation)
