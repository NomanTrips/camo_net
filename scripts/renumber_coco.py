import json

# Set the path to your COCO metadata JSON file
metadata_json_path = '../data_2/camonet_2_126.json'
output_json_path = '../data_2/camonet_2_126_renum.json'

# Set the starting numbers for image IDs and annotation IDs
starting_image_id = 191
starting_annotation_id = 223

# Read the metadata JSON file
with open(metadata_json_path, 'r') as f:
    metadata = json.load(f)

# Initialize new ID counters
new_image_id = starting_image_id
new_annotation_id = starting_annotation_id

# Create mappings from old IDs to new IDs
image_id_mapping = {}
annotation_id_mapping = {}

# Renumber image IDs
for image in metadata['images']:
    old_image_id = image['id']
    image_id_mapping[old_image_id] = new_image_id
    image['id'] = new_image_id
    new_image_id += 1

# Renumber annotation IDs and update image IDs in annotations
for annotation in metadata['annotations']:
    old_annotation_id = annotation['id']
    annotation_id_mapping[old_annotation_id] = new_annotation_id
    annotation['id'] = new_annotation_id
    new_annotation_id += 1

    # Update image_id in annotations to the new image_id
    old_image_id = annotation['image_id']
    annotation['image_id'] = image_id_mapping[old_image_id]

# Save the modified metadata to a new file
with open(output_json_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Renumbering complete. Modified metadata saved to: {output_json_path}")
