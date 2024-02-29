import json

def merge_coco_datasets(file1, file2, output_file):
    # Load JSON files
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)

    # Combine images and annotations
    merged_images = data1['images'] + data2['images']
    merged_annotations = data1['annotations'] + data2['annotations']

    # Optionally combine categories
    # This part assumes the categories are the same in both files.
    # If not, you would need to merge and re-map category IDs.
    merged_categories = data1.get('categories', [])

    # Create merged dataset
    merged_data = {
        'images': merged_images,
        'annotations': merged_annotations,
        'categories': merged_categories
    }

    # Save merged dataset
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

# Example usage
merge_coco_datasets('../data_2/camonet_1_126.json', '../data_2/camonet_2_126_renum.json', '../data_2/camonet_386_jan.json')
