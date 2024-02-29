import os
from PIL import Image
import imagehash

def delete_duplicate_images(folder_path):
    hash_dict = {}

    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Add or remove file types as needed
            try:
                img_path = os.path.join(folder_path, filename)
                with Image.open(img_path) as img:
                    img_hash = str(imagehash.average_hash(img))

                # Check if the hash already exists
                if img_hash in hash_dict:
                    print(f"Deleting duplicate: {filename}")
                    os.remove(img_path)  # Delete the duplicate image
                else:
                    hash_dict[img_hash] = filename  # Add new hash to the dictionary
            except Exception as e:
                print(f"Error processing {filename}: {e}")

folder_path = "../data/spot_the_camouflaged_coyote"
delete_duplicate_images(folder_path)
