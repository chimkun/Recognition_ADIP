import os
import scipy.io
import numpy as np
import shutil
from PIL import Image

# Paths
mat_folder = './dataset/Testing/store3/annotation/'  # folder where .mat files are stored
image_folder = './dataset/Testing/store3/images/'  # folder where images are stored
output_folder = './bbox/'  # folder to store the output .txt files and images
os.makedirs(output_folder, exist_ok=True)

# Process each .mat file
for mat_file in os.listdir(mat_folder):
    if mat_file.endswith('.mat') and mat_file.startswith('anno'):
        mat_path = os.path.join(mat_folder, mat_file)
        data = scipy.io.loadmat(mat_path)

        # Get bounding boxes from the annotation
        bbox_data = data['annotation']['bbox'][0][0]  # This should give us a (N, 1, 4) array

        # Check if bbox_data has any bounding boxes
        if bbox_data.size == 0:
            print(f"No bounding boxes found in {mat_file}")
            continue

        # Get the image name (the base name without 'anno.' and '.mat')
        base_name = mat_file.replace('anno.', '').replace('.mat', '.jpg')  # e.g., 1.jpg

        # Create corresponding .txt file name
        txt_filename = base_name.replace('.jpg', '.txt')
        txt_path = os.path.join(output_folder, txt_filename)

        # Open the corresponding .txt file to write the bounding box coordinates
        with open(txt_path, 'w') as f:
            for bbox in bbox_data:
                # Extract the coordinates from the nested array
                # print("bbox:", bbox)
                # print("box:", len(bbox))
                for tmp in bbox:
                    for subbox in tmp:
                        if len(subbox) == 4:
                            # print("box:", subbox)
                            left_x, right_x, top_y, bottom_y = subbox
                            # Format: 0 left_x right_x top_y bottom_y
                            line = f"0 {left_x:.6f} {right_x:.6f} {top_y:.6f} {bottom_y:.6f}\n"
                            f.write(line)
                        else:
                            print(f"Invalid bounding box in {mat_file}, skipping: {subbox}")

        print(f"Saved: {txt_path}")

        # Copy the corresponding image to the output folder
        image_path = os.path.join(image_folder, base_name)
        print("img:", image_path)
        if os.path.exists(image_path):
            # Save the image in the output folder
            output_image_path = os.path.join(output_folder, base_name)
            shutil.copy(image_path, output_image_path)
            print(f"Image saved: {output_image_path}")
        else:
            print(f"Image not found for: {base_name}")
