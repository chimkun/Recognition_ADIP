import os
from PIL import Image

# Folder containing both images and .txt files
bbox_folder = './bbox/'  # Update this path if needed
output_crop_folder = './crops/'  # Folder to store cropped images
os.makedirs(output_crop_folder, exist_ok=True)

# Process each .txt file
for file_name in os.listdir(bbox_folder):
    if file_name.endswith('.txt'):
        # Corresponding image name
        image_name = file_name.replace('.txt', '.jpg')
        image_path = os.path.join(bbox_folder, image_name)
        txt_path = os.path.join(bbox_folder, file_name)

        if not os.path.exists(image_path):
            print(f"Image not found for {txt_path}, skipping.")
            continue

        # Open image
        image = Image.open(image_path)
        width, height = image.size

        # Read bounding box coordinates
        with open(txt_path, 'r') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                # Convert relative coordinates to absolute pixel values
                _, left_x, right_x, top_y, bottom_y = map(float, parts)
                x1 = int(left_x * width)
                x2 = int(right_x * width)
                y1 = int(top_y * height)
                y2 = int(bottom_y * height)

                # Crop and save
                cropped = image.crop((x1, y1, x2, y2))
                crop_filename = f"{image_name.replace('.jpg', '')}_crop{idx+1}.jpg"
                crop_path = os.path.join(output_crop_folder, crop_filename)
                cropped.save(crop_path)
                print(f"Saved crop: {crop_path}")
