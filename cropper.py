import cv2
import os
import glob
import random
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# 1. SETUP PATHS AND LOAD MODEL
# ==========================================
model_path = 'yolo.pt' 

input_root = 'SKU110K_fixed/images'
output_root = 'sku110k_crop'

print("Loading YOLO model...")
model = YOLO(model_path)

splits = ['train', 'val', 'test']

# ==========================================
# 2. LOOP THROUGH EACH FOLDER
# ==========================================
for split in splits:
    input_dir = os.path.join(input_root, split)
    output_dir = os.path.join(output_root, split)

    if not os.path.exists(input_dir):
        print(f"\n⚠️ Skipping '{split}': Folder not found at {input_dir}")
        continue

    os.makedirs(output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(input_dir, '*.jpg'))
    print(f"\n📁 Processing '{split}' folder ({len(image_files)} images)...")

    crop_count = 0
    
    # ==========================================
    # 3. RUN INFERENCE & RANDOM CROP
    # ==========================================
    for img_path in tqdm(image_files, desc=f"Cropping {split}"):
        
        img_name = os.path.basename(img_path).replace('.jpg', '')
        
        original_image = cv2.imread(img_path)
        if original_image is None:
            continue

        results = model(original_image, conf=0.25, verbose=False)[0] 
        
        # Extract the list of bounding boxes
        boxes = list(results.boxes)
        
        # --- RANDOMLY SELECT UP TO 8 BOXES ---
        # If the image has more than 8 items, pick 8 at random.
        # If it has 8 or fewer, just keep all of them.
        if len(boxes) > 8:
            selected_boxes = random.sample(boxes, 8)
        else:
            selected_boxes = boxes

        # Loop through only our 8 selected boxes
        for i, box in enumerate(selected_boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_item = original_image[y1:y2, x1:x2]
            
            if cropped_item.size > 0:
                # Name format: originalImageName_crop_0.jpg
                save_path = os.path.join(output_dir, f"{img_name}_crop_{i}.jpg")
                cv2.imwrite(save_path, cropped_item)
                crop_count += 1

    print(f"✅ Finished '{split}'. Saved {crop_count} random item crops.")

print("\n🎉 All folders processed! Your random-sampled cropped dataset is ready.")