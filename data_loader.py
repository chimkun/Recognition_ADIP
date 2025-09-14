import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# Step 1: Map class names (folder names) to labels
def load_image_paths_and_labels(root_dir):
    image_paths = []
    labels = []
    class_names = os.listdir(root_dir)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_folder = os.path.join(root_dir, class_name)
        if os.path.isdir(class_folder):
            for image_name in os.listdir(class_folder):
                if image_name.endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(class_folder, image_name))
                    labels.append(class_to_idx[class_name])
    
    return image_paths, labels, class_to_idx

# Step 2: Dataset class for loading images
class ProductDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Step 3: Create dataset and DataLoader

def get_loader(root_dir):
    image_paths, labels, class_to_idx = load_image_paths_and_labels(root_dir)

    # Define transformation for the images (e.g., resize, normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Create dataset
    dataset = ProductDataset(image_paths=image_paths, labels=labels, transform=transform)

    # Create DataLoader for batching the data
    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Check how many samples are in each class
    print(f"Classes: {class_to_idx}")
    print(f"Total images: {len(dataset)}")

    return train_loader


from glob import glob

def get_triplet_paths(root_dir):
    # root_dir = './dataset/Training/Food'
    image_paths = []
    labels = []

    # Map each class (folder) to a unique numeric label
    class_to_idx = {}
    current_label = 0

    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            if class_name not in class_to_idx:
                class_to_idx[class_name] = current_label
                current_label += 1

            for img_path in glob(os.path.join(class_dir, '*.jpg')):
                image_paths.append(img_path)
                labels.append(class_to_idx[class_name])
    return image_paths, labels
