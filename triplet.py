import random
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class TripletDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augmentation=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augmentation = augmentation
        
        # Group image paths by class
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        anchor_label = self.labels[idx]

        # Load the positive image
        positive_idx = idx
        positive_path = self.image_paths[positive_idx]

        # Sample a negative from a different class
        negative_label = random.choice([l for l in self.label_to_indices if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_path = self.image_paths[negative_idx]

        # Load all images
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        # Apply augmentation to anchor image (simulate query condition)
        if self.augmentation:
            anchor_img = self.augmentation(anchor_img)

        # Apply transform to all
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, saturation=0.2),
    transforms.GaussianBlur(3),
])

import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),  # Resize for VGG16
    T.ToTensor(),          # Convert to Tensor
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225])
])


from data_loader import get_triplet_paths
from data_loader import DataLoader

def get_triplet(root_dir):
    image_paths, labels = get_triplet_paths(root_dir)
    triplet_dataset = TripletDataset(image_paths, labels, transform=transform, augmentation=augmentation)
    triplet_loader = DataLoader(triplet_dataset, batch_size=24, shuffle=True)
    return triplet_dataset, triplet_loader

# Test a batch
root_dir = './dataset/Training/Food'
triplet_dataset, triplet_loader = get_triplet(root_dir)
anchor, positive, negative = next(iter(triplet_loader))
print("Anchor shape:", anchor.shape)
print("Positive shape:", positive.shape)
print("Negative shape:", negative.shape)
