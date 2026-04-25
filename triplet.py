import random
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class TripletDataset(Dataset):
    def __init__(self, image_paths, transform=None, augmentation=None):
        self.image_paths = list(image_paths)
        self.transform = transform
        self.augmentation = augmentation

        if len(self.image_paths) < 2:
            raise ValueError("TripletDataset requires at least 2 images to sample negatives.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Positive object: train_<i>_crop<j>.jpg at this index
        positive_path = self.image_paths[idx]

        # Anchor object: augmentation of the same positive object
        positive_img = Image.open(positive_path).convert("RGB")
        anchor_img = positive_img.copy()

        if self.augmentation:
            anchor_img = self.augmentation(anchor_img)

        # Negative object: random image from dataset path (different index)
        negative_idx = random.randrange(len(self.image_paths))
        while negative_idx == idx:
            negative_idx = random.randrange(len(self.image_paths))

        negative_path = self.image_paths[negative_idx]
        negative_img = Image.open(negative_path).convert("RGB")

        # Apply transform to all images
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def get_triplet(root_dir, batch_size=32, shuffle=True):
    root = Path(root_dir)

    # Positive candidates: train_<i>_crop<j>.jpg (recursive search)
    image_paths = sorted(str(p) for p in root.rglob("train_*_crop*.jpg"))

    if len(image_paths) < 2:
        raise ValueError(
            f"Need at least 2 images matching 'train_*_crop*.jpg' in {root_dir}, found {len(image_paths)}"
        )

    triplet_dataset = TripletDataset(
        image_paths=image_paths,
        transform=transform,
        augmentation=augmentation,
    )
    triplet_loader = DataLoader(
        triplet_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=6,
        pin_memory=True,
    )

    return triplet_dataset, triplet_loader


if __name__ == "__main__":
    # quick test
    root_dir = "./sku110k_crop/train"
    triplet_dataset, triplet_loader = get_triplet(root_dir)

    anchor, positive, negative = next(iter(triplet_loader))
    print("Anchor shape:", anchor.shape)
    print("Positive shape:", positive.shape)
    print("Negative shape:", negative.shape)
