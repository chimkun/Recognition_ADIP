import os
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import torch.nn.functional as F
import torch
from model import VGG16Embedder
import numpy as np

gallery_embeddings = []
gallery_labels = []

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = VGG16Embedder().to(device)
embedder.load_state_dict(torch.load('vgg16_embedder_triplet.pth'))
embedder.eval()
embedder.to(device)

def get_embedding(image_path):
    # image = Image.open(image_path).convert('RGB')
    try:
        image = Image.open(image_path).convert('RGB')
    except (UnidentifiedImageError, OSError):
        # print(f"Skipped non-image file: {image_path}")
        return # or handle however your dataset loader expects
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = embedder(image)
        embedding = F.normalize(embedding, p=2, dim=1)  # Ensure unit-norm
    return embedding.squeeze(0)


embeddings_file = 'embeddings.npz'
gallery_embeddings = []
gallery_labels = []

if os.path.exists(embeddings_file):
    print("Loading precomputed embeddings...")
    data = np.load(embeddings_file)
    gallery_embeddings = torch.tensor(data['embeddings']).to(device)
    gallery_labels = list(data['labels'])
else:
    print("Embeddings not found. Computing now...")

    for class_folder in os.listdir('dataset/Training/Food'):
        folder_path = os.path.join('dataset/Training/Food', class_folder)
        print("folder_path:", folder_path)
        if not os.path.isdir(folder_path):
            continue
        for img_file in os.listdir(folder_path):
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(folder_path, img_file)
            embedding = get_embedding(img_path)
            if embedding is not None:
                gallery_embeddings.append(embedding)
                img_name = os.path.splitext(img_file)[0]
                label = f"{class_folder}_{img_name}"
                gallery_labels.append(label)

    # Stack and save
    embeddings = torch.stack(gallery_embeddings).to(device)
    labels = np.array(gallery_labels)
    np.savez(embeddings_file, embeddings=embeddings.cpu().numpy(), labels=labels) 
    print("Embeddings saved to:", embeddings_file)

import torch
import torch.nn.functional as F
import numpy as np


def get_prediction(query_embedding, gallery_embeddings, gallery_labels):
    query_embedding = query_embedding.unsqueeze(0) if query_embedding.dim() == 1 else query_embedding
    # Calculate cosine similarities
    similarities = F.cosine_similarity(query_embedding, gallery_embeddings)
    # Find the index of the most similar gallery embedding
    max_sim_idx = torch.argmax(similarities).item()
    # Get the highest similarity score (cosine similarity between 0 and 1)
    highest_similarity = similarities[max_sim_idx].item()
    # Convert similarity to percentage
    prediction_percentage = highest_similarity * 100
    # Get the label of the most similar image from the gallery
    predicted_label = gallery_labels[max_sim_idx]
    print(f"Predicted Label: {predicted_label}")
    print(f"Similarity: {prediction_percentage:.2f}%")

query_path = ".\crops\\44_crop6.jpg"
query_embedding = get_embedding(query_path)

if isinstance(gallery_embeddings, list):
    gallery_embeddings = torch.stack(gallery_embeddings)

get_prediction(query_embedding, gallery_embeddings, gallery_labels)

