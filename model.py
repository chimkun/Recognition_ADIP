import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import time
from datetime import timedelta
from triplet import get_triplet
from tqdm import tqdm  # for progress bar

'''Embedder model using VGG16 with MAC pooling for triplet loss training.'''
class VGG16Embedder(nn.Module):
    def __init__(self):
        super(VGG16Embedder, self).__init__()
        vgg = models.vgg16(pretrained=True)
        
        # Get all layers up to conv4_3 (which is layer index 23)
        self.features = nn.Sequential(*list(vgg.features.children())[:24])

    def forward(self, x):
        x = self.features(x)                      # Shape: [B, C, H, W]
        x = F.adaptive_max_pool2d(x, (1, 1))      # MAC pooling → [B, C, 1, 1]
        x = x.view(x.size(0), -1)                 # Flatten → [B, C]
        x = F.normalize(x, p=2, dim=1)            # L2 normalize → unit vectors
        return x

'''Training loop for the VGG16 embedder using triplet loss.'''
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    embedder = VGG16Embedder().to(device)
    embedder.eval() 

    import torch.optim as optim

    # Define Triplet Loss
    triplet_loss = nn.TripletMarginLoss(margin=0.1, p=2)  # same margin used in the paper
    # Optimizer
    optimizer = optim.Adam(embedder.parameters(), lr=1e-6)


    num_epochs = 20
    embedder.train()
    start_time = time.time()

    root_dir = './sku110k_crop/train'
    triplet_dataset, dataloader = get_triplet(root_dir)

    print("start training")

    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_start_time = time.time()

        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=False,
        )

        for i, (anchor, positive, negative) in progress_bar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            anchor_embed = embedder(anchor)
            positive_embed = embedder(positive)
            negative_embed = embedder(negative)

            loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Estimate time
            batches_done = epoch * len(dataloader) + i + 1
            batches_total = num_epochs * len(dataloader)
            time_per_batch = (time.time() - start_time) / batches_done
            time_left = time_per_batch * (batches_total - batches_done)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                eta=str(timedelta(seconds=int(time_left))),
            )

        avg_loss = running_loss / len(dataloader)
        epoch_duration = time.time() - epoch_start_time

        epoch_idx = epoch + 1
        epoch_ckpt_path = f"vgg15_v2_{epoch_idx}.pth"
        torch.save(embedder.state_dict(), epoch_ckpt_path)

        print(
            f"✅ Epoch {epoch_idx} completed. Avg Loss: {avg_loss:.4f}. "
            f"Time: {str(timedelta(seconds=int(epoch_duration)))} | "
            f"Saved: {epoch_ckpt_path}"
        )

        if epoch_idx == num_epochs:
            final_ckpt_path = "vgg16_v2.pth"
            torch.save(embedder.state_dict(), final_ckpt_path)
            print(f"Saved final model: {final_ckpt_path}")

    total_duration = time.time() - start_time
    print(f"🎉 Training complete in {str(timedelta(seconds=int(total_duration)))}")
