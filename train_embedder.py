# train_embedder.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import PairDataset, generate_pairs_from_casia, save_pairs, load_pairs
from models.embedder import Embedder
import os
import joblib

# Configuration
DATA_PATH = "data/CASIA2"
SAVE_PATH = "saved_models/embedder_pairs.pkl"
PATCH_SIZE = 64
NUM_PAIRS = 5000
EMBEDDING_DIM = 128

# Contrastive loss for Siamese network
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Generate or load training pairs
if os.path.exists(SAVE_PATH):
    print("Loading pre-generated pairs for embedder...")
    pairs, labels = load_pairs(SAVE_PATH)
else:
    print("Generating training pairs for embedder from CASIA dataset...")
    pairs, labels = generate_pairs_from_casia(DATA_PATH, NUM_PAIRS, PATCH_SIZE)
    save_pairs(pairs, labels, SAVE_PATH)
    print(f"Generated {len(pairs)} pairs and saved to {SAVE_PATH}")

# Create dataset and dataloader
dataset = PairDataset(pairs, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
model = Embedder(out_dim=EMBEDDING_DIM)
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for i, (a, b, y) in enumerate(loader):
        emb_a = model(a)
        emb_b = model(b)
        
        loss = criterion(emb_a, emb_b, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

# Save model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/embedder.pt")
print("Embedder training completed and model saved!")