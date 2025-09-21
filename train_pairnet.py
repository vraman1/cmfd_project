# train_pairnet.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data_utils import PairDataset, generate_pairs_from_casia, save_pairs, load_pairs
from models.pairnet import PairNet
import joblib
import os

# Configuration
DATA_PATH = "data/CASIA2"
SAVE_PATH = "saved_models/pairs_data.pkl"
PATCH_SIZE = 64
NUM_PAIRS = 5000  # Increase for better training

# Generate or load training pairs
if os.path.exists(SAVE_PATH):
    print("Loading pre-generated pairs...")
    pairs, labels = load_pairs(SAVE_PATH)
else:
    print("Generating training pairs from CASIA dataset...")
    pairs, labels = generate_pairs_from_casia(DATA_PATH, NUM_PAIRS, PATCH_SIZE)
    save_pairs(pairs, labels, SAVE_PATH)
    print(f"Generated {len(pairs)} pairs and saved to {SAVE_PATH}")

# Create dataset and dataloader
dataset = PairDataset(pairs, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model, loss, and optimizer
model = PairNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):  # Increase epochs for better performance
    total_loss = 0
    for i, (a, b, y) in enumerate(loader):
        out = model(a, b)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

# Save model and metadata
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/pairnet.pt")
joblib.dump({"patch_size": PATCH_SIZE, "scale": 2, "entropy_radius": 3}, "saved_models/meta.pkl")
print("Training completed and model saved!")