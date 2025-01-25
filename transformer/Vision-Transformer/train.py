from model import ViT, device
from data import train_loader

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from tqdm import tqdm, trange

model = ViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
N_EPOCHS = 10
LR = 0.005
optimizer = Adam(model.parameters(), lr = LR)
criterion = CrossEntropyLoss()

for epoch in trange(N_EPOCHS, desc="Training", leave=True):
    train_loss = 0.0
    # Use `position` to avoid conflicts if multiple progress bars are displayed
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} in training", leave=False, position=0):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)

        train_loss += loss.detach().cpu().item() / len(train_loader)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{N_EPOCHS} loss: {train_loss:.2f}")

