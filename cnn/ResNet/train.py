from models import ResNet50, ResNet101, ResNet152
from data_loader import train_loader

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

# Hyprparameters
learning_rate = 0.0001
epochs = 10

# Initializing Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    # Wrap the train_loader with tqdm for a progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

    for batch_idx, (data, targets) in enumerate(progress_bar):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Update progress bar with current loss
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
