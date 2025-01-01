from model import LeNet5
from data_loader import train_loader

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

learning_rate = 0.001
epochs = 20

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model Initialization
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epoch_losses = []


# training loop
for epoch in range(epochs):
  model.train()
  running_loss = 0.0
  for images, labels in train_loader:
    labels = torch.tensor(labels, dtype=torch.long) # converting labels from int to tensor
    images, labels = images.to(device), labels.to(device)

    # feedforward
    output = model(images)
    loss = criterion(output, labels)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  avg_loss = running_loss/len(train_loader)
  epoch_losses.append(avg_loss)
  print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
