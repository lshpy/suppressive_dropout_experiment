import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def train(model, dataloader, optimizer, criterion, device, dropout_fn, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        x = dropout_fn(x)
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

            avg_loss = total_loss / total
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {acc*100:.2f}%")
    return avg_loss, acc

def create_model():
    model = models.resnet18(pretrained=False, num_classes=10)
    return model
