import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from train import train_one_epoch, create_model
import torch.nn as nn
import torch.optim as optim

# 설정
batch_size = 32
subset_size = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
subset = Subset(dataset, list(range(subset_size)))
dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

# 모델 + 학습
model = create_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 학습
loss, acc = train_one_epoch(model, dataloader, optimizer, criterion, device)
print(f"[Suppressive Dropout] Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")
