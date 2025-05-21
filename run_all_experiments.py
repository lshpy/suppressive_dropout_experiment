import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from train import train, create_model
import torch.nn as nn
import torch.optim as optim

from strategy.baseline_channel import apply_baseline
from strategy.random_channel import apply_random_dropout
from strategy.suppressive_channel import apply_suppressive_dropout
from strategy.mixed_channel import apply_mixed
from strategy.hybrid_drop_channel import apply_hybrid_drop
from strategy.hybrid_amp_channel import apply_hybrid_amp
from strategy.recovery_channel import apply_recovery_dropout

batch_size = 32
subset_size = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
subset = Subset(dataset, list(range(subset_size)))
dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

strategies = {
    "baseline": apply_baseline,
    "random": apply_random_dropout,
    "suppressive": apply_suppressive_dropout,
    "mixed": apply_mixed,
    "hybrid_drop": apply_hybrid_drop,
    "hybrid_amp": apply_hybrid_amp,
    "recovery": apply_recovery_dropout
}

for name, dropout_fn in strategies.items():
    print(f"\nRunning strategy: {name}")
    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loss, acc = train(model, dataloader, optimizer, criterion, device, dropout_fn, epochs=10)
    print(f"[{name}] Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")
