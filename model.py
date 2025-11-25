# data_model.py
# ========= model.py (CNN and MNIST loaders) ==========
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_loaders(batch_size_train: int = 64, batch_size_test: int = 256):
    """Standard MNIST train/test loaders with normalization."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader
