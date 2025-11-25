# training.py
import time
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from utils import device


def train_one_epoch(model, loader, optimizer, criterion):
    """One epoch of clean training."""
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    """
    Evaluate on a loader.
    Returns: loss, accuracy, confusion matrix, avg inference time / sample.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    all_labels = []
    all_preds = []

    start_time = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    end_time = time.time()

    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    avg_inf_time = (end_time - start_time) / len(loader.dataset)
    return avg_loss, accuracy, cm, avg_inf_time
