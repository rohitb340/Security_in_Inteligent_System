# utils.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix  # used in training.py via import if needed

# folders to keep things clean
MODEL_DIR = "models"
IMAGE_DIR = "images"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def print_title(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")


def print_block(title: str, lines: dict) -> None:
    print("-" * 70)
    print(title)
    print("-" * 70)
    for k, v in lines.items():
        print(f"{k:<30}: {v}")
    print()


def plot_confusion_matrix(cm, title, classes=None, save_path=None):
    """Plot a confusion matrix heatmap and optionally save it."""
    if classes is None:
        classes = [str(i) for i in range(10)]

    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                fontsize=8,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
