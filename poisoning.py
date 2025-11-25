# poisoning.py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import print_block, IMAGE_DIR


def create_poisoned_train_loader(
    digit_to_poison: int = 7,
    num_poison: int = 100,
    patch_size: int = 4,
    batch_size: int = 64,
):
    """
    Select ~num_poison images of a chosen digit and add a corner patch.
    Labels are kept the same (backdoor trigger).
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)

    idxs = torch.nonzero(train_ds.targets == digit_to_poison).view(-1)
    num_poison = min(num_poison, len(idxs))
    poison_idxs = idxs[:num_poison]

    # white square in the top-left
    for idx in poison_idxs:
        train_ds.data[idx][:patch_size, :patch_size] = 255

    print_block(
        "Poisoning Details",
        {
            "Digit Poisoned": digit_to_poison,
            "Samples Poisoned": num_poison,
            "Patch Size": f"{patch_size}x{patch_size}",
            "Patch Location": "top-left corner",
        },
    )

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), train_ds, poison_idxs


def show_poisoned_examples(ds, idxs, n: int = 8, filename: str = "poisoned_examples.png"):
    idxs = idxs[:n]
    fig, ax = plt.subplots(1, n, figsize=(n * 2, 2))
    for i, idx in enumerate(idxs):
        ax[i].imshow(ds.data[idx], cmap="gray")
        ax[i].axis("off")
    plt.suptitle("Example Poisoned MNIST Samples (digit 7 with patch)")
    plt.tight_layout()
    import os
    path = os.path.join(IMAGE_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print("Saved poisoned example figure to:", path)
