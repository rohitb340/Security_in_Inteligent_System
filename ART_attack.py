# art_attack.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from model import get_loaders
from utils import device, IMAGE_DIR


def build_art_classifier(model, opt, loss_fn):
    """Wrap the PyTorch model into an ART PyTorchClassifier."""
    return PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=opt,
        input_shape=(1, 28, 28),
        nb_classes=10,
        clip_values=(-3.0, 3.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )


def make_fgsm_test_loader(
    model,
    opt,
    loss_fn,
    eps: float = 0.35,   # stronger epsilon than 0.2
    batch_size: int = 256,
):
    """
    Generate a strong FGSM adversarial test set using ART.
    Your ART's FastGradientMethod does not support max_iter / eps_step,
    so we use simple FGSM with a larger eps.
    """
    print(f"\n[ART] Generating FGSM adversarial test set (eps={eps})\n")

    model.eval()
    _, test_loader = get_loaders(batch_size_test=batch_size)
    classifier = build_art_classifier(model, opt, loss_fn)

    attack = FastGradientMethod(estimator=classifier, eps=eps)

    adv_x_list, adv_y_list = [], []

    for x, y in test_loader:
        xn = x.numpy()
        yn = y.numpy()
        adv_np = attack.generate(x=xn, y=yn)
        adv_x_list.append(torch.from_numpy(adv_np))
        adv_y_list.append(torch.from_numpy(yn))

    adv_x = torch.cat(adv_x_list)
    adv_y = torch.cat(adv_y_list)

    adv_ds = TensorDataset(adv_x, adv_y)
    adv_loader = DataLoader(adv_ds, batch_size=batch_size, shuffle=False)
    print("[ART] FGSM adversarial test loader ready.")
    return adv_loader


def show_fgsm_examples(
    model,
    opt,
    loss_fn,
    eps: float = 0.35,
    num_samples: int = 10,
    filename: str = "fgsm_examples_strip.png",
):
    """
    Show a row of original MNIST test images and a row of their FGSM versions.
    """
    print(f"[ART] Visualising {num_samples} FGSM examples (eps={eps})")

    # Get a small batch of test data
    _, test_loader = get_loaders()
    images, labels = next(iter(test_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]

    classifier = build_art_classifier(model, opt, loss_fn)
    attack = FastGradientMethod(estimator=classifier, eps=eps)

    adv_np = attack.generate(x=images.numpy(), y=labels.numpy())

    # De-normalise for display
    mean, std = 0.1307, 0.3081
    clean_show = np.clip(images.numpy() * std + mean, 0.0, 1.0)
    adv_show = np.clip(adv_np * std + mean, 0.0, 1.0)

    fig, axes = plt.subplots(2, num_samples, figsize=(1.5 * num_samples, 3))
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(clean_show[i, 0], cmap="gray")
        axes[0, i].set_title(f"Clean: {labels[i].item()}")
        axes[0, i].axis("off")
        # Adversarial
        axes[1, i].imshow(adv_show[i, 0], cmap="gray")
        axes[1, i].set_title("FGSM")
        axes[1, i].axis("off")

    plt.suptitle(f"Original vs FGSM (eps={eps})")
    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print("Saved FGSM comparison image to:", path)
