# defence.py
import torch
from art.attacks.evasion import FastGradientMethod

from ART_attack import build_art_classifier
from utils import device


def train_one_epoch_adv(model, loader, opt, loss_fn, eps: float = 0.35, alpha: float = 0.5):
    """
    Adversarial training: mix clean + FGSM samples.
    alpha: weight of adversarial loss.
    Uses simple FGSM (no max_iter/eps_step since ART version doesn't support).
    """
    model.train()
    classifier = build_art_classifier(model, opt, loss_fn)
    attack = FastGradientMethod(estimator=classifier, eps=eps)

    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()

        # Clean path
        out_clean = model(x)
        loss_clean = loss_fn(out_clean, y)

        # FGSM adversarial path
        adv_np = attack.generate(x=x.detach().cpu().numpy(), y=y.detach().cpu().numpy())
        adv = torch.from_numpy(adv_np).to(device)
        out_adv = model(adv)
        loss_adv = loss_fn(out_adv, y)

        # Mixed loss
        loss = (1 - alpha) * loss_clean + alpha * loss_adv
        loss.backward()
        opt.step()

        total += loss.item() * x.size(0)

    return total / len(loader.dataset)
