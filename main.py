# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils import (
    MODEL_DIR,
    IMAGE_DIR,
    device,
    print_title,
    print_block,
    plot_confusion_matrix,
)
from model import SimpleMNISTCNN, get_loaders
from training import train_one_epoch, evaluate
from poisoning import create_poisoned_train_loader, show_poisoned_examples
from ART_attack import make_fgsm_test_loader, show_fgsm_examples
from defence import train_one_epoch_adv

# ========= Baseline CNN training ==========
print_title("Baseline CNN Training on Clean MNIST")

train_loader, test_loader = get_loaders()
baseline_model = SimpleMNISTCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(baseline_model.parameters(), lr=1e-3)

num_epochs = 5
train_losses = []
test_losses = []
test_accs = []

for ep in range(num_epochs):
    tr_loss = train_one_epoch(baseline_model, train_loader, optimizer, criterion)
    te_loss, te_acc, _, _ = evaluate(baseline_model, test_loader, criterion)
    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    test_accs.append(te_acc)

    print_block(
        f"Epoch {ep+1}/{num_epochs}",
        {
            "Train Loss": f"{tr_loss:.4f}",
            "Test Loss": f"{te_loss:.4f}",
            "Test Accuracy": f"{te_acc*100:.2f}%",
        },
    )

baseline_test_loss, baseline_test_acc, baseline_cm, baseline_inf = evaluate(
    baseline_model, test_loader, criterion
)
print_block(
    "Baseline Performance on Clean Test Set",
    {
        "Loss": f"{baseline_test_loss:.4f}",
        "Accuracy": f"{baseline_test_acc*100:.2f}%",
        "Avg Inference Time / Sample (s)": f"{baseline_inf:.6f}",
    },
)

plot_confusion_matrix(
    baseline_cm,
    "Baseline Confusion Matrix (Clean Test Set)",
    save_path=os.path.join(IMAGE_DIR, "baseline_confusion_matrix.png"),
)
torch.save(
    baseline_model.state_dict(),
    os.path.join(MODEL_DIR, "baseline_model.pth"),
)

# ========= Data poisoning experiment ==========
print_title("Data Poisoning – Method 1 (Corner Patch on Digit '7')")

poison_loader, poison_ds, poison_idxs = create_poisoned_train_loader(
    digit_to_poison=7, num_poison=100, patch_size=4, batch_size=64
)
show_poisoned_examples(poison_ds, poison_idxs, n=8)

poisoned_model = SimpleMNISTCNN().to(device)
poison_opt = optim.Adam(poisoned_model.parameters(), lr=1e-3)

poison_epochs = 5
for ep in range(poison_epochs):
    p_tr_loss = train_one_epoch(poisoned_model, poison_loader, poison_opt, criterion)
    p_te_loss, p_te_acc, _, _ = evaluate(poisoned_model, test_loader, criterion)
    print_block(
        f"Poisoned Epoch {ep+1}/{poison_epochs}",
        {
            "Train Loss (poisoned data)": f"{p_tr_loss:.4f}",
            "Clean Test Accuracy": f"{p_te_acc*100:.2f}%",
        },
    )

poisoned_test_loss, poisoned_test_acc, poisoned_cm, poisoned_inf = evaluate(
    poisoned_model, test_loader, criterion
)
print_block(
    "Poisoned Model on Clean Test Set",
    {
        "Loss": f"{poisoned_test_loss:.4f}",
        "Accuracy": f"{poisoned_test_acc*100:.2f}%",
        "Avg Inference Time / Sample (s)": f"{poisoned_inf:.6f}",
    },
)
plot_confusion_matrix(
    poisoned_cm,
    "Confusion Matrix – Model Trained on Poisoned Data",
    save_path=os.path.join(IMAGE_DIR, "poisoned_confusion_matrix.png"),
)
torch.save(
    poisoned_model.state_dict(),
    os.path.join(MODEL_DIR, "poisoned_model.pth"),
)

# ========= FGSM attack with ART (Red Team) ==========
print_title("Red Team – Strong FGSM Adversarial Attack (ART)")

fgsm_eps = 0.7  # strong attack to drop accuracy a lot

adv_loader = make_fgsm_test_loader(
    baseline_model,
    optimizer,
    criterion,
    eps=fgsm_eps,
    batch_size=256,
)
adv_loss, adv_acc, adv_cm, adv_inf = evaluate(baseline_model, adv_loader, criterion)

print_block(
    "Baseline Model on FGSM Adversarial Test Set",
    {
        "FGSM Loss": f"{adv_loss:.4f}",
        "FGSM Accuracy": f"{adv_acc*100:.2f}%",
        "Avg Inference Time / Sample (s)": f"{adv_inf:.6f}",
    },
)
plot_confusion_matrix(
    adv_cm,
    "Confusion Matrix – Baseline on FGSM Test Set",
    save_path=os.path.join(IMAGE_DIR, "baseline_fgsm_confusion_matrix.png"),
)

# show some original vs FGSM images
show_fgsm_examples(
    baseline_model,
    optimizer,
    criterion,
    eps=fgsm_eps,
    num_samples=10,
    filename="fgsm_examples_strip.png",
)

# ========= Adversarial training defence (Blue Team) ==========
print_title("Blue Team – Adversarial Training Defence (FGSM)")

defended_model = SimpleMNISTCNN().to(device)
defended_model.load_state_dict(baseline_model.state_dict())
def_opt = optim.Adam(defended_model.parameters(), lr=1e-3)

def_epochs = 3
def_clean_accs = []

for ep in range(def_epochs):
    adv_tr_loss = train_one_epoch_adv(
        defended_model, train_loader, def_opt, criterion, eps=fgsm_eps, alpha=0.5
    )
    d_clean_loss, d_clean_acc, _, _ = evaluate(defended_model, test_loader, criterion)
    def_clean_accs.append(d_clean_acc)

    print_block(
        f"Defence Epoch {ep+1}/{def_epochs}",
        {
            "Train Loss (clean+FGSM)": f"{adv_tr_loss:.4f}",
            "Clean Test Accuracy": f"{d_clean_acc*100:.2f}%",
        },
    )

# Evaluate defended model on FGSM test set
def_adv_loader = make_fgsm_test_loader(
    defended_model,
    def_opt,
    criterion,
    eps=fgsm_eps,
    batch_size=256,
)
def_adv_loss, def_adv_acc, def_adv_cm, def_adv_inf = evaluate(
    defended_model, def_adv_loader, criterion
)
plot_confusion_matrix(
    def_adv_cm,
    "Confusion Matrix – Defended Model on FGSM Test Set",
    save_path=os.path.join(IMAGE_DIR, "defended_fgsm_confusion_matrix.png"),
)

torch.save(
    defended_model.state_dict(),
    os.path.join(MODEL_DIR, "defended_model.pth"),
)

# ========= Final summary ==========
print_title("Summary – Baseline vs Poisoned vs FGSM vs Defended")

# 1) Baseline model
print_block(
    "1) Baseline Model (Clean Training)",
    {
        "Clean Test Accuracy": f"{baseline_test_acc*100:.2f}%",
        "Clean Test Loss": f"{baseline_test_loss:.4f}",
        "FGSM Test Accuracy": f"{adv_acc*100:.2f}%",
        "FGSM Test Loss": f"{adv_loss:.4f}",
        "Clean Inference (ms/sample)": f"{baseline_inf*1000:.4f}",
        "FGSM Inference (ms/sample)": f"{adv_inf*1000:.4f}",
    },
)

# 2) Poisoned-training model
print_block(
    "2) Poisoned Model (Trained on Corner-Patch Data)",
    {
        "Clean Test Accuracy": f"{poisoned_test_acc*100:.2f}%",
        "Clean Test Loss": f"{poisoned_test_loss:.4f}",
        "Clean Inference (ms/sample)": f"{poisoned_inf*1000:.4f}",
    },
)

# 3) Defended model
print_block(
    "3) Defended Model (FGSM Adversarial Training)",
    {
        "Clean Test Accuracy (last epoch)": f"{def_clean_accs[-1]*100:.2f}%",
        "FGSM Test Accuracy": f"{def_adv_acc*100:.2f}%",
        "FGSM Test Loss": f"{def_adv_loss:.4f}",
        "FGSM Inference (ms/sample)": f"{def_adv_inf*1000:.4f}",
    },
)

print("\nAll plots saved inside 'images/' and models saved inside 'models/'.")
