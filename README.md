# 🛡️ Secure AI Systems: Red & Blue Teaming an MNIST Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)
![Security](https://img.shields.io/badge/Security-Bandit%20%7C%20ART-red)

## Team Detail

📌 This project is submitted for the assignment towards the course “Security in Intelligent Systems” by the following team members:

* Shivansh Agarwal(CS25MTECH14013)
* Rohit B(CS25MTECH14012)

## 📖 Project Overview

This project is a submission for the **"Secure AI Systems – Red and Blue Teaming an MNIST Classifier"** assignment. It demonstrates the complete lifecycle of securing a deep learning model, moving beyond standard performance metrics to evaluate robustness against active adversarial threats.

We implement a Convolutional Neural Network (CNN) to classify MNIST handwritten digits and subject it to a rigorous security assessment:
1.  **Baseline Training:** Establishing standard performance benchmarks.
2.  **🔴 Red Teaming:** Attacking the model using **Data Poisoning** (Backdoor attacks) and **Adversarial Examples** (FGSM).
3.  **🔵 Blue Teaming:** Defending the model using **Adversarial Training**.
4.  **Security Auditing:** Applying **SAST** (Static Application Security Testing) with Bandit and **STRIDE** threat modeling.

---
## 📂 Project Structure

```text
├── data                    # This contains the MNIST dataset
├── secure_mnist.ipynb      # Interactive Jupyter notebook (ideal for presentation/demo)
├── bandit_report           # This contains the SAST report
├── main.py                 # Orchestrator script for training, attacking, and defending
├── model.py                # CNN architecture definition & MNIST data loaders
├── training.py             # Standard training & evaluation loops
├── poisoning.py            # Data poisoning utilities (Trigger generation)
├── ART_attack.py           # Adversarial attack implementation (FGSM via ART)
├── defence.py              # Adversarial training logic
├── utils.py                # Helper functions for visualization & logging
│
├── models/                 # Serialized model weights (.pth)
│   ├── baseline_model.pth
│   ├── poisoned_model.pth
│   └── defended_model.pth
│
└── images/                 # Generated visualizations for the report
    ├── baseline_confusion_matrix.png
    ├── poisoned_confusion_matrix.png
    └── defended_fgsm_confusion_matrix.png

```

## Installation & Requirements

 The project requires Python 3.10+ and the following dependencies. The Adversarial Robustness Toolbox (ART) is used for generating attacks, and Bandit is used for security scanning.

```bash
pip install torch torchvision matplotlib scikit-learn adversarial-robustness-toolbox bandit
```



## ▶️ How to Run

### 1) Full Pipeline (Script)

From the project root, run:

```bash
python main.py
```
This will:

* Train the baseline CNN on clean MNIST and save `models/baseline_model.pth`.

* Save training curves and confusion matrices to `images/`.

* Perform data poisoning by adding a 4×4 corner patch to selected training samples of digit `7`, retrain, and save `models/poisoned_model.pth`

* Generate FGSM adversarial examples using ART and evaluate the baseline under attack.

* Perform adversarial training (FGSM-based) to produce a defended `model models/defended_model.pth`.

* Save all plots and a printed summary comparing baseline, poisoned, and defended models.


### 2) Notebook (interactive)

Open:

```bash
secure_mnist.ipynb
```
Run the cells sequentially for an interactive demo of the entire experiment (ideal for presentations).


## 🔬 Experiments & Key Metrics

Summary of results (see detailed figures in `images/` and the full report):

### Baseline (clean MNIST)

* Test accuracy: 99.20%

* Test loss: 0.0254

* Inference time: ~0.71 ms/sample

* Confusion matrix nearly diagonal (strong performance on clean data).

### Data Poisoning (Backdoor — corner patch on digit 7)

* Inserted a 4×4 white corner patch into 100 training images of digit 7.

* Clean test accuracy: 99.27% (no significant drop).

* Demonstrates stealthy backdoor behaviour: global accuracy remains high while targeted trigger behaviour is introduced.

### FGSM Adversarial Attack (ART, ε = 0.2)

* Baseline model accuracy drops to ~89.43% on FGSM adversarial test set.

* FGSM loss: 0.3618

* Confirms susceptibility of standard CNNs to gradient-based adversarial perturbations.

### Adversarial Training (Blue Team Defence)

* After adversarial training against FGSM:

* * Clean accuracy: 99.22%

* * FGSM accuracy: improved to ~96.30%

* * FGSM loss: 0.1399

* Shows strong robustness gains with minimal impact on clean-data performance.

(Full metric tables and confusion matrices are included in the project images and the attached report.)

## 🧪 Static Application Security Testing (SAST)

We ran Bandit across the codebase:

```bash
bandit -r .
```

### Findings (summary):

* No High-severity issues found.

* One Medium-severity issue flagged (B614): use of `torch.load` may cause unsafe deserialization due to pickle-based loading of `.pth` files.

In this assignment context, practical risk is low because:

* * `.pth` files are generated locally by our scripts.

* * Paths are hardcoded and user uploads are not accepted.

* * Code runs in a controlled offline environment.

### Mitigations applied/recommended:

* Path whitelisting and explicit model_path construction.

* Document trust assumptions in code and README.

* For production: avoid loading arbitrary user-supplied models; use weights_only=True (where available) or safer serialization formats.



## 🛡 STRIDE Threat Model (Summary)


| **STRIDE**              | **Threat Description**                              | **Mitigation**                                       |
|-------------------------|------------------------------------------------------|-------------------------------------------------------|
| **Spoofing**            | Maliciously generated/adversarial inputs             | Input validation, training-data validation            |
| **Tampering**           | Training-data poisoning or model file modification   | Dataset hashing, integrity checks, secure storage     |
| **Repudiation**         | Lack of logging for adversarial events               | Comprehensive logging & dataset/version tracking      |
| **Information Disclosure** | Model leaking sensitive patterns / overfitting   | Regularization, dropout, differential privacy (if needed) |
| **Denial of Service**   | Heavy/rapid adversarial queries exhausting compute   | Rate limiting, input throttling                       |
| **Elevation of Privilege** | Misuse of notebook/VM to access other resources | Least-privilege, notebook isolation, containerization |
                                               

## 📁 Files of Interest

* `models/baseline_model.pth` — baseline weights

* `models/poisoned_model.pth` — model trained on poisoned dataset

* `models/defended_model.pth` — adversarially-trained model

* `images/` — confusion matrices, loss/accuracy curves and poisoned examples

* `secure_mnist.ipynb` — interactive walkthrough

You can also review the assignment final report included in the repository:

* Local copy: `/mnt/data/Copy of Final_Report.pdf`




## ✅ Key Takeaways

* High clean-data accuracy **does not** guarantee robustness to attacks.

* **Data poisoning** can be stealthy while enabling targeted misclassification via small trigger patterns.

* **FGSM adversarial** examples noticeably reduce model performance, but adversarial training recovers a large portion of adversarial accuracy.

* Integrating **SAST** and **threat modelling (STRIDE)** helps operationalize ML security best practices.

