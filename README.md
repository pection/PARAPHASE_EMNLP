# SPLINT

This code is associated with the **SPLINT: SParse Learning for INterpretable Tuning** paper.  
Our work extends the research presented in [**SparseFit: Few-shot Prompting with Sparse Fine-tuning for Jointly Generating Predictions and Natural Language Explanations**](https://arxiv.org/abs/2305.13235). We have used their codebase as a foundation for this project and improved it for usability, reproducibility, and ease of experimentation.

---

SPLINT is a **refactored and extended version** of a larger framework for training and evaluating T5-based models (like UnifiedQA) across various explanation and reasoning tasks using multiple random seeds.

> **Why this version?**  
> This is a **lightweight, single-GPU version** of SPLINT designed for testing and demonstration. It already includes improvements like **custom loss functions** and a cleaner training workflow.  

> 🔓 The **full version**, will be released upon paper acceptance, supports **multi-GPU training**, **flexible experiment configuration**, and **scalable runs across datasets and models**.

---

## 📦 Installation

### 1. Create and activate a virtual environment

```bash
python3.10 -m venv SPLINT
source SPLINT/bin/activate
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Run Training

### Quick Start

To start training and evaluating with default settings:

```bash
python scripts/traning.py
```

This will:
- Train selected models on supported datasets
- Automatically apply optimal IO formats
- Run experiments using 60 random seeds

---
## 📈 Collect Results

After completing all 60 seed runs:

```bash
mkdir out
python scripts/benchmark_result.py \
  --exp_root checkpoints \
  --output out
```

This will generate a report of **mean** and **standard deviation** for all experimental setups.

> 💡 If you get an `AssertionError`, some seed runs likely failed. Identify the missing runs, re-run them, and rerun the results collection script.

---

## 📁 Directory Structure

```
..
├── data/              # External folder containing raw data files
└── SPLINT/
    ├── data/           # Inside folder containing raw data files
    ├── checkpoints/           
    ├── scripts/               # All training and utility scripts
    │   ├── __init__.py
    │   ├── benchmark_result.py
    │   ├── compute_kappa.py
    │   ├── custom_args.py
    │   ├── custom_loss_class.py
    │   ├── exp.py
    │   ├── feature_conversion_methods.py
    │   ├── metrics_custom_loss.py
    │   ├── preprocess_data.py
    │   ├── samples_for_human_eval.py
    │   ├── training.py
    ├── requirements.txt
    ├── README.md
    └── .gitignore
```