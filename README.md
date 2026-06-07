# Self-Supervised EMG

Exploring self-supervised pre-training (VICReg) for zero-shot cross-user EMG gesture classification on the **EMG-EPN-612** dataset.

## Dataset

**EMG-EPN-612** — 612 subjects, Myo armband, 8-channel surface EMG at 200 Hz, 5 gestures: rest (NM), hand close (HC), hand open (HO), flexion (FX), and extension (EX). Sensors are 8-bit signed integers; all values are scaled by `/ 128.0` (bit-depth normalization, not statistical normalization).

Fixed split: users 1–306 training, 307–332 validation, 333–612 testing (280 test users). The split is global and never randomized.

A secondary real-world dataset (`SPACESHIP`, 16 subjects) is included as a LibEMG-compatible dataset class for unlabeled continuous EMg data using OyMotion sensor. Data collected during an EMG-controlled spaceship game, without labels.

## Approach

1. **SSL pre-training** — a 1D CNN encoder is pre-trained on unlabeled EMG windows (1 s, 200 samples) using VICReg, with stochastic signal augmentation applied to construct positive pairs.
2. **Supervised fine-tuning** — the pre-trained encoder is fine-tuned on labeled windows (stride 5 samples, ~25 ms) with balanced cross-entropy to account for rest-class imbalance.

Five fine-tuning strategies are benchmarked against each other and against a fully supervised (no SSL) baseline:

| Experiment | Encoder | Classifier |
|---|---|---|
| Full fine-tune | trainable | trainable |
| Frozen CNN blocks | frozen conv layers | trainable |
| Frozen encoder | frozen | MLP head |
| Linear probe | frozen | linear |
| No SSL (baseline) | random init, trainable | trainable |

Evaluation uses macro F1, overall accuracy, and balanced accuracy. Each configuration is run over multiple seeds; results are saved to CSV.

## Repository Structure

```
├── Models/
│   └── CNN.py           # 1D CNN encoder with projection head
├── Losses/
│   └── VICReg.py        # VICReg loss + augmentation
├── utils.py             # Training loop, loaders, evaluation
├── cross_cnn.py         # Cross-user SSL pipeline (main entry point)
├── within_cnn.py        # Within-user CNN baseline
├── within_mlp.py        # Within-user MLP baseline
├── gForceSGT.py         # Real-time gForce SGT demo (LibEMG)
├── SPACESHIP.py         # LibEMG dataset class for SPACESHIP data
├── Prepare_Data.ipynb   # Data windowing and preprocessing
└── Analyse.ipynb        # Results analysis
```

## Key Configuration

| Parameter | Value |
|---|---|
| Window (SSL) | 200 samples (1 s) |
| Window (FT) | 200 samples (1 s) |
| Stride (SSL) | 40 samples |
| Stride (FT) | 5 samples |
| Channels | 8 |
| Classes | 5 |
| Embedding dim | 128 |
| Projection dim | 128 |

## Dependencies

```
torch
libemg
numpy
pandas
scikit-learn
matplotlib
tqdm
h5py
```

## Relation to Main Work

This repository is a side branch of the main zero-shot cross-user EMG project ([`EPN612_Cross_User`](https://github.com/A-R-Hariri/EPN612_Cross_User)), which uses a Multi-Head CNN (MHCNN) with dilated convolutions trained end-to-end. The SSL branch here investigates whether VICReg pre-training on unlabeled data can improve generalization across users before supervised fine-tuning.
