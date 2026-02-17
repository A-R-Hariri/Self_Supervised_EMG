import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch; print(torch.cuda.is_available())

import libemg
from libemg.datasets import get_dataset_list
from libemg.feature_extractor import FeatureExtractor

import numpy as np, pandas as pd
import random, copy, time
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from utils import *
from Models.CNN import CNN


MMAP_MODE = 'r'


# ======== LOAD ========

path = join(PATH, 'ssl')
ssl_windows = np.load(join(path, 'ssl_windows.npy'), mmap_mode=MMAP_MODE)

path = join(PATH, 'sgt')
train_windows = np.load(join(path, 'train_windows.npy'), mmap_mode=MMAP_MODE)
train_meta = np.load(join(path, 'train_meta.npy'), allow_pickle=True).item()
val_windows = np.load(join(path, 'val_windows.npy'), mmap_mode=MMAP_MODE)
val_meta = np.load(join(path, 'val_meta.npy'), allow_pickle=True).item()
test_windows = np.load(join(path, 'test_windows.npy'), mmap_mode=MMAP_MODE)
test_meta = np.load(join(path, 'test_meta.npy'), allow_pickle=True).item()



# ======== PIPELINE ========

results = []
for SEED in [42, 117]:
    random.seed(SEED); np.random.seed(SEED)
    GENERATOR = torch.manual_seed(SEED)


    # ======== DATA LOADERS ========

    # ---- SSL ----
    ssl_loader = create_ssl_loader(ssl_windows, batch=BATCH_SIZE, shuffle=True)

    # ---- FT ---
    ft_train_loader = create_sup_loader(train_windows, train_meta["classes"], batch=BATCH_SIZE, shuffle=True)
    ft_val_loader = create_sup_loader(val_windows, val_meta["classes"], batch=BATCH_SIZE, shuffle=True)
    ft_test_loader = create_sup_loader(test_windows, test_meta["classes"], batch=BATCH_SIZE, shuffle=True)

    # ---- class weights for FT ----
    ft_weights = compute_class_weight(class_weight="balanced", 
                classes=np.arange(len(FT_CLASSES)), y=train_meta["classes"]).astype(np.float32)
    ft_weights = torch.from_numpy(ft_weights).to(DEVICE)
    ft_loss = nn.CrossEntropyLoss(weight=ft_weights)
    print(ft_weights)


    # ======== MODELS ========

    # ---- BASE: Pretraining ----
    pretrained = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    pretrained = pretrain_vicreg(pretrained, ssl_loader, name=f"cnn_vicreg_ssl_seed{SEED}")


    # ---- EXP 1: Full FT ----
    model_1 = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    model_1.load_state_dict(copy.deepcopy(pretrained.state_dict()))
    model_1.set_classifier(num_classes=len(FT_CLASSES))
    for p in model_1.parameters():
        p.requires_grad = True
    model_1 = train_supervised(
        model_1, ft_train_loader, ft_val_loader,
        name=f"cnn_pretrained_then_ft_seed{SEED}",
        loss_fn=ft_loss)
    acc_1, _, f1_1, bal_1 = evaluate_sup(model_1, ft_test_loader, ft_loss, DEVICE)


    # ---- EXP 2: FT with Frozen CNN ----
    model_2 = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    model_2.load_state_dict(copy.deepcopy(pretrained.state_dict()))
    model_2.set_classifier(num_classes=len(FT_CLASSES))
    for p in model_2.parameters():
        p.requires_grad = True
    for p in model_2.conv1.parameters(): p.requires_grad = False
    for p in model_2.conv2.parameters(): p.requires_grad = False
    for p in model_2.conv3.parameters(): p.requires_grad = False
    model_2 = train_supervised(
        model_2, ft_train_loader, ft_val_loader,
        name=f"cnn_pretrained_frozen_cnn_then_ft_seed{SEED}",
        loss_fn=ft_loss)
    acc_2, _, f1_2, bal_2 = evaluate_sup(model_2, ft_test_loader, ft_loss, DEVICE)


    # ---- EXP 3: FT with Fully Frozen Encoder ----
    model_3 = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    model_3.load_state_dict(copy.deepcopy(pretrained.state_dict()))
    model_3.set_classifier(num_classes=len(FT_CLASSES))
    for p in model_3.parameters():
        p.requires_grad = False
    for p in model_3.classifier.parameters():
        p.requires_grad = True
    model_3 = train_supervised(
        model_3, ft_train_loader, ft_val_loader,
        name=f"cnn_linear_probe_seed{SEED}",
        loss_fn=ft_loss)
    acc_3, _, f1_3, bal_3 = evaluate_sup(model_3, ft_test_loader, ft_loss, DEVICE)


    # ---- EXP 4: FT with Fully Frozen Encoder and linear probe ----
    model_4 = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    model_4.load_state_dict(copy.deepcopy(pretrained.state_dict()))
    model_4.set_linear_probe(num_classes=len(FT_CLASSES))
    for p in model_4.parameters():
        p.requires_grad = False
    for p in model_4.classifier.parameters():
        p.requires_grad = True
    model_4 = train_supervised(
        model_4, ft_train_loader, ft_val_loader,
        name=f"cnn_linear_probe_ssl5_seed{SEED}",
        loss_fn=ft_loss)

    acc_4, _, f1_4, bal_4 = evaluate_sup(
        model_4, ft_test_loader, ft_loss, DEVICE)

    # ---- EXP 5: No SSL ----
    model_5 = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    model_5.set_classifier(num_classes=len(FT_CLASSES))
    for p in model_5.parameters():
        p.requires_grad = True
    model_5 = train_supervised(
        model_5, ft_train_loader, ft_val_loader,
        name=f"cnn_raw_ft_only_seed{SEED}",
        loss_fn=ft_loss)
    acc_5, _, f1_5, bal_5 = evaluate_sup(model_5, ft_test_loader, ft_loss, DEVICE)


    # ======== LOGGING ========

    results.append({
        "seed": SEED,

        # ---- EXP 1: Pretrained -> Full Fine-Tuning ----
        "exp1_full_ft_acc": acc_1,
        "exp1_full_ft_f1": f1_1,
        "exp1_full_ft_bal": bal_1,

        # ---- EXP 2: Pretrained -> FT (Frozen CNN) ----
        "exp2_frozen_cnn_acc": acc_2,
        "exp2_frozen_cnn_f1": f1_2,
        "exp2_frozen_cnn_bal": bal_2,

        # ---- EXP 3: Pretrained -> Classifier (Frozen Encoder) ----
        "exp3_frozen_enc_acc": acc_3,
        "exp3_frozen_enc_f1": f1_3,
        "exp3_frozen_enc_bal": bal_3,

        # ---- EXP 4: Pretrained -> Linear Probe (Frozen Encoder) ----
        "exp4_frozen_enc_linprb_acc": acc_4,
        "exp4_frozen_enc_linprb_f1": f1_4,
        "exp4_frozen_enc_linprb_bal": bal_4,

        # ---- EXP 5: No SSL -> Full Fine-Tuning ----
        "exp5_raw_acc": acc_5,
        "exp5_raw_f1": f1_5,
        "exp5_raw_bal": bal_5,
    })


df = pd.DataFrame(results)
# ---- save full per-seed results ----
out_csv = "results_all_seeds3.csv"
df.to_csv(out_csv, index=False)
# ---- summary ----
print(df.describe())
print(acc_1, acc_2, acc_3, acc_4, acc_5)