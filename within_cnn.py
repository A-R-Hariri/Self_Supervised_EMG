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
SUBJECTS = 6
REPS = 4


# ======== LOAD ========
path = join(PATH, 'ssl')
ssl_windows = np.load(join(path, 'ssl_windows.npy'), mmap_mode=MMAP_MODE)

path = join(PATH, 'sgt')
sgt_data = np.load(join(path, 'sgt_data.npy'), allow_pickle=True).item()


# ======== PIPELINE ========
results = []
for SEED in [13]:
    random.seed(SEED); np.random.seed(SEED)
    GENERATOR = torch.manual_seed(SEED)


    # ======== SSL ========
    # ---- SSL Data ----
    ssl_loader = create_ssl_loader(ssl_windows, 
                        batch=SSL_BATCH_SIZE, shuffle=True)

    # ---- BASE: Pretraining ----
    pretrained = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    pretrained = pretrain_vicreg(pretrained, ssl_loader, 
                                 name=f"cnn_vicreg_ssl_seed{SEED}")
    

    for s in range(SUBJECTS):
        
        # ======== SUPERVISED ========
        for rep in range(REPS - 1):

            # ---- FT Data ---
            _data = sgt_data.isolate_data("subjects", list(range(s + 1)), fast=True)

            train_data = _data.isolate_data("rep_forms", [0], fast=True)
            train_data = _data.isolate_data("reps", list(range(rep + 1)), fast=True)
            train_windows, train_meta = train_data.parse_windows(SEQ, INC)

            val_data = _data.isolate_data("rep_forms", [0], fast=True)
            val_data = _data.isolate_data("reps", [REPS - 1], fast=True)
            val_windows, val_meta = val_data.parse_windows(SEQ, INC)

            test_data = _data.isolate_data("rep_forms", [1], fast=True)
            test_windows, test_meta = test_data.parse_windows(SEQ, INC)

            ft_train_loader = create_sup_loader(train_windows, train_meta["classes"], 
                                                batch=BATCH_SIZE, shuffle=True)
            ft_val_loader = create_sup_loader(val_windows, val_meta["classes"], 
                                              batch=BATCH_SIZE, shuffle=True)
            ft_test_loader = create_sup_loader(test_windows, test_meta["classes"], 
                                               batch=BATCH_SIZE, shuffle=True)

            # ---- class weights for FT ----
            ft_weights = compute_class_weight(class_weight="balanced", 
                        classes=np.arange(len(FT_CLASSES)), 
                        y=train_meta["classes"]).astype(np.float32)
            ft_weights = torch.from_numpy(ft_weights).to(DEVICE)
            ft_loss = nn.CrossEntropyLoss(weight=ft_weights)
            print(ft_weights)


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
                "subject": s,
                "reps": rep,

                # ---- EXP 1 ----
                "exp1_acc": acc_1,
                "exp1_f1": f1_1,
                "exp1_bal": bal_1,

                # ---- EXP 2 ----
                "exp2_acc": acc_2,
                "exp2_f1": f1_2,
                "exp2_bal": bal_2,

                # ---- EXP 3 ----
                "exp3_acc": acc_3,
                "exp3_f1": f1_3,
                "exp3_bal": bal_3,

                # ---- EXP 4 ----
                "exp4_acc": acc_4,
                "exp4_f1": f1_4,
                "exp4_bal": bal_4,

                # ---- EXP 5 ----
                "exp5_acc": acc_5,
                "exp5_f1": f1_5,
                "exp5_bal": bal_5,
            })

        results.append({
            "seed": '',
            "subject": '',
            "reps": '',
            "exp1_acc": '',
            "exp1_f1": '',
            "exp1_bal": '',
            "exp2_acc": '',
            "exp2_f1": '',
            "exp2_bal": '',
            "exp3_acc": '',
            "exp3_f1": '',
            "exp3_bal": '',
            "exp4_acc": '',
            "exp4_f1": '',
            "exp4_bal": '',
            "exp5_acc": '',
            "exp5_f1": '',
            "exp5_bal": ''})


df = pd.DataFrame(results)
# ---- save full per-seed results ----
out_csv = f"cnn_seed{SEED}.csv"
df.to_csv(out_csv, index=False)
# ---- summary ----
print(df.describe())
print(acc_1, acc_2, acc_3, acc_4, acc_5)