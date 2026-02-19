import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
_empty = [{
        "seed": '',
        "subject": '',
        "exp1_accs": '',
        "exp1_accl": '',
        "exp1_acct": '',
        "exp2_accs": '',
        "exp2_accl": '',
        "exp2_acct": '',
        "exp3_accs": '',
        "exp3_accl": '',
        "exp3_acct": '',
        "exp4_accs": '',
        "exp4_accl": '',
        "exp4_acct": '',
        "exp5_accs": '',
        "exp5_accl": '',
        "exp5_acct": ''}]
df = pd.DataFrame(_empty)
# ---- save full per-seed results ----
out_csv = f"within_cnn.csv"
df.to_csv(out_csv, mode='a', index=False,
          header=not os.path.exists(out_csv))

for SEED in [7, 13, 42, 67, 69]:
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
        # ---- FT Data ---
        _data = sgt_data.isolate_data("subjects", [s], fast=True)

        train_data = _data.isolate_data("rep_forms", [0], fast=True)
        train_data = _data.isolate_data("reps", [0], fast=True)
        X, y = train_data.parse_windows(SEQ, INC)

        val_data = _data.isolate_data("rep_forms", [0], fast=True)
        val_data = _data.isolate_data("reps", [1], fast=True)
        X_v, y_v = val_data.parse_windows(SEQ, INC)

        test_data = _data.isolate_data("rep_forms", [0], fast=True)
        val_data = _data.isolate_data("reps", [2, 3], fast=True)
        X_t_static, y_t_static = test_data.parse_windows(SEQ, INC)

        test_data = _data.isolate_data("rep_forms", [1], fast=True)
        X_t_limb, y_t_limb = test_data.parse_windows(SEQ, INC)

        test_data = _data.isolate_data("rep_forms", [2], fast=True)
        X_t_trans, y_t_trans = test_data.parse_windows(SEQ, INC)

        ft_train_loader = create_sup_loader(X, y["classes"], 
                                            batch=BATCH_SIZE, shuffle=True)
        ft_val_loader = create_sup_loader(X_v, y_v["classes"], 
                                            batch=BATCH_SIZE, shuffle=True)
        ft_test_loader_static = create_sup_loader(X_t_static, y_t_static["classes"], 
                                            batch=BATCH_SIZE, shuffle=True)
        ft_test_loader_limb = create_sup_loader(X_t_static, y_t_static["classes"], 
                                            batch=BATCH_SIZE, shuffle=True)
        ft_test_loader_trans = create_sup_loader(X_t_static, y_t_static["classes"], 
                                            batch=BATCH_SIZE, shuffle=True)

        # ---- class weights for FT ----
        ft_weights = compute_class_weight(class_weight="balanced", 
                    classes=np.arange(len(FT_CLASSES)), 
                    y=y["classes"]).astype(np.float32)
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
        acc_1s, _, f1_1s, bal_1s = evaluate_sup(model_1, ft_test_loader_static, ft_loss, DEVICE)
        acc_1l, _, f1_1l, bal_1l = evaluate_sup(model_1, ft_test_loader_limb, ft_loss, DEVICE)
        acc_1t, _, f1_1t, bal_1t = evaluate_sup(model_1, ft_test_loader_trans, ft_loss, DEVICE)


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
        acc_2s, _, f1_2s, bal_2s = evaluate_sup(model_2, ft_test_loader_static, ft_loss, DEVICE)
        acc_2l, _, f1_2l, bal_2l = evaluate_sup(model_2, ft_test_loader_limb, ft_loss, DEVICE)
        acc_2t, _, f1_2t, bal_2t = evaluate_sup(model_2, ft_test_loader_trans, ft_loss, DEVICE)


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
        acc_3s, _, f1_3s, bal_3s = evaluate_sup(model_3, ft_test_loader_static, ft_loss, DEVICE)
        acc_3l, _, f1_3l, bal_3l = evaluate_sup(model_3, ft_test_loader_limb, ft_loss, DEVICE)
        acc_3t, _, f1_3t, bal_3t = evaluate_sup(model_3, ft_test_loader_trans, ft_loss, DEVICE)


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

        acc_4s, _, f1_4s, bal_4s = evaluate_sup(model_4, ft_test_loader_static, ft_loss, DEVICE)
        acc_4l, _, f1_4l, bal_4l = evaluate_sup(model_4, ft_test_loader_limb, ft_loss, DEVICE)
        acc_4t, _, f1_4t, bal_4t = evaluate_sup(model_4, ft_test_loader_trans, ft_loss, DEVICE)


        # ---- EXP 5: No SSL ----
        model_5 = CNN(ch=CH, seq=SEQ, emb_dim=128, proj_dim=128, dropout=DROPOUT)
        model_5.set_classifier(num_classes=len(FT_CLASSES))
        for p in model_5.parameters():
            p.requires_grad = True
        model_5 = train_supervised(
            model_5, ft_train_loader, ft_val_loader,
            name=f"cnn_raw_ft_only_seed{SEED}",
            loss_fn=ft_loss)
        acc_5s, _, f1_5s, bal_5s = evaluate_sup(model_5, ft_test_loader_static, ft_loss, DEVICE)
        acc_5l, _, f1_5l, bal_5l = evaluate_sup(model_5, ft_test_loader_limb, ft_loss, DEVICE)
        acc_5t, _, f1_5t, bal_5t = evaluate_sup(model_5, ft_test_loader_trans, ft_loss, DEVICE)


        # ======== LOGGING ========
        result = [{
            "seed": SEED,
            "subject": s,
            # ---- EXP 1 ----
            "exp1_accs": acc_1l,
            "exp1_accl": acc_1s,
            "exp1_acct": acc_1t,
            # ---- EXP 2 ----
            "exp2_accs": acc_2l,
            "exp2_accl": acc_2s,
            "exp2_acct": acc_2t,
            # ---- EXP 3 ----
            "exp3_accs": acc_3l,
            "exp3_accl": acc_3s,
            "exp3_acct": acc_3t,
            # ---- EXP 4 ----
            "exp4_accs": acc_4l,
            "exp4_accl": acc_4s,
            "exp4_acct": acc_4t,
            # ---- EXP 5 ----
            "exp5_accs": acc_5l,
            "exp5_accl": acc_5s,
            "exp5_acct": acc_5t,
        }]
        df_new = pd.DataFrame(result)
        df_new.to_csv(out_csv, mode='a', index=False,
        header=not os.path.exists(out_csv))

        df_new = pd.DataFrame(_empty)
        df_new.to_csv(out_csv, mode='a', index=False, 
              header=not os.path.exists(out_csv))