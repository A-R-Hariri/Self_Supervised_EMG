import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"

import torch; print(torch.cuda.is_available())
import torch.nn as nn

import libemg
from libemg.datasets import get_dataset_list
from libemg.feature_extractor import FeatureExtractor

import numpy as np, pandas as pd
import random, copy, time
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from utils import *
from Models.MLP import MLP


MMAP_MODE = 'r'
DAY = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SUBJECTS = 5 if DAY else 6

SAMPLING_RATE = 1000
FEATURE_LIST = ['WENG']
FEATURE_DIC = {'WENG_fs': SAMPLING_RATE}


# ======== LOAD ========
path = join(PATH, 'ssl')
ssl_windows = np.load(join(path, 'ssl_windows.npy'), mmap_mode=MMAP_MODE)
sample_features = extract_features(ssl_windows[:10], 
                                   feature_list=FEATURE_LIST, 
                                   feature_dic=FEATURE_DIC)
n_features = sample_features.shape[1]

path = join(PATH, 'sgt')
sgt_data = np.load(join(path, f'sgt_data{DAY}.npy'), allow_pickle=True).item()


# ======== PIPELINE ========
_empty = [{
        "seed": '',
        "subject": '',
        "exp1_acc_s": '',
        "exp1_acc_l": '',
        "exp1_acc_t": '',
        "exp2_acc_s": '',
        "exp2_acc_l": '',
        "exp2_acc_t": '',
        "exp3_acc_s": '',
        "exp3_acc_l": '',
        "exp3_acc_t": '',
        "exp4_acc_s": '',
        "exp4_acc_l": '',
        "exp4_acc_t": '',
        "exp5_acc_s": '',
        "exp5_acc_l": '',
        "exp5_acc_t": ''}]
df = pd.DataFrame(_empty)
# ---- save full per-seed results ----
out_csv = f"within_mlp_d{DAY}.csv"
df.to_csv(out_csv)

for SEED in [7, 13, 42, 67, 127]:
    seed_everything(SEED)

    # ======== SSL ========
    # ---- SSL Data ----
    ssl_loader = create_ssl_loader(ssl_windows, 
                        batch=SSL_BATCH_SIZE, shuffle=True)

    # ---- BASE: Pretraining ----
    pretrained = MLP(n_features, emb_dim=128, proj_dim=128, dropout=DROPOUT)
    pretrained = pretrain_vicreg(pretrained, ssl_loader, 
                                 name=f"mlp_vicreg_ssl_seed{SEED}",
                                 feature_list=FEATURE_LIST,
                                 feature_dict=FEATURE_DIC)
    
    for s in range(SUBJECTS):
        # ======== SUPERVISED ========
        # ---- FT Data ---
        _data = sgt_data.isolate_data("subjects", [s], fast=True)

        train_data = _data.isolate_data("rep_forms", [0], fast=True)
        train_data = train_data.isolate_data("reps", [0], fast=True)
        X, y = train_data.parse_windows(SEQ, INC)
        X = extract_features(X, FEATURE_LIST, FEATURE_DIC, True)

        val_data = _data.isolate_data("rep_forms", [0], fast=True)
        val_data = val_data.isolate_data("reps", [1], fast=True)
        X_v, y_v = val_data.parse_windows(SEQ, INC)
        X_v = extract_features(X_v, FEATURE_LIST, FEATURE_DIC, True)

        test_data = _data.isolate_data("rep_forms", [0], fast=True)
        test_data = test_data.isolate_data("reps", [2, 3, 4], fast=True)
        X_t_static, y_t_static = test_data.parse_windows(SEQ, INC)
        X_t_static = extract_features(X_t_static, FEATURE_LIST, FEATURE_DIC, True)

        test_data = _data.isolate_data("rep_forms", [1], fast=True)
        X_t_limb, y_t_limb = test_data.parse_windows(SEQ, INC)
        X_t_limb = extract_features(X_t_limb, FEATURE_LIST, FEATURE_DIC, True)

        test_data = _data.isolate_data("rep_forms", [2], fast=True)
        X_t_trans, y_t_trans = test_data.parse_windows(SEQ, INC)
        X_t_trans = extract_features(X_t_trans, FEATURE_LIST, FEATURE_DIC, True)

        ft_train_loader = create_sup_loader(X, y["classes"], 
                                            batch=BATCH_SIZE, shuffle=True)
        ft_val_loader = create_sup_loader(X_v, y_v["classes"], 
                                            batch=BATCH_SIZE, shuffle=False)
        ft_test_loader_static = create_sup_loader(X_t_static, y_t_static["classes"], 
                                            batch=BATCH_SIZE, shuffle=False)
        ft_test_loader_limb = create_sup_loader(X_t_limb, y_t_limb["classes"], 
                                            batch=BATCH_SIZE, shuffle=False)
        ft_test_loader_trans = create_sup_loader(X_t_trans, y_t_trans["classes"], 
                                            batch=BATCH_SIZE, shuffle=False)

        # ---- class weights for FT ----
        ft_weights = compute_class_weight(class_weight="balanced", 
                    classes=np.arange(len(FT_CLASSES)), 
                    y=y["classes"]).astype(np.float32)
        ft_weights = torch.from_numpy(ft_weights).to(DEVICE)
        ft_loss = nn.CrossEntropyLoss(weight=ft_weights)
        print(ft_weights)


        # ---- EXP 1: Full FT ----
        model_1 = MLP(n_features, emb_dim=128, proj_dim=128, dropout=DROPOUT)
        model_1.load_state_dict(copy.deepcopy(pretrained.state_dict()))
        model_1.set_classifier(num_classes=len(FT_CLASSES))
        for p in model_1.parameters():
            p.requires_grad = True
        model_1 = train_supervised(
            model_1, ft_train_loader, ft_val_loader,
            name=f"mlp_pretrained_then_ft_seed{SEED}",
            loss_fn=ft_loss)
        acc_1_s, _, f1_1_s, bal_1_s = evaluate_sup(model_1, ft_test_loader_static, ft_loss, DEVICE)
        acc_1_l, _, f1_1_l, bal_1_l = evaluate_sup(model_1, ft_test_loader_limb, ft_loss, DEVICE)
        acc_1_t, _, f1_1_t, bal_1_t = evaluate_sup(model_1, ft_test_loader_trans, ft_loss, DEVICE)


        # ---- EXP 2: FT with Frozen CNN ----
        model_2 = MLP(n_features, emb_dim=128, proj_dim=128, dropout=DROPOUT)
        model_2.load_state_dict(copy.deepcopy(pretrained.state_dict()))
        model_2.set_classifier(num_classes=len(FT_CLASSES))
        for p in model_2.parameters():
            p.requires_grad = True
        for p in model_2.fc1.parameters(): p.requires_grad = False
        for p in model_2.fc2.parameters(): p.requires_grad = False
        for p in model_2.fc3.parameters(): p.requires_grad = False
        model_2 = train_supervised(
            model_2, ft_train_loader, ft_val_loader,
            name=f"mlp_pretrained_frozen_mlp_then_ft_seed{SEED}",
            loss_fn=ft_loss, disable_bn=True)
        acc_2_s, _, f1_2_s, bal_2_s = evaluate_sup(model_2, ft_test_loader_static, ft_loss, DEVICE)
        acc_2_l, _, f1_2_l, bal_2_l = evaluate_sup(model_2, ft_test_loader_limb, ft_loss, DEVICE)
        acc_2_t, _, f1_2_t, bal_2_t = evaluate_sup(model_2, ft_test_loader_trans, ft_loss, DEVICE)


        # ---- EXP 3: FT with Fully Frozen Encoder ----
        model_3 = MLP(n_features, emb_dim=128, proj_dim=128, dropout=DROPOUT)
        model_3.load_state_dict(copy.deepcopy(pretrained.state_dict()))
        model_3.set_classifier(num_classes=len(FT_CLASSES))
        for p in model_3.parameters():
            p.requires_grad = False
        for p in model_3.classifier.parameters():
            p.requires_grad = True
        model_3 = train_supervised(
            model_3, ft_train_loader, ft_val_loader,
            name=f"mlp_linear_probe_seed{SEED}",
            loss_fn=ft_loss, disable_bn=True)
        acc_3_s, _, f1_3_s, bal_3_s = evaluate_sup(model_3, ft_test_loader_static, ft_loss, DEVICE)
        acc_3_l, _, f1_3_l, bal_3_l = evaluate_sup(model_3, ft_test_loader_limb, ft_loss, DEVICE)
        acc_3_t, _, f1_3_t, bal_3_t = evaluate_sup(model_3, ft_test_loader_trans, ft_loss, DEVICE)


        # ---- EXP 4: FT with Fully Frozen Encoder and linear probe ----
        model_4 = MLP(n_features, emb_dim=128, proj_dim=128, dropout=DROPOUT)
        model_4.load_state_dict(copy.deepcopy(pretrained.state_dict()))
        model_4.set_linear_probe(num_classes=len(FT_CLASSES))
        for p in model_4.parameters():
            p.requires_grad = False
        for p in model_4.classifier.parameters():
            p.requires_grad = True
        model_4 = train_supervised(
            model_4, ft_train_loader, ft_val_loader,
            name=f"mlp_linear_probe_ssl5_seed{SEED}",
            loss_fn=ft_loss, disable_bn=True)

        acc_4_s, _, f1_4_s, bal_4_s = evaluate_sup(model_4, ft_test_loader_static, ft_loss, DEVICE)
        acc_4_l, _, f1_4_l, bal_4_l = evaluate_sup(model_4, ft_test_loader_limb, ft_loss, DEVICE)
        acc_4_t, _, f1_4_t, bal_4_t = evaluate_sup(model_4, ft_test_loader_trans, ft_loss, DEVICE)


        # ---- EXP 5: No SSL ----
        model_5 = MLP(n_features, emb_dim=128, proj_dim=128, dropout=DROPOUT)
        model_5.set_classifier(num_classes=len(FT_CLASSES))
        for p in model_5.parameters():
            p.requires_grad = True
        model_5 = train_supervised(
            model_5, ft_train_loader, ft_val_loader,
            name=f"mlp_raw_ft_only_seed{SEED}",
            loss_fn=ft_loss)
        acc_5_s, _, f1_5_s, bal_5_s = evaluate_sup(model_5, ft_test_loader_static, ft_loss, DEVICE)
        acc_5_l, _, f1_5_l, bal_5_l = evaluate_sup(model_5, ft_test_loader_limb, ft_loss, DEVICE)
        acc_5_t, _, f1_5_t, bal_5_t = evaluate_sup(model_5, ft_test_loader_trans, ft_loss, DEVICE)

        del model_1, model_2, model_3, model_4, model_5
        torch.cuda.empty_cache()
        gc.collect()

        # ======== LOGGING ========
        result = [{
            "seed": SEED,
            "subject": s,
            # ---- EXP 1 ----
            "exp1_acc_s": acc_1_s,
            "exp1_acc_l": acc_1_l,
            "exp1_acc_t": acc_1_t,
            # ---- EXP 2 ----
            "exp2_acc_s": acc_2_s,
            "exp2_acc_l": acc_2_l,
            "exp2_acc_t": acc_2_t,
            # ---- EXP 3 ----
            "exp3_acc_s": acc_3_s,
            "exp3_acc_l": acc_3_l,
            "exp3_acc_t": acc_3_t,
            # ---- EXP 4 ----
            "exp4_acc_s": acc_4_s,
            "exp4_acc_l": acc_4_l,
            "exp4_acc_t": acc_4_t,
            # ---- EXP 5 ----
            "exp5_acc_s": acc_5_s,
            "exp5_acc_l": acc_5_l,
            "exp5_acc_t": acc_5_t,
        }]
        df_new = pd.DataFrame(result)
        df_new.to_csv(out_csv, mode='a', index=False,
        header=not os.path.exists(out_csv))

        df_new = pd.DataFrame(_empty)
        df_new.to_csv(out_csv, mode='a', index=False, 
              header=not os.path.exists(out_csv))