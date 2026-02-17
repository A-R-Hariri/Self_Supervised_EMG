import torch
import torch.nn as nn; import torch.nn.functional as F

from utils import *


# ======== MODEL (EMBEDDING + PROJECTOR + CLASSIFIER HEAD) ========
class MLP(nn.Module):
    def __init__(self, ch, seq, layers=[512, 256, 128], lstm_layers=0,
                 emb_dim=128, proj_dim=128, dropout=DROPOUT):
        super().__init__()
        self.lstm_layers = lstm_layers
        
        # (1) Optional LSTM Backbone
        if lstm_layers > 0:
            # Input: (B, C, S) -> Transpose to (B, S, C) for LSTM
            self.lstm = nn.LSTM(
                input_size=ch,
                hidden_size=ch,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0,
                bidirectional=False)
        
        # (2) MLP Backbone
        mlp_modules = []
        in_dim = ch * seq
        for h_dim in layers:
            mlp_modules.append(nn.Linear(in_dim, h_dim))
            mlp_modules.append(nn.ReLU(inplace=True))
            mlp_modules.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        self.backbone = nn.Sequential(*mlp_modules)
        self.fc_emb = nn.Linear(in_dim, emb_dim)
        self.drop = nn.Dropout(dropout)

        # projector for SSL
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, proj_dim))

        self.classifier = None
        self.apply(self._init)
        print(f"Parameters count: {sum(p.numel() for p in self.parameters()):,}")

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

    def set_classifier(self, num_classes: int):
        self.classifier = nn.Sequential(
                        nn.Linear(self.fc_emb.out_features, 
                                self.fc_emb.out_features // 2),
                        nn.ReLU(),
                        nn.Linear(self.fc_emb.out_features // 2, num_classes))
    
    def set_linear_probe(self, num_classes: int):
        self.classifier = nn.Linear(self.fc_emb.out_features, num_classes)

    def forward(self, x, return_emb=False, return_proj=False):
        # x shape: (B, Ch, S)
        
        if self.lstm_layers > 0:
            # LSTM expects (B, S, Ch)
            x = x.transpose(1, 2)
            x, _ = self.lstm(x)
            # Transpose back to (B, Ch, S) for flattening
            x = x.transpose(1, 2)
        
        x = x.flatten(1)
        x = self.backbone(x)
        emb = self.fc_emb(x)

        if return_proj:
            return self.proj(emb)
        if return_emb:
            return emb
        return self.classifier(emb)