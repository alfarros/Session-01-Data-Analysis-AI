"""
Tahap 4: Deep Learning (MLP) untuk prediksi invoice telat bayar > 30 hari.

Ide: sama seperti baseline, tapi modelnya jaringan saraf tiruan (MLP) dengan
  - pos_weight  -> menangani data tidak seimbang (late hanya ~13%)
  - dropout     -> mengurangi overfitting (data kecil: 4.000 baris)
  - early stopping berdasarkan PR-AUC data validasi
  - threshold dipilih di data validasi, BUKAN di data test
Diulang dengan beberapa seed karena positif di test set cuma puluhan (hasil bisa goyang).
"""
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from common import *

SEEDS = [0, 1, 2, 3, 4]
EPOCHS, PATIENCE, BATCH = 200, 20, 64

df = add_history_features(load_raw())
train, val, test = time_split(df)
Xtr, ytr = make_features(train); Xva, yva = make_features(val); Xte, yte = make_features(test)

# Scaler HANYA di-fit di train (kalau di-fit di semua data = data leakage juga)
scaler = StandardScaler().fit(Xtr)
tensor = lambda X: torch.tensor(scaler.transform(X), dtype=torch.float32)
Xtr_t, Xva_t, Xte_t = tensor(Xtr), tensor(Xva), tensor(Xte)
ytr_t = torch.tensor(ytr, dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),            # 1 output = logit (belum probabilitas)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).numpy()


rows = []
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(Xtr_t.shape[1])
    pos_weight = torch.tensor((ytr == 0).sum() / (ytr == 1).sum(), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)   # sigmoid + BCE + bobot kelas positif
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=BATCH, shuffle=True)

    best_ap, best_state, wait = -1, None, 0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        ap = average_precision_score(yva, predict_proba(model, Xva_t))   # PR-AUC validasi
        if ap > best_ap:
            best_ap, best_state, wait = ap, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)

    th = best_threshold_f1(yva, predict_proba(model, Xva_t))
    r = evaluate(f"MLP (seed {seed})", yte, predict_proba(model, Xte_t), th)
    r["epochs_run"] = epoch + 1
    rows.append(r)
    print(f"seed {seed}: epochs={epoch + 1}  recall={r['recall']}  precision={r['precision']}  pr_auc={r['pr_auc']}")

res = pd.DataFrame(rows)
res.to_csv("results/mlp_results_per_seed.csv", index=False)
num = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
summary = res[num].agg(["mean", "std"]).round(3)
summary.to_csv("results/mlp_results_summary.csv")
print("\nRingkasan MLP (mean & std dari semua seed):")
print(summary)
