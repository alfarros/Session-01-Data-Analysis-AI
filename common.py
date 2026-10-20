"""Fungsi bersama: load data, buat fitur point-in-time, split berdasarkan waktu, evaluasi."""
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)

TARGET = "late30"
# Kolom yang baru diketahui SETELAH invoice terbit -> DATA LEAKAGE, wajib dibuang
LEAK_COLS = ["paid_date", "days_overdue"]


def load_raw(path="dataset/invoices_synthetic.csv"):
    return pd.read_csv(path, parse_dates=["issue_date", "due_date", "paid_date"])


def add_history_features(df):
    """Riwayat customer yang SUDAH diketahui pada saat invoice terbit (point-in-time).
    Invoice lama hanya dihitung kalau sudah LUNAS sebelum issue_date invoice sekarang."""
    df = df.sort_values("issue_date").reset_index(drop=True)
    n_prev, late_rate_prev, avg_late_prev, open_amt = [], [], [], []
    for _, row in df.iterrows():
        past = df[(df.customer_id == row.customer_id) & (df.issue_date < row.issue_date)]
        settled = past[past.paid_date.notna() & (past.paid_date <= row.issue_date)]
        n_prev.append(len(settled))
        late_rate_prev.append(settled[TARGET].mean() if len(settled) else np.nan)
        avg_late_prev.append(settled.days_overdue.mean() if len(settled) else np.nan)
        still_open = past[~past.index.isin(settled.index)]
        open_amt.append(still_open.amount.sum())
    df["hist_n_paid"] = n_prev
    df["hist_late_rate"] = late_rate_prev
    df["hist_avg_days_late"] = avg_late_prev
    df["open_amount"] = open_amt
    # Customer baru (belum punya riwayat): isi nilai umum + beri penanda
    df["is_new_customer"] = (df.hist_n_paid == 0).astype(int)
    df["hist_late_rate"] = df.hist_late_rate.fillna(0.13)
    df["hist_avg_days_late"] = df.hist_avg_days_late.fillna(0.0)
    return df


def make_features(df):
    X = pd.DataFrame(index=df.index, data={
        "log_amount": np.log(df.amount.values),
        "n_items": df.n_items,
        "discount_pct": df.discount_pct,
        "terms_days": df.terms_days,
        "issue_month": df.issue_date.dt.month,
        "hist_n_paid": df.hist_n_paid,
        "hist_late_rate": df.hist_late_rate,
        "hist_avg_days_late": df.hist_avg_days_late,
        "log_open_amount": np.log1p(df.open_amount),
        "is_new_customer": df.is_new_customer,
    })
    # kolom dummy harus konsisten di train/val/test -> dipaksa lewat kategori tetap
    seg = pd.Categorical(df.segment, ["retail", "grosir", "horeca"])
    pay = pd.Categorical(df.payment_method, ["transfer", "cash", "giro"])
    X = pd.concat([X,
                   pd.get_dummies(seg, prefix="segment", dtype=int).set_index(df.index),
                   pd.get_dummies(pay, prefix="pay", dtype=int).set_index(df.index)], axis=1)
    return X, df[TARGET].values


def time_split(df, train=0.70, val=0.15):
    """Split BERDASARKAN WAKTU (bukan acak): masa lalu -> masa depan, seperti kondisi nyata."""
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def evaluate(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": name, "threshold": round(float(threshold), 3),
        "accuracy": round((tp + tn) / len(y_true), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 3),
        "pr_auc": round(average_precision_score(y_true, y_prob), 3),
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
    }


def best_threshold_f1(y_true, y_prob):
    """Pilih threshold terbaik di data VALIDASI (bukan test) supaya tidak curang."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in grid]
    return grid[int(np.argmax(scores))]
