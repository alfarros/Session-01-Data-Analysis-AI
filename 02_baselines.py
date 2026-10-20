"""Tahap 3: Baseline ML klasik (seperti demo pemateri: LogReg + Random Forest)."""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import *

df = add_history_features(load_raw())
train, val, test = time_split(df)
Xtr, ytr = make_features(train); Xva, yva = make_features(val); Xte, yte = make_features(test)
print(f"train={len(train)} val={len(val)} test={len(test)} | late rate test={yte.mean():.1%}")

models = {
    "Dummy (selalu 'tidak late')": DummyClassifier(strategy="most_frequent"),
    "LogReg (tanpa weighting)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "LogReg (class_weight)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "RandomForest (class_weight)": RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=5, random_state=42),
}
rows = []
for name, m in models.items():
    m.fit(Xtr, ytr)
    pva, pte = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    # tanpa weighting -> threshold default 0.5 (perilaku seperti demo pemateri)
    th = 0.5 if ("tanpa" in name or "Dummy" in name) else best_threshold_f1(yva, pva)
    rows.append(evaluate(name, yte, pte, th))

res = pd.DataFrame(rows)
res.to_csv("results/baseline_results.csv", index=False)
print(res.drop(columns=["threshold"]).to_string(index=False))
