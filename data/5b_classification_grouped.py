"""
5b: expr_id 単位のグループ分割による分類評価(Path C の B 側)。
5_classification.py は順序による前70%/30%分割のため、同一 expr_id の行が
訓練/テストにまたがりデータリークが生じる。本スクリプトは GroupShuffleSplit
で問題単位に分割し、真の汎化性能を複数シードで測定する。
元スクリプトは A(公表値との直接比較用)として保持し、本結果と併記する。
"""
import json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

FEATURES_PATH = "gemma_mbpp_features.jsonl"
N_SEEDS = 20
TEST_SIZE = 0.3

rows = []
with open(FEATURES_PATH, "r") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

# GroupShuffleSplit does not care about order, but keeping both variants on the
# same canonical ordering means any difference between them is the split, not
# the row order.
rows.sort(key=lambda r: (r.get("row_index", -1), r["expr_id"], r["step_number"]))

X = np.array([r["feature"] for r in rows])
y = np.array([r["step_labels"] for r in rows])
g = np.array([r["expr_id"] for r in rows])
groups = list(g)

print(f"records={len(X)}  groups={len(set(groups))}  "
      f"labels={dict(zip(*np.unique(y, return_counts=True)))}")

aucs, auprs = [], []
skip_train, skip_test = 0, 0
for seed in range(N_SEEDS):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    tr_idx, te_idx = next(gss.split(X, y, groups=g))
    assert not (set(g[tr_idx]) & set(g[te_idx])), f"seed={seed}: group leak"

    if len(np.unique(y[tr_idx])) < 2:
        skip_train += 1
        continue
    if len(np.unique(y[te_idx])) < 2:
        skip_test += 1
        continue

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X[tr_idx], y[tr_idx])
    y_prob = clf.predict_proba(X[te_idx])[:, 1]

    aucs.append(roc_auc_score(y[te_idx], y_prob))
    auprs.append(average_precision_score(y[te_idx], y_prob))

print(f"\nevaluated={len(aucs)}/{N_SEEDS}  "
      f"skipped: train単一クラス={skip_train}, test単一クラス={skip_test}")
if aucs:
    a = np.array(aucs)
    print(f"AUROC: mean={a.mean():.4f}  std={a.std():.4f}  "
          f"min={a.min():.4f}  max={a.max():.4f}")
else:
    print("有効な分割なし → サンプル数不足。フルデータで再実行してください。")
