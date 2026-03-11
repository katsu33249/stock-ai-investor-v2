"""Discord通知スクリプト - PHASE 2 モデル学習結果"""
import json
import os
import sys
import requests

with open("data/ml/model_info.json", encoding="utf-8") as f:
    info = json.load(f)

ev  = info["evaluation"]
cv  = info["cv_results"]
imp = info["feature_importance_top10"]
run_number = sys.argv[1] if len(sys.argv) > 1 else "?"

# 特徴量重要度TOP5
top5 = list(imp.items())[:5]
imp_text = "\n".join([f"  {i+1}. {k}: {v:.0f}" for i, (k, v) in enumerate(top5)])

# Foldごとの結果
fold_text = "\n".join([
    f"  Fold{f['fold']}: AUC={f['auc']:.3f} Prec={f['precision']:.3f} ({f['val_from']}～{f['val_to']})"
    for f in cv["folds"]
])

judge = ev["判定"]
lines = [
    f"📊 **ML PHASE 2 - モデル学習完了**",
    "",
    f"**評価結果** {judge}",
    f"・AUC:       {ev['auc']:.3f}  （目標: 0.55以上）",
    f"・Precision: {ev['precision']:.3f}",
    f"・Recall:    {ev['recall']:.3f}",
    f"・F1:        {ev['f1']:.3f}",
    "",
    f"**時系列CV（{len(cv['folds'])}分割）**",
    fold_text,
    "",
    "**特徴量重要度 TOP5**",
    imp_text,
    "",
    f"・学習データ: {info['total_records']:,}レコード",
    f"・閾値: {info['threshold']}",
    f"・モデル: Artifact ml-model-{run_number}",
]

msg = "\n".join(lines)
webhook = os.environ["DISCORD_WEBHOOK_URL"]
resp = requests.post(webhook, json={"content": msg})
print(f"Discord通知: {resp.status_code}")
