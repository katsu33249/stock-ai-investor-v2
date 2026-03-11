"""
PHASE 2: LightGBMモデル学習スクリプト
=====================================
目的: PHASE 1で収集した学習データでMLモデルを構築する

学習方法:
  - TimeSeriesSplit（時系列クロスバリデーション）
  - ランダム分割は未来データのリークが起きるため使用不可

評価指標:
  - AUC（ROC）: 0.55以上で有効
  - Precision: 予測した銘柄のうち実際に上昇した割合
  - Recall: 実際に上昇した銘柄のうち予測できた割合

出力:
  - data/ml/model.pkl（学習済みモデル）
  - data/ml/model_info.json（精度・特徴量重要度）

実行:
  python ml/train_model.py
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

warnings.filterwarnings("ignore")

# ============================================================
# 設定
# ============================================================
DATA_PATH   = Path("data/ml/training_data.csv")
OUTPUT_DIR  = Path("data/ml")
MODEL_PATH  = OUTPUT_DIR / "model.pkl"
INFO_PATH   = OUTPUT_DIR / "model_info.json"

N_SPLITS    = 5       # 時系列CVの分割数
EARLY_STOP  = 50      # Early stopping rounds
THRESHOLD   = 0.35    # 買いシグナルの確率閾値

# ============================================================
# ロガー設定
# ============================================================
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}\n",
    level="INFO"
)
logger.add(
    "data/logs/train_model_{time:YYYYMMDD}.log",
    rotation="1 day", retention="30 days", level="DEBUG"
)

# ============================================================
# 特徴量リスト
# ============================================================
FEATURE_COLS = [
    "return_1d", "return_5d", "return_20d", "return_60d",
    "ma5_dev", "ma25_dev", "ma75_dev", "above_ma75",
    "rsi14", "bb_pct",
    "macd_hist", "macd_golden",
    "vol_ratio",
    "gc_25_75",
    "from_high", "from_low",
    "margin_ratio",
    "topix_return_5d", "topix_return_20d",
    "per", "pbr", "roe", "roa",
    "operating_margin", "revenue_growth",
    "equity_ratio", "debt_to_equity",
    "dividend_yield", "credit_score",
]


# ============================================================
# 1. データ読み込み・前処理
# ============================================================
def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """学習データを読み込み、特徴量と目的変数を返す"""
    logger.info(f"データ読み込み中: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    logger.info(f"読み込み完了: {len(df):,}レコード")

    # 使用する特徴量（存在するもののみ）
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    logger.info(f"特徴量数: {len(feat_cols)}")

    # 無限値をNaNに変換
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)

    # 日付でソート（時系列順を保証）
    df = df.sort_values("date").reset_index(drop=True)

    X = df[feat_cols].values
    y = df["target"].values
    dates = df["date"].values

    logger.info(f"正例率: {y.mean():.1%}")
    return df, X, y, dates, feat_cols


# ============================================================
# 2. 時系列クロスバリデーション
# ============================================================
def time_series_cv(X, y, dates, feat_cols: list) -> dict:
    """TimeSeriesSplitでクロスバリデーション"""
    try:
        import lightgbm as lgb
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    except ImportError as e:
        raise ImportError(f"必要なライブラリが不足しています: {e}")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    aucs, precisions, recalls, f1s = [], [], [], []
    fold_results = []

    lgb_params = {
        "objective":        "binary",
        "metric":           "auc",
        "boosting_type":    "gbdt",
        "num_leaves":       63,
        "learning_rate":    0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples": 50,
        "lambda_l1":        0.1,
        "lambda_l2":        0.1,
        "verbose":          -1,
        "random_state":     42,
    }

    logger.info(f"時系列CV開始（{N_SPLITS}分割）")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_date = pd.Timestamp(dates[train_idx[-1]]).date()
        val_date_from = pd.Timestamp(dates[val_idx[0]]).date()
        val_date_to   = pd.Timestamp(dates[val_idx[-1]]).date()

        logger.info(f"  Fold {fold}: 訓練〜{train_date} | 検証 {val_date_from}〜{val_date_to}")

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)
        dval   = lgb.Dataset(X_val,   label=y_val,   feature_name=feat_cols, reference=dtrain)

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
        )

        y_pred_prob = model.predict(X_val)
        y_pred      = (y_pred_prob >= THRESHOLD).astype(int)

        auc  = roc_auc_score(y_val, y_pred_prob)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec  = recall_score(y_val, y_pred, zero_division=0)
        f1   = f1_score(y_val, y_pred, zero_division=0)

        aucs.append(auc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        logger.info(f"    AUC:{auc:.3f} Precision:{prec:.3f} Recall:{rec:.3f} F1:{f1:.3f}")

        fold_results.append({
            "fold": fold,
            "auc": round(auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "val_from": str(val_date_from),
            "val_to":   str(val_date_to),
        })

    cv_results = {
        "auc_mean":       round(float(np.mean(aucs)), 4),
        "auc_std":        round(float(np.std(aucs)), 4),
        "precision_mean": round(float(np.mean(precisions)), 4),
        "recall_mean":    round(float(np.mean(recalls)), 4),
        "f1_mean":        round(float(np.mean(f1s)), 4),
        "folds":          fold_results,
    }

    logger.info(f"CV結果: AUC={cv_results['auc_mean']:.3f}±{cv_results['auc_std']:.3f} "
                f"Precision={cv_results['precision_mean']:.3f} "
                f"Recall={cv_results['recall_mean']:.3f}")

    return cv_results


# ============================================================
# 3. 全データで最終モデル学習
# ============================================================
def train_final_model(X, y, feat_cols: list):
    """全データで最終モデルを学習"""
    import lightgbm as lgb

    logger.info("最終モデル学習中（全データ使用）...")

    lgb_params = {
        "objective":        "binary",
        "metric":           "auc",
        "boosting_type":    "gbdt",
        "num_leaves":       63,
        "learning_rate":    0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples": 50,
        "lambda_l1":        0.1,
        "lambda_l2":        0.1,
        "verbose":          -1,
        "random_state":     42,
    }

    dtrain = lgb.Dataset(X, label=y, feature_name=feat_cols)
    model  = lgb.train(lgb_params, dtrain, num_boost_round=500)

    # 特徴量重要度（gainベース）
    importance = dict(zip(
        feat_cols,
        model.feature_importance(importance_type="gain").tolist()
    ))
    # 上位10件でソート
    importance_sorted = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    )

    logger.info("特徴量重要度 TOP10:")
    for feat, imp in importance_sorted.items():
        logger.info(f"  {feat:<25}: {imp:.1f}")

    return model, importance_sorted


# ============================================================
# メイン処理
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    # データ読み込み
    df, X, y, dates, feat_cols = load_data()

    # 時系列CV
    cv_results = time_series_cv(X, y, dates, feat_cols)

    # 最終モデル学習
    model, importance = train_final_model(X, y, feat_cols)

    # モデル保存
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feat_cols": feat_cols, "threshold": THRESHOLD}, f)
    logger.info(f"モデル保存: {MODEL_PATH}")

    # メタ情報保存
    model_info = {
        "created_at":       datetime.now().isoformat(),
        "total_records":    len(df),
        "feature_cols":     feat_cols,
        "threshold":        THRESHOLD,
        "cv_results":       cv_results,
        "feature_importance_top10": importance,
        "evaluation": {
            "auc":       cv_results["auc_mean"],
            "precision": cv_results["precision_mean"],
            "recall":    cv_results["recall_mean"],
            "f1":        cv_results["f1_mean"],
            "判定": "✅ 有効" if cv_results["auc_mean"] >= 0.55 else "⚠️ 要改善",
        }
    }
    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    # サマリー
    ev = model_info["evaluation"]
    logger.success(f"""
========================================
  PHASE 2 完了
========================================
  AUC:       {ev['auc']:.3f}  {ev['判定']}
  Precision: {ev['precision']:.3f}
  Recall:    {ev['recall']:.3f}
  F1:        {ev['f1']:.3f}
  モデル: {MODEL_PATH}
========================================
    """)


if __name__ == "__main__":
    main()
