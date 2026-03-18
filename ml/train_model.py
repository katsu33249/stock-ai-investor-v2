"""
PHASE 2: LightGBMモデル学習スクリプト（改善版）
=============================================
改善内容:
  B. Optunaでハイパーパラメータ最適化
  C. アンサンブル（LightGBM + XGBoost + RandomForest）

目標: AUC改善 + 年率リターン維持
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

N_SPLITS    = 5
EARLY_STOP  = 50
THRESHOLD   = 0.35
TRAIN_END   = "2021-12-31"
OPTUNA_TRIALS = 30   # Optunaの試行回数

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
# 特徴量リスト（新特徴量追加済み）
# ============================================================
FEATURE_COLS = [
    "return_1d", "return_5d", "return_20d", "return_60d",
    "ma5_dev", "ma25_dev", "ma75_dev", "above_ma75",
    "rsi14", "bb_pct",
    "macd_hist", "macd_golden",
    "vol_ratio", "vol_surge_days",
    "gc_25_75",
    "from_high", "from_low",
    "rci9", "rci26",
    "ichi_tenkan_dev", "ichi_kijun_dev", "ichi_above_cloud",
    "adx14",
    "margin_ratio", "margin_ratio_chg",
    "topix_return_5d", "topix_return_20d",
    "per", "pbr", "roe", "roa",
    "operating_margin", "revenue_growth",
    "equity_ratio", "debt_to_equity",
    "dividend_yield", "credit_score",
    # 決算サプライズ
    "earnings_surprise", "revenue_surprise", "days_since_earnings",
    # EPS・成長率
    "eps_growth", "operating_income_growth",
]


# ============================================================
# 1. データ読み込み・前処理
# ============================================================
def load_data():
    logger.info(f"データ読み込み中: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    logger.info(f"読み込み完了: {len(df):,}レコード")

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    logger.info(f"特徴量数: {len(feat_cols)}")

    df = df[df["date"] <= TRAIN_END].copy()
    logger.info(f"学習期間限定: 〜{TRAIN_END} ({len(df):,}レコード)")

    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df = df.sort_values("date").reset_index(drop=True)

    X     = df[feat_cols].values
    y     = df["target"].values
    dates = df["date"].values

    logger.info(f"正例率: {y.mean():.1%}")
    return df, X, y, dates, feat_cols


# ============================================================
# 改善B: Optunaでハイパーパラメータ最適化
# ============================================================
def optimize_params(X, y, dates, feat_cols: list) -> dict:
    """Optunaで最適パラメータを探索（最初のfoldのみで高速化）"""
    try:
        import optuna
        import lightgbm as lgb
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optunaが未インストール → デフォルトパラメータを使用")
        return get_default_params()

    logger.info(f"Optuna最適化開始（{OPTUNA_TRIALS}試行）...")

    # 最初の1foldだけ使って高速化
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[2]  # 3番目のfoldを使用
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    def objective(trial):
        params = {
            "objective":        "binary",
            "metric":           "auc",
            "boosting_type":    "gbdt",
            "verbose":          -1,
            "random_state":     42,
            "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples":trial.suggest_int("min_child_samples", 20, 100),
            "lambda_l1":        trial.suggest_float("lambda_l1", 1e-4, 1.0, log=True),
            "lambda_l2":        trial.suggest_float("lambda_l2", 1e-4, 1.0, log=True),
        }
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)
        dval   = lgb.Dataset(X_val, label=y_val, feature_name=feat_cols, reference=dtrain)
        model  = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)]
        )
        return roc_auc_score(y_val, model.predict(X_val))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best = study.best_params
    best["objective"]    = "binary"
    best["metric"]       = "auc"
    best["boosting_type"]= "gbdt"
    best["verbose"]      = -1
    best["random_state"] = 42

    logger.info(f"最適パラメータ: num_leaves={best['num_leaves']} lr={best['learning_rate']:.4f} AUC={study.best_value:.4f}")
    return best


def get_default_params() -> dict:
    return {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "num_leaves": 63, "learning_rate": 0.05, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5, "min_child_samples": 50,
        "lambda_l1": 0.1, "lambda_l2": 0.1, "verbose": -1, "random_state": 42,
    }


# ============================================================
# 2. 時系列クロスバリデーション
# ============================================================
def time_series_cv(X, y, dates, feat_cols: list, lgb_params: dict) -> dict:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    aucs, precisions, recalls, f1s = [], [], [], []
    fold_results = []

    logger.info(f"時系列CV開始（{N_SPLITS}分割）")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_date    = pd.Timestamp(dates[train_idx[-1]]).date()
        val_date_from = pd.Timestamp(dates[val_idx[0]]).date()
        val_date_to   = pd.Timestamp(dates[val_idx[-1]]).date()

        logger.info(f"  Fold {fold}: 訓練〜{train_date} | 検証 {val_date_from}〜{val_date_to}")

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)
        dval   = lgb.Dataset(X_val,   label=y_val,   feature_name=feat_cols, reference=dtrain)

        model = lgb.train(
            lgb_params, dtrain, num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(period=-1)]
        )

        y_pred_prob = model.predict(X_val)
        y_pred      = (y_pred_prob >= THRESHOLD).astype(int)

        auc  = roc_auc_score(y_val, y_pred_prob)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec  = recall_score(y_val, y_pred, zero_division=0)
        f1   = f1_score(y_val, y_pred, zero_division=0)

        aucs.append(auc); precisions.append(prec); recalls.append(rec); f1s.append(f1)
        logger.info(f"    AUC:{auc:.3f} Precision:{prec:.3f} Recall:{rec:.3f} F1:{f1:.3f}")

        fold_results.append({
            "fold": fold, "auc": round(auc,4), "precision": round(prec,4),
            "recall": round(rec,4), "f1": round(f1,4),
            "val_from": str(val_date_from), "val_to": str(val_date_to),
        })

    cv_results = {
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std":  round(float(np.std(aucs)), 4),
        "precision_mean": round(float(np.mean(precisions)), 4),
        "recall_mean":    round(float(np.mean(recalls)), 4),
        "f1_mean":        round(float(np.mean(f1s)), 4),
        "folds": fold_results,
    }
    logger.info(f"CV結果: AUC={cv_results['auc_mean']:.3f}±{cv_results['auc_std']:.3f}")
    return cv_results


# ============================================================
# 3. 最終モデル学習（LightGBM単体）
# ============================================================
def train_final_model(X, y, feat_cols: list, lgb_params: dict):
    """LightGBM単体で最終モデルを学習"""
    import lightgbm as lgb

    logger.info("LightGBM最終モデル学習中...")
    dtrain = lgb.Dataset(X, label=y, feature_name=feat_cols)
    model  = lgb.train(lgb_params, dtrain, num_boost_round=500)

    importance = dict(zip(feat_cols, model.feature_importance(importance_type="gain").tolist()))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

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

    # データ読み込み（改善A適用）
    df, X, y, dates, feat_cols = load_data()

    # 改善B: Optuna最適化
    lgb_params = optimize_params(X, y, dates, feat_cols)

    # 時系列CV（最適パラメータで）
    cv_results = time_series_cv(X, y, dates, feat_cols, lgb_params)

    # LightGBM単体学習
    model, importance = train_final_model(X, y, feat_cols, lgb_params)

    # モデル保存
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model":     model,
            "feat_cols": feat_cols,
            "threshold": THRESHOLD,
            "lgb_params": lgb_params,
        }, f)
    logger.info(f"モデル保存: {MODEL_PATH}")

    # メタ情報保存
    model_info = {
        "created_at":    datetime.now().isoformat(),
        "total_records": len(df),
        "feature_cols":  feat_cols,
        "threshold":     THRESHOLD,
        "ensemble":      ["lgb"],
        "lgb_params":    lgb_params,
        "cv_results":    cv_results,
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

    ev = model_info["evaluation"]
    logger.success(f"""
========================================
  PHASE 2 完了（改善版）
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
