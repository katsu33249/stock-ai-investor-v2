"""
PHASE 3: バックテストスクリプト
================================
目的: LightGBMモデルの予測シグナルを使い、過去データで損益をシミュレーション

ルール:
  - モデルの予測確率 >= threshold → 買いシグナル
  - 保有期間: 5日（target_daysと一致）
  - 同日に複数シグナルが出た場合: スコア上位N銘柄のみ
  - 売買コスト: 往復0.2%（証券会社手数料＋スプレッド想定）
  - ポジションサイズ: 均等割り（1銘柄あたり資金/N）

評価指標:
  - 年率リターン
  - 最大ドローダウン（MDD）
  - シャープレシオ
  - 勝率
  - プロフィットファクター

出力:
  - data/ml/backtest_result.json
  - data/ml/backtest_equity.csv（資産推移）
"""

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
DATA_PATH    = Path("data/ml/training_data.csv")
MODEL_PATH   = Path("data/ml/model.pkl")
OUTPUT_DIR   = Path("data/ml")
RESULT_PATH  = OUTPUT_DIR / "backtest_result.json"
EQUITY_PATH  = OUTPUT_DIR / "backtest_equity.csv"

INITIAL_CAPITAL  = 1_000_000   # 初期資金（円）
MAX_POSITIONS    = 10          # 同時保有最大銘柄数
POSITION_RATIO   = 0.10        # 1銘柄あたり最大投資比率（10%）
HOLD_DAYS        = 5           # 保有期間（日）
TRADE_COST       = 0.002       # 往復コスト（0.2%）
SIGNAL_THRESHOLD = 0.50        # シグナル閾値（モデルの閾値より高く設定）
BACKTEST_START   = "2022-01-01"  # バックテスト開始日（学習期間外）

# ============================================================
# ロガー設定
# ============================================================
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}\n",
    level="INFO"
)
Path("data/logs").mkdir(parents=True, exist_ok=True)
logger.add("data/logs/backtest_{time:YYYYMMDD}.log", rotation="1 day", level="DEBUG")


# ============================================================
# 1. データ・モデル読み込み
# ============================================================
def load_data_and_model():
    logger.info("データ・モデル読み込み中...")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model     = saved["model"]
    feat_cols = saved["feat_cols"]
    threshold = saved["threshold"]

    logger.info(f"データ: {len(df):,}レコード | 特徴量: {len(feat_cols)}個 | 閾値: {threshold}")
    return df, model, feat_cols, threshold


# ============================================================
# 2. 全データに予測確率を付与
# ============================================================
def predict_all(df: pd.DataFrame, model, feat_cols: list) -> pd.DataFrame:
    logger.info("予測確率を計算中...")
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).values
    df = df.copy()
    df["pred_prob"] = model.predict(X)
    return df


# ============================================================
# 3. バックテスト
# ============================================================
def run_backtest(df: pd.DataFrame, threshold: float) -> dict:
    logger.info(f"バックテスト開始（{BACKTEST_START}以降）")

    # バックテスト期間のみ
    bt = df[df["date"] >= BACKTEST_START].copy()
    bt = bt.sort_values("date").reset_index(drop=True)
    logger.info(f"バックテスト対象: {len(bt):,}レコード | 期間: {bt['date'].min().date()}〜{bt['date'].max().date()}")

    # 日付リスト
    dates = sorted(bt["date"].unique())

    capital   = float(INITIAL_CAPITAL)
    equity    = []      # (date, total_assets)
    trades    = []      # 個別取引記録
    open_pos  = []      # [{ticker, entry_date, exit_date, entry_idx, size}]

    # ★ 正しい将来リターンをticker×dateで引けるようインデックス作成
    # entry日のreturn_5dは「過去5日」なので使えない
    # 代わりに: entry日のreturn_1dを5日分累積して将来リターンを近似
    # より正確に: 同銘柄の5営業日後の行のreturn_1dを使う
    for today in dates:
        today_pd = pd.Timestamp(today)

        # ① ポジションのクローズチェック（保有期間満了）
        still_open = []
        for pos in open_pos:
            if today_pd >= pos["exit_date"]:
                # 将来リターン = entry日から5営業日分のreturn_1dを累積
                ticker = pos["ticker"]
                entry_idx = pos["entry_idx"]
                ticker_rows = bt[bt["ticker"] == ticker].sort_values("date").reset_index(drop=True)
                end_idx = min(entry_idx + HOLD_DAYS, len(ticker_rows) - 1)
                daily_rets = ticker_rows.loc[entry_idx+1:end_idx, "return_1d"].values
                if len(daily_rets) > 0:
                    ret = float(np.prod(1 + daily_rets) - 1)
                else:
                    ret = 0.0
                pnl = pos["size"] * ret - pos["size"] * TRADE_COST
                capital += pos["size"] + pnl
                trades.append({
                    "entry_date": str(pos["entry_date"].date()),
                    "exit_date":  str(today_pd.date()),
                    "ticker":     ticker,
                    "ret":        round(ret, 4),
                    "pnl":        round(pnl, 2),
                    "win":        1 if pnl > 0 else 0,
                })
            else:
                still_open.append(pos)
        open_pos = still_open

        # ② 新規エントリー（スロットが空いている場合）
        slots = MAX_POSITIONS - len(open_pos)
        if slots > 0:
            today_signals = bt[
                (bt["date"] == today) &
                (bt["pred_prob"] >= SIGNAL_THRESHOLD)
            ].sort_values("pred_prob", ascending=False)

            held_tickers = {p["ticker"] for p in open_pos}
            today_signals = today_signals[~today_signals["ticker"].isin(held_tickers)]

            for _, sig in today_signals.head(slots).iterrows():
                size = capital * POSITION_RATIO
                if size <= 0:
                    continue
                # 銘柄の行インデックスを記録（将来リターン計算用）
                ticker_rows = bt[bt["ticker"] == sig["ticker"]].sort_values("date").reset_index(drop=True)
                idx_matches = ticker_rows[ticker_rows["date"] == today].index.tolist()
                if not idx_matches:
                    continue
                entry_idx = idx_matches[0]
                capital -= size
                open_pos.append({
                    "ticker":     sig["ticker"],
                    "entry_date": today_pd,
                    "exit_date":  today_pd + pd.Timedelta(days=HOLD_DAYS),
                    "entry_idx":  entry_idx,
                    "size":       size,
                    "entry_prob": round(float(sig["pred_prob"]), 4),
                })

        # ③ 資産記録 = 現金 + 保有ポジション簿価（DDの正確な計算のため）
        pos_value = sum(p["size"] for p in open_pos)
        equity.append({"date": today_pd, "capital": capital + pos_value})

    # 残りポジションを強制クローズ（最終日）
    for pos in open_pos:
        capital += pos["size"] * (1 - TRADE_COST)

    logger.info(f"取引数: {len(trades)}")
    return trades, equity, capital


# ============================================================
# 4. パフォーマンス評価
# ============================================================
def evaluate(trades: list, equity: list, final_capital: float) -> dict:
    eq_df = pd.DataFrame(equity).set_index("date")["capital"]

    # 年率リターン
    days = (eq_df.index[-1] - eq_df.index[0]).days
    total_ret = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    annual_ret = (1 + total_ret) ** (365 / max(days, 1)) - 1

    # 最大ドローダウン
    rolling_max = eq_df.cummax()
    drawdown    = (eq_df - rolling_max) / rolling_max
    mdd         = float(drawdown.min())

    # シャープレシオ（日次リターン）
    daily_ret = eq_df.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)

    # 勝率・プロフィットファクター
    if trades:
        wins   = [t for t in trades if t["win"] == 1]
        losses = [t for t in trades if t["win"] == 0]
        win_rate = len(wins) / len(trades)
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss   = abs(sum(t["pnl"] for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_ret = np.mean([t["ret"] for t in trades])
    else:
        win_rate = avg_ret = pf = 0.0

    result = {
        "period":         f"{eq_df.index[0].date()} 〜 {eq_df.index[-1].date()}",
        "initial_capital": INITIAL_CAPITAL,
        "final_capital":   round(final_capital, 0),
        "total_return":    round(total_ret, 4),
        "annual_return":   round(annual_ret, 4),
        "max_drawdown":    round(mdd, 4),
        "sharpe_ratio":    round(float(sharpe), 3),
        "win_rate":        round(win_rate, 4),
        "profit_factor":   round(pf, 3),
        "total_trades":    len(trades),
        "avg_return_per_trade": round(float(avg_ret), 4) if trades else 0.0,
        "判定": _judge(annual_ret, mdd, sharpe),
    }

    logger.info(f"年率リターン: {annual_ret:.1%}")
    logger.info(f"最大DD:       {mdd:.1%}")
    logger.info(f"シャープ:     {sharpe:.2f}")
    logger.info(f"勝率:         {win_rate:.1%}")
    logger.info(f"PF:           {pf:.2f}")
    logger.info(f"判定:         {result['判定']}")
    return result


def _judge(annual_ret, mdd, sharpe) -> str:
    if annual_ret >= 0.15 and mdd > -0.20 and sharpe >= 1.0:
        return "✅ 優秀（PHASE 4へ進む）"
    elif annual_ret >= 0.05 and mdd > -0.30:
        return "⚠️ 合格（パラメータ調整推奨）"
    else:
        return "❌ 要改善（モデル再学習推奨）"


# ============================================================
# メイン
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # データ・モデル読み込み
    df, model, feat_cols, threshold = load_data_and_model()

    # 予測確率付与
    df = predict_all(df, model, feat_cols)

    # バックテスト実行
    trades, equity, final_capital = run_backtest(df, threshold)

    # 評価
    result = evaluate(trades, equity, final_capital)

    # 保存
    output = {
        "created_at": datetime.now().isoformat(),
        "settings": {
            "initial_capital": INITIAL_CAPITAL,
            "max_positions":   MAX_POSITIONS,
            "hold_days":       HOLD_DAYS,
            "trade_cost":      TRADE_COST,
            "backtest_start":  BACKTEST_START,
        },
        "result": result,
        "sample_trades": trades[:20],  # 最初の20件のみ保存
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 資産推移CSV
    eq_df = pd.DataFrame(equity)
    eq_df.to_csv(EQUITY_PATH, index=False)

    logger.success(f"""
========================================
  PHASE 3 バックテスト完了
========================================
  期間:         {result['period']}
  年率リターン: {result['annual_return']:.1%}
  最大DD:       {result['max_drawdown']:.1%}
  シャープ:     {result['sharpe_ratio']:.2f}
  勝率:         {result['win_rate']:.1%}
  取引数:       {result['total_trades']}
  判定:         {result['判定']}
========================================
    """)


if __name__ == "__main__":
    main()
