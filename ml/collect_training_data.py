"""
PHASE 1: 学習データ収集スクリプト
=====================================
目的: MLモデルの学習データを収集・整備する

収集データ:
  - 株価履歴（10年）: /v2/equities/bars/daily
  - 信用倍率（週次）: /v2/markets/margin-interest
  - TOPIX（地合い）: /v2/indices/topix
  - 財務データ: EDINETキャッシュから読込

目的変数:
  - 5営業日後に+3%以上上昇 = 1 / それ以外 = 0

出力:
  - data/ml/training_data.csv（特徴量 + 目的変数）
  - data/ml/feature_info.json（特徴量メタ情報）

実行:
  python collect_training_data.py
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

# ============================================================
# 設定
# ============================================================
JQUANTS_BASE_URL = "https://api.jquants.com/v2"
HISTORY_YEARS    = 10          # 取得する過去データ年数
TARGET_DAYS      = 5           # 目的変数: N営業日後
TARGET_RETURN    = 0.03        # 目的変数: +3%以上で正例
SLEEP_SEC        = 0.5         # APIコール間隔（レート制限対策）
OUTPUT_DIR       = Path("data/ml")
CACHE_PATH       = Path("data/cache/fundamental_cache.json")

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
    "data/logs/collect_training_{time:YYYYMMDD}.log",
    rotation="1 day", retention="30 days", level="DEBUG"
)


def _headers() -> dict:
    api_key = os.environ.get("JQUANTS_API_KEY", "")
    if not api_key:
        raise ValueError("JQUANTS_API_KEY が未設定です")
    return {"x-api-key": api_key}


def _to_code(ticker: str) -> str:
    """7011.T → 70110"""
    code = ticker.replace(".T", "")
    return code + "0" if len(code) == 4 else code


def get_tickers() -> list:
    """policy_keywords.yaml から対象銘柄リストを取得"""
    import yaml
    path = Path("config/policy_keywords.yaml")
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tickers = set()
    for theme in config.get("themes", {}).values():
        for t in theme.get("tickers", []):
            tickers.add(t)
    return sorted(list(tickers))


# ============================================================
# 1. 株価履歴取得（10年）
# ============================================================
def fetch_price_history(ticker: str) -> pd.DataFrame:
    """J-Quants から株価履歴を取得（10年分）"""
    code      = _to_code(ticker)
    end_date  = datetime.now()
    start_date = end_date - timedelta(days=365 * HISTORY_YEARS)

    all_data = []
    params = {
        "code": code,
        "from": start_date.strftime("%Y%m%d"),
        "to":   end_date.strftime("%Y%m%d"),
    }

    while True:
        res = requests.get(
            f"{JQUANTS_BASE_URL}/equities/bars/daily",
            headers=_headers(),
            params=params,
            timeout=30
        )
        if res.status_code == 429:
            logger.warning(f"レート制限 → 60秒待機")
            time.sleep(60)
            continue
        if res.status_code != 200:
            logger.warning(f"株価取得失敗({ticker}): {res.status_code}")
            return pd.DataFrame()

        body = res.json()
        data = body.get("data", [])
        all_data.extend(data)

        pagination_key = body.get("pagination_key")
        if not pagination_key:
            break
        params["pagination_key"] = pagination_key
        time.sleep(SLEEP_SEC)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.rename(columns={
        "Date": "date", "O": "open", "H": "high",
        "L": "low",  "C": "close", "V": "volume",
        "Vo": "volume", "Va": "turnover"
    })
    df["date"]   = pd.to_datetime(df["date"])
    df["ticker"] = ticker
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ============================================================
# 2. TOPIX取得（地合い特徴量用）
# ============================================================
def fetch_topix(start_date: str, end_date: str) -> pd.DataFrame:
    """TOPIXの日次データを取得"""
    all_data = []
    params = {"from": start_date, "to": end_date}

    while True:
        res = requests.get(
            f"{JQUANTS_BASE_URL}/indices/topix",
            headers=_headers(),
            params=params,
            timeout=30
        )
        if res.status_code != 200:
            logger.warning(f"TOPIX取得失敗: {res.status_code}")
            return pd.DataFrame()

        body = res.json()
        data = body.get("topix", body.get("data", []))
        all_data.extend(data)

        pagination_key = body.get("pagination_key")
        if not pagination_key:
            break
        params["pagination_key"] = pagination_key
        time.sleep(SLEEP_SEC)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.rename(columns={"Date": "date", "C": "topix_close", "Close": "topix_close"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "topix_close"]].sort_values("date").reset_index(drop=True)
    df["topix_return_5d"] = df["topix_close"].pct_change(5)
    df["topix_return_20d"] = df["topix_close"].pct_change(20)
    return df


# ============================================================
# 3. 信用倍率取得
# ============================================================
def fetch_margin_history(ticker: str) -> pd.DataFrame:
    """信用倍率の履歴を取得"""
    code = _to_code(ticker)
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=365 * HISTORY_YEARS)

    res = requests.get(
        f"{JQUANTS_BASE_URL}/markets/margin-interest",
        headers=_headers(),
        params={
            "code": code,
            "from": start_date.strftime("%Y%m%d"),
            "to":   end_date.strftime("%Y%m%d"),
        },
        timeout=30
    )
    if res.status_code != 200:
        return pd.DataFrame()

    data = res.json().get("data", [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])

    def safe_float(v):
        try: return float(v) if v else None
        except: return None

    df["long_vol"]  = df.get("LongVol",  pd.Series()).apply(safe_float)
    df["short_vol"] = df.get("ShrtVol",  pd.Series()).apply(safe_float)
    df["margin_ratio"] = df.apply(
        lambda r: round(r["long_vol"] / r["short_vol"], 2)
        if r["short_vol"] and r["short_vol"] > 0 else None, axis=1
    )
    return df[["date", "margin_ratio"]].sort_values("date").reset_index(drop=True)


# ============================================================
# 4. テクニカル特徴量の計算
# ============================================================
def calc_features(df: pd.DataFrame) -> pd.DataFrame:
    """株価DataFrameからテクニカル特徴量を計算"""
    d = df.copy()

    # --- リターン系 ---
    d["return_1d"]  = d["close"].pct_change(1)
    d["return_5d"]  = d["close"].pct_change(5)
    d["return_20d"] = d["close"].pct_change(20)
    d["return_60d"] = d["close"].pct_change(60)

    # --- 移動平均 ---
    d["ma5"]  = d["close"].rolling(5).mean()
    d["ma25"] = d["close"].rolling(25).mean()
    d["ma75"] = d["close"].rolling(75).mean()

    # 移動平均乖離率
    d["ma5_dev"]  = (d["close"] - d["ma5"])  / d["ma5"]
    d["ma25_dev"] = (d["close"] - d["ma25"]) / d["ma25"]
    d["ma75_dev"] = (d["close"] - d["ma75"]) / d["ma75"]

    # MA位置（75日MAより上か）
    d["above_ma75"] = (d["close"] > d["ma75"]).astype(int)

    # --- RSI(14) ---
    delta = d["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    d["rsi14"] = 100 - (100 / (1 + rs))

    # --- ボリンジャーバンド（25日）---
    bb_mid   = d["close"].rolling(25).mean()
    bb_std   = d["close"].rolling(25).std()
    d["bb_upper"] = bb_mid + 2 * bb_std
    d["bb_lower"] = bb_mid - 2 * bb_std
    d["bb_pct"]   = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"])

    # --- MACD ---
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    d["macd"]        = macd
    d["macd_signal"] = signal
    d["macd_hist"]   = macd - signal
    d["macd_golden"] = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(int)

    # --- 出来高系 ---
    d["vol_ma5"]   = d["volume"].rolling(5).mean()
    d["vol_ma20"]  = d["volume"].rolling(20).mean()
    d["vol_ratio"] = d["volume"] / d["vol_ma20"].replace(0, np.nan)

    # --- ゴールデンクロス ---
    d["gc_25_75"] = ((d["ma25"] > d["ma75"]) & (d["ma25"].shift(1) <= d["ma75"].shift(1))).astype(int)

    # --- 高値・安値からの乖離 ---
    d["high_52w"] = d["high"].rolling(252).max()
    d["low_52w"]  = d["low"].rolling(252).min()
    d["from_high"] = (d["close"] - d["high_52w"]) / d["high_52w"]
    d["from_low"]  = (d["close"] - d["low_52w"])  / d["low_52w"]

    return d


# ============================================================
# 5. 目的変数の計算
# ============================================================
def calc_target(df: pd.DataFrame, days: int = TARGET_DAYS, threshold: float = TARGET_RETURN) -> pd.DataFrame:
    """N営業日後のリターンが threshold 以上なら 1"""
    future_close  = df["close"].shift(-days)
    future_return = (future_close - df["close"]) / df["close"]
    df["future_return"] = future_return
    df["target"]        = (future_return >= threshold).astype(int)
    return df


# ============================================================
# 6. ファンダメンタルデータの結合
# ============================================================
def load_fundamental_cache() -> dict:
    """EDINETキャッシュから財務データを読込"""
    if not CACHE_PATH.exists():
        logger.warning("EDINETキャッシュが見つかりません。財務特徴量はスキップします")
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def add_fundamental_features(df: pd.DataFrame, ticker: str, fund_cache: dict) -> pd.DataFrame:
    """財務特徴量を追加（全行に同じ値を付与）"""
    fund = fund_cache.get(ticker, {})
    df["per"]            = fund.get("per")
    df["roe"]            = fund.get("roe")
    df["roa"]            = fund.get("roa")
    df["operating_margin"] = fund.get("operating_margin")
    df["revenue_growth"] = fund.get("revenue_growth")
    df["equity_ratio"]   = fund.get("equity_ratio")
    df["debt_to_equity"] = fund.get("debt_to_equity")
    df["dividend_yield"] = fund.get("dividend_yield")
    df["credit_score"]   = fund.get("credit_score")

    # PBR = PER × ROE / 100
    per = fund.get("per")
    roe = fund.get("roe")
    if per and roe and per > 0 and roe > 0:
        df["pbr"] = round(per * (roe / 100), 2)
    else:
        df["pbr"] = None

    return df


# ============================================================
# メイン処理
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    tickers   = get_tickers()
    fund_cache = load_fundamental_cache()
    logger.info(f"対象銘柄数: {len(tickers)}")
    logger.info(f"財務キャッシュ: {len(fund_cache)}銘柄")

    # TOPIX取得（地合い特徴量）
    end_date   = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365 * HISTORY_YEARS)).strftime("%Y%m%d")
    logger.info("TOPIX取得中...")
    topix_df = fetch_topix(start_date, end_date)
    logger.info(f"TOPIX: {len(topix_df)}レコード")

    all_frames = []
    success, skip, fail = 0, 0, 0

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i}/{len(tickers)}] {ticker} 処理中...")

        # 株価履歴
        price_df = fetch_price_history(ticker)
        if price_df.empty:
            logger.warning(f"  株価データなし → スキップ")
            fail += 1
            continue

        if len(price_df) < 100:
            logger.warning(f"  データ不足({len(price_df)}行) → スキップ")
            skip += 1
            continue

        # テクニカル特徴量
        price_df = calc_features(price_df)

        # 目的変数
        price_df = calc_target(price_df)

        # 信用倍率
        margin_df = fetch_margin_history(ticker)
        if not margin_df.empty:
            # 週次データを日次にマージ（前方補完）
            price_df = pd.merge_asof(
                price_df.sort_values("date"),
                margin_df.sort_values("date"),
                on="date",
                direction="backward"
            )
        else:
            price_df["margin_ratio"] = None

        # TOPIX結合
        if not topix_df.empty:
            price_df = pd.merge_asof(
                price_df.sort_values("date"),
                topix_df.sort_values("date"),
                on="date",
                direction="backward"
            )

        # ファンダメンタル特徴量
        price_df = add_fundamental_features(price_df, ticker, fund_cache)

        all_frames.append(price_df)
        success += 1

        time.sleep(SLEEP_SEC)

    if not all_frames:
        logger.error("データが取得できませんでした")
        return

    # 全銘柄を結合
    logger.info("データを結合中...")
    full_df = pd.concat(all_frames, ignore_index=True)

    # 末尾（TARGET_DAYS行）は目的変数がNaN → 除去
    full_df = full_df.dropna(subset=["target"])

    # 特徴量リスト
    feature_cols = [
        # テクニカル
        "return_1d", "return_5d", "return_20d", "return_60d",
        "ma5_dev", "ma25_dev", "ma75_dev", "above_ma75",
        "rsi14", "bb_pct",
        "macd_hist", "macd_golden",
        "vol_ratio",
        "gc_25_75",
        "from_high", "from_low",
        # 需給
        "margin_ratio",
        # 地合い
        "topix_return_5d", "topix_return_20d",
        # ファンダメンタル
        "per", "pbr", "roe", "roa",
        "operating_margin", "revenue_growth",
        "equity_ratio", "debt_to_equity",
        "dividend_yield", "credit_score",
    ]

    output_cols = ["date", "ticker", "close"] + feature_cols + ["future_return", "target"]
    output_df   = full_df[[c for c in output_cols if c in full_df.columns]]

    # 保存
    output_path = OUTPUT_DIR / "training_data.csv"
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 特徴量メタ情報を保存
    feature_info = {
        "created_at":    datetime.now().isoformat(),
        "tickers":       tickers,
        "total_records": len(output_df),
        "positive_rate": float(output_df["target"].mean()),
        "feature_cols":  feature_cols,
        "target_days":   TARGET_DAYS,
        "target_return": TARGET_RETURN,
        "date_range": {
            "from": str(output_df["date"].min()),
            "to":   str(output_df["date"].max()),
        }
    }
    info_path = OUTPUT_DIR / "feature_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)

    # サマリー
    logger.success(f"""
========================================
  PHASE 1 完了
========================================
  成功: {success}銘柄
  スキップ: {skip}銘柄
  失敗: {fail}銘柄
  総レコード数: {len(output_df):,}
  正例率（上昇）: {output_df['target'].mean():.1%}
  期間: {output_df['date'].min().date()} 〜 {output_df['date'].max().date()}
  出力: {output_path}
========================================
    """)


if __name__ == "__main__":
    main()
