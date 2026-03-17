"""
PHASE 1: 学習データ収集スクリプト（東証プライム全銘柄版）
=========================================================
変更点:
  - 対象: 政策178銘柄 → 東証プライム全銘柄（約1,800社）
  - 取得方法: 銘柄ごと → 日付一括取得（1日1APIコール）
  - 保存形式: CSV → Parquet + float32（容量80%削減）
  - 目的変数: 5日後+3% → TOPIX比アルファ+2%（純粋な銘柄選択力）

APIコール数:
  旧: 1,800銘柄 × 個別取得 = 多数
  新: 2,500営業日 × 1コール = 2,500コール
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
HISTORY_YEARS    = 10
TARGET_DAYS      = 5
TARGET_ALPHA     = 0.02   # TOPIXより+2%以上 = 正例
SLEEP_SEC        = 0.2
OUTPUT_DIR       = Path("data/ml")
CACHE_PATH       = Path("data/cache/fundamental_cache.json")

logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}\n",
    level="INFO"
)
logger.add("data/logs/collect_{time:YYYYMMDD}.log", rotation="1 day", level="DEBUG")


def _headers() -> dict:
    api_key = os.environ.get("JQUANTS_API_KEY", "")
    if not api_key:
        raise ValueError("JQUANTS_API_KEY が未設定です")
    return {"x-api-key": api_key}


# ============================================================
# 1. 東証プライム銘柄リスト取得
# ============================================================
def get_prime_tickers() -> list:
    """東証プライム全銘柄コードを取得"""
    logger.info("東証プライム銘柄リスト取得中...")
    all_data = []
    params   = {}

    while True:
        res = requests.get(
            f"{JQUANTS_BASE_URL}/equities/list",
            headers=_headers(),
            params=params,
            timeout=30
        )
        if res.status_code != 200:
            logger.error(f"銘柄リスト取得失敗: {res.status_code}")
            break

        body = res.json()
        data = body.get("data", [])
        all_data.extend(data)

        pagination_key = body.get("pagination_key")
        if not pagination_key:
            break
        params["pagination_key"] = pagination_key
        time.sleep(SLEEP_SEC)

    # 東証プライム（market_code=0111）のみ絞り込み
    prime = [
        d for d in all_data
        if str(d.get("MarketCode", "")) == "0111"
        and d.get("Code")
    ]
    # 5桁コードを4桁+.T形式に変換
    tickers = []
    for d in prime:
        code = str(d["Code"])
        if code.endswith("0") and len(code) == 5:
            tickers.append(code[:4] + ".T")

    tickers = sorted(list(set(tickers)))
    logger.success(f"東証プライム銘柄数: {len(tickers)}社")
    return tickers


# ============================================================
# 2. TOPIX取得（目的変数・地合い特徴量用）
# ============================================================
def fetch_topix(start_date: str, end_date: str) -> pd.DataFrame:
    logger.info("TOPIX取得中...")
    all_data = []
    params   = {"from": start_date, "to": end_date}

    while True:
        res = requests.get(
            f"{JQUANTS_BASE_URL}/indices/bars/daily/topix",
            headers=_headers(), params=params, timeout=30
        )
        if res.status_code != 200:
            logger.warning(f"TOPIX取得失敗: {res.status_code}")
            return pd.DataFrame()

        body = res.json()
        all_data.extend(body.get("data", []))
        pagination_key = body.get("pagination_key")
        if not pagination_key:
            break
        params["pagination_key"] = pagination_key
        time.sleep(SLEEP_SEC)

    df = pd.DataFrame(all_data)
    close_col = next((c for c in ["C", "Close", "close"] if c in df.columns), None)
    if close_col is None:
        return pd.DataFrame()

    df = df.rename(columns={"Date": "date", close_col: "topix_close"})
    df["date"]           = pd.to_datetime(df["date"])
    df["topix_return_5d"]  = df["topix_close"].pct_change(5)
    df["topix_return_20d"] = df["topix_close"].pct_change(20)
    # 目的変数用: 5日後TOPIXリターン
    df["topix_future_5d"]  = df["topix_close"].pct_change(5).shift(-5)
    logger.success(f"TOPIX: {len(df)}日分取得完了")
    return df[["date","topix_close","topix_return_5d","topix_return_20d","topix_future_5d"]].sort_values("date").reset_index(drop=True)


# ============================================================
# 3. 日付一括取得（1日1コールで全銘柄）
# ============================================================
def fetch_all_tickers_by_date(date_str: str) -> pd.DataFrame:
    """指定日の全上場銘柄データを一括取得（1APIコール）"""
    all_data = []
    params   = {"date": date_str}

    while True:
        res = requests.get(
            f"{JQUANTS_BASE_URL}/equities/bars/daily",
            headers=_headers(), params=params, timeout=60
        )
        if res.status_code == 429:
            logger.warning("レート制限 → 60秒待機")
            time.sleep(60)
            continue
        if res.status_code != 200:
            return pd.DataFrame()

        body = res.json()
        all_data.extend(body.get("data", []))
        pagination_key = body.get("pagination_key")
        if not pagination_key:
            break
        params["pagination_key"] = pagination_key

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    return df


# ============================================================
# 4. テクニカル特徴量の計算
# ============================================================
def calc_features(df: pd.DataFrame) -> pd.DataFrame:
    """株価DataFrameからテクニカル特徴量を計算"""
    d = df.copy()

    # リターン系
    d["return_1d"]  = d["close"].pct_change(1)
    d["return_5d"]  = d["close"].pct_change(5)
    d["return_20d"] = d["close"].pct_change(20)
    d["return_60d"] = d["close"].pct_change(60)

    # 移動平均
    d["ma5"]  = d["close"].rolling(5).mean()
    d["ma25"] = d["close"].rolling(25).mean()
    d["ma75"] = d["close"].rolling(75).mean()
    d["ma5_dev"]  = (d["close"] - d["ma5"])  / d["ma5"]
    d["ma25_dev"] = (d["close"] - d["ma25"]) / d["ma25"]
    d["ma75_dev"] = (d["close"] - d["ma75"]) / d["ma75"]
    d["above_ma75"] = (d["close"] > d["ma75"]).astype(int)

    # RSI(14)
    delta = d["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # ボリンジャーバンド
    bb_mid = d["close"].rolling(25).mean()
    bb_std = d["close"].rolling(25).std()
    d["bb_upper"] = bb_mid + 2 * bb_std
    d["bb_lower"] = bb_mid - 2 * bb_std
    d["bb_pct"]   = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"])

    # MACD
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = macd - signal
    d["macd_golden"] = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(int)

    # 出来高
    d["vol_ma20"]  = d["volume"].rolling(20).mean()
    d["vol_ratio"] = d["volume"] / d["vol_ma20"].replace(0, np.nan)

    # GC
    d["gc_25_75"] = ((d["ma25"] > d["ma75"]) & (d["ma25"].shift(1) <= d["ma75"].shift(1))).astype(int)

    # 高値・安値乖離
    d["high_52w"]  = d["high"].rolling(252).max()
    d["low_52w"]   = d["low"].rolling(252).min()
    d["from_high"] = (d["close"] - d["high_52w"]) / d["high_52w"]
    d["from_low"]  = (d["close"] - d["low_52w"])  / d["low_52w"]

    # ① RCI
    def calc_rci(series, period):
        def _rci(x):
            n  = len(x)
            pr = pd.Series(x).rank(ascending=False)
            dr = pd.Series(range(1, n+1))
            return (1 - 6 * ((dr - pr)**2).sum() / (n*(n**2-1))) * 100
        return series.rolling(period).apply(_rci, raw=True)

    d["rci9"]  = calc_rci(d["close"], 9)
    d["rci26"] = calc_rci(d["close"], 26)

    # ② 一目均衡表
    h9  = d["high"].rolling(9).max();  l9  = d["low"].rolling(9).min()
    h26 = d["high"].rolling(26).max(); l26 = d["low"].rolling(26).min()
    h52 = d["high"].rolling(52).max(); l52 = d["low"].rolling(52).min()
    tenkan  = (h9  + l9)  / 2
    kijun   = (h26 + l26) / 2
    span_a  = ((tenkan + kijun) / 2).shift(26)
    span_b  = ((h52 + l52) / 2).shift(26)
    d["ichi_tenkan_dev"]  = (d["close"] - tenkan) / tenkan
    d["ichi_kijun_dev"]   = (d["close"] - kijun)  / kijun
    cloud_top = span_a.combine(span_b, max)
    d["ichi_above_cloud"] = (d["close"] > cloud_top).astype(int)

    # ③ ADX(14)
    hdiff = d["high"].diff(); ldiff = d["low"].diff()
    pdm   = hdiff.where((hdiff > 0) & (hdiff > -ldiff), 0.0)
    mdm   = (-ldiff).where((-ldiff > 0) & (-ldiff > hdiff), 0.0)
    tr    = pd.concat([d["high"]-d["low"],
                       (d["high"]-d["close"].shift()).abs(),
                       (d["low"] -d["close"].shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    pdi   = 100 * pdm.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
    mdi   = 100 * mdm.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
    dx    = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    d["adx14"] = dx.ewm(span=14, adjust=False).mean()

    # ⑤ 出来高急増持続日数
    surge = (d["vol_ratio"] >= 2.0).astype(int)
    d["vol_surge_days"] = surge.groupby(
        (surge != surge.shift()).cumsum()
    ).cumcount() * surge

    return d


# ============================================================
# 5. 目的変数（TOPIX比アルファ）
# ============================================================
def calc_target_alpha(df: pd.DataFrame, topix_df: pd.DataFrame) -> pd.DataFrame:
    """
    目的変数: 5日後リターン - TOPIX5日後リターン > TARGET_ALPHA = 1
    市場全体の動きを除いた「純粋な銘柄選択力」を学習
    """
    future_close  = df["close"].shift(-TARGET_DAYS)
    stock_return  = (future_close - df["close"]) / df["close"]

    # TOPIXの5日後リターンをマージ
    df = df.merge(
        topix_df[["date", "topix_future_5d"]],
        on="date", how="left"
    )
    alpha = stock_return - df["topix_future_5d"].fillna(0)
    df["future_return"] = stock_return
    df["alpha_return"]  = alpha
    df["target"]        = (alpha >= TARGET_ALPHA).astype(int)
    return df


# ============================================================
# 6. ファンダメンタルデータ
# ============================================================
def load_fundamental_cache() -> dict:
    if not CACHE_PATH.exists():
        logger.warning("EDINETキャッシュが見つかりません")
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def add_fundamental_features(df: pd.DataFrame, ticker: str, fund_cache: dict) -> pd.DataFrame:
    fund = fund_cache.get(ticker, {})
    for col, key, default in [
        ("per",             "per",             None),
        ("roe",             "roe",             None),
        ("roa",             "roa",             None),
        ("operating_margin","operating_margin", None),
        ("revenue_growth",  "revenue_growth",   None),
        ("equity_ratio",    "equity_ratio",     None),
        ("debt_to_equity",  "debt_to_equity",   None),
        ("dividend_yield",  "dividend_yield",   None),
        ("credit_score",    "credit_score",     None),
    ]:
        df[col] = fund.get(key, default)

    per = fund.get("per"); roe = fund.get("roe")
    df["pbr"] = round(per * (roe/100), 2) if per and roe and per > 0 and roe > 0 else None
    return df


# ============================================================
# メイン処理
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    # 東証プライム銘柄リスト取得
    tickers = get_prime_tickers()
    if not tickers:
        logger.error("銘柄リストが取得できませんでした")
        return

    prime_codes = set()
    for t in tickers:
        code = t.replace(".T", "") + "0"
        prime_codes.add(code)

    fund_cache = load_fundamental_cache()
    logger.info(f"財務キャッシュ: {len(fund_cache)}銘柄")

    # 日付範囲
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=365 * HISTORY_YEARS)
    start_str  = start_date.strftime("%Y%m%d")
    end_str    = end_date.strftime("%Y%m%d")

    # TOPIX取得
    topix_df = fetch_topix(start_str, end_str)
    logger.info(f"TOPIX: {len(topix_df)}日分")

    # 営業日リスト（TOPIXの日付を使用）
    trade_dates = sorted(topix_df["date"].dt.strftime("%Y%m%d").tolist())
    logger.info(f"取得対象営業日: {len(trade_dates)}日")

    # 銘柄ごとの価格データ蓄積用
    ticker_data = {t: [] for t in tickers}
    ticker_map  = {t.replace(".T","")+"0": t for t in tickers}

    # 日付ループで一括取得
    total = len(trade_dates)
    for i, date_str in enumerate(trade_dates):
        if i % 100 == 0:
            logger.info(f"日付取得中: {i}/{total} ({date_str})")

        day_df = fetch_all_tickers_by_date(date_str)
        if day_df.empty:
            time.sleep(SLEEP_SEC)
            continue

        # 列名正規化
        rename_map = {}
        for old, new in [("Date","date"),("O","open"),("H","high"),("L","low"),
                         ("C","close"),("Vo","volume"),("AdjC","close"),("AdjVo","volume")]:
            if old in day_df.columns:
                rename_map[old] = new
        day_df = day_df.rename(columns=rename_map)
        day_df["date"] = pd.to_datetime(day_df["date"])

        # プライム銘柄のみ抽出
        if "Code" in day_df.columns:
            day_df = day_df[day_df["Code"].astype(str).isin(prime_codes)]

        for _, row in day_df.iterrows():
            code   = str(row.get("Code",""))
            ticker = ticker_map.get(code)
            if ticker and ticker in ticker_data:
                ticker_data[ticker].append(row.to_dict())

        time.sleep(SLEEP_SEC)

    logger.info("特徴量計算・データ結合中...")

    feature_cols = [
        "return_1d","return_5d","return_20d","return_60d",
        "ma5_dev","ma25_dev","ma75_dev","above_ma75",
        "rsi14","bb_pct","macd_hist","macd_golden",
        "vol_ratio","vol_surge_days","gc_25_75","from_high","from_low",
        "rci9","rci26",
        "ichi_tenkan_dev","ichi_kijun_dev","ichi_above_cloud",
        "adx14",
        "margin_ratio","margin_ratio_chg",
        "topix_return_5d","topix_return_20d",
        "per","pbr","roe","roa",
        "operating_margin","revenue_growth",
        "equity_ratio","debt_to_equity","dividend_yield","credit_score",
    ]

    all_frames = []
    success = 0

    for ticker, rows in ticker_data.items():
        if len(rows) < 100:
            continue
        try:
            price_df = pd.DataFrame(rows)
            price_df["date"] = pd.to_datetime(price_df["date"])
            price_df = price_df.sort_values("date").reset_index(drop=True)

            # 必須列確認
            for col in ["open","high","low","close","volume"]:
                if col not in price_df.columns:
                    price_df[col] = np.nan

            price_df[["open","high","low","close","volume"]] = \
                price_df[["open","high","low","close","volume"]].apply(pd.to_numeric, errors="coerce")

            # テクニカル特徴量
            price_df = calc_features(price_df)

            # 信用倍率変化率（デフォルトNone）
            price_df["margin_ratio"]     = None
            price_df["margin_ratio_chg"] = None

            # TOPIX結合
            price_df = pd.merge_asof(
                price_df.sort_values("date"),
                topix_df.sort_values("date"),
                on="date", direction="backward"
            )

            # 目的変数（TOPIX比アルファ）
            price_df = calc_target_alpha(price_df, topix_df)

            # ファンダメンタル
            price_df = add_fundamental_features(price_df, ticker, fund_cache)

            price_df["ticker"] = ticker
            all_frames.append(price_df)
            success += 1

        except Exception as e:
            logger.warning(f"{ticker}: {e}")

    if not all_frames:
        logger.error("データが取得できませんでした")
        return

    logger.info(f"結合中... {success}銘柄")
    full_df = pd.concat(all_frames, ignore_index=True)
    full_df = full_df.dropna(subset=["target"])

    output_cols = ["date","ticker","close"] + \
                  [c for c in feature_cols if c in full_df.columns] + \
                  ["alpha_return","future_return","target"]
    output_df = full_df[[c for c in output_cols if c in full_df.columns]].copy()

    # float64 → float32（容量削減）
    float_cols = output_df.select_dtypes(include="float64").columns
    output_df[float_cols] = output_df[float_cols].astype(np.float32)

    # Parquet保存（CSVより80%小さい）
    parquet_path = OUTPUT_DIR / "training_data.parquet"
    output_df.to_parquet(parquet_path, index=False, compression="snappy")

    # 後方互換でCSVも保存（バックテスト用）
    csv_path = OUTPUT_DIR / "training_data.csv"
    output_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 特徴量メタ情報
    feat_cols_actual = [c for c in feature_cols if c in output_df.columns]
    feature_info = {
        "created_at":    datetime.now().isoformat(),
        "tickers":       list(ticker_data.keys()),
        "total_records": len(output_df),
        "positive_rate": float(output_df["target"].mean()),
        "feature_cols":  feat_cols_actual,
        "target_type":   "topix_alpha",
        "target_alpha":  TARGET_ALPHA,
        "parquet_size_mb": round(parquet_path.stat().st_size / 1024 / 1024, 1),
    }
    with open(OUTPUT_DIR / "feature_info.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)

    pos_rate = output_df["target"].mean()
    logger.success(f"""
========================================
  PHASE 1 完了
========================================
  銘柄数:     {success}社（東証プライム）
  レコード数: {len(output_df):,}件
  正例率:     {pos_rate:.1%}
  Parquet:    {feature_info['parquet_size_mb']}MB
  特徴量:     {len(feat_cols_actual)}個
========================================
    """)


if __name__ == "__main__":
    main()
