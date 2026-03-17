"""
PHASE 1: 学習データ収集スクリプト
===================================
銘柄: config/stock_names.json（178社）
取得: 銘柄ごと個別取得（実績あり）
特徴量: 37個（RCI・一目・ADX追加）
目的変数: 5日後リターン - TOPIX5日後リターン > 2%（TOPIXアルファ）
保存: Parquet + CSV
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
SLEEP_SEC        = 0.5
OUTPUT_DIR       = Path("data/ml")
CACHE_PATH       = Path("data/cache/fundamental_cache.json")
STOCK_NAMES_PATH = Path("config/stock_names.json")
POLICY_YAML_PATH = Path("config/policy_keywords.yaml")

logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}\n",
    level="INFO"
)
Path("data/logs").mkdir(parents=True, exist_ok=True)
logger.add("data/logs/collect_{time:YYYYMMDD}.log", rotation="1 day", level="DEBUG")


def _headers() -> dict:
    api_key = os.environ.get("JQUANTS_API_KEY", "")
    if not api_key:
        raise ValueError("JQUANTS_API_KEY が未設定です")
    return {"x-api-key": api_key}


# ============================================================
# 1. 銘柄リスト取得
# ============================================================
def get_tickers() -> list:
    # stock_names.json から取得
    if STOCK_NAMES_PATH.exists():
        with open(STOCK_NAMES_PATH, encoding="utf-8") as f:
            names = json.load(f)
        tickers = sorted([f"{k}.T" for k in names.keys()])
        logger.success(f"stock_names.json から {len(tickers)}銘柄取得")
        return tickers

    # フォールバック: policy_keywords.yaml
    import yaml
    if POLICY_YAML_PATH.exists():
        with open(POLICY_YAML_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        tickers = set()
        for sector in config.get("policy_sectors", {}).values():
            for t in sector.get("ticker_list", []):
                tickers.add(t)
        tickers = sorted(list(tickers))
        logger.success(f"policy_keywords.yaml から {len(tickers)}銘柄取得")
        return tickers

    logger.error("銘柄リストが見つかりません")
    return []


# ============================================================
# 2. TOPIX取得
# ============================================================
def fetch_topix(start_str: str, end_str: str) -> pd.DataFrame:
    logger.info("TOPIX取得中...")
    all_data = []
    params = {"from": start_str, "to": end_str}

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
        pkey = body.get("pagination_key")
        if not pkey:
            break
        params["pagination_key"] = pkey
        time.sleep(0.2)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    # 終値列を特定
    close_col = next((c for c in ["AdjC","C","Close","close"] if c in df.columns), None)
    if close_col is None:
        logger.error(f"TOPIX終値列が見つかりません: {list(df.columns)}")
        return pd.DataFrame()

    df["date"]              = pd.to_datetime(df["Date"])
    df["topix_close"]       = pd.to_numeric(df[close_col], errors="coerce")
    df["topix_return_5d"]   = df["topix_close"].pct_change(5, fill_method=None)
    df["topix_return_20d"]  = df["topix_close"].pct_change(20, fill_method=None)
    df["topix_future_5d"]   = df["topix_close"].pct_change(5, fill_method=None).shift(-5)

    result = df[["date","topix_close","topix_return_5d","topix_return_20d","topix_future_5d"]]
    result = result.sort_values("date").reset_index(drop=True)
    logger.success(f"TOPIX: {len(result)}日分取得完了")
    return result


# ============================================================
# 3. 株価履歴取得（銘柄ごと）
# ============================================================
def fetch_price_history(ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
    code = ticker.replace(".T", "") + "0"
    all_data = []
    params = {"code": code, "from": start_str, "to": end_str}

    while True:
        res = requests.get(
            f"{JQUANTS_BASE_URL}/equities/bars/daily",
            headers=_headers(), params=params, timeout=30
        )
        if res.status_code == 429:
            logger.warning(f"レート制限({ticker}) 60秒待機")
            time.sleep(60)
            continue
        if res.status_code != 200:
            return pd.DataFrame()

        body = res.json()
        all_data.extend(body.get("data", []))
        pkey = body.get("pagination_key")
        if not pkey:
            break
        params["pagination_key"] = pkey
        time.sleep(0.1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # 調整済み列優先: AdjC > C
    if "AdjC" in df.columns:
        df["close"] = pd.to_numeric(df["AdjC"], errors="coerce")
    elif "C" in df.columns:
        df["close"] = pd.to_numeric(df["C"], errors="coerce")
    else:
        return pd.DataFrame()

    # その他の列
    if "AdjVo" in df.columns:
        df["volume"] = pd.to_numeric(df["AdjVo"], errors="coerce")
    elif "Vo" in df.columns:
        df["volume"] = pd.to_numeric(df["Vo"], errors="coerce")
    else:
        df["volume"] = 0.0

    df["high"]   = pd.to_numeric(df.get("H", df.get("High",  df["close"])), errors="coerce")
    df["low"]    = pd.to_numeric(df.get("L", df.get("Low",   df["close"])), errors="coerce")
    df["open"]   = pd.to_numeric(df.get("O", df.get("Open",  df["close"])), errors="coerce")
    df["date"]   = pd.to_datetime(df["Date"])
    df["ticker"] = ticker

    cols = ["date","ticker","open","high","low","close","volume"]
    return df[cols].sort_values("date").reset_index(drop=True)


# ============================================================
# 4. 信用倍率取得
# ============================================================
def fetch_margin_history(ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
    code = ticker.replace(".T", "") + "0"
    res = requests.get(
        f"{JQUANTS_BASE_URL}/markets/margin-interest",
        headers=_headers(),
        params={"code": code, "from": start_str, "to": end_str},
        timeout=30
    )
    if res.status_code != 200:
        return pd.DataFrame()

    data = res.json().get("data", [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["Date"])

    def sf(v):
        try: return float(v) if v else None
        except: return None

    long_vol  = df.get("LongVol",  pd.Series([None]*len(df))).apply(sf)
    short_vol = df.get("ShrtVol",  pd.Series([None]*len(df))).apply(sf)
    df["margin_ratio"] = long_vol / short_vol.replace(0, np.nan)

    df = df[["date","margin_ratio"]].sort_values("date").reset_index(drop=True)
    df["margin_ratio_chg"] = df["margin_ratio"].pct_change(1, fill_method=None)
    return df


# ============================================================
# 5. テクニカル特徴量
# ============================================================
def calc_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["close"]
    h = d["high"]
    l = d["low"]
    v = d["volume"]

    # リターン
    d["return_1d"]  = c.pct_change(1, fill_method=None)
    d["return_5d"]  = c.pct_change(5, fill_method=None)
    d["return_20d"] = c.pct_change(20, fill_method=None)
    d["return_60d"] = c.pct_change(60, fill_method=None)

    # 移動平均
    ma5  = c.rolling(5).mean()
    ma25 = c.rolling(25).mean()
    ma75 = c.rolling(75).mean()
    d["ma5_dev"]    = (c - ma5)  / ma5
    d["ma25_dev"]   = (c - ma25) / ma25
    d["ma75_dev"]   = (c - ma75) / ma75
    d["above_ma75"] = (c > ma75).astype(int)
    d["gc_25_75"]   = ((ma25 > ma75) & (ma25.shift(1) <= ma75.shift(1))).astype(int)

    # RSI(14)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # ボリンジャーバンド
    bb_mid = c.rolling(25).mean()
    bb_std = c.rolling(25).std()
    bb_u   = bb_mid + 2 * bb_std
    bb_l   = bb_mid - 2 * bb_std
    d["bb_pct"] = (c - bb_l) / (bb_u - bb_l).replace(0, np.nan)

    # MACD
    ema12  = c.ewm(span=12, adjust=False).mean()
    ema26  = c.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    sig    = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = macd - sig
    d["macd_golden"] = ((macd > sig) & (macd.shift(1) <= sig.shift(1))).astype(int)

    # 出来高
    vol_ma20       = v.rolling(20).mean()
    d["vol_ratio"] = v / vol_ma20.replace(0, np.nan)

    # 高値・安値乖離
    d["from_high"] = (c - c.rolling(252).max())  / c.rolling(252).max()
    d["from_low"]  = (c - c.rolling(252).min())  / c.rolling(252).min()

    # ① RCI(9, 26)
    def calc_rci(series: pd.Series, period: int) -> pd.Series:
        def _rci(x):
            n  = len(x)
            pr = pd.Series(x).rank(ascending=False)
            dr = pd.Series(range(1, n + 1))
            return (1 - 6 * ((dr - pr) ** 2).sum() / (n * (n**2 - 1))) * 100
        return series.rolling(period).apply(_rci, raw=True)

    d["rci9"]  = calc_rci(c, 9)
    d["rci26"] = calc_rci(c, 26)

    # ② 一目均衡表
    h9    = h.rolling(9).max();  l9  = l.rolling(9).min()
    h26   = h.rolling(26).max(); l26 = l.rolling(26).min()
    h52   = h.rolling(52).max(); l52 = l.rolling(52).min()
    tenkan = (h9 + l9) / 2
    kijun  = (h26 + l26) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h52 + l52) / 2).shift(26)
    d["ichi_tenkan_dev"]  = (c - tenkan) / tenkan.replace(0, np.nan)
    d["ichi_kijun_dev"]   = (c - kijun)  / kijun.replace(0, np.nan)
    cloud_top = span_a.combine(span_b, max)
    d["ichi_above_cloud"] = (c > cloud_top).astype(int)

    # ③ ADX(14)
    hdiff = h.diff(); ldiff = l.diff()
    pdm   = hdiff.where((hdiff > 0) & (hdiff > -ldiff), 0.0)
    mdm   = (-ldiff).where((-ldiff > 0) & (-ldiff > hdiff), 0.0)
    tr    = pd.concat([h - l,
                       (h - c.shift()).abs(),
                       (l - c.shift()).abs()], axis=1).max(axis=1)
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
# 6. 目的変数（TOPIXアルファ）
# ============================================================
def calc_target(df: pd.DataFrame, topix_df: pd.DataFrame) -> pd.DataFrame:
    future_close = df["close"].shift(-TARGET_DAYS)
    stock_return = (future_close - df["close"]) / df["close"]

    # topix_future_5d はmerge_asof済みで既にdf内にある
    topix_future = df["topix_future_5d"].fillna(0) if "topix_future_5d" in df.columns else 0.0
    alpha = stock_return - topix_future

    df = df.copy()
    df["future_return"] = stock_return
    df["alpha_return"]  = alpha
    df["target"]        = (alpha >= TARGET_ALPHA).astype(int)
    return df


# ============================================================
# 7. 決算サプライズ取得
# ============================================================
def fetch_earnings_surprise(ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
    """J-Quants /v2/fins/statements から決算サプライズを計算"""
    code = ticker.replace(".T", "")
    res = requests.get(
        f"{JQUANTS_BASE_URL}/fins/statements",
        headers=_headers(),
        params={"code": code},
        timeout=30
    )
    if res.status_code != 200:
        return pd.DataFrame()
    data = res.json().get("data", [])
    if not data:
        return pd.DataFrame()

    rows = []
    for d in data:
        try:
            disc_date = pd.to_datetime(d.get("DisclosedDate",""))
            if pd.isna(disc_date):
                continue
            net_actual   = float(d.get("NetIncome", 0) or 0)
            net_forecast = float(d.get("ForecastNetIncome", 0) or 0)
            earn_surp = (net_actual / net_forecast - 1) if net_forecast != 0 else None
            rev_actual   = float(d.get("NetSales", 0) or 0)
            rev_forecast = float(d.get("ForecastNetSales", 0) or 0)
            rev_surp = (rev_actual / rev_forecast - 1) if rev_forecast != 0 else None
            rows.append({
                "date":              disc_date,
                "earnings_surprise": earn_surp,
                "revenue_surprise":  rev_surp,
                "earnings_date":     disc_date,
            })
        except:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def add_earnings_features(price_df: pd.DataFrame, ticker: str,
                          start_str: str, end_str: str) -> pd.DataFrame:
    """決算サプライズ特徴量を追加"""
    earn_df = fetch_earnings_surprise(ticker, start_str, end_str)
    if earn_df.empty:
        price_df["earnings_surprise"]   = None
        price_df["revenue_surprise"]    = None
        price_df["days_since_earnings"] = None
        return price_df

    price_df = pd.merge_asof(
        price_df.sort_values("date"),
        earn_df.sort_values("date"),
        on="date", direction="backward"
    )
    price_df["days_since_earnings"] = (
        price_df["date"] - price_df["earnings_date"]
    ).dt.days.clip(upper=90)
    price_df = price_df.drop(columns=["earnings_date"], errors="ignore")
    return price_df


# ============================================================
# 8. ファンダメンタル
# ============================================================
def load_fundamental_cache() -> dict:
    if not CACHE_PATH.exists():
        logger.warning("EDINETキャッシュなし → 財務特徴量はNullになります")
        return {}
    with open(CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def add_fundamental(df: pd.DataFrame, ticker: str, fund_cache: dict) -> pd.DataFrame:
    fund = fund_cache.get(ticker, {})
    for col, key in [
        ("per",             "per"),
        ("roe",             "roe"),
        ("roa",             "roa"),
        ("operating_margin","operating_margin"),
        ("revenue_growth",  "revenue_growth"),
        ("equity_ratio",    "equity_ratio"),
        ("debt_to_equity",  "debt_to_equity"),
        ("dividend_yield",  "dividend_yield"),
        ("credit_score",    "credit_score"),
    ]:
        df[col] = fund.get(key)

    per = fund.get("per"); roe = fund.get("roe")
    df["pbr"] = round(per * (roe / 100), 2) if per and roe and per > 0 and roe > 0 else None
    return df


# ============================================================
# メイン処理
# ============================================================
FEATURE_COLS = [
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
    # 決算サプライズ
    "earnings_surprise","revenue_surprise","days_since_earnings",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers = get_tickers()
    if not tickers:
        logger.error("銘柄リストが取得できませんでした")
        return
    logger.info(f"対象銘柄数: {len(tickers)}")

    fund_cache = load_fundamental_cache()
    logger.info(f"財務キャッシュ: {len(fund_cache)}銘柄")

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=365 * HISTORY_YEARS)
    start_str  = start_date.strftime("%Y%m%d")
    end_str    = end_date.strftime("%Y%m%d")

    topix_df = fetch_topix(start_str, end_str)
    if topix_df.empty:
        logger.error("TOPIX取得失敗")
        return
    logger.info(f"TOPIX: {len(topix_df)}日分")

    all_frames = []
    success = skip = fail = 0
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        if i % 20 == 0 or i == 1:
            logger.info(f"取得中: {i}/{total} ({ticker})")

        try:
            # 株価取得
            price_df = fetch_price_history(ticker, start_str, end_str)
            if price_df.empty or len(price_df) < 100:
                skip += 1
                continue

            # テクニカル特徴量
            price_df = calc_features(price_df)

            # 信用倍率
            margin_df = fetch_margin_history(ticker, start_str, end_str)
            if not margin_df.empty:
                price_df = pd.merge_asof(
                    price_df.sort_values("date"),
                    margin_df.sort_values("date"),
                    on="date", direction="backward"
                )
            else:
                price_df["margin_ratio"]     = None
                price_df["margin_ratio_chg"] = None

            # TOPIX結合
            price_df = pd.merge_asof(
                price_df.sort_values("date"),
                topix_df.sort_values("date"),
                on="date", direction="backward"
            )

            # 目的変数（TOPIXアルファ）
            price_df = calc_target(price_df, topix_df)

            # 決算サプライズ
            price_df = add_earnings_features(price_df, ticker, start_str, end_str)

            # ファンダメンタル
            price_df = add_fundamental(price_df, ticker, fund_cache)

            all_frames.append(price_df)
            success += 1

        except Exception as e:
            logger.warning(f"{ticker}: {e}")
            fail += 1

        time.sleep(SLEEP_SEC)

    logger.info(f"取得完了: 成功{success} / スキップ{skip} / 失敗{fail}")

    if not all_frames:
        logger.error("データが取得できませんでした")
        return

    full_df = pd.concat(all_frames, ignore_index=True)
    full_df = full_df.dropna(subset=["target"])

    out_cols = ["date","ticker","close"] + \
               [c for c in FEATURE_COLS if c in full_df.columns] + \
               ["alpha_return","future_return","target"]
    out_df = full_df[[c for c in out_cols if c in full_df.columns]].copy()

    # float32に変換（容量削減）
    for col in out_df.select_dtypes(include="float64").columns:
        out_df[col] = out_df[col].astype(np.float32)

    # 保存
    parquet_path = OUTPUT_DIR / "training_data.parquet"
    csv_path     = OUTPUT_DIR / "training_data.csv"
    out_df.to_parquet(parquet_path, index=False, compression="snappy")
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    feat_cols_actual = [c for c in FEATURE_COLS if c in out_df.columns]
    feature_info = {
        "created_at":      datetime.now().isoformat(),
        "tickers":         tickers,
        "total_records":   len(out_df),
        "positive_rate":   float(out_df["target"].mean()),
        "feature_cols":    feat_cols_actual,
        "target_type":     "topix_alpha",
        "target_alpha":    TARGET_ALPHA,
        "parquet_size_mb": round(parquet_path.stat().st_size / 1024 / 1024, 1),
        "csv_size_mb":     round(csv_path.stat().st_size / 1024 / 1024, 1),
    }
    with open(OUTPUT_DIR / "feature_info.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)

    logger.success(f"""
========================================
  PHASE 1 完了
========================================
  銘柄数:     {success}社
  レコード数: {len(out_df):,}件
  正例率:     {float(out_df['target'].mean()):.1%}
  Parquet:    {feature_info['parquet_size_mb']}MB
  CSV:        {feature_info['csv_size_mb']}MB
  特徴量:     {len(feat_cols_actual)}個
========================================
    """)


if __name__ == "__main__":
    main()
