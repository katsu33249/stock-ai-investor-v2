"""
PHASE 4: 日次MLシグナル予測スクリプト
=======================================
目的: 毎朝、当日の特徴量を計算してモデルで予測し、
      高確率銘柄をDiscordに通知する

実行タイミング: daily_scan.yml から呼び出される（毎朝6時）

出力:
  - data/ml/today_signals.json（当日のMLシグナル）
  - Discord通知（prob >= SIGNAL_THRESHOLD の銘柄）
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

warnings.filterwarnings("ignore")

# ============================================================
# 設定
# ============================================================
MODEL_PATH       = Path("data/ml/model.pkl")
CACHE_PATH       = Path("data/cache/fundamental_cache.json")
CONFIG_PATH      = Path("config/policy_keywords.yaml")
SIGNAL_PATH      = Path("data/ml/today_signals.json")
PRICE_CACHE_PATH = Path("data/cache/price_cache.pkl")
CACHE_EXPIRE_HOURS = 18  # キャッシュ有効時間
TRADES_PATH      = Path("data/ml/demo_trades.csv")
SIGNAL_THRESHOLD = 0.55   # バックテストで最適化済み
TOP_N            = 5      # Discord通知する上位銘柄数

JQUANTS_API_KEY  = os.environ.get("JQUANTS_API_KEY", "")
DISCORD_WEBHOOK  = os.environ.get("DISCORD_WEBHOOK_URL", "")

def _headers() -> dict:
    return {"x-api-key": JQUANTS_API_KEY}

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
logger.add("data/logs/predict_{time:YYYYMMDD}.log", rotation="1 day", level="DEBUG")


# ============================================================
# 1. 銘柄リスト取得
# ============================================================
def get_tickers() -> list[str]:
    """policy_keywords.yaml から銘柄リストを取得"""
    import yaml
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tickers = []
    sectors = config.get("policy_sectors", {})
    # policy_sectors は辞書形式: {sector_name: {ticker_list: [...]}}
    for sector_name, sector_data in sectors.items():
        if isinstance(sector_data, dict):
            tickers.extend(sector_data.get("ticker_list", []))
    tickers = [t.replace(".T", "") for t in tickers]
    tickers = list(dict.fromkeys(tickers))  # 重複除去
    logger.info(f"銘柄数: {len(tickers)}")
    return tickers


# ============================================================
# 2. J-Quants 銘柄マスタから会社名を取得
# ============================================================
# ★ 銘柄名マスタ（config/stock_names.json から一元管理）
def _load_stock_names() -> dict:
    """stock_names.json を読み込み {4桁コード: 会社名} で返す"""
    import json
    paths = [
        Path("config/stock_names.json"),
        Path(__file__).parent.parent / "config/stock_names.json",
    ]
    for p in paths:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return {}

STOCK_NAME_MAP = _load_stock_names()

def fetch_company_names() -> dict[str, str]:
    """4桁コード → 会社名 の辞書を返す（.T付き・なし両対応）"""
    # .T付きでもアクセスできるよう両方のキーを持つ辞書を返す
    result = dict(STOCK_NAME_MAP)
    for k, v in STOCK_NAME_MAP.items():
        result[f"{k}.T"] = v  # "7011.T" → "三菱重工業" も追加
    return result


def get_company_name_yf(ticker: str) -> str:
    # .T を除いた4桁コードで検索
    code = ticker.replace(".T", "")
    return STOCK_NAME_MAP.get(code, STOCK_NAME_MAP.get(ticker, ticker))


# ============================================================
# 3. 価格キャッシュ管理
# ============================================================
def load_price_cache() -> dict | None:
    """キャッシュが有効なら返す（18時間以内）"""
    if not PRICE_CACHE_PATH.exists():
        return None
    try:
        with open(PRICE_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        cached_at = cache.get("_cached_at")
        if cached_at is None:
            return None
        elapsed = (datetime.now() - cached_at).total_seconds() / 3600
        if elapsed < CACHE_EXPIRE_HOURS:
            logger.info(f"価格キャッシュ使用 (経過:{elapsed:.1f}時間 / 有効:{CACHE_EXPIRE_HOURS}時間) 銘柄数:{len(cache)-1}")
            return {k: v for k, v in cache.items() if k != "_cached_at"}
        else:
            logger.info(f"価格キャッシュ期限切れ (経過:{elapsed:.1f}時間) → 再取得")
            return None
    except Exception as e:
        logger.warning(f"キャッシュ読み込みエラー: {e}")
        return None


def save_price_cache(prices: dict):
    """価格データをキャッシュに保存"""
    try:
        PRICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache = {"_cached_at": datetime.now(), **prices}
        with open(PRICE_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
        logger.info(f"価格キャッシュ保存: {len(prices)}銘柄 → {PRICE_CACHE_PATH}")
    except Exception as e:
        logger.warning(f"キャッシュ保存エラー: {e}")


# ============================================================
# 4. J-Quants 価格データ取得（直近90日）
# ============================================================
def fetch_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """各銘柄の直近90日の価格を取得"""
    import requests, time

    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=130)).strftime("%Y-%m-%d")

    prices = {}
    # 日付フォーマットをYYYYMMDDに変換（data_fetcher.pyと同じ）
    start_fmt = start_date.replace("-", "")
    end_fmt   = end_date.replace("-", "")

    for i, ticker in enumerate(tickers):
        try:
            # 4桁コード → 5桁（末尾に0追加）
            code = ticker + "0" if len(ticker) == 4 else ticker
            resp = requests.get(
                "https://api.jquants.com/v2/equities/bars/daily",
                headers=_headers(),
                params={"code": code, "from": start_fmt, "to": end_fmt},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    df = pd.DataFrame(data)
                    # AdjC（調整後終値）またはC（終値）を使用
                    if "AdjC" in df.columns:
                        df = df.rename(columns={"Date": "Date", "AdjC": "AdjustmentClose", "AdjVo": "AdjustmentVolume"})
                    else:
                        df = df.rename(columns={"Date": "Date", "C": "AdjustmentClose", "Vo": "AdjustmentVolume"})
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date").reset_index(drop=True)
                    prices[ticker] = df
            if i % 30 == 0:
                logger.info(f"  価格取得中: {i}/{len(tickers)}")
            time.sleep(0.1)
        except Exception as e:
            logger.warning(f"  {ticker}: {e}")

    logger.info(f"価格取得完了: {len(prices)}/{len(tickers)}銘柄")
    return prices


# ============================================================
# 3. 特徴量計算（学習時と同じロジック）
# ============================================================
def calc_features(ticker: str, df: pd.DataFrame, fund: dict) -> dict | None:
    """1銘柄の当日特徴量を計算"""
    try:
        if len(df) < 75:
            return None

        close = df["AdjustmentClose"].astype(float)
        vol   = df["AdjustmentVolume"].astype(float)

        # リターン
        r = {
            "return_1d":  float(close.pct_change(1).iloc[-1]),
            "return_5d":  float(close.pct_change(5).iloc[-1]),
            "return_20d": float(close.pct_change(20).iloc[-1]),
            "return_60d": float(close.pct_change(60).iloc[-1]),
        }

        # 移動平均乖離
        ma5  = close.rolling(5).mean()
        ma25 = close.rolling(25).mean()
        ma75 = close.rolling(75).mean()
        c    = close.iloc[-1]

        r["ma5_dev"]    = float((c / ma5.iloc[-1] - 1))
        r["ma25_dev"]   = float((c / ma25.iloc[-1] - 1))
        r["ma75_dev"]   = float((c / ma75.iloc[-1] - 1))
        r["above_ma75"] = float(c > ma75.iloc[-1])
        r["gc_25_75"]   = float(ma25.iloc[-1] > ma75.iloc[-1])

        # 5MA関連（フィルター用）
        c_prev        = close.iloc[-2]
        ma5_today     = float(ma5.iloc[-1])
        ma5_prev      = float(ma5.iloc[-2])
        # 5MA上抜け: 前日終値が5MA以下 → 当日終値が5MA以上
        r["ma5_breakout"] = float(c_prev <= ma5_prev and c >= ma5_today)
        # 5MAからの乖離（すでにma5_devで計算済み）
        # 5MA上に位置しているか
        r["above_ma5"] = float(c >= ma5_today)

        # RSI14
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
        r["rsi14"] = float(100 - 100 / (1 + rs))

        # ボリンジャーバンド %B
        m20  = close.rolling(20).mean()
        s20  = close.rolling(20).std()
        r["bb_pct"] = float((c - (m20.iloc[-1] - 2*s20.iloc[-1])) /
                             (4*s20.iloc[-1]) if s20.iloc[-1] != 0 else 0.5)

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9).mean()
        r["macd_hist"]   = float(macd.iloc[-1] - sig.iloc[-1])
        r["macd_golden"] = float(
            (macd.iloc[-1] > sig.iloc[-1]) and (macd.iloc[-2] <= sig.iloc[-2])
        )

        # 出来高比率
        vol_ma = vol.rolling(20).mean()
        r["vol_ratio"] = float(vol.iloc[-1] / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 1.0

        # 52週高値・安値からの乖離
        high52 = close.rolling(min(252, len(close))).max().iloc[-1]
        low52  = close.rolling(min(252, len(close))).min().iloc[-1]
        r["from_high"] = float(c / high52 - 1)
        r["from_low"]  = float(c / low52 - 1)

        # 信用倍率（デフォルト値）
        r["margin_ratio"] = float(fund.get("margin_ratio", 2.0))

        # TOPIX（後で追加）
        r["topix_return_5d"]  = 0.0
        r["topix_return_20d"] = 0.0

        # ファンダメンタル
        r["per"]              = float(fund.get("per", 15.0) or 15.0)
        r["pbr"]              = float(fund.get("pbr", 1.0) or 1.0)
        r["roe"]              = float(fund.get("roe", 5.0) or 5.0)
        r["roa"]              = float(fund.get("roa", 3.0) or 3.0)
        r["operating_margin"] = float(fund.get("operating_margin", 5.0) or 5.0)
        r["revenue_growth"]   = float(fund.get("revenue_growth_rate", 0.0) or 0.0)
        r["equity_ratio"]     = float(fund.get("equity_ratio", 40.0) or 40.0)
        r["debt_to_equity"]   = float(fund.get("debt_to_equity", 1.0) or 1.0)
        r["dividend_yield"]   = float(fund.get("dividend_yield", 2.0) or 2.0)
        r["credit_score"]     = float(fund.get("credit_score", 50.0) or 50.0)

        # 異常値チェック
        for k, v in r.items():
            if not np.isfinite(v):
                r[k] = 0.0

        r["ticker"] = ticker
        r["name"]   = str(fund.get("name", ticker))
        return r

    except Exception as e:
        logger.debug(f"  {ticker} 特徴量計算エラー: {e}")
        return None


# ============================================================
# 4. TOPIX直近リターンを取得して追加
# ============================================================
def fetch_topix_data() -> dict:
    """TOPIXデータ取得（5d/20dリターン + 25MA判定）APIコール1回のみ"""
    import requests
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    result = {"r5": 0.0, "r20": 0.0, "above_ma25": True, "current": 0.0}
    try:
        resp = requests.get(
            "https://api.jquants.com/v2/indices/bars/daily/topix",
            headers=_headers(),
            params={"from": start_date, "to": end_date},
            timeout=10
        )
        body = resp.json()
        data = body.get("data", body.get("indices", body.get("topix", [])))
        if not data:
            return result
        sample    = data[0]
        close_key = next((k for k in ["C", "Close", "close", "AdjustmentClose"] if k in sample), None)
        if not close_key:
            logger.warning(f"TOPIX closeキー不明: {list(sample.keys())}")
            return result
        closes = [float(d[close_key]) for d in data if d.get(close_key)]
        if len(closes) < 6:
            return result
        current   = closes[-1]
        ma25      = sum(closes[-25:]) / min(25, len(closes))
        r5        = current / closes[-6]  - 1 if len(closes) >= 6  else 0.0
        r20       = current / closes[-21] - 1 if len(closes) >= 21 else 0.0
        above_ma25 = current >= ma25
        logger.info(f"TOPIX: {current:.0f} 25MA:{ma25:.0f} {'✅上' if above_ma25 else '⚠️下'} 5d:{r5:+.2%} 20d:{r20:+.2%}")
        result = {"r5": round(r5,4), "r20": round(r20,4), "above_ma25": above_ma25, "current": round(current,2)}
    except Exception as e:
        logger.warning(f"TOPIX取得エラー: {e}")
    return result





# ============================================================
# 5. 予測実行
# ============================================================
def predict(features_list: list[dict], topix: dict) -> pd.DataFrame:
    """モデルで予測確率を計算（アンサンブル対応）"""
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    feat_cols = saved["feat_cols"]
    df = pd.DataFrame(features_list)

    # TOPIX追加
    df["topix_return_5d"]  = topix.get("r5", 0.0)
    df["topix_return_20d"] = topix.get("r20", 0.0)

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

    # アンサンブル or 単体モデル対応
    if "models" in saved:
        # 新方式: アンサンブル
        models  = saved["models"]
        preds   = []
        if "lgb" in models:
            preds.append(models["lgb"].predict(X))
        if "xgb" in models:
            import xgboost as xgb
            dmat = xgb.DMatrix(X, feature_names=feat_cols)
            preds.append(models["xgb"].predict(dmat))
        if "rf" in models:
            preds.append(models["rf"].predict_proba(np.nan_to_num(X))[:, 1])
        df["pred_prob"] = np.mean(preds, axis=0) if preds else np.zeros(len(X))
    else:
        # 旧方式: 単体モデル（後方互換）
        df["pred_prob"] = saved["model"].predict(X)
    df = df.sort_values("pred_prob", ascending=False)
    return df


# ============================================================
# 6. Discord通知
# ============================================================
def notify_discord(signals: pd.DataFrame, today_str: str, name_map: dict = {}, topix: dict = {}):
    import requests

    # 市場環境ヘッダー
    topix_mark  = "✅25MA上" if topix.get("above_ma25", True) else "⚠️25MA下"
    market_line = f"📊 TOPIX:{topix_mark}"

    if signals.empty:
        msg = f"🤖 **ML シグナル {today_str}**\n{market_line}\n該当銘柄なし（閾値: {SIGNAL_THRESHOLD}）"
    else:
        lines = [f"🤖 **ML シグナル {today_str}** （閾値: {SIGNAL_THRESHOLD}）", market_line, ""]
        for i, (_, row) in enumerate(signals.iterrows(), 1):
            ticker    = row["ticker"]
            name      = name_map.get(ticker) or get_company_name_yf(ticker)
            prob      = row["pred_prob"]
            ret_1d    = row.get("return_1d", 0)
            rsi       = row.get("rsi14", 0)
            above_ma5 = row.get("above_ma5", 1.0)
            ma5_break = row.get("ma5_breakout", 0.0)
            vol_ratio = row.get("vol_ratio", 1.0)
            ma5_mark  = "🔼5MA抜け " if ma5_break == 1.0 else ("📈5MA上 " if above_ma5 == 1.0 else "")
            vol_mark  = "🔥出来高急増 " if vol_ratio >= 2.0 else ""

            # ボリンジャーバンドシグナル
            bb_pct_val = row.get("bb_pct", 0.5)
            if bb_pct_val < 0.1 and ret_1d > 0:
                bb_mark = "📉-2σ反発 "
            elif bb_pct_val > 0.9 and ret_1d < 0:
                bb_mark = "📈+2σ反落 "
            elif 0.45 <= bb_pct_val <= 0.55 and ret_1d > 0:
                bb_mark = "↩中央反発 "
            else:
                bb_mark = ""

            lines.append(
                f"**{i}. {name} ({ticker}.T)**  確率:{prob:.0%}  "
                f"前日:{ret_1d:+.1%}  RSI:{rsi:.0f}  {ma5_mark}{vol_mark}{bb_mark}"
            )
        msg = "\n".join(lines)

    resp = requests.post(DISCORD_WEBHOOK, json={"content": msg})
    logger.info(f"Discord通知: {resp.status_code} | {len(signals)}銘柄")


# ============================================================
# デモ取引管理
# ============================================================
HOLD_DAYS_TRADE = 5  # 保有期間（営業日ではなく暦日で近似）

def get_current_price(ticker: str, prices: dict = {}) -> float | None:
    """当日の終値を取得（価格キャッシュから）"""
    try:
        df = prices.get(ticker)
        if df is not None and not df.empty:
            # AdjustmentClose列があれば使用
            if "AdjustmentClose" in df.columns:
                return float(df["AdjustmentClose"].iloc[-1])
            elif "close" in df.columns:
                return float(df["close"].iloc[-1])
    except Exception as e:
        logger.warning(f"{ticker} 価格取得エラー: {e}")
    return None


def record_entry(signals: pd.DataFrame, today_str: str, name_map: dict, prices: dict = {}):
    """新規シグナルをdemo_trades.csvに追記（重複チェック付き）"""
    if signals.empty:
        return

    # 既存CSVを読み込み
    if TRADES_PATH.exists():
        existing = pd.read_csv(TRADES_PATH, dtype=str)
        # 当日・同銘柄の重複チェック
        already = set(
            zip(existing["entry_date"].tolist(), existing["ticker"].tolist())
        )
    else:
        existing = pd.DataFrame()
        already  = set()

    rows = []
    for _, row in signals.iterrows():
        ticker = row["ticker"]

        # 重複スキップ
        if (today_str, ticker) in already:
            logger.info(f"重複スキップ: {ticker} ({today_str})")
            continue

        price      = get_current_price(ticker, prices) or 0.0
        exit_date  = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        ma5_signal = "🔼抜け" if row.get("ma5_breakout", 0) == 1.0 else ("📈上" if row.get("above_ma5", 0) == 1.0 else "")

        rows.append({
            "entry_date":   today_str,
            "ticker":       ticker,
            "name":         name_map.get(ticker, ticker),
            "prob":         round(float(row["pred_prob"]), 3),
            "ma5_signal":   ma5_signal,
            "entry_price":  price,
            "exit_date":    exit_date,
            "exit_price":   "",
            "return":       "",
            "win":          "",
        })

    if not rows:
        logger.info("新規追記なし（全て重複）")
        return

    new_df   = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined.to_csv(TRADES_PATH, index=False, encoding="utf-8-sig")
    logger.info(f"デモ取引記録: {len(rows)}件追記 → {TRADES_PATH}")


def update_exits(today_str: str):
    """決済予定日を過ぎた未決済行の損益を自動計算"""
    if not TRADES_PATH.exists():
        return

    df = pd.read_csv(TRADES_PATH, dtype=str)
    updated = 0

    for idx, row in df.iterrows():
        # 未決済かつ決済予定日を過ぎている
        if str(row.get("exit_price", "")).strip() != "":
            continue
        try:
            exit_date = datetime.strptime(str(row["exit_date"]), "%Y-%m-%d")
        except Exception:
            continue
        if datetime.now() < exit_date:
            continue

        ticker      = str(row["ticker"])
        entry_price = float(row["entry_price"]) if str(row["entry_price"]) not in ("", "0.0", "0") else None
        if entry_price is None or entry_price == 0:
            continue

        exit_price = get_current_price(ticker)
        if exit_price is None or exit_price == 0:
            continue

        ret = (exit_price - entry_price) / entry_price
        win = "✅" if ret > 0 else "❌"

        df.at[idx, "exit_price"] = round(exit_price, 2)
        df.at[idx, "return"]     = round(ret, 4)
        df.at[idx, "win"]        = win
        updated += 1
        logger.info(f"決済更新: {ticker} {ret:+.1%} {win}")

    if updated > 0:
        df.to_csv(TRADES_PATH, index=False, encoding="utf-8-sig")
        logger.info(f"決済更新: {updated}件完了")


# ============================================================
# メイン処理
# ============================================================
def main():
    Path("data/ml").mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        logger.error(f"モデルが見つかりません: {MODEL_PATH}")
        logger.error("PHASE 2 を先に実行してください")
        return

    today_str = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"=== ML日次予測 {today_str} ===")

    # 銘柄リスト
    tickers = get_tickers()

    # 会社名マップ取得
    name_map  = fetch_company_names()

    # 価格データ取得（キャッシュ優先）
    raw_cache = load_price_cache()
    if raw_cache is None:
        prices = fetch_prices(tickers)
        save_price_cache(prices)
    else:
        logger.info("→ APIコールをスキップしました")
        # main.py のキャッシュ形式に対応:
        # {ticker: DataFrame} または {ticker: {"price_history": DataFrame, ...}}
        prices = {}
        for ticker, val in raw_cache.items():
            if isinstance(val, dict) and "price_history" in val:
                # main.py 形式 → DataFrameを取り出す
                # predict.py のfetch_prices形式に変換
                ph = val["price_history"]
                if ph is not None and not ph.empty:
                    # predict.py はAdjustmentClose列を期待
                    import pandas as pd
                    df2 = ph.reset_index()
                    df2 = df2.rename(columns={
                        "date": "Date",
                        "close": "AdjustmentClose",
                        "volume": "AdjustmentVolume",
                    })
                    df2["Date"] = pd.to_datetime(df2["Date"])
                    prices[ticker] = df2
            else:
                # predict.py 形式（そのまま使用）
                prices[ticker] = val

    # TOPIX
    topix    = fetch_topix_data()

    # ファンダメンタルキャッシュ読み込み
    fund_cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            fund_cache = json.load(f)
    logger.info(f"ファンダメンタルキャッシュ: {len(fund_cache)}銘柄")

    # 特徴量計算
    features_list = []
    for ticker, df in prices.items():
        fund = fund_cache.get(ticker, {})
        feat = calc_features(ticker, df, fund)
        if feat:
            features_list.append(feat)

    logger.info(f"特徴量計算完了: {len(features_list)}銘柄")

    if not features_list:
        logger.error("特徴量が計算できませんでした")
        return

    # 予測
    pred_df = predict(features_list, topix)

    # 前日シグナルを読み込み（重複除外用）
    prev_tickers = set()
    if SIGNAL_PATH.exists():
        try:
            with open(SIGNAL_PATH, encoding="utf-8") as f:
                prev = json.load(f)
            # 前日のデータなら除外対象にする
            if prev.get("date") != today_str:
                prev_tickers = {s["ticker"] for s in prev.get("signals", [])}
                logger.info(f"前日シグナル除外: {prev_tickers}")
        except Exception:
            pass

    # シグナル抽出（前日と重複しない新規銘柄のみ・上位TOP_N件）
    all_signals = pred_df[pred_df["pred_prob"] >= SIGNAL_THRESHOLD]

    # 5MAフィルター: 5MA上に位置している銘柄のみ（上抜け or 上位にある）
    if "above_ma5" in all_signals.columns:
        ma5_filtered = all_signals[all_signals["above_ma5"] == 1.0]
        logger.info(f"5MAフィルター後: {len(ma5_filtered)} / {len(all_signals)}銘柄")
        # フィルター後に5件未満なら緩めてフィルター前に戻す
        if len(ma5_filtered) >= 3:
            all_signals = ma5_filtered

    new_signals = all_signals[~all_signals["ticker"].isin(prev_tickers)]
    # 市場環境フィルター: TOPIX25MA以下なら最大3件に絞る
    max_signals = 3 if not topix.get("above_ma25", True) else TOP_N
    signals = new_signals.head(max_signals)
    logger.info(f"シグナル銘柄数: {len(signals)} 新規 / {len(all_signals)} 全体 (閾値:{SIGNAL_THRESHOLD})")

    # 保存
    output = {
        "date":       today_str,
        "threshold":  SIGNAL_THRESHOLD,
        "total_analyzed": len(pred_df),
        "signals": signals[["ticker", "pred_prob", "return_1d", "rsi14", "ma25_dev"]].to_dict("records") if not signals.empty else [],
        "top20": pred_df.head(20)[["ticker", "pred_prob"]].to_dict("records"),
    }
    with open(SIGNAL_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Discord通知
    if DISCORD_WEBHOOK:
        notify_discord(signals, today_str, name_map, topix)
    else:
        logger.warning("DISCORD_WEBHOOK_URL が未設定")

    # デモ取引記録
    update_exits(today_str)          # 既存の未決済行を更新
    record_entry(signals, today_str, name_map, prices)  # 新規シグナルを追記

    logger.success(f"完了: {len(signals)}件のMLシグナルを通知")


if __name__ == "__main__":
    main()
