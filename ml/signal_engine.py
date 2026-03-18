"""
統合シグナルエンジン
====================
① 地合い判定
② セクター分析
③ 銘柄スクリーニング
④ AIスコアリング（短期・中期・長期）
⑤ Discord統合通知

データ取得: 重複なし・キャッシュ最優先
"""

import os
import json
import pickle
import requests
import time
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

# ============================================================
# パス設定
# ============================================================
BASE_DIR          = Path(".")
CONFIG_PATH       = BASE_DIR / "config" / "signal_config.yaml"
SELECTED_PATH     = BASE_DIR / "config" / "selected_tickers.json"
STOCK_NAMES_PATH  = BASE_DIR / "config" / "stock_names.json"
POLICY_YAML_PATH  = BASE_DIR / "config" / "policy_keywords.yaml"
FUND_CACHE_PATH   = BASE_DIR / "data" / "cache" / "fundamental_cache.json"
PRICE_CACHE_PATH  = BASE_DIR / "data" / "cache" / "price_cache.pkl"
MODEL_PATH        = BASE_DIR / "data" / "ml" / "model.pkl"
DEMO_TRADES_PATH  = BASE_DIR / "data" / "ml" / "demo_trades.csv"
JQUANTS_BASE_URL  = "https://api.jquants.com/v2"

logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}\n",
    level="INFO"
)


# ============================================================
# 設定読み込み
# ============================================================
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _headers() -> dict:
    api_key = os.environ.get("JQUANTS_API_KEY", "")
    if not api_key:
        raise ValueError("JQUANTS_API_KEY が未設定")
    return {"x-api-key": api_key}


# ============================================================
# 1. データ取得（重複なし）
# ============================================================
def load_price_cache() -> dict:
    """価格キャッシュ読み込み（main.py と共有）"""
    if not PRICE_CACHE_PATH.exists():
        logger.warning("価格キャッシュなし")
        return {}
    with open(PRICE_CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    elapsed = (datetime.now() - cache.get("timestamp", datetime.min)).total_seconds() / 3600
    if elapsed > 18:
        logger.warning(f"価格キャッシュ期限切れ({elapsed:.1f}h)")
        return {}
    data = cache.get("data", {})
    logger.info(f"価格キャッシュ使用: {len(data)}銘柄 ({elapsed:.1f}h経過)")
    return data


def fetch_topix() -> dict:
    """TOPIX取得（1コール）"""
    end_str   = datetime.now().strftime("%Y%m%d")
    start_str = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")
    all_data  = []
    params    = {"from": start_str, "to": end_str}

    while True:
        res = requests.get(
            f"{JQUANTS_BASE_URL}/indices/bars/daily/topix",
            headers=_headers(), params=params, timeout=30
        )
        if res.status_code != 200:
            logger.warning(f"TOPIX取得失敗: {res.status_code}")
            return {}
        body = res.json()
        all_data.extend(body.get("data", []))
        pkey = body.get("pagination_key")
        if not pkey:
            break
        params = {"pagination_key": pkey}
        time.sleep(0.1)

    if not all_data:
        return {}

    df = pd.DataFrame(all_data)
    close_col = next((c for c in ["AdjC","C","Close"] if c in df.columns), None)
    if not close_col:
        return {}

    df["date"]  = pd.to_datetime(df["Date"])
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    closes = df["close"].values
    ma25   = float(pd.Series(closes).rolling(25).mean().iloc[-1])
    c      = float(closes[-1])

    return {
        "close":          c,
        "ma25":           ma25,
        "above_ma25":     c > ma25,
        "return_5d":      float(closes[-1] / closes[-6]  - 1) if len(closes) >= 6  else 0,
        "return_20d":     float(closes[-1] / closes[-21] - 1) if len(closes) >= 21 else 0,
        "closes":         closes.tolist(),
    }


def fetch_nikkei_return() -> float:
    """日経平均20日リターン取得（1コール）"""
    end_str   = datetime.now().strftime("%Y%m%d")
    start_str = (datetime.now() - timedelta(days=40)).strftime("%Y%m%d")
    try:
        res = requests.get(
            f"{JQUANTS_BASE_URL}/indices/bars/daily/nikkei225",
            headers=_headers(),
            params={"from": start_str, "to": end_str},
            timeout=30
        )
        if res.status_code != 200:
            return 0.0
        data = res.json().get("data", [])
        if len(data) < 21:
            return 0.0
        closes = [float(d.get("AdjC") or d.get("C") or 0) for d in data]
        closes = [c for c in closes if c > 0]
        return float(closes[-1] / closes[-21] - 1) if len(closes) >= 21 else 0.0
    except:
        return 0.0


def calc_advance_decline(price_dict: dict) -> float:
    """価格キャッシュから騰落レシオを計算"""
    advances = declines = 0
    for ticker, data in price_dict.items():
        closes = data.get("close", [])
        if len(closes) < 2:
            continue
        if closes[-1] > closes[-2]:
            advances += 1
        elif closes[-1] < closes[-2]:
            declines += 1
    total = advances + declines
    return advances / total if total > 0 else 1.0


def fetch_earnings_announcements() -> list:
    """決算発表予定取得（1コール）"""
    today     = datetime.now()
    end_date  = today + timedelta(days=3)
    try:
        res = requests.get(
            f"{JQUANTS_BASE_URL}/fins/announcement",
            headers=_headers(),
            params={
                "from": today.strftime("%Y-%m-%d"),
                "to":   end_date.strftime("%Y-%m-%d"),
            },
            timeout=30
        )
        if res.status_code != 200:
            return []
        return res.json().get("data", [])
    except:
        return []


def load_tickers() -> list:
    """銘柄リスト読み込み（selected_tickers.json 優先）"""
    if SELECTED_PATH.exists():
        with open(SELECTED_PATH, encoding="utf-8") as f:
            data = json.load(f)
        tickers = [t.replace(".T", "") for t in data.get("tickers", [])]
        if tickers:
            logger.info(f"銘柄数: {len(tickers)} (selected_tickers.json)")
            return tickers

    # フォールバック: policy_keywords.yaml
    if POLICY_YAML_PATH.exists():
        with open(POLICY_YAML_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        tickers = []
        for sector_data in config.get("policy_sectors", {}).values():
            if isinstance(sector_data, dict):
                tickers.extend([t.replace(".T","") for t in sector_data.get("ticker_list",[])])
        tickers = list(dict.fromkeys(tickers))
        logger.info(f"銘柄数: {len(tickers)} (policy_keywords.yaml)")
        return tickers
    return []


def load_fundamentals() -> dict:
    """ファンダキャッシュ読み込み"""
    if not FUND_CACHE_PATH.exists():
        return {}
    with open(FUND_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_stock_names() -> dict:
    """銘柄名辞書読み込み"""
    if not STOCK_NAMES_PATH.exists():
        return {}
    with open(STOCK_NAMES_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_ml_model():
    """MLモデル読み込み"""
    if not MODEL_PATH.exists():
        logger.warning("MLモデルなし")
        return None, None, None
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model     = saved.get("model")
    feat_cols = saved.get("feature_cols", [])
    threshold = saved.get("threshold", 0.55)
    return model, feat_cols, threshold


# ============================================================
# 2. シグナルエンジン
# ============================================================
class SignalEngine:

    def __init__(self, config: dict):
        self.config = config
        self.mf_cfg = config.get("market_filter", {})
        self.sc_cfg = config.get("scoring", {})
        self.al_cfg = config.get("alerts", {})
        self.tr_cfg = config.get("trade_rules", {})

    # ① 地合い判定
    def market_filter(self, topix: dict, nikkei_r20: float, ad_ratio: float) -> dict:
        r20_topix  = topix.get("return_20d", 0)
        above_ma25 = topix.get("above_ma25", True)
        conds      = self.mf_cfg.get("risk_off_conditions", {})

        risk_flags = {
            "topix_r20":  r20_topix  <= conds.get("topix_return_20d",  -0.02),
            "nikkei_r20": nikkei_r20 <= conds.get("nikkei_return_20d", -0.02),
            "ad_ratio":   ad_ratio   <  conds.get("advance_decline_ratio", 0.90),
            "below_ma25": not above_ma25,
        }
        risk_off      = any(risk_flags.values())
        ps_cfg        = self.mf_cfg.get("position_size", {})
        position_size = ps_cfg.get("risk_off", 0.3) if risk_off else ps_cfg.get("normal", 1.0)

        logger.info(
            f"地合い: {'⚠️リスクオフ' if risk_off else '✅通常'} "
            f"TOPIX={r20_topix:.1%} 騰落={ad_ratio:.2f} "
            f"25MA={'上' if above_ma25 else '下'}"
        )
        return {
            "risk_off":      risk_off,
            "risk_flags":    risk_flags,
            "position_size": position_size,
            "topix_r20":     r20_topix,
            "ad_ratio":      ad_ratio,
            "above_ma25":    above_ma25,
        }

    # ② セクター強度
    def sector_scores(self, price_dict: dict, sector_map: dict) -> dict:
        from src.screener.sector_analyzer import calc_sector_scores
        weights = self.config.get("sector", {}).get("metrics", {
            "return_5d": 0.5, "vol_ratio": 0.3, "policy_hit": 0.2
        })
        return calc_sector_scores(price_dict, sector_map, weights)

    # ③ 総合スコア計算
    def combine_score(self, row: dict, sector_scores: dict) -> float:
        mid   = float(row.get("mid_prob",   0) or 0)
        long_ = float(row.get("long_prob",  0) or 0)
        short = float(row.get("short_prob", 0) or 0)

        w = self.sc_cfg.get("weights", {"mid":0.5,"long":0.3,"short":0.2})
        score = mid * w["mid"] + long_ * w["long"] + short * w["short"]

        # セクター強度ブースト
        sector = row.get("sector", "")
        sector_strength = sector_scores.get(sector, {}).get("score", 1.0) if isinstance(sector_scores.get(sector), dict) else 1.0
        score *= sector_strength

        # 出来高ブースト
        vb = self.sc_cfg.get("vol_boost", {})
        if float(row.get("vol_ratio", 1) or 1) >= vb.get("threshold", 2.0):
            score += vb.get("bonus", 0.05)

        # 信用倍率ペナルティ
        mp = self.sc_cfg.get("margin_penalty", {})
        if float(row.get("margin_ratio", 0) or 0) > mp.get("threshold", 8.0):
            score -= mp.get("penalty", 0.10)

        return round(max(score, 0), 4)

    # ④ 短期シグナル（MLモデル）
    def short_signals(
        self, tickers: list, price_dict: dict, fund_cache: dict,
        model, feat_cols: list, threshold: float
    ) -> list:
        if model is None:
            return []

        records = []
        for ticker in tickers:
            data = price_dict.get(ticker) or price_dict.get(f"{ticker}.T")
            if not data:
                continue
            closes  = data.get("close", [])
            volumes = data.get("volume", [])
            if len(closes) < 75:
                continue

            try:
                r = self._calc_tech_features(ticker, closes, volumes, fund_cache)
                if r is None:
                    continue
                records.append(r)
            except:
                continue

        if not records:
            return []

        df = pd.DataFrame(records)
        missing = [c for c in feat_cols if c not in df.columns]
        for c in missing:
            df[c] = 0.0

        X     = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        probs = model.predict(X)
        df["short_prob"] = probs

        result = df[df["short_prob"] >= threshold].sort_values(
            "short_prob", ascending=False
        ).head(self.config.get("models",{}).get("short",{}).get("top_n", 5))

        return result.to_dict("records")

    # ⑤ 中期シグナル（財務×テクニカル）
    def mid_signals(
        self, tickers: list, price_dict: dict, fund_cache: dict,
        sector_scores: dict
    ) -> list:
        cfg     = self.config.get("models", {}).get("mid", {})
        top_n   = cfg.get("top_n", 5)
        records = []

        for ticker in tickers:
            data = price_dict.get(ticker) or price_dict.get(f"{ticker}.T")
            if not data:
                continue
            closes  = data.get("close", [])
            volumes = data.get("volume", [])
            if len(closes) < 25:
                continue

            fund = fund_cache.get(ticker) or fund_cache.get(f"{ticker}.T") or {}
            try:
                r = self._calc_tech_features(ticker, closes, volumes, fund_cache)
                if r is None:
                    continue

                # 財務スコア
                roe  = float(fund.get("roe", 0) or 0)
                per  = float(fund.get("per", 999) or 999)
                pbr  = float(fund.get("pbr", 999) or 999)
                rev_growth = float(fund.get("revenue_growth", 0) or 0)
                op_margin  = float(fund.get("operating_margin", 0) or 0)
                eq_ratio   = float(fund.get("equity_ratio", 0) or 0)
                div_yield  = float(fund.get("dividend_yield", 0) or 0)

                # 財務スコア（0〜1に正規化）
                fund_score = np.mean([
                    min(roe / 0.20, 1.0),
                    min(max(1 - per / 30, 0), 1.0),
                    min(max(1 - pbr / 3.0, 0), 1.0),
                    min(max(rev_growth / 0.10, 0), 1.0),
                    min(op_margin / 0.20, 1.0),
                    min(eq_ratio / 100, 1.0),
                    min(div_yield / 0.05, 1.0),
                ])

                # テクニカルスコア
                tech_score = np.mean([
                    min(float(r.get("vol_ratio", 1)) / 3.0, 1.0),
                    float(r.get("rsi14", 50)) / 100,
                    min(max(float(r.get("ma25_dev", 0)) + 0.1, 0) / 0.2, 1.0),
                    float(r.get("above_ma75", 0)),
                ])

                sector = _get_sector(ticker)
                ss     = sector_scores.get(sector, {})
                sector_boost = ss.get("score", 1.0) if isinstance(ss, dict) else 1.0

                mid_prob = (fund_score * 0.6 + tech_score * 0.4) * sector_boost
                r["mid_prob"]    = round(mid_prob, 4)
                r["fund_score"]  = round(fund_score, 4)
                r["tech_score"]  = round(tech_score, 4)
                r["roe"]         = roe
                r["per"]         = per
                r["pbr"]         = pbr
                r["div_yield"]   = div_yield
                r["sector"]      = sector
                records.append(r)
            except:
                continue

        if not records:
            return []

        df = pd.DataFrame(records).sort_values("mid_prob", ascending=False)
        return df.head(top_n).to_dict("records")

    # ⑥ 長期シグナル（財務優良×高配当×国策）
    def long_signals(
        self, tickers: list, price_dict: dict, fund_cache: dict
    ) -> list:
        cfg     = self.config.get("models", {}).get("long", {})
        filters = cfg.get("filters", {})
        top_n   = cfg.get("top_n", 3)
        records = []

        for ticker in tickers:
            fund = fund_cache.get(ticker) or fund_cache.get(f"{ticker}.T") or {}
            if not fund:
                continue

            roe        = float(fund.get("roe", 0) or 0)
            equity     = float(fund.get("equity_ratio", 0) or 0)
            pbr        = float(fund.get("pbr", 999) or 999)
            per        = float(fund.get("per", 999) or 999)
            div_yield  = float(fund.get("dividend_yield", 0) or 0)
            rev_growth = float(fund.get("revenue_growth", 0) or 0)

            # 長期フィルター
            if roe        < filters.get("roe_min",            0.10): continue
            if equity     < filters.get("equity_ratio_min",   0.50) * 100: continue
            if pbr        > filters.get("pbr_max",            1.50): continue
            if div_yield  < filters.get("dividend_yield_min", 0.03): continue

            # 長期スコア
            long_score = np.mean([
                min(roe / 0.20, 1.0),
                min(equity / 80, 1.0),
                min(max(1 - pbr / 2.0, 0), 1.0),
                min(div_yield / 0.06, 1.0),
                min(max(rev_growth / 0.10, 0), 1.0),
            ])

            sector = _get_sector(ticker)
            records.append({
                "ticker":     ticker,
                "sector":     sector,
                "long_prob":  round(long_score, 4),
                "roe":        roe,
                "pbr":        pbr,
                "per":        per,
                "div_yield":  div_yield,
                "equity":     equity,
            })

        if not records:
            return []

        df = pd.DataFrame(records).sort_values("long_prob", ascending=False)
        return df.head(top_n).to_dict("records")

    # ⑦ 出来高急増アラート
    def volume_surge_alert(self, price_dict: dict, tickers: list) -> list:
        cfg       = self.al_cfg.get("volume_surge", {})
        threshold = cfg.get("threshold", 3.0)
        min_price = cfg.get("min_price", 500)
        surges    = []

        for ticker in tickers:
            data = price_dict.get(ticker) or price_dict.get(f"{ticker}.T")
            if not data:
                continue
            closes  = data.get("close", [])
            volumes = data.get("volume", [])
            if len(closes) < 21 or len(volumes) < 21:
                continue
            if float(closes[-1]) < min_price:
                continue
            vol_ma20  = np.mean(volumes[-21:-1])
            if vol_ma20 <= 0:
                continue
            vol_ratio = volumes[-1] / vol_ma20
            if vol_ratio >= threshold:
                surges.append({
                    "ticker":    ticker,
                    "vol_ratio": round(float(vol_ratio), 1),
                    "close":     closes[-1],
                })

        return sorted(surges, key=lambda x: x["vol_ratio"], reverse=True)[:5]

    # ⑧ デモトレード損益
    def demo_pnl_summary(self) -> dict:
        if not DEMO_TRADES_PATH.exists():
            return {}
        try:
            df = pd.read_csv(DEMO_TRADES_PATH)
            if df.empty:
                return {}

            open_pos  = df[df["status"] == "open"]  if "status" in df.columns else pd.DataFrame()
            closed    = df[df["status"] == "closed"] if "status" in df.columns else df

            total_pnl = float(closed["pnl"].sum()) if "pnl" in closed.columns else 0.0
            wins      = int((closed["win"] == 1).sum()) if "win" in closed.columns else 0
            total     = len(closed)
            win_rate  = wins / total if total > 0 else 0.0

            worst = {}
            if not open_pos.empty and "unrealized_pnl_pct" in open_pos.columns:
                worst_row = open_pos.loc[open_pos["unrealized_pnl_pct"].idxmin()]
                worst = {
                    "ticker": worst_row.get("ticker", ""),
                    "pnl_pct": float(worst_row["unrealized_pnl_pct"]),
                }

            return {
                "open_count":  len(open_pos),
                "total_pnl":   round(total_pnl, 2),
                "win_rate":    round(win_rate, 3),
                "total_trades":total,
                "worst":       worst,
            }
        except:
            return {}

    # ヘルパー: テクニカル特徴量計算
    def _calc_tech_features(
        self, ticker: str, closes: list, volumes: list, fund_cache: dict
    ) -> dict:
        c_arr = np.array(closes, dtype=float)
        v_arr = np.array(volumes, dtype=float)
        c     = c_arr[-1]

        if c < 500:
            return None

        ma5   = np.mean(c_arr[-5:])
        ma25  = np.mean(c_arr[-25:])
        ma75  = np.mean(c_arr[-75:]) if len(c_arr) >= 75 else ma25

        vol_ma20 = np.mean(v_arr[-21:-1]) if len(v_arr) >= 21 else 1
        vol_ratio = float(v_arr[-1] / vol_ma20) if vol_ma20 > 0 else 1.0

        delta = np.diff(c_arr)
        gain  = np.mean(np.where(delta[-14:] > 0, delta[-14:], 0))
        loss  = np.mean(np.where(delta[-14:] < 0, -delta[-14:], 0))
        rsi14 = float(100 - 100 / (1 + gain / loss)) if loss > 0 else 50.0

        fund = fund_cache.get(ticker) or fund_cache.get(f"{ticker}.T") or {}

        return {
            "ticker":      ticker,
            "close":       float(c),
            "ma5_dev":     float((c - ma5)  / ma5)  if ma5  > 0 else 0,
            "ma25_dev":    float((c - ma25) / ma25) if ma25 > 0 else 0,
            "ma75_dev":    float((c - ma75) / ma75) if ma75 > 0 else 0,
            "above_ma75":  float(c > ma75),
            "rsi14":       rsi14,
            "vol_ratio":   vol_ratio,
            "margin_ratio":float(fund.get("margin_ratio", 2.0) or 2.0),
            "return_5d":   float(c_arr[-1] / c_arr[-6]  - 1) if len(c_arr) >= 6  else 0,
            "return_20d":  float(c_arr[-1] / c_arr[-21] - 1) if len(c_arr) >= 21 else 0,
        }


# ============================================================
# ヘルパー
# ============================================================
_TICKER_SECTOR_MAP = {}

def _build_sector_map():
    global _TICKER_SECTOR_MAP
    if _TICKER_SECTOR_MAP:
        return
    if not POLICY_YAML_PATH.exists():
        return
    with open(POLICY_YAML_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for sector, data in config.get("policy_sectors", {}).items():
        if isinstance(data, dict):
            for t in data.get("ticker_list", []):
                _TICKER_SECTOR_MAP[t.replace(".T", "")] = sector

def _get_sector(ticker: str) -> str:
    _build_sector_map()
    return _TICKER_SECTOR_MAP.get(ticker, "その他")


# ============================================================
# Discord 通知
# ============================================================
def _post_discord(content: str):
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        logger.warning("DISCORD_WEBHOOK_URL 未設定")
        return
    for chunk in [content[i:i+1990] for i in range(0, len(content), 1990)]:
        requests.post(url, json={"content": chunk}, timeout=10)
        time.sleep(0.5)


def build_discord_message(
    date_str: str,
    market: dict,
    topix: dict,
    earnings: list,
    surges: list,
    short_sigs: list,
    mid_sigs: list,
    long_sigs: list,
    demo: dict,
    names: dict,
) -> str:

    def name(ticker):
        return names.get(ticker, ticker)

    lines = []

    # ヘッダー
    topix_close = topix.get("close", 0)
    topix_r5    = topix.get("return_5d", 0)
    ma25_flag   = "✅25MA上" if topix.get("above_ma25") else "⚠️25MA下"
    risk_label  = "⚠️リスクオフ" if market["risk_off"] else "✅通常"
    ps          = market["position_size"]
    lines.append(f"📈 市場: TOPIX {topix_close:,.0f} ({topix_r5:+.1%}) {ma25_flag} | {risk_label} → ポジション{ps:.0%}")
    lines.append("━━━━━━━━━━━━━━━━━━━━")

    # 決算アラート
    if earnings:
        lines.append("📅 決算注意（3日以内）")
        for e in earnings[:3]:
            code  = str(e.get("Code", "")).rstrip("0")
            ddate = e.get("DisclosedDate", e.get("Date", ""))
            lines.append(f"  ・{name(code)}({code}.T) {ddate}")

    # 出来高急増
    if surges:
        lines.append("🔥 出来高急増")
        for s in surges[:3]:
            t = s["ticker"]
            lines.append(f"  ・{name(t)}({t}.T) 通常比{s['vol_ratio']}倍")

    if earnings or surges:
        lines.append("━━━━━━━━━━━━━━━━━━━━")

    # 短期シグナル
    lines.append("🤖 短期シグナル（1〜5日 MLモデル）")
    if short_sigs:
        for i, s in enumerate(short_sigs, 1):
            t    = s["ticker"]
            prob = s.get("short_prob", 0)
            vr   = s.get("vol_ratio", 1)
            vol_mark = "🔥" if vr >= 2.0 else ""
            lines.append(f"  {i}. {name(t)}({t}.T) 確率:{prob:.0%} {vol_mark}")
    else:
        lines.append("  （シグナルなし）")

    lines.append("━━━━━━━━━━━━━━━━━━━━")

    # 中期シグナル
    lines.append("📊 中期シグナル（1〜3ヶ月 国策×財務）")
    if mid_sigs:
        for i, s in enumerate(mid_sigs, 1):
            t     = s["ticker"]
            prob  = s.get("mid_prob", 0)
            roe   = s.get("roe", 0)
            pbr   = s.get("pbr", 0)
            sec   = s.get("sector", "")
            lines.append(f"  {i}. {name(t)}({t}.T) スコア:{prob:.2f} ROE:{roe:.0%} PBR:{pbr:.1f} {sec}")
    else:
        lines.append("  （シグナルなし）")

    lines.append("━━━━━━━━━━━━━━━━━━━━")

    # 長期シグナル
    lines.append("🏆 長期候補（6ヶ月〜 テンバガー×高配当）")
    if long_sigs:
        for i, s in enumerate(long_sigs, 1):
            t    = s["ticker"]
            prob = s.get("long_prob", 0)
            div  = s.get("div_yield", 0)
            roe  = s.get("roe", 0)
            pbr  = s.get("pbr", 0)
            sec  = s.get("sector", "")
            lines.append(f"  {i}. {name(t)}({t}.T) スコア:{prob:.2f} 配当:{div:.1%} ROE:{roe:.0%} PBR:{pbr:.1f} {sec}")
    else:
        lines.append("  （シグナルなし）")

    # デモトレード
    if demo:
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("💼 デモトレード")
        lines.append(
            f"  保有:{demo.get('open_count',0)}件 "
            f"累計:{demo.get('total_pnl',0):+.1f}% "
            f"勝率:{demo.get('win_rate',0):.0%}"
        )
        worst = demo.get("worst", {})
        if worst:
            lines.append(f"  ⚠️ 含み損: {name(worst['ticker'])}({worst['ticker']}.T) {worst['pnl_pct']:+.1%}")

    return "\n".join(lines)


# ============================================================
# メイン
# ============================================================
def main():
    logger.info("=== シグナルエンジン起動 ===")
    config = load_config()
    engine = SignalEngine(config)

    # データ取得（重複なし）
    price_dict = load_price_cache()
    tickers    = load_tickers()
    fund_cache = load_fundamentals()
    names      = load_stock_names()

    logger.info(f"銘柄数: {len(tickers)} | 価格: {len(price_dict)}銘柄 | ファンダ: {len(fund_cache)}銘柄")

    # TOPIX・日経
    topix      = fetch_topix()
    nikkei_r20 = fetch_nikkei_return()
    ad_ratio   = calc_advance_decline(price_dict)

    # 地合い判定
    market = engine.market_filter(topix, nikkei_r20, ad_ratio)

    # セクター分析
    if POLICY_YAML_PATH.exists():
        from src.screener.sector_analyzer import load_sector_tickers, calc_sector_scores
        sector_map    = load_sector_tickers()
        sector_scores = calc_sector_scores(price_dict, sector_map)
    else:
        sector_scores = {}

    # MLモデル
    model, feat_cols, threshold = load_ml_model()

    # シグナル生成
    short_sigs = engine.short_signals(tickers, price_dict, fund_cache, model, feat_cols, threshold)
    mid_sigs   = engine.mid_signals(tickers, price_dict, fund_cache, sector_scores)
    long_sigs  = engine.long_signals(tickers, price_dict, fund_cache)

    # アラート
    surges   = engine.volume_surge_alert(price_dict, tickers)
    earnings = fetch_earnings_announcements()
    demo     = engine.demo_pnl_summary()

    logger.info(
        f"シグナル: 短期{len(short_sigs)} 中期{len(mid_sigs)} 長期{len(long_sigs)} "
        f"出来高急増{len(surges)} 決算{len(earnings)}"
    )

    # Discord通知
    date_str = datetime.now().strftime("%Y-%m-%d")
    msg = build_discord_message(
        date_str, market, topix, earnings,
        surges, short_sigs, mid_sigs, long_sigs, demo, names
    )
    _post_discord(msg)
    logger.success("=== シグナルエンジン完了 ===")

    # 結果保存
    result = {
        "date":        date_str,
        "market":      market,
        "short_sigs":  short_sigs,
        "mid_sigs":    mid_sigs,
        "long_sigs":   long_sigs,
        "surges":      surges,
        "earnings":    earnings[:10],
        "demo":        demo,
    }
    out_path = BASE_DIR / "data" / "ml" / "signal_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"結果保存: {out_path}")


if __name__ == "__main__":
    main()
