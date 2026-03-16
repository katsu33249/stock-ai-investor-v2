"""
data_fetcher.py - J-Quants API V2対応版

キャッシュ機能:
- data/cache/price_cache.pkl に価格データを保存
- 18時間以内は再取得しない（APIコール節約）
- predict.py と共有
"""

import requests
import pandas as pd
import pickle
import time
import os
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
from pathlib import Path


PRICE_CACHE_PATH   = Path("data/cache/price_cache.pkl")
CACHE_EXPIRE_HOURS = 18

# 銘柄名マップ（config/stock_names.json から一元管理）
def _load_stock_names() -> dict:
    """stock_names.jsonを読み込み {ticker.T: 名前} 形式で返す"""
    import json
    paths = [
        Path("config/stock_names.json"),
        Path(__file__).parent.parent.parent / "config/stock_names.json",
    ]
    for p in paths:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            # キーを "7011" → "7011.T" 形式に変換
            return {f"{k}.T": v for k, v in data.items()}
    return {}

STOCK_NAME_MAP = _load_stock_names()


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
            logger.info(f"💾 価格キャッシュ使用 (経過:{elapsed:.1f}h / 有効:{CACHE_EXPIRE_HOURS}h) 銘柄数:{len(cache)-1}")
            return {k: v for k, v in cache.items() if k != "_cached_at"}
        else:
            logger.info(f"⏰ 価格キャッシュ期限切れ (経過:{elapsed:.1f}h) → 再取得")
            return None
    except Exception as e:
        logger.warning(f"キャッシュ読み込みエラー: {e}")
        return None


def save_price_cache(data: dict):
    """価格データをキャッシュに保存"""
    try:
        PRICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache = {"_cached_at": datetime.now(), **data}
        with open(PRICE_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
        logger.info(f"💾 価格キャッシュ保存: {len(data)}銘柄 → {PRICE_CACHE_PATH}")
    except Exception as e:
        logger.warning(f"キャッシュ保存エラー: {e}")


class DataFetcher:

    BASE_URL = "https://api.jquants.com"

    def __init__(self, history_days: int = 180):
        self.history_days = history_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=history_days)
        self.api_key = os.environ.get("JQUANTS_API_KEY", "")
        if not self.api_key:
            logger.error("JQUANTS_API_KEY が設定されていません")
        else:
            logger.success("J-Quants API V2 初期化完了")

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key}

    def _to_code(self, ticker: str) -> str:
        code = ticker.replace(".T", "")
        return code + "0" if len(code) == 4 else code

    def get_price_history(self, ticker: str) -> Optional[pd.DataFrame]:
        if not self.api_key:
            return None
        try:
            code = self._to_code(ticker)
            res = requests.get(
                f"{self.BASE_URL}/v2/equities/bars/daily",
                headers=self._headers(),
                params={
                    "code": code,
                    "from": self.start_date.strftime("%Y%m%d"),
                    "to": self.end_date.strftime("%Y%m%d"),
                },
                timeout=15
            )
            if res.status_code != 200:
                logger.warning(f"株価取得失敗({ticker}): {res.status_code} {res.text[:100]}")
                return None
            data = res.json().get("data", [])
            if not data:
                logger.warning(f"株価データなし: {ticker}")
                return None
            df = pd.DataFrame(data)
            raw_cols = df.columns.tolist()
            if "AdjC" in raw_cols:
                rename = {"Date": "date", "AdjO": "open", "AdjH": "high",
                          "AdjL": "low", "AdjC": "close", "AdjVo": "volume"}
            else:
                rename = {"Date": "date", "O": "open", "H": "high",
                          "L": "low", "C": "close", "Vo": "volume"}
            df = df.rename(columns=rename)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            return df
        except Exception as e:
            logger.error(f"株価履歴エラー({ticker}): {e}")
            return None

    def get_company_name(self, ticker: str) -> str:
        if ticker in STOCK_NAME_MAP:
            return STOCK_NAME_MAP[ticker]
        if self.api_key:
            try:
                code = self._to_code(ticker)
                today = datetime.now().strftime("%Y%m%d")
                res = requests.get(
                    f"{self.BASE_URL}/v2/equities/master",
                    headers=self._headers(),
                    params={"code": code, "date": today},
                    timeout=10
                )
                if res.status_code == 200:
                    items = res.json().get("data", [])
                    if items:
                        item = items[0]
                        name = (item.get("CompanyName") or item.get("CompanyNameEnglish")
                                or item.get("Name") or item.get("name"))
                        if name:
                            return name
            except Exception as e:
                logger.warning(f"銘柄名API取得エラー({ticker}): {e}")
        return ticker

    def get_stock_info(self, ticker: str) -> Optional[dict]:
        if not self.api_key:
            return None
        try:
            code = self._to_code(ticker)
            today = datetime.now().strftime("%Y%m%d")
            res = requests.get(
                f"{self.BASE_URL}/v2/equities/master",
                headers=self._headers(),
                params={"code": code, "date": today},
                timeout=10
            )
            info = {}
            if res.status_code == 200:
                items = res.json().get("data", [])
                if items:
                    item = items[0]
                    api_name = (item.get("CompanyName") or item.get("CompanyNameEnglish")
                                or item.get("Name") or item.get("name"))
                    info = {
                        "name": api_name or STOCK_NAME_MAP.get(ticker, ticker),
                        "sector": item.get("Sector17CodeName", "不明"),
                        "industry": item.get("Sector33CodeName", "不明"),
                        "market": item.get("MarketCodeName", ""),
                    }
            history = self.get_price_history(ticker)
            current_price = volume = avg_volume = week52_high = week52_low = 0
            if history is not None and not history.empty:
                current_price = float(history["close"].iloc[-1])
                volume        = float(history["volume"].iloc[-1])
                avg_volume    = float(history["volume"].mean())
                week52_high   = float(history["high"].max())
                week52_low    = float(history["low"].min())
            return {
                "ticker": ticker, "name": info.get("name", ticker),
                "sector": info.get("sector", "不明"), "industry": info.get("industry", "不明"),
                "current_price": current_price, "market_cap": 0,
                "volume": volume, "avg_volume": avg_volume,
                "per": None, "pbr": None, "psr": None, "ev_ebitda": None,
                "roe": None, "roa": None, "profit_margin": None, "operating_margin": None,
                "revenue_growth": None, "earnings_growth": None,
                "dividend_yield": None, "debt_to_equity": None, "current_ratio": None,
                "week52_high": week52_high, "week52_low": week52_low,
            }
        except Exception as e:
            logger.error(f"銘柄情報エラー({ticker}): {e}")
            return None

    def get_margin_trading(self, ticker: str) -> Optional[dict]:
        if not self.api_key:
            return None
        try:
            code = self._to_code(ticker)
            from_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
            to_date   = datetime.now().strftime("%Y-%m-%d")
            res = requests.get(
                f"{self.BASE_URL}/v2/markets/margin-interest",
                headers=self._headers(),
                params={"code": code, "from": from_date, "to": to_date},
                timeout=10
            )
            if res.status_code != 200:
                return None
            data = res.json().get("data", [])
            if not data:
                return None
            latest     = data[-1]
            long_vol   = float(latest.get("LongVol", 0) or 0)
            shrt_vol   = float(latest.get("ShrtVol", 0) or 0)
            margin_ratio = round(long_vol / shrt_vol, 2) if shrt_vol > 0 else None
            return {"margin_ratio": margin_ratio, "long_margin": long_vol,
                    "short_margin": shrt_vol, "date": latest.get("Date", "")}
        except Exception as e:
            logger.warning(f"信用残取得エラー({ticker}): {e}")
            return None

    def get_valid_tse_codes(self) -> set:
        if not self.api_key:
            return set()
        try:
            today = datetime.now().strftime("%Y%m%d")
            res = requests.get(
                f"{self.BASE_URL}/v2/equities/master",
                headers=self._headers(),
                params={"date": today},
                timeout=30
            )
            if res.status_code != 200:
                return set()
            items = res.json().get("data", [])
            codes = {str(item.get("Code", "")) for item in items if item.get("Code")}
            logger.info(f"東証上場銘柄数: {len(codes)}")
            return codes
        except Exception as e:
            logger.warning(f"masterAPI取得エラー: {e}")
            return set()

    def get_multiple_stocks(self, tickers: list) -> dict:
        """複数銘柄を一括取得（キャッシュ対応・東証銘柄フィルタリング）"""

        # ① キャッシュ確認（18時間以内なら再取得しない）
        cached = load_price_cache()
        if cached is not None:
            results = {t: cached[t] for t in tickers if t in cached}
            logger.info(f"💾 キャッシュから{len(results)}銘柄を取得 (APIコール: 0回)")
            return results

        # ② キャッシュなし → J-Quantsから取得
        results = {}
        valid_codes = self.get_valid_tse_codes()
        if valid_codes:
            tse_tickers = []
            skipped = []
            for t in tickers:
                code = self._to_code(t)
                if code in valid_codes:
                    tse_tickers.append(t)
                else:
                    skipped.append(t)
            if skipped:
                logger.warning(f"非東証銘柄をスキップ({len(skipped)}件): "
                                f"{skipped[:5]}{'...' if len(skipped)>5 else ''}")
            tickers = tse_tickers
        else:
            logger.warning("masterAPI取得失敗。フィルタリングなしで実行します")

        total = len(tickers)
        logger.info(f"対象銘柄数（東証のみ）: {total}")

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"データ取得中... ({i}/{total}): {ticker}")
            info = self.get_stock_info(ticker)
            if not info:
                continue
            history = self.get_price_history(ticker)
            if history is None:
                continue
            info["price_history"] = history
            margin = self.get_margin_trading(ticker)
            info["margin_ratio"] = margin["margin_ratio"] if margin else None
            results[ticker] = info
            time.sleep(0.6)

        logger.success(f"取得完了: {len(results)}/{total} 銘柄")

        # ③ キャッシュに保存（predict.py と共有）
        if results:
            save_price_cache(results)

        return results

    def get_market_overview(self) -> dict:
        overview = {}
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        to_date   = datetime.now().strftime("%Y%m%d")
        try:
            res = requests.get(
                f"{self.BASE_URL}/v2/indices/bars/daily/topix",
                headers=self._headers(),
                params={"from": from_date, "to": to_date},
                timeout=10
            )
            if res.status_code == 200:
                data = res.json().get("data", [])
                if len(data) >= 2:
                    close = float(data[-1].get("C", 0))
                    prev  = float(data[-2].get("C", 1))
                    overview["TOPIX"] = {
                        "price": round(close, 2),
                        "change_pct": round((close - prev) / prev * 100, 2),
                    }
        except Exception as e:
            logger.warning(f"TOPIX取得エラー: {e}")
        try:
            res2 = requests.get(
                f"{self.BASE_URL}/v2/indices/bars/daily",
                headers=self._headers(),
                params={"code": "0028", "from": from_date, "to": to_date},
                timeout=10
            )
            if res2.status_code == 200:
                data2 = res2.json().get("data", [])
                if len(data2) >= 2:
                    close2 = float(data2[-1].get("C", 0))
                    prev2  = float(data2[-2].get("C", 1))
                    overview["日経平均"] = {
                        "price": round(close2, 2),
                        "change_pct": round((close2 - prev2) / prev2 * 100, 2),
                    }
        except Exception as e:
            logger.warning(f"日経平均取得エラー: {e}")
        return overview


class MarginScorer:
    def score(self, margin_ratio: float) -> tuple:
        if margin_ratio is None:    return 0,  "信用倍率データなし"
        if margin_ratio <= 1.0:     return 10, f"🟢 信用倍率良好({margin_ratio:.1f}倍）"
        elif margin_ratio <= 2.0:   return 7,  f"🟡 信用倍率普通({margin_ratio:.1f}倍）"
        elif margin_ratio <= 3.0:   return 4,  f"🟠 信用倍率やや過熱({margin_ratio:.1f}倍）"
        else:                        return -5, f"🔴 信用倍率過熱({margin_ratio:.1f}倍）要注意"
