"""
data_fetcher.py - J-Quants API V2対応版

V2変更点：
- 認証：APIキー（x-api-key ヘッダー）
- エンドポイント：/v2/equities/bars/daily
- カラム名：Close→C, Open→O, High→H, Low→L, Volume→Vo
- 銘柄マスタ：dateパラメータ必須
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional


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
        """7011.T → 70110 形式に変換"""
        code = ticker.replace(".T", "")
        return code + "0" if len(code) == 4 else code

    def get_price_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """株価履歴を取得（V2 API）"""
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

            # V2: 調整後カラムがあれば優先使用
            raw_cols = pd.DataFrame(data).columns.tolist()
            if "AdjC" in raw_cols:
                rename = {
                    "Date": "date", "AdjO": "open", "AdjH": "high",
                    "AdjL": "low", "AdjC": "close", "AdjVo": "volume",
                }
            else:
                rename = {
                    "Date": "date", "O": "open", "H": "high",
                    "L": "low", "C": "close", "Vo": "volume",
                }

            df = df.rename(columns=rename)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            return df

        except Exception as e:
            logger.error(f"株価履歴エラー({ticker}): {e}")
            return None

    def get_company_name(self, ticker: str) -> str:
        """銘柄名を取得（V2 API・dateパラメータ付き）"""
        if not self.api_key:
            return ticker
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
                    # 日本語名 → 英語名 の順で取得
                    name = (
                        item.get("CompanyName") or
                        item.get("CompanyNameEnglish") or
                        item.get("Name") or
                        ticker
                    )
                    return name
        except Exception as e:
            logger.warning(f"銘柄名取得エラー({ticker}): {e}")
        return ticker

    def get_stock_info(self, ticker: str) -> Optional[dict]:
        """銘柄情報を取得（V2 API）"""
        if not self.api_key:
            return None
        try:
            code = self._to_code(ticker)
            today = datetime.now().strftime("%Y%m%d")

            # 銘柄マスタ（V2・dateパラメータ付き）
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
                    info = {
                        "name": (
                            item.get("CompanyName") or
                            item.get("CompanyNameEnglish") or
                            item.get("Name") or
                            ticker
                        ),
                        "sector": item.get("Sector17CodeName", "不明"),
                        "industry": item.get("Sector33CodeName", "不明"),
                        "market": item.get("MarketCodeName", ""),
                    }

            # 株価履歴取得
            history = self.get_price_history(ticker)
            current_price = 0
            volume = 0
            avg_volume = 0
            week52_high = 0
            week52_low = 0
            if history is not None and not history.empty:
                current_price = float(history["close"].iloc[-1])
                volume = float(history["volume"].iloc[-1])
                avg_volume = float(history["volume"].mean())
                week52_high = float(history["high"].max())
                week52_low = float(history["low"].min())

            return {
                "ticker": ticker,
                "name": info.get("name", ticker),
                "sector": info.get("sector", "不明"),
                "industry": info.get("industry", "不明"),
                "current_price": current_price,
                "market_cap": 0,
                "volume": volume,
                "avg_volume": avg_volume,
                "per": None, "pbr": None, "psr": None,
                "ev_ebitda": None, "roe": None, "roa": None,
                "profit_margin": None, "operating_margin": None,
                "revenue_growth": None, "earnings_growth": None,
                "dividend_yield": None, "debt_to_equity": None,
                "current_ratio": None,
                "week52_high": week52_high,
                "week52_low": week52_low,
            }

        except Exception as e:
            logger.error(f"銘柄情報エラー({ticker}): {e}")
            return None

    def get_margin_trading(self, ticker: str) -> Optional[dict]:
        """信用取引週末残高を取得（V2 API・Standardプラン）"""
        if not self.api_key:
            return None
        try:
            code = self._to_code(ticker)
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            to_date = datetime.now().strftime("%Y%m%d")

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

            latest = data[-1]
            long_margin  = float(latest.get("LongMarginTradeVolume",  0) or 0)
            short_margin = float(latest.get("ShortMarginTradeVolume", 0) or 0)
            margin_ratio = round(long_margin / short_margin, 2) if short_margin > 0 else None

            return {
                "margin_ratio": margin_ratio,
                "long_margin":  long_margin,
                "short_margin": short_margin,
                "date": latest.get("Date", ""),
            }

        except Exception as e:
            logger.warning(f"信用残取得エラー({ticker}): {e}")
            return None

    def get_multiple_stocks(self, tickers: list) -> dict:
        """複数銘柄を一括取得"""
        results = {}
        total = len(tickers)

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
        return results

    def get_market_overview(self) -> dict:
        """市場概況（日経225・TOPIX）を取得（V2 API）"""
        overview = {}
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        to_date = datetime.now().strftime("%Y%m%d")

        # TOPIX
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

        # 日経225
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
                    overview["日経225"] = {
                        "price": round(close2, 2),
                        "change_pct": round((close2 - prev2) / prev2 * 100, 2),
                    }
        except Exception as e:
            logger.warning(f"日経225取得エラー: {e}")

        return overview


class MarginScorer:
    """信用倍率スコアリング"""

    def score(self, margin_ratio: float) -> tuple:
        if margin_ratio is None:    return 0,  "信用倍率データなし"
        if margin_ratio <= 1.0:     return 10, f"🟢 信用倍率良好({margin_ratio:.1f}倍）"
        elif margin_ratio <= 2.0:   return 7,  f"🟡 信用倍率普通({margin_ratio:.1f}倍）"
        elif margin_ratio <= 3.0:   return 4,  f"🟠 信用倍率やや過熱({margin_ratio:.1f}倍）"
        else:                        return -5, f"🔴 信用倍率過熱({margin_ratio:.1f}倍）要注意"
