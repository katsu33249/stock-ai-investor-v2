"""
data_fetcher.py - J-Quants Standard対応データ取得モジュール

J-Quants Standardプラン（月額3,300円）で取得できるデータ：
- 株価四本値（日通し）← Yahoo Financeを完全置き換え
- 信用取引週末残高   ← 信用倍率スコアに使用（2.0新機能）
- 業種別空売り比率   ← 将来的に活用可能
- 過去10年分データ
- 120件/分のAPIコール（429エラーなし）
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional


class DataFetcher:

    BASE_URL = "https://api.jquants.com/v1"

    def __init__(self, history_days: int = 180):
        self.history_days = history_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=history_days)
        self.id_token = None
        self.refresh_token = os.environ.get("JQUANTS_REFRESH_TOKEN", "")

        if not self.refresh_token:
            logger.error("JQUANTS_REFRESH_TOKEN が設定されていません")
        else:
            self._get_id_token()

    def _get_id_token(self):
        """リフレッシュトークンからIDトークンを取得"""
        try:
            res = requests.post(
                f"{self.BASE_URL}/token/auth_refresh",
                params={"refreshtoken": self.refresh_token},
                timeout=10
            )
            if res.status_code == 200:
                self.id_token = res.json().get("idToken")
                logger.success("J-Quants 認証成功")
            else:
                logger.error(f"J-Quants 認証失敗: {res.status_code}")
        except Exception as e:
            logger.error(f"J-Quants 認証エラー: {e}")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.id_token}"}

    def _to_code(self, ticker: str) -> str:
        """7011.T → 70110 形式に変換"""
        code = ticker.replace(".T", "")
        return code + "0" if len(code) == 4 else code

    def get_price_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """株価履歴を取得（J-Quants公式データ）"""
        if not self.id_token:
            return None
        try:
            code = self._to_code(ticker)
            res = requests.get(
                f"{self.BASE_URL}/prices/daily_quotes",
                headers=self._headers(),
                params={
                    "code": code,
                    "from": self.start_date.strftime("%Y%m%d"),
                    "to": self.end_date.strftime("%Y%m%d"),
                },
                timeout=15
            )
            if res.status_code != 200:
                logger.warning(f"株価取得失敗({ticker}): {res.status_code}")
                return None

            data = res.json().get("daily_quotes", [])
            if not data:
                logger.warning(f"株価データなし: {ticker}")
                return None

            df = pd.DataFrame(data)
            df = df.rename(columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume",
                "TurnoverValue": "turnover",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            return df

        except Exception as e:
            logger.error(f"株価履歴エラー({ticker}): {e}")
            return None

    def get_stock_info(self, ticker: str) -> Optional[dict]:
        """銘柄情報を取得"""
        if not self.id_token:
            return None
        try:
            code = self._to_code(ticker)

            # 銘柄マスタ
            res = requests.get(
                f"{self.BASE_URL}/listed/info",
                headers=self._headers(),
                params={"code": code},
                timeout=10
            )
            info = {}
            if res.status_code == 200:
                items = res.json().get("info", [])
                if items:
                    item = items[0]
                    info = {
                        "name": item.get("CompanyName", ticker),
                        "sector": item.get("Sector17CodeName", "不明"),
                        "industry": item.get("Sector33CodeName", "不明"),
                        "market": item.get("MarketCodeName", ""),
                    }

            # 最新株価・出来高
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
                "per": None,
                "pbr": None,
                "psr": None,
                "ev_ebitda": None,
                "roe": None,
                "roa": None,
                "profit_margin": None,
                "operating_margin": None,
                "revenue_growth": None,
                "earnings_growth": None,
                "dividend_yield": None,
                "debt_to_equity": None,
                "current_ratio": None,
                "week52_high": week52_high,
                "week52_low": week52_low,
            }

        except Exception as e:
            logger.error(f"銘柄情報エラー({ticker}): {e}")
            return None

    def get_margin_trading(self, ticker: str) -> Optional[dict]:
        """
        信用取引週末残高を取得（Standardプラン限定）

        信用倍率 = 信用買い残 / 信用売り残
        → 2.0の信用倍率スコアリングに使用
        """
        if not self.id_token:
            return None
        try:
            code = self._to_code(ticker)
            # 直近4週間の信用残を取得
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            to_date = datetime.now().strftime("%Y%m%d")

            res = requests.get(
                f"{self.BASE_URL}/markets/weekly_margin_interest",
                headers=self._headers(),
                params={
                    "code": code,
                    "from": from_date,
                    "to": to_date,
                },
                timeout=10
            )
            if res.status_code != 200:
                return None

            data = res.json().get("weekly_margin_interest", [])
            if not data:
                return None

            latest = data[-1]
            long_margin = float(latest.get("LongMarginTradeVolume", 0) or 0)
            short_margin = float(latest.get("ShortMarginTradeVolume", 0) or 0)

            if short_margin == 0:
                margin_ratio = None
            else:
                margin_ratio = round(long_margin / short_margin, 2)

            return {
                "margin_ratio": margin_ratio,         # 信用倍率
                "long_margin": long_margin,            # 信用買い残
                "short_margin": short_margin,          # 信用売り残
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

            # 信用倍率取得（Standardプラン）
            margin = self.get_margin_trading(ticker)
            if margin:
                info["margin_ratio"] = margin["margin_ratio"]
                info["long_margin"] = margin["long_margin"]
                info["short_margin"] = margin["short_margin"]
                logger.debug(f"  信用倍率: {margin['margin_ratio']}")
            else:
                info["margin_ratio"] = None

            results[ticker] = info
            time.sleep(0.3)  # J-Quants Standard: 120件/分なので余裕

        logger.success(f"取得完了: {len(results)}/{total} 銘柄")
        return results

    def get_market_overview(self) -> dict:
        """市場概況（日経225・TOPIX）を取得"""
        overview = {}
        try:
            # TOPIX
            res = requests.get(
                f"{self.BASE_URL}/indices/topix",
                headers=self._headers(),
                params={
                    "from": (datetime.now() - timedelta(days=7)).strftime("%Y%m%d"),
                    "to": datetime.now().strftime("%Y%m%d"),
                },
                timeout=10
            )
            if res.status_code == 200:
                data = res.json().get("topix", [])
                if len(data) >= 2:
                    latest = data[-1]
                    prev = data[-2]
                    close = float(latest.get("Close", 0))
                    prev_close = float(prev.get("Close", 1))
                    change_pct = (close - prev_close) / prev_close * 100
                    overview["TOPIX"] = {
                        "price": round(close, 2),
                        "change_pct": round(change_pct, 2),
                    }
        except Exception as e:
            logger.warning(f"市場概況取得エラー: {e}")
        return overview


class MarginScorer:
    """
    信用倍率スコアリング（2.0新機能）

    信用倍率 = 信用買い残 / 信用売り残

    低倍率 → 売り圧力少ない → 上昇しやすい
    高倍率 → 売り圧力大きい → 上昇しにくい（過熱サイン）
    """

    def score(self, margin_ratio: float) -> tuple:
        """
        信用倍率スコア（10点満点）

        1倍以下  → +10点（売り圧力なし・最高）
        1〜2倍   → +7点（良好）
        2〜3倍   → +4点（やや過熱）
        3倍超    → −5点（過熱・ペナルティ）
        データなし→  0点
        """
        if margin_ratio is None:
            return 0, "信用倍率データなし"

        if margin_ratio <= 1.0:
            return 10, f"🟢 信用倍率良好({margin_ratio:.1f}倍）売り圧力なし"
        elif margin_ratio <= 2.0:
            return 7, f"🟡 信用倍率普通({margin_ratio:.1f}倍）"
        elif margin_ratio <= 3.0:
            return 4, f"🟠 信用倍率やや過熱({margin_ratio:.1f}倍）"
        else:
            return -5, f"🔴 信用倍率過熱({margin_ratio:.1f}倍）要注意"
