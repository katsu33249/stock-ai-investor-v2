"""
data_fetcher.py - J-Quants API対応データ取得モジュール

J-Quants（JPX公式）からデータを取得します。
Yahoo Financeより安定・高信頼性です。

初心者メモ:
- J-QuantsはJPX（日本取引所グループ）公式のデータAPI
- Freeプランで株価四本値・財務情報が取得可能
- リフレッシュトークンを使ってIDトークンを自動取得します
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
    """J-Quants API対応データ取得クラス"""

    BASE_URL = "https://api.jquants.com/v1"

    def __init__(self, history_days: int = 180):
        self.history_days = history_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=history_days)
        self.id_token = None

        # 環境変数からリフレッシュトークンを取得
        self.refresh_token = os.environ.get("JQUANTS_REFRESH_TOKEN", "")
        if not self.refresh_token:
            logger.warning("JQUANTS_REFRESH_TOKEN が設定されていません")
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
                logger.success("J-Quants IDトークン取得成功")
            else:
                logger.error(f"IDトークン取得失敗: {res.status_code}")
        except Exception as e:
            logger.error(f"IDトークン取得エラー: {e}")

    def _headers(self) -> dict:
        """認証ヘッダーを返す"""
        return {"Authorization": f"Bearer {self.id_token}"}

    def _ticker_to_code(self, ticker: str) -> str:
        """7011.T → 70110 形式に変換"""
        code = ticker.replace(".T", "")
        return code + "0" if len(code) == 4 else code

    def get_price_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """株価履歴データを取得"""
        if not self.id_token:
            return None
        try:
            code = self._ticker_to_code(ticker)
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
                logger.warning(f"価格取得失敗({ticker}): {res.status_code}")
                return None

            data = res.json().get("daily_quotes", [])
            if not data:
                logger.warning(f"価格データなし: {ticker}")
                return None

            df = pd.DataFrame(data)
            df = df.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            return df

        except Exception as e:
            logger.error(f"価格履歴エラー({ticker}): {e}")
            return None

    def get_stock_info(self, ticker: str) -> Optional[dict]:
        """銘柄の基本情報・財務データを取得"""
        if not self.id_token:
            return None
        try:
            code = self._ticker_to_code(ticker)

            # 銘柄情報
            res_info = requests.get(
                f"{self.BASE_URL}/listed/info",
                headers=self._headers(),
                params={"code": code},
                timeout=10
            )
            # 財務情報
            res_fin = requests.get(
                f"{self.BASE_URL}/fins/statements",
                headers=self._headers(),
                params={"code": code},
                timeout=10
            )

            info = {}
            if res_info.status_code == 200:
                items = res_info.json().get("info", [])
                if items:
                    item = items[0]
                    info["name"] = item.get("CompanyNameEn", ticker)
                    info["sector"] = item.get("Sector17CodeName", "不明")
                    info["industry"] = item.get("Sector33CodeName", "不明")

            # 財務サマリーから指標を取得
            fin_data = {}
            if res_fin.status_code == 200:
                stmts = res_fin.json().get("statements", [])
                if stmts:
                    latest = stmts[-1]
                    fin_data = {
                        "per": self._safe_float(latest.get("ForecastPER")),
                        "pbr": self._safe_float(latest.get("BookValuePerShare")),
                        "roe": self._safe_float(latest.get("ROE")),
                        "revenue_growth": self._safe_float(latest.get("NetSalesGrowthRate")),
                        "earnings_growth": self._safe_float(latest.get("OrdinaryIncomeGrowthRate")),
                    }

            # 最新株価から現在値・出来高を取得
            price_history = self.get_price_history(ticker)
            current_price = 0
            volume = 0
            if price_history is not None and not price_history.empty:
                current_price = float(price_history["close"].iloc[-1])
                volume = float(price_history["volume"].iloc[-1])

            return {
                "ticker": ticker,
                "name": info.get("name", ticker),
                "sector": info.get("sector", "不明"),
                "industry": info.get("industry", "不明"),
                "current_price": current_price,
                "market_cap": 0,
                "volume": volume,
                "avg_volume": float(price_history["volume"].mean()) if price_history is not None else 0,
                "per": fin_data.get("per"),
                "pbr": fin_data.get("pbr"),
                "psr": None,
                "ev_ebitda": None,
                "roe": fin_data.get("roe"),
                "roa": None,
                "profit_margin": None,
                "operating_margin": None,
                "revenue_growth": fin_data.get("revenue_growth"),
                "earnings_growth": fin_data.get("earnings_growth"),
                "dividend_yield": None,
                "debt_to_equity": None,
                "current_ratio": None,
                "week52_high": float(price_history["high"].max()) if price_history is not None else 0,
                "week52_low": float(price_history["low"].min()) if price_history is not None else 0,
            }

        except Exception as e:
            logger.error(f"銘柄情報エラー({ticker}): {e}")
            return None

    def _safe_float(self, value) -> Optional[float]:
        """安全にfloatに変換"""
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    def get_multiple_stocks(self, tickers: list) -> dict:
        """複数銘柄を一括取得"""
        results = {}
        total = len(tickers)

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"データ取得中... ({i}/{total}): {ticker}")
            info = self.get_stock_info(ticker)
            history = self.get_price_history(ticker)
            time.sleep(0.5)  # J-Quantsは安定しているので短めの待機でOK

            if info and history is not None:
                info["price_history"] = history
                results[ticker] = info

        logger.success(f"取得完了: {len(results)}/{total} 銘柄")
        return results

    def get_market_overview(self) -> dict:
        """市場概況（日経225・TOPIX）を取得"""
        overview = {}
        try:
            res = requests.get(
                f"{self.BASE_URL}/indices",
                headers=self._headers(),
                params={"code": "0000"},  # TOPIX
                timeout=10
            )
            if res.status_code == 200:
                data = res.json().get("indices", [])
                if data:
                    latest = data[-1]
                    prev = data[-2] if len(data) > 1 else latest
                    close = float(latest.get("Close", 0))
                    prev_close = float(prev.get("Close", 1))
                    change_pct = (close - prev_close) / prev_close * 100
                    overview["TOPIX"] = {
                        "price": close,
                        "change_pct": round(change_pct, 2)
                    }
        except Exception as e:
            logger.warning(f"市場概況取得エラー: {e}")
        return overview
