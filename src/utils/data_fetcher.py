"""
data_fetcher.py - 株価・財務データ取得モジュール

yfinanceを使用して日本株の株価・財務情報を取得します。
初心者メモ: yfinanceは無料でYahoo Financeのデータを取得できるライブラリです。
           日本株のティッカーは末尾に「.T」をつけます（例: 7011.T = 三菱重工業）
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional


class DataFetcher:
    """株価・財務データ取得クラス"""

    def __init__(self, history_days: int = 180):
        """
        Args:
            history_days: 取得する過去データの日数
        """
        self.history_days = history_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=history_days)

    def get_stock_info(self, ticker: str) -> Optional[dict]:
        """
        銘柄の基本情報と財務データを取得

        Args:
            ticker: 銘柄コード（例: "7011.T"）

        Returns:
            銘柄情報の辞書、取得失敗時はNone
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or info.get("regularMarketPrice") is None:
                logger.warning(f"データ取得失敗: {ticker}")
                return None

            # 必要な情報を整理
            return {
                "ticker": ticker,
                "name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", "不明"),
                "industry": info.get("industry", "不明"),
                # 価格情報
                "current_price": info.get("regularMarketPrice", 0),
                "market_cap": info.get("marketCap", 0),
                "volume": info.get("regularMarketVolume", 0),
                "avg_volume": info.get("averageVolume", 0),
                # バリュエーション指標
                "per": info.get("trailingPE", None),          # PER
                "pbr": info.get("priceToBook", None),          # PBR
                "psr": info.get("priceToSalesTrailing12Months", None),  # PSR
                "ev_ebitda": info.get("enterpriseToEbitda", None),
                # 収益性指標
                "roe": info.get("returnOnEquity", None),       # ROE
                "roa": info.get("returnOnAssets", None),       # ROA
                "profit_margin": info.get("profitMargins", None),
                "operating_margin": info.get("operatingMargins", None),
                # 成長性指標
                "revenue_growth": info.get("revenueGrowth", None),
                "earnings_growth": info.get("earningsGrowth", None),
                # 配当
                "dividend_yield": info.get("dividendYield", None),
                # 財務健全性
                "debt_to_equity": info.get("debtToEquity", None),
                "current_ratio": info.get("currentRatio", None),
                # 52週高安値
                "week52_high": info.get("fiftyTwoWeekHigh", 0),
                "week52_low": info.get("fiftyTwoWeekLow", 0),
            }

        except Exception as e:
            logger.error(f"エラー({ticker}): {e}")
            return None

    def get_price_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        株価履歴データを取得

        Args:
            ticker: 銘柄コード

        Returns:
            OHLCVデータのDataFrame
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=self.start_date, end=self.end_date)

            if df.empty:
                logger.warning(f"価格履歴なし: {ticker}")
                return None

            # カラム名を日本語対応に統一
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            return df

        except Exception as e:
            logger.error(f"価格履歴取得エラー({ticker}): {e}")
            return None

    def get_multiple_stocks(self, tickers: list) -> dict:
        """
        複数銘柄のデータを一括取得

        Args:
            tickers: 銘柄コードのリスト

        Returns:
            {ticker: info_dict} の辞書
        """
        results = {}
        total = len(tickers)

　　　　　　　　for i, ticker in enumerate(tickers, 1):
            logger.info(f"データ取得中... ({i}/{total}): {ticker}")
            info = self.get_stock_info(ticker)
            history = self.get_price_history(ticker)
            time.sleep(random.uniform(1.5, 3.0))

            if info and history is not None:
           　info["price_history"] = history
            results[ticker] = info

        logger.success(f"取得完了: {len(results)}/{total} 銘柄")
        return results

    def get_market_overview(self) -> dict:
        """
        市場全体の概況を取得（日経225、TOPIXなど）

        Returns:
            市場指数の辞書
        """
        indices = {
            "日経225": "^N225",
            "TOPIX": "1306.T",
            "マザーズ": "2516.T",
        }

        overview = {}
        for name, ticker in indices.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                if not hist.empty:
                    latest = hist.iloc[-1]
                    prev = hist.iloc[-2]
                    change_pct = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
                    overview[name] = {
                        "price": latest["Close"],
                        "change_pct": round(change_pct, 2),
                    }
            except Exception as e:
                logger.warning(f"指数取得エラー({name}): {e}")

        return overview
