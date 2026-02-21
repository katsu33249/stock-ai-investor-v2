import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional


class DataFetcher:
    def __init__(self, history_days: int = 180):
        self.history_days = history_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=history_days)

    def get_stock_info(self, ticker: str) -> Optional[dict]:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or info.get("regularMarketPrice") is None:
                logger.warning(f"データ取得失敗: {ticker}")
                return None
            return {
                "ticker": ticker,
                "name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", "不明"),
                "industry": info.get("industry", "不明"),
                "current_price": info.get("regularMarketPrice", 0),
                "market_cap": info.get("marketCap", 0),
                "volume": info.get("regularMarketVolume", 0),
                "avg_volume": info.get("averageVolume", 0),
                "per": info.get("trailingPE", None),
                "pbr": info.get("priceToBook", None),
                "psr": info.get("priceToSalesTrailing12Months", None),
                "ev_ebitda": info.get("enterpriseToEbitda", None),
                "roe": info.get("returnOnEquity", None),
                "roa": info.get("returnOnAssets", None),
                "profit_margin": info.get("profitMargins", None),
                "operating_margin": info.get("operatingMargins", None),
                "revenue_growth": info.get("revenueGrowth", None),
                "earnings_growth": info.get("earningsGrowth", None),
                "dividend_yield": info.get("dividendYield", None),
                "debt_to_equity": info.get("debtToEquity", None),
                "current_ratio": info.get("currentRatio", None),
                "week52_high": info.get("fiftyTwoWeekHigh", 0),
                "week52_low": info.get("fiftyTwoWeekLow", 0),
            }
        except Exception as e:
            logger.error(f"エラー({ticker}): {e}")
            return None

    def get_price_history(self, ticker: str) -> Optional[pd.DataFrame]:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=self.start_date, end=self.end_date)
            if df.empty:
                logger.warning(f"価格履歴なし: {ticker}")
                return None
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.error(f"価格履歴取得エラー({ticker}): {e}")
            return None

    def get_multiple_stocks(self, tickers: list) -> dict:
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
        indices = {
            "日経225": "^N225",
            "TOPIX": "1306.T",
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
