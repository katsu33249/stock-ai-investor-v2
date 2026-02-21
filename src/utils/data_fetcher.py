"""
data_fetcher.py - Yahoo Finance 安定版（リトライ強化）

J-Quants Freeプランは直近12週間のデータが取得できないため
Yahoo Financeを使用します。リトライ処理で429エラーに対応。
"""

import yfinance as yf
import pandas as pd
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
        """銘柄情報を取得（3回リトライ）"""
        for attempt in range(3):
            try:
                time.sleep(random.uniform(2.0, 4.0))
                stock = yf.Ticker(ticker)
                info = stock.info

                # 有効なデータかチェック
                price = (
                    info.get("currentPrice") or
                    info.get("regularMarketPrice") or
                    info.get("previousClose")
                )
                if not info or not price:
                    logger.warning(f"データなし: {ticker}")
                    return None

                return {
                    "ticker": ticker,
                    "name": info.get("longName", info.get("shortName", ticker)),
                    "sector": info.get("sector", "不明"),
                    "industry": info.get("industry", "不明"),
                    "current_price": float(price),
                    "market_cap": info.get("marketCap", 0),
                    "volume": info.get("regularMarketVolume", 0),
                    "avg_volume": info.get("averageVolume", 0),
                    "per": info.get("trailingPE"),
                    "pbr": info.get("priceToBook"),
                    "psr": info.get("priceToSalesTrailing12Months"),
                    "ev_ebitda": info.get("enterpriseToEbitda"),
                    "roe": info.get("returnOnEquity"),
                    "roa": info.get("returnOnAssets"),
                    "profit_margin": info.get("profitMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "earnings_growth": info.get("earningsGrowth"),
                    "dividend_yield": info.get("dividendYield"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "week52_high": info.get("fiftyTwoWeekHigh", 0),
                    "week52_low": info.get("fiftyTwoWeekLow", 0),
                }
            except Exception as e:
                wait = 5 * (attempt + 1)
                logger.warning(f"リトライ {attempt+1}/3 ({ticker}) {wait}秒待機: {e}")
                time.sleep(wait)
        return None

    def get_price_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """株価履歴を取得（3回リトライ）"""
        for attempt in range(3):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True
                )
                if df.empty:
                    logger.warning(f"価格履歴なし: {ticker}")
                    return None
                df.columns = [c.lower() for c in df.columns]
                df.index = pd.to_datetime(df.index)
                return df
            except Exception as e:
                wait = 5 * (attempt + 1)
                logger.warning(f"価格リトライ {attempt+1}/3 ({ticker}): {e}")
                time.sleep(wait)
        return None

    def get_multiple_stocks(self, tickers: list) -> dict:
        """複数銘柄を一括取得"""
        results = {}
        total = len(tickers)

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"データ取得中... ({i}/{total}): {ticker}")
            info = self.get_stock_info(ticker)
            if info is None:
                continue
            history = self.get_price_history(ticker)
            if history is not None:
                info["price_history"] = history
                results[ticker] = info
                logger.debug(f"✅ 取得成功: {ticker} ({info.get('name','')})")

        logger.success(f"取得完了: {len(results)}/{total} 銘柄")
        return results

    def get_market_overview(self) -> dict:
        """市場概況を取得"""
        indices = {"日経225": "^N225", "TOPIX": "1306.T"}
        overview = {}
        for name, ticker in indices.items():
            try:
                time.sleep(2)
                hist = yf.Ticker(ticker).history(period="5d")
                if not hist.empty and len(hist) >= 2:
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
