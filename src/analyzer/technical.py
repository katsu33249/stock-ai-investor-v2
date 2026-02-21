"""
technical.py - テクニカル分析モジュール（プロ仕様出来高分析）

出来高分析を3条件で強化：
① 出来高条件: 5日平均 >= 20日平均 × 1.5
② 価格条件:   株価が25日線より上 かつ 直近高値の3%以内
③ 陽線率:     直近5日で陽線3本以上
→ 「下げの出来高増加」を排除します
"""

import pandas as pd
import numpy as np
from loguru import logger


class TechnicalAnalyzer:

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.rsi_period = cfg.get("rsi_period", 14)
        self.rsi_oversold = cfg.get("rsi_oversold", 30)
        self.rsi_overbought = cfg.get("rsi_overbought", 70)
        self.macd_fast = cfg.get("macd_fast", 12)
        self.macd_slow = cfg.get("macd_slow", 26)
        self.macd_signal = cfg.get("macd_signal", 9)
        self.sma_short = cfg.get("sma_short", 25)
        self.sma_long = cfg.get("sma_long", 75)

    def calculate_rsi(self, prices: pd.Series) -> float:
        if len(prices) < self.rsi_period + 1:
            return 50.0
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    def calculate_macd(self, prices: pd.Series) -> dict:
        if len(prices) < self.macd_slow + self.macd_signal:
            return {"macd": 0, "signal": 0, "histogram": 0, "prev_histogram": 0}
        ema_fast = prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            "macd": float(macd_line.iloc[-1]),
            "signal": float(signal_line.iloc[-1]),
            "histogram": float(histogram.iloc[-1]),
            "prev_histogram": float(histogram.iloc[-2]) if len(histogram) > 1 else 0,
        }

    def calculate_moving_averages(self, prices: pd.Series) -> dict:
        current_price = float(prices.iloc[-1])
        result = {"current_price": current_price}
        for period in [5, 25, 75, 200]:
            if len(prices) >= period:
                ma = float(prices.rolling(period).mean().iloc[-1])
                result[f"sma{period}"] = ma
                result[f"above_sma{period}"] = current_price > ma
        if f"sma{self.sma_short}" in result and f"sma{self.sma_long}" in result:
            result["golden_cross"] = (
                result[f"sma{self.sma_short}"] > result[f"sma{self.sma_long}"]
            )
        return result

    def calculate_volume_trend(self, df: pd.DataFrame) -> dict:
        """
        プロ仕様の出来高分析（3条件）

        ① 出来高条件: 5日平均 >= 20日平均 × 1.5
        ② 価格条件:   25日線より上 かつ 直近高値の3%以内
        ③ 陽線率:     直近5日で陽線3本以上

        3条件すべて満たす = 本物の上昇出来高（最高評価）
        出来高増加でも価格下落 = 下げの出来高（低評価）
        """
        if len(df) < 25:
            return {
                "volume_ratio": 1.0,
                "volume_trend": "neutral",
                "pro_conditions": {},
                "passed_conditions": 0,
            }

        # ① 出来高条件
        vol_5d = df["volume"].tail(5).mean()
        vol_20d = df["volume"].tail(20).mean()
        volume_ratio = vol_5d / (vol_20d + 1e-10)
        cond_volume = volume_ratio >= 1.5

        # ② 価格条件
        current_price = float(df["close"].iloc[-1])
        sma25 = float(df["close"].rolling(25).mean().iloc[-1])
        recent_high = float(df["high"].tail(20).max())
        cond_above_sma25 = current_price > sma25
        cond_near_high = current_price >= recent_high * 0.97
        cond_price = cond_above_sma25 and cond_near_high

        # ③ 陽線率
        recent_5 = df.tail(5)
        bullish_candles = int((recent_5["close"] > recent_5["open"]).sum())
        cond_bullish = bullish_candles >= 3

        passed = sum([cond_volume, cond_price, cond_bullish])

        if passed == 3:
            trend = "strong_up"
        elif passed == 2:
            trend = "high"
        elif cond_volume and not cond_price:
            trend = "down_volume"
        elif passed == 1:
            trend = "neutral"
        else:
            trend = "low"

        return {
            "volume_ratio": round(float(volume_ratio), 2),
            "volume_trend": trend,
            "pro_conditions": {
                "volume_surge": cond_volume,
                "above_sma25": cond_above_sma25,
                "near_recent_high": cond_near_high,
                "bullish_candles": bullish_candles,
                "bullish_candle_ok": cond_bullish,
            },
            "passed_conditions": passed,
        }

    def calculate_price_momentum(self, prices: pd.Series) -> dict:
        result = {}
        for days in [5, 20, 60]:
            if len(prices) > days:
                momentum = (prices.iloc[-1] / prices.iloc[-days - 1] - 1) * 100
                result[f"momentum_{days}d"] = float(momentum)
        return result

    def calculate_score(self, df: pd.DataFrame) -> dict:
        """
        テクニカル総合スコアを計算（0〜100点）

        - RSI        : 20点
        - MACD       : 20点
        - 移動平均線 : 25点
        - 出来高     : 15点（プロ仕様3条件）
        - モメンタム : 20点
        """
        if df is None or len(df) < 30:
            return {"total_score": 50, "details": {}}

        prices = df["close"]
        score = 0
        details = {}

        # ===== RSIスコア (20点) =====
        rsi = self.calculate_rsi(prices)
        details["rsi"] = round(rsi, 1)
        if 40 <= rsi <= 55:
            rsi_score = 20
        elif 35 <= rsi < 40:
            rsi_score = 16
        elif 55 < rsi <= 65:
            rsi_score = 14
        elif rsi < 35:
            rsi_score = 12
        elif 65 < rsi <= 70:
            rsi_score = 8
        else:
            rsi_score = 3
        score += rsi_score

        # ===== MACDスコア (20点) =====
        macd_data = self.calculate_macd(prices)
        details["macd"] = macd_data
        macd_score = 0
        if macd_data["macd"] > 0:
            macd_score += 8
        if macd_data["histogram"] > macd_data["prev_histogram"]:
            macd_score += 7
        if macd_data["macd"] > macd_data["signal"]:
            macd_score += 5
        score += macd_score

        # ===== 移動平均線スコア (25点) =====
        ma_data = self.calculate_moving_averages(prices)
        details["moving_averages"] = ma_data
        ma_score = 0
        if ma_data.get("above_sma5"):   ma_score += 5
        if ma_data.get("above_sma25"):  ma_score += 7
        if ma_data.get("above_sma75"):  ma_score += 8
        if ma_data.get("golden_cross"): ma_score += 5
        score += ma_score

        # ===== 出来高スコア (15点) プロ仕様 =====
        vol_data = self.calculate_volume_trend(df)
        details["volume"] = vol_data
        passed = vol_data.get("passed_conditions", 0)
        trend = vol_data["volume_trend"]

        if trend == "strong_up":    # 3条件すべて満たす
            vol_score = 15
        elif trend == "high":       # 2条件
            vol_score = 11
        elif trend == "down_volume": # 下げの出来高（ペナルティ）
            vol_score = 0
        elif trend == "neutral":
            vol_score = 6
        else:
            vol_score = 2
        score += vol_score

        # ===== モメンタムスコア (20点) =====
        momentum = self.calculate_price_momentum(prices)
        details["momentum"] = momentum
        mom_score = 0
        if momentum.get("momentum_5d", 0) > 0:  mom_score += 5
        if momentum.get("momentum_20d", 0) > 0: mom_score += 8
        if momentum.get("momentum_60d", 0) > 0: mom_score += 7
        score += mom_score

        # 出来高シグナルの表示用テキスト
        volume_signal = {
            "strong_up":   "🔥 本物の上昇出来高（3条件クリア）",
            "high":        "📈 良好な出来高（2条件クリア）",
            "down_volume": "⚠️ 下げの出来高増加（要注意）",
            "neutral":     "➡️ 普通",
            "low":         "📉 出来高減少",
        }.get(trend, "")

        return {
            "total_score": min(100, max(0, score)),
            "details": details,
            "signals": {
                "rsi_signal": "oversold" if rsi < 35 else "overbought" if rsi > 70 else "neutral",
                "trend": "uptrend" if ma_data.get("golden_cross") else "downtrend",
                "volume_increasing": trend in ["strong_up", "high"],
                "volume_signal": volume_signal,
                "volume_passed_conditions": passed,
            }
        }
