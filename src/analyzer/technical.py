"""
technical.py - Stock AI 2.0 テクニカル分析モジュール

2.0判定ルール：
- RSI: 25日MAフィルター付き条件分岐
- MACD: 75日MA以下の場合 ×0.7補正
- 出来高: 5日中3本以上陰線で -5点ペナルティ
- 移動平均: 25日・75日MA相対位置
"""

import pandas as pd
import numpy as np
from loguru import logger


class TechnicalAnalyzer:

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.rsi_period = cfg.get("rsi_period", 14)
        self.macd_fast = cfg.get("macd_fast", 12)
        self.macd_slow = cfg.get("macd_slow", 26)
        self.macd_signal = cfg.get("macd_signal", 9)

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
        if "sma25" in result and "sma75" in result:
            result["golden_cross"] = result["sma25"] > result["sma75"]
        return result

    def calculate_volume_trend(self, df: pd.DataFrame) -> dict:
        if len(df) < 20:
            return {"volume_ratio": 1.0, "volume_trend": "neutral", "red_candle_count": 0}
        recent_volume = df["volume"].tail(5).mean()
        avg_volume = df["volume"].tail(20).mean()
        volume_ratio = recent_volume / (avg_volume + 1e-10)
        trend = "high" if volume_ratio > 1.5 else "low" if volume_ratio < 0.7 else "neutral"

        # 2.0: 直近5日の陰線カウント
        recent = df.tail(5)
        red_candle_count = int((recent["close"] < recent["open"]).sum())

        return {
            "volume_ratio": float(volume_ratio),
            "volume_trend": trend,
            "red_candle_count": red_candle_count,
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
        Stock AI 2.0 テクニカルスコア（0〜100点）

        RSI    : 最大25点（25日MAフィルター付き）
        MACD   : 最大25点（75日MA以下は×0.7補正）
        移動平均: 最大25点
        出来高  : 最大15点（陰線3本以上で-5点ペナルティ）
        モメンタム: 最大10点
        """
        if df is None or len(df) < 30:
            return {"total_score": 50, "details": {}}

        prices = df["close"]
        score = 0
        details = {}

        rsi = self.calculate_rsi(prices)
        details["rsi"] = round(rsi, 1)
        ma_data = self.calculate_moving_averages(prices)
        details["moving_averages"] = ma_data

        # ===== RSIスコア 最大25点（2.0: 25日MAフィルター） =====
        above_sma25 = ma_data.get("above_sma25", False)

        if above_sma25:
            # 25日MA上：強気フィルター通過
            if 40 <= rsi <= 60:    rsi_score = 25
            elif 35 <= rsi < 40:   rsi_score = 20
            elif 60 < rsi <= 70:   rsi_score = 15
            elif rsi < 35:         rsi_score = 12  # 売られすぎ（底値圏）
            elif 70 < rsi <= 80:   rsi_score = 8
            else:                   rsi_score = 3   # 過熱域
        else:
            # 25日MA下：弱気フィルター（スコアを絞る）
            if rsi < 30:           rsi_score = 15  # 大底圏のみ加点
            elif rsi < 40:         rsi_score = 8
            elif 40 <= rsi <= 55:  rsi_score = 5
            else:                   rsi_score = 2

        score += rsi_score
        details["rsi_score"] = rsi_score
        details["rsi_filter"] = "25日MA上（強気）" if above_sma25 else "25日MA下（弱気）"

        # ===== MACDスコア 最大25点（2.0: 75日MA以下は×0.7補正） =====
        macd_data = self.calculate_macd(prices)
        details["macd"] = macd_data
        above_sma75 = ma_data.get("above_sma75", False)

        raw_macd_score = 0
        if macd_data["macd"] > 0:                                      raw_macd_score += 10
        if macd_data["histogram"] > macd_data["prev_histogram"]:       raw_macd_score += 10
        if macd_data["macd"] > macd_data["signal"]:                    raw_macd_score += 5

        # 2.0: 75日MA以下は×0.7補正
        if not above_sma75:
            macd_score = int(raw_macd_score * 0.7)
            details["macd_correction"] = "75日MA下 ×0.7補正"
        else:
            macd_score = raw_macd_score
            details["macd_correction"] = "補正なし"

        score += macd_score
        details["macd_score"] = macd_score

        # ===== 移動平均スコア 最大25点 =====
        ma_score = 0
        if ma_data.get("above_sma5"):    ma_score += 5
        if ma_data.get("above_sma25"):   ma_score += 7
        if ma_data.get("above_sma75"):   ma_score += 8
        if ma_data.get("golden_cross"):  ma_score += 5
        score += ma_score
        details["ma_score"] = ma_score

        # ===== 出来高スコア 最大15点（2.0: 陰線ペナルティ） =====
        vol_data = self.calculate_volume_trend(df)
        details["volume"] = vol_data

        if vol_data["volume_trend"] == "high":      vol_score = 15
        elif vol_data["volume_trend"] == "neutral":  vol_score = 8
        else:                                         vol_score = 3

        # 2.0: 直近5日中3本以上陰線で -5点
        red_count = vol_data["red_candle_count"]
        if red_count >= 3:
            vol_score -= 5
            details["volume_penalty"] = f"陰線{red_count}本 -5点"
        else:
            details["volume_penalty"] = "なし"

        vol_score = max(0, vol_score)
        score += vol_score
        details["volume_score"] = vol_score

        # ===== モメンタムスコア 最大10点 =====
        momentum = self.calculate_price_momentum(prices)
        details["momentum"] = momentum
        mom_score = 0
        if momentum.get("momentum_5d", 0) > 0:   mom_score += 3
        if momentum.get("momentum_20d", 0) > 0:  mom_score += 4
        if momentum.get("momentum_60d", 0) > 0:  mom_score += 3
        score += mom_score
        details["momentum_score"] = mom_score

        return {
            "total_score": min(100, max(0, score)),
            "details": details,
            "signals": {
                "rsi_signal": "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral",
                "trend": "uptrend" if ma_data.get("golden_cross") else "downtrend",
                "above_sma25": above_sma25,
                "above_sma75": above_sma75,
                "volume_increasing": vol_data["volume_trend"] == "high",
                "red_candles": red_count,
            }
        }
