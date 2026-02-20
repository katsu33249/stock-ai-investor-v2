"""
technical.py - テクニカル分析モジュール

RSI、MACD、移動平均線などのテクニカル指標を計算し、
0〜100のスコアに変換します。

初心者メモ:
- RSI: 30以下が売られすぎ（買いシグナル）、70以上が買われすぎ（売りシグナル）
- MACD: シグナルラインを上抜けると買いシグナル
- ゴールデンクロス: 短期移動平均が長期移動平均を上抜けること（強い買いシグナル）
"""

import pandas as pd
import numpy as np
from loguru import logger


class TechnicalAnalyzer:
    """テクニカル分析クラス"""

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
        """RSIを計算"""
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
        """MACDを計算"""
        if len(prices) < self.macd_slow + self.macd_signal:
            return {"macd": 0, "signal": 0, "histogram": 0}

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
        """移動平均線を計算"""
        current_price = float(prices.iloc[-1])
        result = {"current_price": current_price}

        for period in [5, 25, 75, 200]:
            if len(prices) >= period:
                ma = float(prices.rolling(period).mean().iloc[-1])
                result[f"sma{period}"] = ma
                result[f"above_sma{period}"] = current_price > ma

        # ゴールデンクロス/デッドクロス判定
        if f"sma{self.sma_short}" in result and f"sma{self.sma_long}" in result:
            result["golden_cross"] = (
                result[f"sma{self.sma_short}"] > result[f"sma{self.sma_long}"]
            )

        return result

    def calculate_volume_trend(self, df: pd.DataFrame) -> dict:
        """出来高トレンドを計算"""
        if len(df) < 20:
            return {"volume_ratio": 1.0, "volume_trend": "neutral"}

        recent_volume = df["volume"].tail(5).mean()
        avg_volume = df["volume"].tail(20).mean()
        volume_ratio = recent_volume / (avg_volume + 1e-10)

        trend = "high" if volume_ratio > 1.5 else "low" if volume_ratio < 0.7 else "neutral"
        return {"volume_ratio": float(volume_ratio), "volume_trend": trend}

    def calculate_price_momentum(self, prices: pd.Series) -> dict:
        """価格モメンタムを計算"""
        result = {}
        for days in [5, 20, 60]:
            if len(prices) > days:
                momentum = (prices.iloc[-1] / prices.iloc[-days - 1] - 1) * 100
                result[f"momentum_{days}d"] = float(momentum)
        return result

    def calculate_score(self, df: pd.DataFrame) -> dict:
        """
        テクニカル総合スコアを計算（0〜100点）

        採点基準:
        - RSI (20点): 適切なゾーン（40-60）に近いほど高得点
        - MACD (20点): ヒストグラム上昇中、ゴールデンクロス近辺
        - 移動平均線 (25点): 上昇トレンド配置
        - 出来高 (15点): 出来高増加を伴う上昇
        - モメンタム (20点): 短中長期の上昇モメンタム
        """
        if df is None or len(df) < 30:
            return {"total_score": 50, "details": {}}

        prices = df["close"]
        score = 0
        details = {}

        # ===== RSIスコア (20点) =====
        rsi = self.calculate_rsi(prices)
        details["rsi"] = round(rsi, 1)

        if 40 <= rsi <= 55:          # 理想的なゾーン
            rsi_score = 20
        elif 35 <= rsi < 40:         # 少し売られすぎ（買いチャンス）
            rsi_score = 16
        elif 55 < rsi <= 65:         # 少し強め
            rsi_score = 14
        elif rsi < 35:               # 売られすぎ（反発期待）
            rsi_score = 12
        elif 65 < rsi <= 70:         # やや過熱
            rsi_score = 8
        else:                         # 買われすぎ（リスク大）
            rsi_score = 3
        score += rsi_score

        # ===== MACDスコア (20点) =====
        macd_data = self.calculate_macd(prices)
        details["macd"] = macd_data

        macd_score = 0
        # MACDがプラス（強気）
        if macd_data["macd"] > 0:
            macd_score += 8
        # ヒストグラムが上昇中（勢い増加）
        if macd_data["histogram"] > macd_data["prev_histogram"]:
            macd_score += 7
        # シグナルとの関係
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

        # ===== 出来高スコア (15点) =====
        vol_data = self.calculate_volume_trend(df)
        details["volume"] = vol_data

        if vol_data["volume_trend"] == "high":
            vol_score = 15
        elif vol_data["volume_trend"] == "neutral":
            vol_score = 8
        else:
            vol_score = 3
        score += vol_score

        # ===== モメンタムスコア (20点) =====
        momentum = self.calculate_price_momentum(prices)
        details["momentum"] = momentum

        mom_score = 0
        if momentum.get("momentum_5d", 0) > 0:  mom_score += 5
        if momentum.get("momentum_20d", 0) > 0: mom_score += 8
        if momentum.get("momentum_60d", 0) > 0: mom_score += 7
        score += mom_score

        return {
            "total_score": min(100, max(0, score)),
            "details": details,
            "signals": {
                "rsi_signal": "oversold" if rsi < 35 else "overbought" if rsi > 70 else "neutral",
                "trend": "uptrend" if ma_data.get("golden_cross") else "downtrend",
                "volume_increasing": vol_data["volume_trend"] == "high",
            }
        }
