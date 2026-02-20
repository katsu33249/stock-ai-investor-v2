"""
scoring_engine.py - 総合スコアリングエンジン

テクニカル・ファンダメンタル・政策の3軸スコアを統合し、
最終的な投資判断スコアと推奨アクションを生成します。
"""

import pandas as pd
from datetime import datetime
from loguru import logger

from src.analyzer.technical import TechnicalAnalyzer
from src.analyzer.fundamental import FundamentalAnalyzer
from src.screener.policy_screener import PolicyScreener


class ScoringEngine:
    """総合スコアリングエンジン"""

    def __init__(self, config: dict = None):
        cfg = config or {}

        # ウェイト設定（合計1.0）
        weights = cfg.get("scoring_weights", {})
        self.w_policy = weights.get("policy", 0.35)
        self.w_technical = weights.get("technical", 0.35)
        self.w_fundamental = weights.get("fundamental", 0.30)

        # 各分析エンジン初期化
        self.technical = TechnicalAnalyzer(cfg.get("technical", {}))
        self.fundamental = FundamentalAnalyzer()
        self.policy = PolicyScreener()

    def evaluate_stock(self, ticker: str, stock_info: dict) -> dict:
        """
        1銘柄の総合評価を実行

        Args:
            ticker: 銘柄コード
            stock_info: get_stock_info()の返り値（price_historyを含む）

        Returns:
            総合スコアと推奨アクションの辞書
        """
        logger.debug(f"評価中: {ticker}")

        price_history = stock_info.get("price_history")

        # 各スコア計算
        tech_result = self.technical.calculate_score(price_history)
        fund_result = self.fundamental.calculate_score(stock_info)
        policy_result = self.policy.calculate_policy_score(
            ticker,
            stock_info.get("sector", "") + " " + stock_info.get("industry", "")
        )

        tech_score = tech_result.get("total_score", 50)
        fund_score = fund_result.get("total_score", 50)
        policy_score = policy_result.get("total_score", 0)

        # 加重平均で総合スコア算出
        total_score = (
            tech_score * self.w_technical +
            fund_score * self.w_fundamental +
            policy_score * self.w_policy
        )
        total_score = round(total_score, 1)

        # 推奨アクション判定
        action = self._determine_action(total_score, tech_result, policy_score)

        # 投資判断コメント生成
        comment = self._generate_comment(
            total_score, tech_score, fund_score, policy_score,
            tech_result, fund_result, policy_result, stock_info
        )

        return {
            "ticker": ticker,
            "name": stock_info.get("name", ticker),
            "timestamp": datetime.now().isoformat(),
            # スコア
            "total_score": total_score,
            "technical_score": tech_score,
            "fundamental_score": fund_score,
            "policy_score": policy_score,
            # 判定
            "action": action,
            "action_emoji": self._action_emoji(action),
            "comment": comment,
            # 基本情報
            "current_price": stock_info.get("current_price", 0),
            "market_cap_B": round(stock_info.get("market_cap", 0) / 1e8, 0),  # 億円
            "per": stock_info.get("per"),
            "pbr": stock_info.get("pbr"),
            "roe": stock_info.get("roe"),
            "dividend_yield": stock_info.get("dividend_yield"),
            "sector": stock_info.get("sector", ""),
            "policy_sectors": policy_result.get("details", {}).get("matching_sectors", []),
            # 詳細スコア
            "score_details": {
                "technical": tech_result.get("details", {}),
                "fundamental": fund_result.get("details", {}),
                "policy": policy_result.get("details", {}),
            },
            "signals": tech_result.get("signals", {}),
        }

    def evaluate_multiple(self, stocks_data: dict) -> list[dict]:
        """
        複数銘柄を一括評価してランキング形式で返す

        Args:
            stocks_data: {ticker: stock_info} の辞書

        Returns:
            スコア降順でソートされた評価結果リスト
        """
        results = []
        total = len(stocks_data)

        for i, (ticker, stock_info) in enumerate(stocks_data.items(), 1):
            logger.info(f"スコアリング ({i}/{total}): {ticker}")
            try:
                result = self.evaluate_stock(ticker, stock_info)
                results.append(result)
            except Exception as e:
                logger.error(f"スコアリングエラー ({ticker}): {e}")

        # スコア降順でソート
        results.sort(key=lambda x: x["total_score"], reverse=True)

        # ランク付け
        for rank, result in enumerate(results, 1):
            result["rank"] = rank

        return results

    def to_dataframe(self, results: list[dict]) -> pd.DataFrame:
        """評価結果をDataFrameに変換"""
        rows = []
        for r in results:
            rows.append({
                "ランク": r.get("rank", "-"),
                "ティッカー": r["ticker"],
                "銘柄名": r["name"],
                "総合スコア": r["total_score"],
                "テクニカル": r["technical_score"],
                "ファンダメンタル": r["fundamental_score"],
                "政策スコア": r["policy_score"],
                "判定": f"{r['action_emoji']} {r['action']}",
                "株価": r.get("current_price", "-"),
                "時価総額(億円)": r.get("market_cap_B", "-"),
                "PER": r.get("per", "-"),
                "PBR": r.get("pbr", "-"),
                "政策セクター": ", ".join(r.get("policy_sectors", [])),
                "コメント": r.get("comment", ""),
            })
        return pd.DataFrame(rows)

    def _determine_action(self, total_score: float, tech_result: dict, policy_score: int) -> str:
        """推奨アクションを決定"""
        signals = tech_result.get("signals", {})

        if total_score >= 80:
            return "強気買い"
        elif total_score >= 70:
            if signals.get("rsi_signal") == "oversold":
                return "買い（RSI底値圏）"
            elif policy_score >= 70:
                return "買い（政策恩恵）"
            return "買い"
        elif total_score >= 60:
            return "監視・買い検討"
        elif total_score >= 50:
            return "様子見"
        elif total_score >= 40:
            return "保有継続"
        else:
            return "売り検討"

    def _action_emoji(self, action: str) -> str:
        """アクションに対応する絵文字"""
        emoji_map = {
            "強気買い": "🔥",
            "買い": "📈",
            "買い（RSI底値圏）": "📈",
            "買い（政策恩恵）": "🏛️",
            "監視・買い検討": "👀",
            "様子見": "⏸️",
            "保有継続": "💼",
            "売り検討": "📉",
        }
        return emoji_map.get(action, "❓")

    def _generate_comment(
        self, total: float, tech: float, fund: float, policy: float,
        tech_result: dict, fund_result: dict, policy_result: dict, info: dict
    ) -> str:
        """投資判断コメントを生成"""
        comments = []

        # 政策コメント
        if policy >= 70:
            sectors = policy_result.get("details", {}).get("matching_sectors", [])
            if sectors:
                comments.append(f"政策重点銘柄（{', '.join(sectors)}）")

        # テクニカルコメント
        signals = tech_result.get("signals", {})
        if signals.get("rsi_signal") == "oversold":
            comments.append("RSI売られすぎ圏（反発期待）")
        elif signals.get("trend") == "uptrend":
            comments.append("ゴールデンクロス形成中")
        if signals.get("volume_increasing"):
            comments.append("出来高急増")

        # ファンダメンタルコメント
        valuation = fund_result.get("valuation_summary", "")
        if valuation == "割安":
            comments.append("バリュエーション割安")
        elif valuation == "割高":
            comments.append("バリュエーション割高注意")

        roe = info.get("roe")
        if roe and (roe * 100 if roe < 1 else roe) >= 15:
            comments.append("高ROE優良企業")

        if not comments:
            if total >= 60:
                comments.append("総合的に良好な状態")
            else:
                comments.append("特段の注目シグナルなし")

        return " / ".join(comments)
