"""
scoring_engine.py - Stock AI 2.0 総合スコアリングエンジン

2.0ウェイト：
- テクニカル  40%（従来35%）
- ファンダメンタル 35%（従来30%）
- 政策スコア  25%（従来35%）

2.0追加ペナルティ：
- 信用倍率3倍超: -5点
- EPS前年比マイナス: -5点
- 営業CF赤字: -5点
- 負債資本倍率200%超: -5点
- 政策銘柄かつ赤字: -10点
"""

import pandas as pd
from datetime import datetime
from loguru import logger

from src.analyzer.technical import TechnicalAnalyzer
from src.analyzer.fundamental import FundamentalAnalyzer
from src.screener.policy_screener import PolicyScreener


class ScoringEngine:

    def __init__(self, config: dict = None):
        cfg = config or {}

        # 2.0ウェイト
        weights = cfg.get("scoring_weights", {})
        self.w_technical    = weights.get("technical",    0.40)
        self.w_fundamental  = weights.get("fundamental",  0.35)
        self.w_policy       = weights.get("policy",       0.25)

        self.technical   = TechnicalAnalyzer(cfg.get("technical", {}))
        self.fundamental = FundamentalAnalyzer()
        self.policy      = PolicyScreener()

    def evaluate_stock(self, ticker: str, stock_info: dict) -> dict:
        price_history = stock_info.get("price_history")

        tech_result   = self.technical.calculate_score(price_history)
        fund_result   = self.fundamental.calculate_score(stock_info)
        policy_result = self.policy.calculate_policy_score(
            ticker,
            stock_info.get("sector", "") + " " + stock_info.get("industry", "")
        )

        tech_score   = tech_result.get("total_score", 50)
        fund_score   = fund_result.get("total_score", 50)
        policy_score = policy_result.get("total_score", 0)

        # 加重平均
        total_score = (
            tech_score   * self.w_technical +
            fund_score   * self.w_fundamental +
            policy_score * self.w_policy
        )

        # ===== 2.0ペナルティ =====
        penalties = []

        # 信用倍率3倍超 -5点
        margin_ratio = stock_info.get("margin_ratio")
        if margin_ratio is not None and margin_ratio > 3.0:
            total_score -= 5
            penalties.append(f"信用倍率過熱({margin_ratio:.1f}倍) -5点")

        # 信用倍率スコア（1倍以下は+加点）
        margin_bonus = self._score_margin(margin_ratio)
        total_score += margin_bonus
        if margin_bonus > 0:
            penalties.append(f"信用倍率良好 +{margin_bonus}点")

        # EPS前年比マイナス -5点
        earnings_growth = stock_info.get("earnings_growth")
        if earnings_growth is not None and earnings_growth < 0:
            total_score -= 5
            penalties.append(f"EPS前年比マイナス({earnings_growth:.1f}%) -5点")

        # 営業CF赤字 -5点（EDINET DBから取得）
        operating_cf = stock_info.get("operating_cf")
        if operating_cf is not None and operating_cf < 0:
            total_score -= 5
            penalties.append("営業CF赤字 -5点")

        # 負債資本倍率200%超 -5点
        debt_to_equity = stock_info.get("debt_to_equity")
        if debt_to_equity is not None and debt_to_equity > 200:
            total_score -= 5
            penalties.append(f"高負債(D/E:{debt_to_equity:.0f}%) -5点")

        # 政策銘柄かつ赤字 -10点
        is_policy = policy_score >= 50
        profit_margin = stock_info.get("profit_margin")
        if is_policy and profit_margin is not None and profit_margin < 0:
            total_score -= 10
            penalties.append("政策銘柄・赤字企業 -10点")

        total_score = round(min(100, max(0, total_score)), 1)
        action = self._determine_action(total_score, tech_result, policy_score)
        comment = self._generate_comment(
            total_score, tech_score, fund_score, policy_score,
            tech_result, fund_result, policy_result, stock_info, penalties
        )

        return {
            "ticker": ticker,
            "name": stock_info.get("name", ticker),
            "timestamp": datetime.now().isoformat(),
            "total_score": total_score,
            "technical_score": tech_score,
            "fundamental_score": fund_score,
            "policy_score": policy_score,
            "action": action,
            "action_emoji": self._action_emoji(action),
            "comment": comment,
            "penalties": penalties,
            "current_price": stock_info.get("current_price", 0),
            "market_cap_B": round(stock_info.get("market_cap", 0) / 1e8, 0),
            "per": stock_info.get("per"),
            "pbr": stock_info.get("pbr"),
            "roe": stock_info.get("roe"),
            "dividend_yield": stock_info.get("dividend_yield"),
            "margin_ratio": margin_ratio,
            "sector": stock_info.get("sector", ""),
            "policy_sectors": policy_result.get("details", {}).get("matching_sectors", []),
            "ai_comment": fund_result.get("ai_comment", ""),
            "data_source": fund_result.get("data_source", ""),
            "score_details": {
                "technical": tech_result.get("details", {}),
                "fundamental": fund_result.get("details", {}),
                "policy": policy_result.get("details", {}),
            },
            "signals": tech_result.get("signals", {}),
        }

    def _score_margin(self, margin_ratio) -> int:
        """信用倍率加点（1倍以下のみ加点）"""
        if margin_ratio is None:
            return 0
        if margin_ratio <= 1.0:
            return 3   # 売り圧力なし
        return 0

    def evaluate_multiple(self, stocks_data: dict) -> list:
        results = []
        total = len(stocks_data)
        for i, (ticker, stock_info) in enumerate(stocks_data.items(), 1):
            logger.info(f"スコアリング ({i}/{total}): {ticker}")
            try:
                result = self.evaluate_stock(ticker, stock_info)
                results.append(result)
            except Exception as e:
                logger.error(f"スコアリングエラー ({ticker}): {e}")

        results.sort(key=lambda x: x["total_score"], reverse=True)
        for rank, result in enumerate(results, 1):
            result["rank"] = rank
        return results

    def to_dataframe(self, results: list) -> pd.DataFrame:
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
                "PER": r.get("per", "-"),
                "PBR": r.get("pbr", "-"),
                "信用倍率": r.get("margin_ratio", "-"),
                "政策セクター": ", ".join(r.get("policy_sectors", [])),
                "コメント": r.get("comment", ""),
            })
        return pd.DataFrame(rows)

    def _determine_action(self, total_score: float, tech_result: dict, policy_score: int) -> str:
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
        return {
            "強気買い": "🔥",
            "買い": "📈",
            "買い（RSI底値圏）": "📈",
            "買い（政策恩恵）": "🏛️",
            "監視・買い検討": "👀",
            "様子見": "⏸️",
            "保有継続": "💼",
            "売り検討": "📉",
        }.get(action, "❓")

    def _generate_comment(
        self, total, tech, fund, policy,
        tech_result, fund_result, policy_result, info, penalties
    ) -> str:
        comments = []

        if policy >= 70:
            sectors = policy_result.get("details", {}).get("matching_sectors", [])
            if sectors:
                comments.append(f"政策重点（{', '.join(sectors[:2])}）")

        signals = tech_result.get("signals", {})
        if signals.get("rsi_signal") == "oversold":
            comments.append("RSI底値圏（反発期待）")
        elif signals.get("trend") == "uptrend":
            comments.append("GC形成中")
        if signals.get("volume_increasing"):
            comments.append("出来高急増")
        if signals.get("above_sma75"):
            comments.append("75日MA上")

        valuation = fund_result.get("valuation_summary", "")
        if valuation == "割安":
            comments.append("バリュエーション割安")

        margin_ratio = info.get("margin_ratio")
        if margin_ratio and margin_ratio <= 1.0:
            comments.append(f"信用倍率良好({margin_ratio:.1f}倍)")
        elif margin_ratio and margin_ratio > 3.0:
            comments.append(f"⚠信用過熱({margin_ratio:.1f}倍)")

        if penalties:
            comments.append(f"⚠ペナルティ:{len(penalties)}件")

        if not comments:
            comments.append("総合的に良好" if total >= 60 else "要観察")

        return " / ".join(comments)
