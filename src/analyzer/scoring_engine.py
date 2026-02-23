"""
scoring_engine.py - Stock AI 2.0 総合スコアリングエンジン

設計方針：
- ファンダは「足切り」用途に留める
- テクニカル×政策テーマの順張りを主軸にする

ウェイト：
- テクニカル  50%（主軸）
- 政策スコア  30%（テーマ）
- ファンダ    20%（足切り）

ペナルティ（足切り機能）：
- 営業CF赤字         → -5点
- D/E 200〜300%     → -3点
- D/E 300%超        → -5点
- 政策銘柄＋赤字     → -10点

政策底上げ：
- 政策80点以上 → ファンダ最低45点
- 政策70点以上 → ファンダ最低35点
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
        weights = cfg.get("scoring_weights", {})
        self.w_technical   = weights.get("technical",   0.50)
        self.w_fundamental = weights.get("fundamental", 0.20)
        self.w_policy      = weights.get("policy",      0.30)

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

        # ===== 政策底上げ（テーマ株の初動を逃さない） =====
        if policy_score >= 80 and fund_score < 45:
            fund_score = 45
        elif policy_score >= 70 and fund_score < 35:
            fund_score = 35

        # 加重平均
        total_score = (
            tech_score   * self.w_technical +
            fund_score   * self.w_fundamental +
            policy_score * self.w_policy
        )

        # ===== ペナルティ（足切り機能） =====
        penalties = []
        edinet_data = fund_result.get("raw_data", {})

        def get_val(key):
            if edinet_data and edinet_data.get(key) is not None:
                return edinet_data[key]
            return stock_info.get(key)

        # 信用倍率
        margin_ratio = stock_info.get("margin_ratio")
        if margin_ratio is not None and margin_ratio > 3.0:
            total_score -= 5
            penalties.append(f"信用倍率過熱({margin_ratio:.1f}倍) -5点")
        margin_bonus = 3 if (margin_ratio is not None and margin_ratio <= 1.0) else 0
        total_score += margin_bonus
        if margin_bonus > 0:
            penalties.append(f"信用倍率良好({margin_ratio:.1f}倍) +3点")

        # 営業CF赤字 -5点
        operating_cf = get_val("operating_cf")
        if operating_cf is not None and operating_cf < 0:
            total_score -= 5
            penalties.append("営業CF赤字 -5点")

        # D/E段階制ペナルティ（業種特性を考慮）
        debt_to_equity = get_val("debt_to_equity")
        if debt_to_equity is not None:
            if debt_to_equity > 300:
                total_score -= 5
                penalties.append(f"D/E過大({debt_to_equity:.0f}%) -5点")
            elif debt_to_equity > 200:
                total_score -= 3
                penalties.append(f"D/E高め({debt_to_equity:.0f}%) -3点")

        # 政策銘柄＋赤字 -10点
        is_policy = policy_score >= 50
        profit_margin = get_val("profit_margin")
        if is_policy and profit_margin is not None and profit_margin < 0:
            total_score -= 10
            penalties.append("政策銘柄・赤字企業 -10点")

        # ===== ボーナス（上限+6点） =====
        bonus_total = 0
        bonus_log = []

        # 出来高加速ボーナス（テーマ初動を拾う）
        price_history = stock_info.get("price_history")
        if price_history is not None and len(price_history) >= 20:
            vol_5  = price_history["volume"].tail(5).mean()
            vol_20 = price_history["volume"].tail(20).mean()
            if vol_20 > 0 and vol_5 / vol_20 >= 1.5:
                bonus_total += 3
                bonus_log.append(f"出来高加速({vol_5/vol_20:.1f}倍) +3点")

        # 売上高成長ボーナス
        revenue_growth = get_val("revenue_growth")
        if revenue_growth is not None:
            rg = revenue_growth * 100 if abs(revenue_growth) < 1 else revenue_growth
            if rg >= 20:
                bonus_total += 3
                bonus_log.append(f"売上成長({rg:.0f}%) +3点")

        # 営業利益成長ボーナス
        earnings_growth = get_val("earnings_growth")
        if earnings_growth is not None:
            eg = earnings_growth * 100 if abs(earnings_growth) < 1 else earnings_growth
            if eg >= 20:
                bonus_total += 3
                bonus_log.append(f"営業利益成長({eg:.0f}%) +3点")

        # ボーナス上限適用（最大+6点）
        bonus_total = min(bonus_total, 6)
        total_score += bonus_total
        penalties.extend(bonus_log)

        total_score = round(min(100, max(0, total_score)), 1)
        action = self._determine_action(total_score, tech_result, policy_score)
        comment = self._generate_comment(
            total_score, tech_score, fund_score, policy_score,
            tech_result, fund_result, policy_result, stock_info, penalties
        )

        per = get_val("per")
        pbr = get_val("pbr")
        roe = get_val("roe")
        dividend_yield = get_val("dividend_yield")

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
            "per": per,
            "pbr": pbr,
            "roe": roe,
            "dividend_yield": dividend_yield,
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
                "ROE": r.get("roe", "-"),
                "信用倍率": r.get("margin_ratio", "-"),
                "政策セクター": ", ".join(r.get("policy_sectors", [])),
                "コメント": r.get("comment", ""),
            })
        return pd.DataFrame(rows)

    def _determine_action(self, total_score, tech_result, policy_score):
        signals = tech_result.get("signals", {})
        if total_score >= 80:   return "強気買い"
        elif total_score >= 70:
            if signals.get("rsi_signal") == "oversold": return "買い（RSI底値圏）"
            elif policy_score >= 70:                     return "買い（政策恩恵）"
            return "買い"
        elif total_score >= 60: return "監視・買い検討"
        elif total_score >= 50: return "様子見"
        elif total_score >= 40: return "保有継続"
        else:                    return "売り検討"

    def _action_emoji(self, action):
        return {
            "強気買い": "🔥", "買い": "📈",
            "買い（RSI底値圏）": "📈", "買い（政策恩恵）": "🏛️",
            "監視・買い検討": "👀", "様子見": "⏸️",
            "保有継続": "💼", "売り検討": "📉",
        }.get(action, "❓")

    def _generate_comment(self, total, tech, fund, policy,
                          tech_result, fund_result, policy_result, info, penalties):
        comments = []
        if policy >= 80:
            sectors = policy_result.get("details", {}).get("matching_sectors", [])
            if sectors:
                comments.append(f"🏛️政策主力({', '.join(sectors[:2])})")
        elif policy >= 70:
            sectors = policy_result.get("details", {}).get("matching_sectors", [])
            if sectors:
                comments.append(f"政策({', '.join(sectors[:1])})")

        signals = tech_result.get("signals", {})
        if signals.get("rsi_signal") == "oversold": comments.append("RSI底値圏")
        elif signals.get("trend") == "uptrend":      comments.append("GC形成中")
        if signals.get("volume_increasing"):          comments.append("出来高急増")
        if signals.get("above_sma75"):                comments.append("75日MA上")

        if fund_result.get("valuation_summary") == "割安":
            comments.append("割安")

        margin_ratio = info.get("margin_ratio")
        if margin_ratio and margin_ratio <= 1.0:
            comments.append(f"信用良好({margin_ratio:.1f}倍)")
        elif margin_ratio and margin_ratio > 3.0:
            comments.append(f"⚠信用過熱({margin_ratio:.1f}倍)")

        if penalties:
            comments.append(f"⚠ペナルティ{len(penalties)}件")

        if not comments:
            comments.append("総合良好" if total >= 60 else "要観察")

        return " / ".join(comments)
