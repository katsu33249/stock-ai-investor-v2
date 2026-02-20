"""
fundamental.py - ファンダメンタル分析モジュール

PER、PBR、ROE等の財務指標を評価し、0〜100のスコアに変換します。

初心者メモ:
- PER (株価収益率): 低いほど割安。15〜20が標準、25以上は割高感
- PBR (株価純資産倍率): 1以下は理論上割安
- ROE (自己資本利益率): 高いほど効率的な経営。10%以上が優良
- 配当利回り: 高いほど株主還元が手厚い
"""

from loguru import logger


class FundamentalAnalyzer:
    """ファンダメンタル分析クラス"""

    def __init__(self):
        # 日本株の業種別平均PER参考値
        self.sector_avg_per = {
            "Technology": 30,
            "Defense": 18,
            "Materials": 12,
            "Energy": 10,
            "Healthcare": 25,
            "Industrials": 16,
            "Consumer Cyclical": 18,
            "Financial Services": 12,
            "Utilities": 14,
            "default": 20,
        }

    def score_per(self, per: float, sector: str = "default") -> tuple[int, str]:
        """PERをスコア化（最大25点）"""
        if per is None or per <= 0:
            return 10, "データなし"

        sector_avg = self.sector_avg_per.get(sector, self.sector_avg_per["default"])

        if per < sector_avg * 0.5:
            return 25, f"非常に割安 (PER:{per:.1f})"
        elif per < sector_avg * 0.8:
            return 20, f"割安 (PER:{per:.1f})"
        elif per < sector_avg * 1.2:
            return 15, f"適正 (PER:{per:.1f})"
        elif per < sector_avg * 1.5:
            return 8, f"やや割高 (PER:{per:.1f})"
        else:
            return 3, f"割高 (PER:{per:.1f})"

    def score_pbr(self, pbr: float) -> tuple[int, str]:
        """PBRをスコア化（最大15点）"""
        if pbr is None or pbr <= 0:
            return 5, "データなし"

        if pbr < 1.0:
            return 15, f"資産価値以下 (PBR:{pbr:.2f})"
        elif pbr < 1.5:
            return 12, f"割安 (PBR:{pbr:.2f})"
        elif pbr < 2.5:
            return 9, f"適正 (PBR:{pbr:.2f})"
        elif pbr < 4.0:
            return 5, f"やや割高 (PBR:{pbr:.2f})"
        else:
            return 2, f"割高 (PBR:{pbr:.2f})"

    def score_roe(self, roe: float) -> tuple[int, str]:
        """ROEをスコア化（最大20点）"""
        if roe is None:
            return 5, "データなし"

        roe_pct = roe * 100 if roe < 1 else roe  # 比率→パーセント変換

        if roe_pct >= 20:
            return 20, f"優秀な収益性 (ROE:{roe_pct:.1f}%)"
        elif roe_pct >= 15:
            return 16, f"良好 (ROE:{roe_pct:.1f}%)"
        elif roe_pct >= 10:
            return 12, f"標準 (ROE:{roe_pct:.1f}%)"
        elif roe_pct >= 5:
            return 7, f"やや低い (ROE:{roe_pct:.1f}%)"
        else:
            return 2, f"低収益 (ROE:{roe_pct:.1f}%)"

    def score_growth(self, revenue_growth: float, earnings_growth: float) -> tuple[int, str]:
        """成長性をスコア化（最大20点）"""
        if revenue_growth is None and earnings_growth is None:
            return 8, "データなし"

        rg = (revenue_growth or 0) * 100
        eg = (earnings_growth or 0) * 100

        # 売上・利益ともに高成長
        if rg > 20 and eg > 20:
            return 20, f"高成長 (売上:{rg:.0f}%, 利益:{eg:.0f}%)"
        elif rg > 10 and eg > 10:
            return 16, f"安定成長 (売上:{rg:.0f}%, 利益:{eg:.0f}%)"
        elif rg > 5 or eg > 10:
            return 12, f"緩やかな成長 (売上:{rg:.0f}%, 利益:{eg:.0f}%)"
        elif rg > 0:
            return 8, f"微増 (売上:{rg:.0f}%, 利益:{eg:.0f}%)"
        else:
            return 3, f"減収 (売上:{rg:.0f}%, 利益:{eg:.0f}%)"

    def score_dividend(self, dividend_yield: float) -> tuple[int, str]:
        """配当利回りをスコア化（最大10点）"""
        if dividend_yield is None or dividend_yield == 0:
            return 3, "無配当"

        dy_pct = dividend_yield * 100 if dividend_yield < 1 else dividend_yield

        if dy_pct >= 4.0:
            return 10, f"高配当 ({dy_pct:.1f}%)"
        elif dy_pct >= 2.5:
            return 8, f"良好 ({dy_pct:.1f}%)"
        elif dy_pct >= 1.5:
            return 6, f"標準 ({dy_pct:.1f}%)"
        elif dy_pct >= 0.5:
            return 4, f"低め ({dy_pct:.1f}%)"
        else:
            return 3, f"ごくわずか ({dy_pct:.1f}%)"

    def score_financial_health(self, debt_to_equity: float, current_ratio: float) -> tuple[int, str]:
        """財務健全性をスコア化（最大10点）"""
        score = 5  # デフォルト
        notes = []

        if debt_to_equity is not None:
            if debt_to_equity < 30:
                score += 3
                notes.append(f"低負債(D/E:{debt_to_equity:.0f}%)")
            elif debt_to_equity < 100:
                score += 1
                notes.append(f"適正(D/E:{debt_to_equity:.0f}%)")
            else:
                score -= 2
                notes.append(f"高負債(D/E:{debt_to_equity:.0f}%)")

        if current_ratio is not None:
            if current_ratio > 2.0:
                score += 2
                notes.append(f"流動性良好({current_ratio:.1f})")
            elif current_ratio > 1.2:
                score += 1

        return min(10, max(0, score)), " / ".join(notes) or "標準的"

    def calculate_score(self, stock_info: dict) -> dict:
        """
        ファンダメンタル総合スコアを計算（0〜100点）

        配点:
        - PER評価: 25点
        - PBR評価: 15点
        - ROE評価: 20点
        - 成長性:  20点
        - 配当:    10点
        - 財務健全性: 10点
        """
        if not stock_info:
            return {"total_score": 50, "details": {}}

        sector = stock_info.get("sector", "default")
        score = 0
        details = {}

        # PERスコア
        per_score, per_note = self.score_per(stock_info.get("per"), sector)
        score += per_score
        details["per"] = {"score": per_score, "note": per_note}

        # PBRスコア
        pbr_score, pbr_note = self.score_pbr(stock_info.get("pbr"))
        score += pbr_score
        details["pbr"] = {"score": pbr_score, "note": pbr_note}

        # ROEスコア
        roe_score, roe_note = self.score_roe(stock_info.get("roe"))
        score += roe_score
        details["roe"] = {"score": roe_score, "note": roe_note}

        # 成長性スコア
        growth_score, growth_note = self.score_growth(
            stock_info.get("revenue_growth"),
            stock_info.get("earnings_growth")
        )
        score += growth_score
        details["growth"] = {"score": growth_score, "note": growth_note}

        # 配当スコア
        div_score, div_note = self.score_dividend(stock_info.get("dividend_yield"))
        score += div_score
        details["dividend"] = {"score": div_score, "note": div_note}

        # 財務健全性スコア
        health_score, health_note = self.score_financial_health(
            stock_info.get("debt_to_equity"),
            stock_info.get("current_ratio")
        )
        score += health_score
        details["financial_health"] = {"score": health_score, "note": health_note}

        # バリュエーション判定サマリー
        per = stock_info.get("per")
        pbr = stock_info.get("pbr")
        valuation = "不明"
        if per and pbr:
            if per < 15 and pbr < 1.2:
                valuation = "割安"
            elif per > 30 or pbr > 4:
                valuation = "割高"
            else:
                valuation = "適正"

        return {
            "total_score": min(100, max(0, score)),
            "details": details,
            "valuation_summary": valuation,
        }
