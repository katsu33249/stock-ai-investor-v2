"""
fundamental.py - EDINET DB対応ファンダメンタル分析モジュール

EDINET DB API（全上場企業3,800社の財務データ）を使用して
正確なPER/PBR/ROE等を取得・スコアリングします。

初心者メモ:
- EDINET DB = 金融庁の有価証券報告書を整理したAPI
- Yahoo Financeより正確で信頼性が高い公式データ
- AI財務分析スコアも取得できます
"""

import requests
import os
from loguru import logger
from typing import Optional


class FundamentalAnalyzer:
    """EDINET DB対応ファンダメンタル分析クラス"""

    BASE_URL = "https://edinetdb.jp/v1"

    def __init__(self):
        self.api_key = os.environ.get("EDINET_DB_API_KEY", "")
        if not self.api_key:
            logger.warning("EDINET_DB_API_KEY が未設定。Yahoo Financeのデータを使用します")
        self.headers = {"X-API-Key": self.api_key}

        # 業種別平均PER
        self.sector_avg_per = {
            "Technology": 30, "Defense": 18, "Materials": 12,
            "Energy": 10, "Healthcare": 25, "Industrials": 16,
            "Consumer Cyclical": 18, "Financial Services": 12,
            "Utilities": 14, "default": 20,
        }

    def _ticker_to_seccode(self, ticker: str) -> str:
        """7011.T → 70110 形式に変換"""
        code = ticker.replace(".T", "")
        return code + "0" if len(code) == 4 else code

    def get_financial_data(self, ticker: str) -> Optional[dict]:
        """
        EDINET DBから財務データを取得

        取得データ:
        - 財務指標（PER/PBR/ROE/売上成長率等）
        - AI財務分析スコア（信用スコア）
        """
        if not self.api_key:
            return None

        sec_code = self._ticker_to_seccode(ticker)

        try:
            # ① 企業検索でEDINETコードを取得
            res_search = requests.get(
                f"{self.BASE_URL}/search",
                params={"q": sec_code},
                headers=self.headers,
                timeout=10
            )
            if res_search.status_code != 200:
                logger.warning(f"企業検索失敗({ticker}): {res_search.status_code}")
                return None

            companies = res_search.json().get("data", [])
            if not companies:
                logger.warning(f"企業が見つかりません: {ticker}")
                return None

            # 証券コードが一致する企業を選択
            edinet_code = None
            for c in companies:
                if str(c.get("sec_code", "")) == sec_code:
                    edinet_code = c.get("edinet_code")
                    break

            if not edinet_code:
                edinet_code = companies[0].get("edinet_code")

            # ② 財務指標を取得
            res_ratios = requests.get(
                f"{self.BASE_URL}/companies/{edinet_code}/ratios",
                headers=self.headers,
                timeout=10
            )

            # ③ AI財務分析を取得
            res_analysis = requests.get(
                f"{self.BASE_URL}/companies/{edinet_code}/analysis",
                headers=self.headers,
                timeout=10
            )

            ratios = {}
            if res_ratios.status_code == 200:
                data = res_ratios.json().get("data", {})
                ratios = {
                    "per": self._safe_float(data.get("per")),
                    "pbr": self._safe_float(data.get("pbr")),
                    "roe": self._safe_float(data.get("roe")),
                    "roa": self._safe_float(data.get("roa")),
                    "profit_margin": self._safe_float(data.get("net_margin")),
                    "operating_margin": self._safe_float(data.get("operating_margin")),
                    "revenue_growth": self._safe_float(data.get("revenue_growth_rate")),
                    "earnings_growth": self._safe_float(data.get("operating_income_growth_rate")),
                    "dividend_yield": self._safe_float(data.get("dividend_yield")),
                    "debt_to_equity": self._safe_float(data.get("debt_to_equity")),
                    "current_ratio": self._safe_float(data.get("current_ratio")),
                    "equity_ratio": self._safe_float(data.get("equity_ratio")),
                }

            # AI財務分析スコア
            credit_score = None
            ai_comment = ""
            if res_analysis.status_code == 200:
                analysis = res_analysis.json().get("data", {})
                credit_score = analysis.get("credit_score")
                ai_comment = analysis.get("summary", "")

            logger.debug(f"EDINET DB取得成功: {ticker} (credit_score: {credit_score})")
            return {**ratios, "credit_score": credit_score, "ai_comment": ai_comment}

        except Exception as e:
            logger.error(f"EDINET DB取得エラー({ticker}): {e}")
            return None

    def _safe_float(self, value) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    def score_per(self, per: float, sector: str = "default") -> tuple:
        if per is None or per <= 0:
            return 10, "データなし"
        sector_avg = self.sector_avg_per.get(sector, self.sector_avg_per["default"])
        if per < sector_avg * 0.5:   return 25, f"非常に割安 (PER:{per:.1f})"
        elif per < sector_avg * 0.8: return 20, f"割安 (PER:{per:.1f})"
        elif per < sector_avg * 1.2: return 15, f"適正 (PER:{per:.1f})"
        elif per < sector_avg * 1.5: return 8,  f"やや割高 (PER:{per:.1f})"
        else:                         return 3,  f"割高 (PER:{per:.1f})"

    def score_pbr(self, pbr: float) -> tuple:
        if pbr is None or pbr <= 0:
            return 5, "データなし"
        if pbr < 1.0:   return 15, f"資産価値以下 (PBR:{pbr:.2f})"
        elif pbr < 1.5: return 12, f"割安 (PBR:{pbr:.2f})"
        elif pbr < 2.5: return 9,  f"適正 (PBR:{pbr:.2f})"
        elif pbr < 4.0: return 5,  f"やや割高 (PBR:{pbr:.2f})"
        else:            return 2,  f"割高 (PBR:{pbr:.2f})"

    def score_roe(self, roe: float) -> tuple:
        if roe is None:
            return 5, "データなし"
        roe_pct = roe * 100 if abs(roe) < 1 else roe
        if roe_pct >= 20:   return 20, f"優秀 (ROE:{roe_pct:.1f}%)"
        elif roe_pct >= 15: return 16, f"良好 (ROE:{roe_pct:.1f}%)"
        elif roe_pct >= 10: return 12, f"標準 (ROE:{roe_pct:.1f}%)"
        elif roe_pct >= 5:  return 7,  f"やや低い (ROE:{roe_pct:.1f}%)"
        else:                return 2,  f"低収益 (ROE:{roe_pct:.1f}%)"

    def score_growth(self, revenue_growth: float, earnings_growth: float) -> tuple:
        if revenue_growth is None and earnings_growth is None:
            return 8, "データなし"
        rg = (revenue_growth or 0) * 100 if abs(revenue_growth or 0) < 1 else (revenue_growth or 0)
        eg = (earnings_growth or 0) * 100 if abs(earnings_growth or 0) < 1 else (earnings_growth or 0)
        if rg > 20 and eg > 20:   return 20, f"高成長 (売上:{rg:.0f}%, 利益:{eg:.0f}%)"
        elif rg > 10 and eg > 10: return 16, f"安定成長 (売上:{rg:.0f}%, 利益:{eg:.0f}%)"
        elif rg > 5 or eg > 10:   return 12, f"緩やかな成長"
        elif rg > 0:               return 8,  f"微増"
        else:                       return 3,  f"減収"

    def score_dividend(self, dividend_yield: float) -> tuple:
        if dividend_yield is None or dividend_yield == 0:
            return 3, "無配当"
        dy = dividend_yield * 100 if dividend_yield < 1 else dividend_yield
        if dy >= 4.0:   return 10, f"高配当 ({dy:.1f}%)"
        elif dy >= 2.5: return 8,  f"良好 ({dy:.1f}%)"
        elif dy >= 1.5: return 6,  f"標準 ({dy:.1f}%)"
        else:            return 3,  f"低め ({dy:.1f}%)"

    def score_financial_health(self, debt_to_equity: float, current_ratio: float, equity_ratio: float = None) -> tuple:
        score = 5
        notes = []
        if equity_ratio is not None:
            if equity_ratio >= 50:   score += 3; notes.append(f"財務優良(自己資本比率:{equity_ratio:.0f}%)")
            elif equity_ratio >= 30: score += 1; notes.append(f"安定({equity_ratio:.0f}%)")
            else:                     score -= 1; notes.append(f"低め({equity_ratio:.0f}%)")
        elif debt_to_equity is not None:
            if debt_to_equity < 30:   score += 3; notes.append(f"低負債(D/E:{debt_to_equity:.0f}%)")
            elif debt_to_equity < 100: score += 1
            else:                      score -= 2; notes.append(f"高負債")
        if current_ratio is not None:
            if current_ratio > 2.0: score += 2; notes.append(f"流動性良好")
            elif current_ratio > 1.2: score += 1
        return min(10, max(0, score)), " / ".join(notes) or "標準的"

    def score_credit(self, credit_score) -> tuple:
        """EDINET DB AI財務スコア（0〜100）をそのまま活用"""
        if credit_score is None:
            return 0, ""
        score = int(credit_score)
        if score >= 80:   return 10, f"🤖 AI財務スコア優秀({score}点)"
        elif score >= 60: return 7,  f"🤖 AI財務スコア良好({score}点)"
        elif score >= 40: return 4,  f"🤖 AI財務スコア普通({score}点)"
        else:              return 1,  f"🤖 AI財務スコア低い({score}点)"

    def calculate_score(self, stock_info: dict) -> dict:
        """
        ファンダメンタル総合スコアを計算（0〜100点）

        EDINET DBが使える場合は正確な財務データを優先使用。
        使えない場合はYahoo Financeのデータにフォールバック。

        配点:
        - PER評価      : 25点
        - PBR評価      : 15点
        - ROE評価      : 20点
        - 成長性       : 20点
        - 配当         : 10点
        - 財務健全性   : 10点
        + AI財務ボーナス: 最大10点（EDINET DB限定）
        """
        if not stock_info:
            return {"total_score": 50, "details": {}}

        # EDINET DBから財務データを取得（利用可能な場合）
        edinet_data = None
        ticker = stock_info.get("ticker", "")
        if self.api_key and ticker:
            edinet_data = self.get_financial_data(ticker)

        # EDINET DBのデータを優先、なければYahoo Financeを使用
        def get_val(key):
            if edinet_data and edinet_data.get(key) is not None:
                return edinet_data[key]
            return stock_info.get(key)

        sector = stock_info.get("sector", "default")
        score = 0
        details = {}
        data_source = "EDINET DB" if edinet_data else "Yahoo Finance"

        # PERスコア
        per_score, per_note = self.score_per(get_val("per"), sector)
        score += per_score
        details["per"] = {"score": per_score, "note": per_note}

        # PBRスコア
        pbr_score, pbr_note = self.score_pbr(get_val("pbr"))
        score += pbr_score
        details["pbr"] = {"score": pbr_score, "note": pbr_note}

        # ROEスコア
        roe_score, roe_note = self.score_roe(get_val("roe"))
        score += roe_score
        details["roe"] = {"score": roe_score, "note": roe_note}

        # 成長性スコア
        growth_score, growth_note = self.score_growth(
            get_val("revenue_growth"), get_val("earnings_growth")
        )
        score += growth_score
        details["growth"] = {"score": growth_score, "note": growth_note}

        # 配当スコア
        div_score, div_note = self.score_dividend(get_val("dividend_yield"))
        score += div_score
        details["dividend"] = {"score": div_score, "note": div_note}

        # 財務健全性スコア
        health_score, health_note = self.score_financial_health(
            get_val("debt_to_equity"),
            get_val("current_ratio"),
            edinet_data.get("equity_ratio") if edinet_data else None
        )
        score += health_score
        details["financial_health"] = {"score": health_score, "note": health_note}

        # AI財務スコア（EDINET DB限定ボーナス）
        ai_comment = ""
        if edinet_data:
            credit_score_val = edinet_data.get("credit_score")
            credit_score_pts, credit_note = self.score_credit(credit_score_val)
            score += credit_score_pts
            ai_comment = edinet_data.get("ai_comment", "")
            details["ai_score"] = {"score": credit_score_pts, "note": credit_note}

        # バリュエーション判定
        per = get_val("per")
        pbr = get_val("pbr")
        valuation = "不明"
        if per and pbr:
            if per < 15 and pbr < 1.2:  valuation = "割安"
            elif per > 30 or pbr > 4:   valuation = "割高"
            else:                         valuation = "適正"

        return {
            "total_score": min(100, max(0, score)),
            "details": details,
            "valuation_summary": valuation,
            "data_source": data_source,
            "ai_comment": ai_comment,
        }
