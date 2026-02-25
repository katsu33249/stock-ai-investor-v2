"""
fundamental.py - EDINET DB対応ファンダメンタル分析モジュール
"""

import requests
import json
import os
from pathlib import Path
from loguru import logger
from typing import Optional


class FundamentalAnalyzer:

    BASE_URL = "https://edinetdb.jp/v1"

    def __init__(self):
        self.api_key = os.environ.get("EDINET_DB_API_KEY", "")
        if not self.api_key:
            logger.warning("EDINET_DB_API_KEY が未設定。キャッシュのみ使用します")
        self.headers = {"X-API-Key": self.api_key}

        # キャッシュ読み込み
        self.cache = {}
        cache_path = Path("data/cache/fundamental_cache.json")
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"財務キャッシュ読み込み: {len(self.cache)}銘柄")
            except Exception as e:
                logger.warning(f"キャッシュ読み込みエラー: {e}")

        self.sector_avg_per = {
            "Technology": 30, "Defense": 18, "Materials": 12,
            "Energy": 10, "Healthcare": 25, "Industrials": 16,
            "Consumer Cyclical": 18, "Financial Services": 12,
            "Utilities": 14, "default": 20,
        }

    def _ticker_to_seccode(self, ticker: str) -> str:
        code = ticker.replace(".T", "")
        return code + "0" if len(code) == 4 else code

    def get_financial_data(self, ticker: str) -> Optional[dict]:
        """キャッシュ → EDINET DB の順で財務データを取得"""

        # キャッシュ優先
        if ticker in self.cache:
            return self.cache[ticker]

        # キャッシュになければAPIから取得
        if not self.api_key:
            return None

        sec_code = self._ticker_to_seccode(ticker)
        try:
            res_search = requests.get(
                f"{self.BASE_URL}/search",
                params={"q": sec_code},
                headers=self.headers,
                timeout=30
            )
            if res_search.status_code != 200:
                return None

            companies = res_search.json().get("data", [])
            if not companies:
                return None

            edinet_code = None
            for c in companies:
                if str(c.get("sec_code", "")) == sec_code:
                    edinet_code = c.get("edinet_code")
                    break
            if not edinet_code:
                edinet_code = companies[0].get("edinet_code")

            res_r = requests.get(
                f"{self.BASE_URL}/companies/{edinet_code}/ratios",
                headers=self.headers, timeout=30
            )
            res_a = requests.get(
                f"{self.BASE_URL}/companies/{edinet_code}/analysis",
                headers=self.headers, timeout=30
            )

            data = {}
            if res_r.status_code == 200:
                raw = res_r.json().get("data", {})
                if isinstance(raw, list):
                    raw = raw[0] if raw else {}
                data = {
                    "per": self._safe_float(raw.get("per")),
                    "pbr": self._safe_float(raw.get("pbr")),
                    "roe": self._safe_float(raw.get("roe")),
                    "roa": self._safe_float(raw.get("roa")),
                    "profit_margin": self._safe_float(raw.get("net_margin")),
                    "operating_margin": self._safe_float(raw.get("operating_margin")),
                    "revenue_growth": self._safe_float(raw.get("revenue_growth_rate")),
                    "earnings_growth": self._safe_float(raw.get("operating_income_growth_rate")),
                    "dividend_yield": self._safe_float(raw.get("dividend_yield")),
                    "debt_to_equity": self._safe_float(raw.get("debt_to_equity")),
                    "current_ratio": self._safe_float(raw.get("current_ratio")),
                    "equity_ratio": self._safe_float(raw.get("equity_ratio")),
                    "operating_cf": self._safe_float(raw.get("operating_cash_flow")),
                }

            if res_a.status_code == 200:
                raw_a = res_a.json().get("data", {})
                if isinstance(raw_a, list):
                    raw_a = raw_a[0] if raw_a else {}
                data["credit_score"] = raw_a.get("credit_score")
                data["ai_comment"] = raw_a.get("summary", "")

            return data if data else None

        except Exception as e:
            logger.error(f"EDINET DB取得エラー({ticker}): {e}")
            return None

    def _safe_float(self, value) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    def score_per(self, per, sector="default"):
        if per is None or per <= 0: return 10, "データなし"
        sector_avg = self.sector_avg_per.get(sector, self.sector_avg_per["default"])
        if per < sector_avg * 0.5:   return 25, f"非常に割安(PER:{per:.1f})"
        elif per < sector_avg * 0.8: return 20, f"割安(PER:{per:.1f})"
        elif per < sector_avg * 1.2: return 15, f"適正(PER:{per:.1f})"
        elif per < sector_avg * 1.5: return 8,  f"やや割高(PER:{per:.1f})"
        else:                         return 3,  f"割高(PER:{per:.1f})"

    def score_pbr(self, pbr):
        if pbr is None or pbr <= 0: return 5, "データなし"
        if pbr < 1.0:   return 15, f"資産価値以下(PBR:{pbr:.2f})"
        elif pbr < 1.5: return 12, f"割安(PBR:{pbr:.2f})"
        elif pbr < 2.5: return 9,  f"適正(PBR:{pbr:.2f})"
        elif pbr < 4.0: return 5,  f"やや割高(PBR:{pbr:.2f})"
        else:            return 2,  f"割高(PBR:{pbr:.2f})"

    def score_roe(self, roe):
        if roe is None: return 5, "データなし"
        roe_pct = roe * 100 if abs(roe) < 1 else roe
        if roe_pct >= 20:   return 20, f"優秀(ROE:{roe_pct:.1f}%)"
        elif roe_pct >= 15: return 16, f"良好(ROE:{roe_pct:.1f}%)"
        elif roe_pct >= 10: return 12, f"標準(ROE:{roe_pct:.1f}%)"
        elif roe_pct >= 5:  return 7,  f"やや低い(ROE:{roe_pct:.1f}%)"
        else:                return 2,  f"低収益(ROE:{roe_pct:.1f}%)"

    def score_growth(self, revenue_growth, earnings_growth):
        if revenue_growth is None and earnings_growth is None:
            return 8, "データなし"
        rg = (revenue_growth or 0) * 100 if abs(revenue_growth or 0) < 1 else (revenue_growth or 0)
        eg = (earnings_growth or 0) * 100 if abs(earnings_growth or 0) < 1 else (earnings_growth or 0)
        if rg > 20 and eg > 20:   return 20, f"高成長(売上:{rg:.0f}%)"
        elif rg > 10 and eg > 10: return 16, f"安定成長(売上:{rg:.0f}%)"
        elif rg > 5 or eg > 10:   return 12, f"緩やか成長"
        elif rg > 0:               return 8,  f"微増"
        else:                       return 3,  f"減収"

    def score_dividend(self, dividend_yield):
        if dividend_yield is None or dividend_yield == 0: return 3, "無配当"
        dy = dividend_yield * 100 if dividend_yield < 1 else dividend_yield
        if dy >= 4.0:   return 10, f"高配当({dy:.1f}%)"
        elif dy >= 2.5: return 8,  f"良好({dy:.1f}%)"
        elif dy >= 1.5: return 6,  f"標準({dy:.1f}%)"
        else:            return 3,  f"低め({dy:.1f}%)"

    def score_financial_health(self, debt_to_equity, current_ratio, equity_ratio=None):
        score = 5
        notes = []
        if equity_ratio is not None:
            if equity_ratio >= 50:    score += 3; notes.append(f"財務優良({equity_ratio:.0f}%)")
            elif equity_ratio >= 30:  score += 1
            else:                      score -= 1
        elif debt_to_equity is not None:
            if debt_to_equity < 30:    score += 3
            elif debt_to_equity < 100: score += 1
            else:                       score -= 2; notes.append("高負債")
        if current_ratio is not None:
            if current_ratio > 2.0: score += 2
            elif current_ratio > 1.2: score += 1
        return min(10, max(0, score)), " / ".join(notes) or "標準"

    def score_credit(self, credit_score):
        if credit_score is None: return 0, ""
        score = int(credit_score)
        if score >= 80:   return 10, f"🤖AIスコア優秀({score}点)"
        elif score >= 60: return 7,  f"🤖AIスコア良好({score}点)"
        elif score >= 40: return 4,  f"🤖AIスコア普通({score}点)"
        else:              return 1,  f"🤖AIスコア低({score}点)"

    def calculate_score(self, stock_info: dict) -> dict:
        """ファンダメンタル総合スコアを計算（0〜100点）"""
        if not stock_info:
            return {"total_score": 50, "details": {}, "raw_data": {}}

        ticker = stock_info.get("ticker", "")
        edinet_data = self.get_financial_data(ticker) if ticker else None

        def get_val(key):
            if edinet_data and edinet_data.get(key) is not None:
                return edinet_data[key]
            return stock_info.get(key)

        sector = stock_info.get("sector", "default")
        score = 0
        details = {}
        data_source = "EDINET DB" if edinet_data else "データなし"

        per_score, per_note = self.score_per(get_val("per"), sector)
        score += per_score
        details["per"] = {"score": per_score, "note": per_note}

        pbr_score, pbr_note = self.score_pbr(get_val("pbr"))
        # PBRがない場合はcurrent_price ÷ BPSで計算
        if get_val("pbr") is None and edinet_data:
            bps = edinet_data.get("bps")
            price = stock_info.get("current_price", 0)
            if bps and bps > 0 and price > 0:
                computed_pbr = round(price / bps, 2)
                pbr_score, pbr_note = self.score_pbr(computed_pbr)
                if edinet_data:
                    edinet_data["pbr"] = computed_pbr  # rawにも反映
        score += pbr_score
        details["pbr"] = {"score": pbr_score, "note": pbr_note}

        roe_score, roe_note = self.score_roe(get_val("roe"))
        score += roe_score
        details["roe"] = {"score": roe_score, "note": roe_note}

        growth_score, growth_note = self.score_growth(
            get_val("revenue_growth"), get_val("earnings_growth")
        )
        score += growth_score
        details["growth"] = {"score": growth_score, "note": growth_note}

        div_score, div_note = self.score_dividend(get_val("dividend_yield"))
        score += div_score
        details["dividend"] = {"score": div_score, "note": div_note}

        health_score, health_note = self.score_financial_health(
            get_val("debt_to_equity"),
            get_val("current_ratio"),
            edinet_data.get("equity_ratio") if edinet_data else None
        )
        score += health_score
        details["financial_health"] = {"score": health_score, "note": health_note}

        ai_comment = ""
        if edinet_data:
            credit_pts, credit_note = self.score_credit(edinet_data.get("credit_score"))
            score += credit_pts
            ai_comment = edinet_data.get("ai_comment", "")
            details["ai_score"] = {"score": credit_pts, "note": credit_note}

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
            # scoring_engineがPER/PBR/ROEを参照するためのraw_data
            "raw_data": edinet_data or {},
        }
