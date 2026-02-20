"""
policy_screener.py - 政策連動スクリーナー

政府予算・政策との連動度をスコア化します。
予算規模、成長率、キーワード合致度から0〜100点を算出します。

初心者メモ: 
日本政府の予算配分は毎年12月頃に概算要求が固まり、3月に成立します。
予算が増加するセクターの企業は受注・売上増が期待できるため、
このスクリーナーでは「予算配分が多く、かつ増加傾向のセクター」を高評価します。
"""

import yaml
from pathlib import Path
from loguru import logger


class PolicyScreener:
    """政策連動スクリーニングクラス"""

    def __init__(self, config_path: str = "config/policy_keywords.yaml"):
        self.config = self._load_config(config_path)
        self.sectors = self.config.get("policy_sectors", {})

    def _load_config(self, config_path: str) -> dict:
        """設定ファイル読み込み"""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"設定ファイルが見つかりません: {config_path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_sector_for_ticker(self, ticker: str) -> list[str]:
        """
        ティッカーコードが属する政策セクターを返す

        Args:
            ticker: 銘柄コード（例: "7011.T"）

        Returns:
            合致する政策セクターのリスト
        """
        matching_sectors = []
        for sector_name, sector_data in self.sectors.items():
            ticker_list = sector_data.get("ticker_list", [])
            if ticker in ticker_list:
                matching_sectors.append(sector_name)
        return matching_sectors

    def calculate_policy_score(self, ticker: str, company_description: str = "") -> dict:
        """
        政策連動スコアを計算（0〜100点）

        採点基準:
        - セクター該当 (40点): 政策予算対象セクターに属するか
        - 予算規模 (30点): 対象セクターの予算規模
        - 予算成長率 (20点): 予算の前年比増加率
        - キーワード合致 (10点): 企業説明文の政策キーワード含有数
        """
        matching_sectors = self.get_sector_for_ticker(ticker)
        score = 0
        details = {
            "matching_sectors": matching_sectors,
            "sector_scores": {},
        }

        if not matching_sectors:
            # 政策セクター非該当でもキーワードで部分点
            keyword_score = self._score_by_keywords(company_description)
            score = keyword_score * 0.5  # 最大5点

            details["note"] = "政策重点セクター外"
            return {
                "total_score": min(100, int(score)),
                "details": details,
            }

        # 最も評価の高いセクターのスコアを採用
        best_score = 0
        best_sector_detail = {}

        for sector_name in matching_sectors:
            sector_data = self.sectors.get(sector_name, {})
            s_score = 0
            s_detail = {}

            # セクター該当点 (40点)
            s_score += 40
            s_detail["sector_match"] = f"✅ {sector_name} セクター該当"

            # 予算規模点 (30点)
            budget = sector_data.get("budget_trillion_yen", 0)
            if budget >= 5.0:
                budget_score = 30
            elif budget >= 3.0:
                budget_score = 24
            elif budget >= 1.5:
                budget_score = 18
            elif budget >= 0.5:
                budget_score = 12
            else:
                budget_score = 6
            s_score += budget_score
            s_detail["budget"] = f"予算規模: {budget}兆円 ({budget_score}点)"

            # 予算成長率点 (20点)
            growth = sector_data.get("growth_rate", 0)
            if growth >= 0.3:
                growth_score = 20
            elif growth >= 0.2:
                growth_score = 16
            elif growth >= 0.1:
                growth_score = 12
            elif growth >= 0.05:
                growth_score = 8
            else:
                growth_score = 4
            s_score += growth_score
            s_detail["growth"] = f"予算成長率: {growth*100:.0f}% ({growth_score}点)"

            # キーワード合致点 (10点)
            keyword_score = self._score_by_keywords(company_description, sector_name)
            s_score += keyword_score
            s_detail["keywords"] = f"キーワード合致: {keyword_score}点"

            if s_score > best_score:
                best_score = s_score
                best_sector_detail = s_detail

        details["best_sector_detail"] = best_sector_detail
        details["policy_aligned"] = True

        return {
            "total_score": min(100, best_score),
            "details": details,
        }

    def _score_by_keywords(self, text: str, sector_name: str = None) -> int:
        """テキスト内のキーワード合致数でスコア計算（最大10点）"""
        if not text:
            return 5  # データなしは中間点

        score = 0
        matched_keywords = []

        # 指定セクターのキーワードのみチェック
        sectors_to_check = (
            [sector_name] if sector_name
            else list(self.sectors.keys())
        )

        for sname in sectors_to_check:
            keywords = self.sectors.get(sname, {}).get("keywords", [])
            for kw in keywords:
                if kw in text:
                    matched_keywords.append(kw)
                    score += 2  # 1キーワード2点

        return min(10, score)

    def get_all_policy_tickers(self) -> list[str]:
        """全政策連動銘柄のティッカーリストを返す"""
        all_tickers = []
        for sector_data in self.sectors.values():
            all_tickers.extend(sector_data.get("ticker_list", []))
        return list(set(all_tickers))  # 重複排除

    def get_sector_summary(self) -> dict:
        """セクター別サマリーを返す"""
        summary = {}
        for sector_name, sector_data in self.sectors.items():
            summary[sector_name] = {
                "budget": f"{sector_data.get('budget_trillion_yen', 0)}兆円",
                "growth_rate": f"{sector_data.get('growth_rate', 0)*100:.0f}%増",
                "ticker_count": len(sector_data.get("ticker_list", [])),
            }
        return summary
