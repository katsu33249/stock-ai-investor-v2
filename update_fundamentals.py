"""
update_fundamentals.py - 月次財務データ更新スクリプト

毎月1日に実行。EDINET DBから全対象銘柄の財務データを取得し
data/cache/fundamental_cache.json に保存します。

次回以降の日次スクリーニングはキャッシュを使用するため
APIコールを節約できます。
"""

import json
import os
import time
import requests
from pathlib import Path
from datetime import datetime
from loguru import logger


BASE_URL = "https://edinetdb.jp/v1"
CACHE_PATH = Path("data/cache/fundamental_cache.json")
CONFIG_PATH = Path("config/policy_keywords.yaml")


def get_all_tickers() -> list:
    """policy_keywords.yaml から全銘柄を取得"""
    import yaml
    if not CONFIG_PATH.exists():
        return []
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tickers = []
    for sector_data in config.get("policy_sectors", {}).values():
        tickers.extend(sector_data.get("ticker_list", []))
    return list(set(tickers))


def fetch_fundamental(ticker: str, api_key: str) -> dict:
    """EDINET DBから1銘柄の財務データを取得"""
    headers = {"X-API-Key": api_key}
    sec_code = ticker.replace(".T", "") + "0"

    try:
        # 企業検索
        res = requests.get(
            f"{BASE_URL}/search",
            params={"q": sec_code},
            headers=headers,
            timeout=10
        )
        if res.status_code != 200:
            return {}

        companies = res.json().get("data", [])
        if not companies:
            return {}

        edinet_code = None
        for c in companies:
            if str(c.get("sec_code", "")) == sec_code:
                edinet_code = c.get("edinet_code")
                break
        if not edinet_code:
            edinet_code = companies[0].get("edinet_code")

        # 財務指標
        res_ratios = requests.get(
            f"{BASE_URL}/companies/{edinet_code}/ratios",
            headers=headers, timeout=10
        )
        # AI財務分析
        res_analysis = requests.get(
            f"{BASE_URL}/companies/{edinet_code}/analysis",
            headers=headers, timeout=10
        )

        result = {}
        if res_ratios.status_code == 200:
            data = res_ratios.json().get("data", {})
            result.update({
                "per": data.get("per"),
                "pbr": data.get("pbr"),
                "roe": data.get("roe"),
                "roa": data.get("roa"),
                "profit_margin": data.get("net_margin"),
                "operating_margin": data.get("operating_margin"),
                "revenue_growth": data.get("revenue_growth_rate"),
                "earnings_growth": data.get("operating_income_growth_rate"),
                "dividend_yield": data.get("dividend_yield"),
                "debt_to_equity": data.get("debt_to_equity"),
                "current_ratio": data.get("current_ratio"),
                "equity_ratio": data.get("equity_ratio"),
            })

        if res_analysis.status_code == 200:
            analysis = res_analysis.json().get("data", {})
            result.update({
                "credit_score": analysis.get("credit_score"),
                "ai_comment": analysis.get("summary", ""),
            })

        return result

    except Exception as e:
        logger.error(f"取得エラー({ticker}): {e}")
        return {}


def send_discord_notification(webhook_url: str, success: int, total: int):
    """更新完了をDiscordに通知"""
    if not webhook_url:
        return
    payload = {
        "username": "Stock AI Investor 🤖",
        "embeds": [{
            "title": "🏦 月次財務データ更新完了",
            "color": 0x4CAF50,
            "fields": [
                {"name": "更新銘柄数", "value": f"{success} / {total} 銘柄", "inline": True},
                {"name": "次回更新", "value": "来月1日 夜9時", "inline": True},
                {"name": "データソース", "value": "EDINET DB（JPX公式）", "inline": False},
            ],
            "footer": {"text": "財務データは四半期決算まで有効です"},
            "timestamp": datetime.utcnow().isoformat(),
        }]
    }
    try:
        requests.post(webhook_url, json=payload, timeout=10)
        logger.success("Discord通知送信完了")
    except Exception as e:
        logger.warning(f"Discord通知エラー: {e}")


def main():
    api_key = os.environ.get("EDINET_DB_API_KEY", "")
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

    if not api_key:
        logger.error("EDINET_DB_API_KEY が設定されていません")
        return

    # キャッシュディレクトリ作成
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 既存キャッシュ読み込み
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)

    tickers = get_all_tickers()
    total = len(tickers)
    success = 0

    logger.info(f"🏦 月次財務データ更新開始: {total} 銘柄")

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"取得中... ({i}/{total}): {ticker}")
        data = fetch_fundamental(ticker, api_key)
        if data:
            cache[ticker] = {
                **data,
                "updated_at": datetime.now().isoformat()
            }
            success += 1
        time.sleep(0.8)  # APIレート制限対応

    # キャッシュ保存
    cache["_meta"] = {
        "last_updated": datetime.now().isoformat(),
        "total_tickers": total,
        "success_count": success,
    }
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    logger.success(f"✅ 更新完了: {success}/{total} 銘柄")
    logger.info(f"💾 キャッシュ保存: {CACHE_PATH}")

    send_discord_notification(webhook_url, success, total)


if __name__ == "__main__":
    main()
