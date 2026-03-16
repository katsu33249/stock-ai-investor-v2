"""
update_fundamentals.py - EDINET DB財務データ更新スクリプト（ランキング一括取得版）

旧方式: 178銘柄 × 2コール = 356回/月
新方式: ランキングAPI × 7回 + 企業マスタ1回 = 8回/月（98%削減）
        銘柄数が増えてもAPIコールは変わらない
"""

import requests
import json
import os
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:{line} - {message}",
    level="INFO"
)

BASE_URL = "https://edinetdb.jp/v1"

RANKING_METRICS = [
    ("per",              "per"),
    ("roe",              "roe"),
    ("roa",              "roa"),
    ("operating-margin", "operating_margin"),
    ("equity-ratio",     "equity_ratio"),
    ("dividend-yield",   "dividend_yield"),
    ("net-margin",       "profit_margin"),
]


def load_tickers() -> list:
    yaml_path = Path("config/policy_keywords.yaml")
    if not yaml_path.exists():
        logger.error("config/policy_keywords.yaml が見つかりません")
        return []
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tickers = set()
    for sector_data in config.get("policy_sectors", {}).values():
        for t in sector_data.get("ticker_list", []):
            tickers.add(t)
    return sorted(tickers)


def load_cache() -> dict:
    cache_path = Path("data/cache/fundamental_cache.json")
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info(f"既存キャッシュ読み込み: {len(cache)}銘柄")
            return cache
        except Exception as e:
            logger.warning(f"キャッシュ読み込みエラー: {e}")
    return {}


def save_cache(cache: dict):
    cache_path = Path("data/cache/fundamental_cache.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.success(f"キャッシュ保存完了: {cache_path}")


def _safe_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def ticker_to_sec_code(ticker: str) -> str:
    code = ticker.replace(".T", "")
    return code + "0" if len(code) == 4 else code


def fetch_all_companies(headers: dict) -> dict:
    """全社マスタ一括取得（1回）→ {sec_code: {edinet_code, credit_score, ...}}"""
    logger.info("全社マスタを一括取得中... (1APIコール)")
    try:
        res = requests.get(
            f"{BASE_URL}/companies",
            headers=headers,
            params={"per_page": 5000},
            timeout=60
        )
        if res.status_code != 200:
            logger.error(f"全社マスタ取得失敗: {res.status_code}")
            return {}
        raw = res.json()
        companies = raw if isinstance(raw, list) else raw.get("data", [])
        result = {}
        for c in companies:
            if not isinstance(c, dict):
                continue
            sec = str(c.get("sec_code", "")).strip()
            edinet = c.get("edinet_code", "")
            if sec and edinet:
                result[sec] = {
                    "edinet_code":   edinet,
                    "credit_score":  _safe_float(c.get("credit_score")),
                    "credit_rating": c.get("credit_rating", ""),
                    "name":          c.get("name", ""),
                }
        logger.success(f"全社マスタ取得完了: {len(result)}社")
        return result
    except Exception as e:
        logger.error(f"全社マスタ取得エラー: {e}")
        return {}


def fetch_ranking(metric: str, headers: dict, limit: int = 5000) -> dict:
    """ランキングAPI一括取得（1回）→ {sec_code: value}"""
    try:
        res = requests.get(
            f"{BASE_URL}/rankings/{metric}",
            headers=headers,
            params={"limit": limit},
            timeout=60
        )
        if res.status_code == 429:
            logger.warning(f"レート制限({metric}) 30秒待機...")
            time.sleep(30)
            res = requests.get(
                f"{BASE_URL}/rankings/{metric}",
                headers=headers,
                params={"limit": limit},
                timeout=60
            )
        if res.status_code != 200:
            logger.warning(f"ランキング取得失敗({metric}): {res.status_code}")
            return {}
        raw = res.json()
        items = raw if isinstance(raw, list) else raw.get("data", [])
        result = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            sec = str(item.get("sec_code", "")).strip()
            val = _safe_float(item.get("value"))
            if sec:
                result[sec] = val
        logger.info(f"  ランキング取得: {metric} → {len(result)}社")
        return result
    except Exception as e:
        logger.warning(f"ランキング取得エラー({metric}): {e}")
        return {}


def send_discord_notification(webhook_url: str, success: int, total: int, api_calls: int):
    if not webhook_url:
        return
    msg = (
        f"🏦 **EDINET DB 財務データ更新完了**\n"
        f"・更新銘柄: {success}/{total}\n"
        f"・APIコール数: {api_calls}回（旧方式: {total*2+1}回）\n"
        f"・更新日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    try:
        requests.post(webhook_url, json={"content": msg}, timeout=10)
    except Exception as e:
        logger.warning(f"Discord通知エラー: {e}")


def main():
    logger.info("🏦 月次財務データ更新開始（ランキング一括取得版）")

    api_key = os.environ.get("EDINET_DB_API_KEY", "")
    if not api_key:
        logger.error("EDINET_DB_API_KEY が設定されていません")
        sys.exit(1)
    headers  = {"X-API-Key": api_key}
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

    tickers = load_tickers()
    if not tickers:
        logger.error("対象銘柄がありません")
        sys.exit(1)
    logger.info(f"対象銘柄数: {len(tickers)}")

    cache = load_cache()

    # STEP 1: 全社マスタ一括取得（1回）
    companies = fetch_all_companies(headers)
    if not companies:
        logger.error("全社マスタ取得失敗。終了します。")
        sys.exit(1)
    api_calls = 1

    # STEP 2: ランキングAPIで各指標一括取得（7回）
    logger.info(f"ランキングAPI一括取得開始（{len(RANKING_METRICS)}指標）")
    rankings = {}
    for metric_api, metric_key in RANKING_METRICS:
        rankings[metric_key] = fetch_ranking(metric_api, headers)
        api_calls += 1
        time.sleep(0.5)

    logger.success(f"全指標取得完了。合計APIコール: {api_calls}回")

    # STEP 3: 対象銘柄のデータを組み立て
    success = 0
    skip    = 0
    updated_at = datetime.now().isoformat()

    for ticker in tickers:
        sec_code = ticker_to_sec_code(ticker)
        company  = companies.get(sec_code, {})

        if not company:
            logger.warning(f"マスタなし: {ticker} (sec:{sec_code})")
            skip += 1
            continue

        data = {
            "updated_at":       updated_at,
            "credit_score":     company.get("credit_score"),
            "credit_rating":    company.get("credit_rating", ""),
            "ai_comment":       "",
            "revenue_growth":   None,
            "earnings_growth":  None,
            "debt_to_equity":   None,
            "current_ratio":    None,
            "operating_cf":     None,
            "pbr":              None,
        }
        for _, metric_key in RANKING_METRICS:
            data[metric_key] = rankings.get(metric_key, {}).get(sec_code)

        cache[ticker] = data
        success += 1

        per = data.get("per")
        roe = data.get("roe")
        logger.info(
            f"  ✅ {ticker} "
            f"PER:{f'{per:.1f}' if per else '-'} "
            f"ROE:{f'{roe*100:.1f}%' if roe else '-'} "
            f"credit:{data.get('credit_score') or '-'}"
        )

    save_cache(cache)

    logger.success(
        f"更新完了: 成功{success} / スキップ{skip} / 合計{len(tickers)} "
        f"| APIコール: {api_calls}回（旧方式: {len(tickers)*2+1}回）"
    )
    send_discord_notification(webhook_url, success, len(tickers), api_calls)


if __name__ == "__main__":
    main()
