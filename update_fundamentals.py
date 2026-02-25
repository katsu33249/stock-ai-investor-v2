"""
update_fundamentals.py - EDINET DB財務データ更新スクリプト

実行方法:
  python update_fundamentals.py

機能:
  - 全政策銘柄のPER/PBR/ROE等をEDINET DBから取得
  - data/cache/fundamental_cache.json に保存
  - 30日以内に更新済みの銘柄はスキップ（API節約）
  - Beta: 1,000回/日 / Free: 100回/日 対応

API消費:
  - STEP1: /v1/companies?per_page=5000 → 1回（全社EDINETコード取得）
  - STEP2: /v1/companies/{code}/ratios + /analysis → 各1回
  - 合計: 1 + 178×2 = 357回
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

# ロガー設定
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:{line} - {message}",
    level="INFO"
)

BASE_URL = "https://edinetdb.jp/v1"


def load_tickers() -> list:
    """policy_keywords.yaml から全銘柄コードを取得"""
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
    """既存キャッシュを読み込む"""
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
    """キャッシュを保存"""
    cache_path = Path("data/cache/fundamental_cache.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.success(f"キャッシュ保存完了: {cache_path}")


def safe_json(res) -> dict | list:
    """レスポンスをJSONとして安全にパース"""
    try:
        return res.json()
    except Exception:
        return {}


def normalize(raw_json) -> dict:
    """
    APIレスポンスを dict に正規化する
    - list → [0] を取る
    - dict → "data" キーを取り出す（なければそのまま）
    """
    if isinstance(raw_json, list):
        return raw_json[0] if raw_json else {}
    if isinstance(raw_json, dict):
        data = raw_json.get("data", raw_json)
        if isinstance(data, list):
            return data[0] if data else {}
        return data if isinstance(data, dict) else {}
    return {}


def get_sec_to_edinet_map(headers: dict) -> dict:
    """
    /v1/companies?per_page=5000 で全社一括取得
    → {sec_code: edinet_code} のマップを返す
    """
    logger.info("全上場企業マスタを一括取得中...")
    try:
        res = requests.get(
            f"{BASE_URL}/companies",
            headers=headers,
            params={"per_page": 5000},
            timeout=60
        )
        if res.status_code != 200:
            logger.error(f"企業マスタ取得失敗: {res.status_code} {res.text[:200]}")
            return {}

        raw = safe_json(res)
        if isinstance(raw, list):
            companies = raw
        else:
            companies = raw.get("data", [])
            if isinstance(companies, dict):
                companies = [companies]

        sec_to_edinet = {}
        for c in companies:
            if not isinstance(c, dict):
                continue
            sc = str(c.get("sec_code", "")).strip()
            ec = c.get("edinet_code", "")
            if sc and ec:
                sec_to_edinet[sc] = ec

        logger.success(f"企業マスタ取得完了: {len(sec_to_edinet)}社")
        return sec_to_edinet

    except Exception as e:
        logger.error(f"企業マスタ取得エラー: {e}")
        return {}


JQUANTS_BASE_URL = "https://api.jquants.com/v2"

def _get_bps_from_jquants(ticker: str) -> float | None:
    """J-Quants fins/statements から最新のBPS（1株純資産）を取得"""
    api_key = os.environ.get("JQUANTS_API_KEY", "")
    if not api_key:
        return None
    try:
        code = ticker.replace(".T", "")
        res = requests.get(
            f"{JQUANTS_BASE_URL}/fins/statements",
            headers={"x-api-key": api_key},
            params={"code": code},
            timeout=30
        )
        if res.status_code != 200:
            return None
        statements = res.json().get("statements", [])
        # TypeOfCurrentPeriod == "FY"（通期）の最新データを使用
        fy_records = [s for s in statements if s.get("TypeOfCurrentPeriod") == "FY"]
        if not fy_records:
            fy_records = statements  # FYがなければ全データから
        if not fy_records:
            return None
        latest = fy_records[-1]
        bps_str = latest.get("BookValuePerShare", "")
        return float(bps_str) if bps_str else None
    except Exception as e:
        logger.debug(f"BPS取得エラー({ticker}): {e}")
        return None


def fetch_fundamental(ticker: str, edinet_code: str, headers: dict) -> dict:
    """
    1銘柄の財務データを取得
    - /ratios  → PER/ROE等（EDINET DB）
    - /analysis → AIスコア・コメント（EDINET DB）
    - J-Quants fins/statements → BPS（PBR計算用）
    """
    try:
        res_r = requests.get(
            f"{BASE_URL}/companies/{edinet_code}/ratios",
            headers=headers,
            timeout=30
        )
        res_a = requests.get(
            f"{BASE_URL}/companies/{edinet_code}/analysis",
            headers=headers,
            timeout=30
        )

        # 429チェック
        if res_r.status_code == 429 or res_a.status_code == 429:
            logger.warning(f"  レート制限({ticker}) 30秒待機してリトライ...")
            time.sleep(30)
            res_r = requests.get(f"{BASE_URL}/companies/{edinet_code}/ratios",
                                 headers=headers, timeout=30)
            res_a = requests.get(f"{BASE_URL}/companies/{edinet_code}/analysis",
                                 headers=headers, timeout=30)

        result = {"updated_at": datetime.now().isoformat()}

        # ratios（PER/PBR/ROE等）
        if res_r.status_code == 200:
            raw = normalize(safe_json(res_r))
            # PBRフィールド確認（初回のみ）
            if "pbr_debug_done" not in globals():
                globals()["pbr_debug_done"] = True
                all_keys = list(raw.keys())
                logger.debug(f"ratiosフィールド一覧: {all_keys}")
            result.update({
                "per":              _safe_float(raw.get("per")),
                "pbr":              _safe_float(raw.get("pbr")),
                "roe":              _safe_float(raw.get("roe")),
                "roa":              _safe_float(raw.get("roa")),
                "profit_margin":    _safe_float(raw.get("net_margin")),
                "operating_margin": _safe_float(raw.get("operating_margin")),
                "revenue_growth":   _safe_float(raw.get("revenue_growth_rate")),
                "earnings_growth":  _safe_float(raw.get("operating_income_growth_rate")),
                "dividend_yield":   _safe_float(raw.get("dividend_yield")),
                "debt_to_equity":   _safe_float(raw.get("debt_to_equity")),
                "current_ratio":    _safe_float(raw.get("current_ratio")),
                "equity_ratio":     _safe_float(raw.get("equity_ratio")),
                "operating_cf":     _safe_float(raw.get("operating_cash_flow")),
            })
        else:
            logger.warning(f"  ratios取得失敗({ticker}): {res_r.status_code}")

        # analysis（AIスコア・コメント）
        if res_a.status_code == 200:
            raw_a = normalize(safe_json(res_a))
            result["credit_score"] = raw_a.get("credit_score")
            result["ai_comment"]   = raw_a.get("summary", "")
        else:
            logger.warning(f"  analysis取得失敗({ticker}): {res_a.status_code}")

        # BPS（J-Quants fins/statements → PBR計算用）
        bps = _get_bps_from_jquants(ticker)
        if bps and bps > 0:
            result["bps"] = bps
            logger.debug(f"  BPS取得({ticker}): {bps:.2f}")
        else:
            result["bps"] = None

        return result

    except Exception as e:
        logger.error(f"取得エラー({ticker}): {e}")
        return {}


def _safe_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def send_discord_notification(webhook_url: str, success: int, skip: int, total: int):
    """更新完了をDiscordに通知"""
    if not webhook_url:
        return
    msg = (
        f"🏦 **EDINET DB 財務データ更新完了**\n"
        f"✅ 取得成功: {success}銘柄\n"
        f"⏭️ スキップ: {skip}銘柄（30日以内更新済み）\n"
        f"⚠️ 失敗: {total - success - skip}銘柄\n"
        f"📅 更新時刻: {datetime.now().strftime('%Y/%m/%d %H:%M')}"
    )
    try:
        requests.post(webhook_url, json={"content": msg}, timeout=10)
    except Exception as e:
        logger.warning(f"Discord通知エラー: {e}")


def main():
    logger.info("🏦 月次財務データ更新開始")

    # APIキー確認
    api_key = os.environ.get("EDINET_DB_API_KEY", "")
    if not api_key:
        logger.error("EDINET_DB_API_KEY が設定されていません")
        sys.exit(1)
    headers = {"X-API-Key": api_key}

    # Discord Webhook
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

    # 銘柄リスト読み込み
    tickers = load_tickers()
    if not tickers:
        logger.error("対象銘柄がありません")
        sys.exit(1)
    logger.info(f"対象銘柄数: {len(tickers)}")

    # キャッシュ読み込み
    cache = load_cache()

    # STEP1: 全社EDINETコードマップ取得（1回）
    sec_to_edinet = get_sec_to_edinet_map(headers)
    if not sec_to_edinet:
        logger.error("EDINETコードマップ取得失敗。終了します。")
        sys.exit(1)

    # STEP2: 各銘柄の財務データ取得
    success = 0
    skip = 0

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"取得中... ({i}/{len(tickers)}): {ticker}")

        # sec_code変換（4桁→5桁）
        sec_code = ticker.replace(".T", "")
        if len(sec_code) == 4:
            sec_code = sec_code + "0"

        # EDINETコード確認
        edinet_code = sec_to_edinet.get(sec_code)
        if not edinet_code:
            logger.warning(f"  EDINETコードなし: {ticker} (sec:{sec_code})")
            skip += 1
            continue

        # 30日以内キャッシュはスキップ
        if ticker in cache:
            updated_at = cache[ticker].get("updated_at", "")
            if updated_at:
                try:
                    age_days = (datetime.now() - datetime.fromisoformat(updated_at)).days
                    if age_days < 30:
                        logger.debug(f"  キャッシュ有効({age_days}日前) スキップ: {ticker}")
                        skip += 1
                        continue
                except Exception:
                    pass

        # 財務データ取得
        data = fetch_fundamental(ticker, edinet_code, headers)
        if data:
            cache[ticker] = data
            success += 1
            per = data.get("per")
            roe = data.get("roe")
            logger.success(
                f"  ✅ {ticker} "
                f"PER:{f'{per:.1f}' if per else '-'} "
                f"ROE:{f'{roe:.1f}' if roe else '-'}"
            )
        else:
            logger.warning(f"  ❌ {ticker} データなし")

        time.sleep(1.0)

    # キャッシュ保存
    save_cache(cache)

    logger.success(
        f"更新完了: 成功 {success} / スキップ {skip} / "
        f"失敗 {len(tickers) - success - skip} / 合計 {len(tickers)}"
    )

    # Discord通知
    send_discord_notification(webhook_url, success, skip, len(tickers))


if __name__ == "__main__":
    main()
