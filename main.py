"""
main.py - Stock AI Investor エントリーポイント

使い方:
  python main.py --mode full                 # フルスキャン（全政策銘柄）
  python main.py --mode policy               # 政策連動銘柄のみ
  python main.py --mode quick                # クイックスキャン（上位30銘柄）
  python main.py --mode update_fundamentals  # EDINET DB財務データ更新
  python main.py --ticker 7011.T             # 特定銘柄の分析
"""

import argparse
import yaml
import sys
import os
import json
import requests
import time
from pathlib import Path
from datetime import datetime
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.data_fetcher import DataFetcher
from src.analyzer.scoring_engine import ScoringEngine
from src.screener.policy_screener import PolicyScreener
from src.notifier.discord_bot import DiscordNotifier


# ============================================================
# ロガー設定
# ============================================================
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO"
)
logger.add(
    "data/logs/run_{time:YYYYMMDD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    rotation="1 day",
    retention="30 days",
    level="DEBUG"
)


def load_config(config_path: str = "config/settings.yaml") -> dict:
    if not Path(config_path).exists():
        logger.warning(f"設定ファイルが見つかりません: {config_path}")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_results(results_df, config: dict):
    output_dir = Path(config.get("output", {}).get("results_dir", "data/results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"screening_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    filepath = output_dir / filename
    results_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.success(f"結果を保存しました: {filepath}")
    return filepath


def print_ranking_table(results: list, top_n: int = 20):
    print("\n" + "=" * 80)
    print(f"{'🏆 株式スクリーニング結果':^78}")
    print(f"{'実行時刻: ' + datetime.now().strftime('%Y-%m-%d %H:%M'):^78}")
    print("=" * 80)
    print(f"{'順位':<4} {'銘柄名':<20} {'コード':<10} {'総合':<6} {'テク':<6} {'ファン':<6} {'政策':<6} {'判定'}")
    print("-" * 80)
    for r in results[:top_n]:
        policy_flag = "🏛️" if r.get("policy_sectors") else "  "
        print(
            f"{r.get('rank', '-'):<4} "
            f"{r['name'][:18]:<20} "
            f"{r['ticker']:<10} "
            f"{r['total_score']:<6.1f} "
            f"{r['technical_score']:<6} "
            f"{r['fundamental_score']:<6} "
            f"{r['policy_score']:<6} "
            f"{policy_flag}{r['action_emoji']} {r['action']}"
        )
    print("=" * 80)
    print(f"合計 {len(results)} 銘柄を評価 | 表示 上位 {min(top_n, len(results))} 銘柄")
    print("⚠️  本ツールは投資判断の参考情報です。投資は自己責任でお願いします。")
    print("=" * 80 + "\n")


def run_full_scan(config: dict, args):
    logger.info("🚀 フルスキャン開始")
    screener = PolicyScreener()
    tickers = screener.get_all_policy_tickers()
    if args.ticker:
        tickers = [args.ticker]
        logger.info(f"単銘柄モード: {args.ticker}")
    logger.info(f"対象銘柄数: {len(tickers)}")

    fetcher = DataFetcher(history_days=config.get("data", {}).get("history_days", 180))
    stocks_data = fetcher.get_multiple_stocks(tickers)

    if not stocks_data:
        logger.error("データ取得できませんでした")
        return

    engine = ScoringEngine(config)
    results = engine.evaluate_multiple(stocks_data)

    top_n = config.get("output", {}).get("top_n_stocks", 20)
    print_ranking_table(results, top_n)

    results_df = engine.to_dataframe(results)
    if config.get("output", {}).get("save_csv", True):
        save_results(results_df, config)

    webhook_url = config.get("notifications", {}).get("discord_webhook_url", "")
    if webhook_url:
        notifier = DiscordNotifier(webhook_url)
        market_overview = fetcher.get_market_overview()
        notifier.send_daily_report(results, market_overview)

    return results


def run_update_fundamentals(config: dict):
    """
    EDINET DB財務データを全銘柄分取得してキャッシュに保存

    ・新規銘柄追加後すぐに実行するとPER/PBR/ROEが反映される
    ・毎月1日の自動実行でも呼ばれる
    """
    logger.info("🏦 EDINET DB財務データ更新開始")

    api_key = os.environ.get("EDINET_DB_API_KEY", "")
    if not api_key:
        logger.error("EDINET_DB_API_KEY が設定されていません")
        return

    screener = PolicyScreener()
    tickers = screener.get_all_policy_tickers()
    logger.info(f"対象銘柄数: {len(tickers)}")

    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "fundamental_cache.json"

    # 既存キャッシュ読み込み
    cache = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"既存キャッシュ読み込み: {len(cache)} 銘柄")

    BASE_URL = "https://edinetdb.jp/v1"
    headers = {"X-API-Key": api_key}
    success = 0
    skip = 0

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"EDINET取得中... ({i}/{len(tickers)}): {ticker}")
        sec_code = ticker.replace(".T", "")
        if len(sec_code) == 4:
            sec_code = sec_code + "0"

        try:
            # 企業検索
            res = requests.get(
                f"{BASE_URL}/search",
                params={"q": sec_code},
                headers=headers,
                timeout=30
            )
            if res.status_code != 200:
                logger.warning(f"  検索失敗({ticker}): {res.status_code}")
                skip += 1
                continue

            companies = res.json().get("data", [])
            if not companies:
                logger.warning(f"  企業データなし: {ticker}")
                skip += 1
                continue

            edinet_code = None
            for c in companies:
                if str(c.get("sec_code", "")) == sec_code:
                    edinet_code = c.get("edinet_code")
                    break
            if not edinet_code:
                edinet_code = companies[0].get("edinet_code")

            # 財務指標取得
            res_r = requests.get(
                f"{BASE_URL}/companies/{edinet_code}/ratios",
                headers=headers, timeout=30
            )
            # AI分析取得
            res_a = requests.get(
                f"{BASE_URL}/companies/{edinet_code}/analysis",
                headers=headers, timeout=30
            )

            data = {"updated_at": datetime.now().isoformat()}

            if res_r.status_code == 200:
                raw = res_r.json().get("data", {})
                if isinstance(raw, list):
                    raw = raw[0] if raw else {}
                data.update({
                    "per": raw.get("per"),
                    "pbr": raw.get("pbr"),
                    "roe": raw.get("roe"),
                    "roa": raw.get("roa"),
                    "profit_margin": raw.get("net_margin"),
                    "operating_margin": raw.get("operating_margin"),
                    "revenue_growth": raw.get("revenue_growth_rate"),
                    "earnings_growth": raw.get("operating_income_growth_rate"),
                    "dividend_yield": raw.get("dividend_yield"),
                    "debt_to_equity": raw.get("debt_to_equity"),
                    "current_ratio": raw.get("current_ratio"),
                    "equity_ratio": raw.get("equity_ratio"),
                    "operating_cf": raw.get("operating_cash_flow"),
                })

            if res_a.status_code == 200:
                raw_a = res_a.json().get("data", {})
                if isinstance(raw_a, list):
                    raw_a = raw_a[0] if raw_a else {}
                data["credit_score"] = raw_a.get("credit_score")
                data["ai_comment"] = raw_a.get("summary", "")

            cache[ticker] = data
            success += 1
            logger.success(f"  ✅ {ticker} 取得完了")
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"  エラー({ticker}): {e}")
            skip += 1

    # キャッシュ保存
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    logger.success(f"EDINET更新完了: 成功 {success} / スキップ {skip} / 合計 {len(tickers)}")
    logger.success(f"キャッシュ保存: {cache_path}")

    # Discord通知
    webhook_url = config.get("notifications", {}).get("discord_webhook_url", "")
    if webhook_url:
        msg = (
            f"🏦 **EDINET DB 財務データ更新完了**\n"
            f"✅ 取得成功: {success}銘柄\n"
            f"⚠️ スキップ: {skip}銘柄\n"
            f"📅 更新時刻: {datetime.now().strftime('%Y/%m/%d %H:%M')}"
        )
        requests.post(webhook_url, json={"content": msg})


def analyze_single_stock(ticker: str, config: dict):
    logger.info(f"📊 単銘柄詳細分析: {ticker}")
    fetcher = DataFetcher()
    stocks_data = fetcher.get_multiple_stocks([ticker])
    if not stocks_data:
        logger.error(f"データ取得失敗: {ticker}")
        return
    engine = ScoringEngine(config)
    result = engine.evaluate_stock(ticker, stocks_data[ticker])
    print(f"\n{'='*60}")
    print(f"📊 {result['name']} ({result['ticker']}) 詳細分析")
    print(f"{'='*60}")
    print(f"総合スコア: {result['total_score']:.1f} / 100点")
    print(f"推奨: {result['action_emoji']} {result['action']}")
    print(f"\nスコア内訳")
    print(f"  テクニカル    : {result['technical_score']}点 (40%)")
    print(f"  ファンダメンタル: {result['fundamental_score']}点 (35%)")
    print(f"  政策スコア    : {result['policy_score']}点 (25%)")
    if result.get("per"):  print(f"  PER: {result['per']:.1f}倍")
    if result.get("pbr"):  print(f"  PBR: {result['pbr']:.2f}倍")
    if result.get("roe"):
        roe = result["roe"]
        print(f"  ROE: {roe*100 if roe < 1 else roe:.1f}%")
    print(f"\nコメント: {result.get('comment', 'なし')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Stock AI Investor 2.0")
    parser.add_argument(
        "--mode",
        choices=["full", "policy", "portfolio", "quick", "update_fundamentals"],
        default="full",
        help="実行モード"
    )
    parser.add_argument("--ticker", type=str, help="特定銘柄コード（例: 7011.T）")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    args = parser.parse_args()

    Path("data/logs").mkdir(parents=True, exist_ok=True)
    Path("data/results").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("🤖 Stock AI Investor 起動")
    logger.info(f"   モード: {args.mode}")
    logger.info(f"   時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    config = load_config(args.config)

    if args.ticker:
        analyze_single_stock(args.ticker, config)
        return

    if args.mode == "update_fundamentals":
        run_update_fundamentals(config)
    elif args.mode in ["full", "policy", "quick"]:
        run_full_scan(config, args)
    else:
        logger.warning(f"未実装モード: {args.mode}")

    logger.info("✅ 処理完了")


if __name__ == "__main__":
    main()
