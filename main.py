"""
main.py - Stock AI Investor エントリーポイント

使い方:
  python main.py --mode full        # フルスキャン（全政策銘柄）
  python main.py --mode policy      # 政策連動銘柄のみ
  python main.py --mode portfolio   # ポートフォリオ評価
  python main.py --mode quick       # クイックスキャン（上位30銘柄）
  python main.py --ticker 7011.T    # 特定銘柄の分析
"""

import argparse
import yaml
import sys
import os
from pathlib import Path
from datetime import datetime
from loguru import logger

# プロジェクトルートをパスに追加
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
    """設定ファイルを読み込む"""
    if not Path(config_path).exists():
        logger.warning(f"設定ファイルが見つかりません: {config_path}")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_results(results_df, config: dict):
    """結果をCSVに保存"""
    output_dir = Path(config.get("output", {}).get("results_dir", "data/results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"screening_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    filepath = output_dir / filename
    results_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.success(f"結果を保存しました: {filepath}")
    return filepath


def print_ranking_table(results: list[dict], top_n: int = 20):
    """ランキング表をコンソールに表示"""
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
    """フルスクリーニングを実行"""
    logger.info("🚀 フルスキャン開始")

    # 政策連動銘柄を取得
    screener = PolicyScreener()
    tickers = screener.get_all_policy_tickers()

    if args.ticker:
        tickers = [args.ticker]
        logger.info(f"単銘柄モード: {args.ticker}")

    logger.info(f"対象銘柄数: {len(tickers)}")

    # データ取得
    fetcher = DataFetcher(
        history_days=config.get("data", {}).get("history_days", 180)
    )
    stocks_data = fetcher.get_multiple_stocks(tickers)

    if not stocks_data:
        logger.error("データ取得できませんでした")
        return

    # スコアリング
    engine = ScoringEngine(config)
    results = engine.evaluate_multiple(stocks_data)

    # 表示
    top_n = config.get("output", {}).get("top_n_stocks", 20)
    print_ranking_table(results, top_n)

    # 保存
    results_df = engine.to_dataframe(results)
    if config.get("output", {}).get("save_csv", True):
        save_results(results_df, config)

    # Discord通知
    webhook_url = config.get("notifications", {}).get("discord_webhook_url", "")
    if webhook_url:
        notifier = DiscordNotifier(webhook_url)
        market_overview = fetcher.get_market_overview()
        notifier.send_daily_report(results, market_overview)

    return results


def analyze_single_stock(ticker: str, config: dict):
    """単一銘柄の詳細分析"""
    logger.info(f"📊 単銘柄詳細分析: {ticker}")

    fetcher = DataFetcher()
    stocks_data = fetcher.get_multiple_stocks([ticker])

    if not stocks_data:
        logger.error(f"データ取得失敗: {ticker}")
        return

    engine = ScoringEngine(config)
    result = engine.evaluate_stock(ticker, stocks_data[ticker])

    # 詳細表示
    print("\n" + "=" * 60)
    print(f"📊 {result['name']} ({result['ticker']}) 詳細分析")
    print("=" * 60)
    print(f"総合スコア: {result['total_score']:.1f} / 100点")
    print(f"推奨アクション: {result['action_emoji']} {result['action']}")
    print(f"\n--- スコア内訳 ---")
    print(f"  テクニカル    : {result['technical_score']} 点 (重み35%)")
    print(f"  ファンダメンタル: {result['fundamental_score']} 点 (重み30%)")
    print(f"  政策スコア    : {result['policy_score']} 点 (重み35%)")
    print(f"\n--- 基本情報 ---")
    print(f"  現在株価 : ¥{result.get('current_price', 0):,.0f}")
    print(f"  時価総額 : {result.get('market_cap_B', 0):,.0f} 億円")
    if result.get("per"):
        print(f"  PER      : {result['per']:.1f} 倍")
    if result.get("pbr"):
        print(f"  PBR      : {result['pbr']:.2f} 倍")
    if result.get("roe"):
        roe = result["roe"]
        roe_pct = roe * 100 if roe < 1 else roe
        print(f"  ROE      : {roe_pct:.1f}%")
    print(f"\n--- 投資コメント ---")
    print(f"  {result.get('comment', 'なし')}")
    print(f"\n--- 政策連動セクター ---")
    sectors = result.get("policy_sectors", [])
    print(f"  {', '.join(sectors) if sectors else '該当なし'}")
    print("=" * 60)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Stock AI Investor - 日本株AI投資スクリーニングシステム"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "policy", "portfolio", "quick"],
        default="full",
        help="実行モード"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="特定銘柄コード（例: 7011.T）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="設定ファイルパス"
    )
    args = parser.parse_args()

    # ログディレクトリ作成
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    Path("data/results").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("🤖 Stock AI Investor 起動")
    logger.info(f"   モード: {args.mode}")
    logger.info(f"   時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    # 設定読み込み
    config = load_config(args.config)

    # 単銘柄分析
    if args.ticker:
        analyze_single_stock(args.ticker, config)
        return

    # モード別実行
    if args.mode in ["full", "policy", "quick"]:
        run_full_scan(config, args)
    else:
        logger.warning(f"未実装モード: {args.mode}")

    logger.info("✅ 処理完了")


if __name__ == "__main__":
    main()
