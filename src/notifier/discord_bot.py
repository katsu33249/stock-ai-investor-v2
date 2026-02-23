"""
discord_bot.py - Discord通知モジュール（Stock AI 2.0対応版）

通知構成：
① 市場概況（日経225・TOPIX）
② 🔥 強気買いシグナル（TOP5）
③ 📈 買いシグナル（次の5件）
④ 🏆 総合TOP10ランキング
"""

import requests
import json
from datetime import datetime
from loguru import logger


class DiscordNotifier:

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url and webhook_url != "YOUR_DISCORD_WEBHOOK_URL")

    def send_daily_report(self, results: list, market_overview: dict = None):
        """日次スクリーニングレポートを送信"""
        if not self.enabled:
            logger.warning("Discord通知: Webhook URLが未設定")
            return

        today = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        messages = []

        # ===== ① 市場概況 =====
        market_lines = [f"📊 **市場概況 - {today}**"]
        if market_overview:
            for name, data in market_overview.items():
                change = data.get("change_pct", 0)
                arrow = "📈" if change > 0 else "📉"
                price = data.get("price", 0)
                market_lines.append(f"{arrow} **{name}**: {price:,.2f} ({change:+.2f}%)")
        else:
            market_lines.append("市場データ取得中...")
        messages.append("\n".join(market_lines))

        # スコア70以上を強気買い、60〜70を買いとして分類
        strong_buy = [r for r in results if r.get("total_score", 0) >= 70][:5]
        buy_signals = [r for r in results if 60 <= r.get("total_score", 0) < 70][:5]

        # ===== ② 強気買いシグナル =====
        if strong_buy:
            lines = ["📈 **買いシグナル**\n"]
            for r in strong_buy:
                lines.append(self._format_stock(r))
            messages.append("\n".join(lines))

        # ===== ③ 買いシグナル =====
        if buy_signals:
            lines = ["👀 **監視銘柄**\n"]
            for r in buy_signals:
                lines.append(self._format_stock_short(r))
            messages.append("\n".join(lines))

        # ===== ④ TOP10ランキング =====
        top10 = results[:10]
        if top10:
            lines = ["🏆 **本日のTOP10ランキング**"]
            for i, r in enumerate(top10, 1):
                name = r.get("name", r["ticker"])
                # 銘柄名がtickerと同じ場合はtickerのみ表示
                if name == r["ticker"]:
                    display = f"{r['ticker']}"
                else:
                    display = f"{name} ({r['ticker']})"
                policy = "🏛️🏛️" if r.get("policy_score", 0) >= 80 else ("🏛️" if r.get("policy_score", 0) >= 60 else "👁️")
                lines.append(
                    f"{i}．{policy} **{display}** — **{r['total_score']}点**\n"
                    f"　テク:{r['technical_score']} / ファン:{r['fundamental_score']} / 政策:{r['policy_score']}"
                )
            messages.append("\n".join(lines))

        # メッセージ送信
        for msg in messages:
            self._send_message(msg)

        logger.success("Discord通知送信完了")

    def _format_stock(self, r: dict) -> str:
        """銘柄の詳細フォーマット（強気買い用）"""
        name = r.get("name", r["ticker"])
        if name == r["ticker"]:
            display = r["ticker"]
        else:
            display = f"{name} ({r['ticker']})"

        # PER/PBR/ROE
        per = r.get("per")
        pbr = r.get("pbr")
        roe = r.get("roe")
        per_str = f"{per:.1f}倍" if per else "-"
        pbr_str = f"{pbr:.2f}倍" if pbr else "-"
        roe_val = (roe * 100 if roe and abs(roe) < 1 else roe) if roe else None
        roe_str = f"{roe_val:.1f}%" if roe_val else "-"

        # 政策セクター
        sectors = r.get("policy_sectors", [])
        sector_str = ", ".join(sectors[:2]) if sectors else "-"

        # 信用倍率
        margin = r.get("margin_ratio")
        margin_str = f"{margin:.1f}倍" if margin else "-"

        # AIコメント
        ai_comment = r.get("ai_comment", "")
        ai_line = f"🤖 {ai_comment[:60]}..." if len(ai_comment) > 60 else (f"🤖 {ai_comment}" if ai_comment else "")

        lines = [
            f"🏛️ **{display}**",
            f"**総合スコア: {r['total_score']}点** — {r.get('action_emoji','')} {r.get('action','')}",
            f"📊 テク:{r['technical_score']} 💰 ファン:{r['fundamental_score']} 🏛️ 政策:{r['policy_score']}",
            f"📈 株価: ¥{r.get('current_price', 0):,.0f} | PER:{per_str} | PBR:{pbr_str} | ROE:{roe_str}",
            f"🏛️ {sector_str}",
            f"📁 {r.get('data_source', 'データなし')} | 信用倍率:{margin_str}",
        ]
        if ai_line:
            lines.append(ai_line)
        lines.append(f"💡 {r.get('comment', '')}")
        lines.append("")  # 区切り
        return "\n".join(lines)

    def _format_stock_short(self, r: dict) -> str:
        """銘柄の短縮フォーマット（監視銘柄用）"""
        name = r.get("name", r["ticker"])
        if name == r["ticker"]:
            display = r["ticker"]
        else:
            display = f"{name} ({r['ticker']})"

        per = r.get("per")
        per_str = f"PER:{per:.1f}" if per else "PER:-"
        sectors = r.get("policy_sectors", [])
        sector_str = sectors[0] if sectors else "-"

        return (
            f"👁️ **{display}** {r['total_score']}点 "
            f"| テク:{r['technical_score']} 政策:{r['policy_score']} "
            f"| {per_str} | {sector_str}\n"
        )

    def _send_message(self, content: str):
        """Discord Webhookにメッセージを送信"""
        # 2000文字制限対応
        if len(content) > 1900:
            content = content[:1900] + "..."

        payload = {
            "username": "Stock AI Investor 🤖",
            "content": content,
        }
        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 204:
                logger.debug("Discord送信成功")
            else:
                logger.warning(f"Discord送信ステータス: {response.status_code}")
        except Exception as e:
            logger.error(f"Discord送信エラー: {e}")
