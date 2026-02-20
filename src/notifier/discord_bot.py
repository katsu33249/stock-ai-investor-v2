"""
discord_bot.py - Discord通知モジュール

スクリーニング結果をDiscordチャンネルに自動送信します。
日次レポート、シグナルアラート、ポートフォリオ更新を通知します。

初心者メモ:
Discord Webhook URLの取得方法:
1. Discordサーバーの設定 → テキストチャンネル → 連携サービス
2. 「ウェブフックを作成」をクリック
3. 「ウェブフックURLをコピー」でURLを取得
4. config/settings.yamlの discord_webhook_url に貼り付け
"""

import requests
import json
from datetime import datetime
from loguru import logger


class DiscordNotifier:
    """Discord通知クラス"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url and webhook_url != "YOUR_DISCORD_WEBHOOK_URL")

    def send_daily_report(self, results: list[dict], market_overview: dict = None):
        """日次スクリーニングレポートを送信"""
        if not self.enabled:
            logger.warning("Discord通知: Webhook URLが未設定")
            return

        # ヘッダーEmbed
        today = datetime.now().strftime("%Y年%m月%d日")
        embeds = []

        # 市場概況
        if market_overview:
            market_fields = []
            for name, data in market_overview.items():
                change = data.get("change_pct", 0)
                arrow = "📈" if change > 0 else "📉"
                market_fields.append({
                    "name": name,
                    "value": f"{data.get('price', 0):,.0f} {arrow} {change:+.2f}%",
                    "inline": True
                })

            embeds.append({
                "title": f"📊 市場概況 - {today}",
                "color": 0x2196F3,
                "fields": market_fields,
            })

        # トップ銘柄（上位5件）
        top_stocks = results[:5]

        if top_stocks:
            fields = []
            for r in top_stocks:
                policy_tag = ""
                if r.get("policy_sectors"):
                    policy_tag = f" 🏛️{', '.join(r['policy_sectors'][:1])}"

                fields.append({
                    "name": f"{r['action_emoji']} {r['name']} ({r['ticker']}){policy_tag}",
                    "value": (
                        f"**総合スコア: {r['total_score']}点**\n"
                        f"📊テクニカル:{r['technical_score']}点 "
                        f"💰ファンダメンタル:{r['fundamental_score']}点 "
                        f"🏛️政策:{r['policy_score']}点\n"
                        f"株価: ¥{r.get('current_price', 0):,.0f} | "
                        f"PER: {r.get('per', '-'):.1f}" if r.get("per") else
                        f"株価: ¥{r.get('current_price', 0):,.0f}\n"
                        f"💡 {r.get('comment', '')}"
                    ),
                    "inline": False
                })

            embeds.append({
                "title": "🔥 本日の注目銘柄 TOP5",
                "color": 0xFF5722,
                "fields": fields,
                "footer": {"text": "※投資判断はご自身でお願いします"},
                "timestamp": datetime.utcnow().isoformat(),
            })

        self._send_embeds(embeds)
        logger.success("Discord通知送信完了")

    def send_signal_alert(self, result: dict):
        """個別シグナルアラートを送信"""
        if not self.enabled:
            return

        score = result.get("total_score", 0)
        color = 0xFF5722 if score >= 80 else 0xFF9800 if score >= 70 else 0x4CAF50

        embed = {
            "title": f"🚨 シグナルアラート: {result.get('name')} ({result.get('ticker')})",
            "color": color,
            "fields": [
                {
                    "name": "判定",
                    "value": f"{result.get('action_emoji')} **{result.get('action')}**",
                    "inline": True
                },
                {
                    "name": "総合スコア",
                    "value": f"**{score}点 / 100点**",
                    "inline": True
                },
                {
                    "name": "内訳",
                    "value": (
                        f"テクニカル: {result.get('technical_score')}点\n"
                        f"ファンダメンタル: {result.get('fundamental_score')}点\n"
                        f"政策スコア: {result.get('policy_score')}点"
                    ),
                    "inline": False
                },
                {
                    "name": "コメント",
                    "value": result.get("comment", "なし"),
                    "inline": False
                },
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Stock AI Investor | ⚠️ 投資は自己責任で"},
        }

        self._send_embeds([embed])

    def send_error_notification(self, error_msg: str):
        """エラー通知を送信"""
        if not self.enabled:
            return

        embed = {
            "title": "⚠️ システムエラー",
            "description": f"```{error_msg}```",
            "color": 0xF44336,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._send_embeds([embed])

    def _send_embeds(self, embeds: list):
        """Discord WebhookにEmbedsを送信"""
        payload = {
            "username": "Stock AI Investor 🤖",
            "avatar_url": "https://cdn.discordapp.com/embed/avatars/0.png",
            "embeds": embeds[:10],  # Discord制限: 最大10 embeds
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
