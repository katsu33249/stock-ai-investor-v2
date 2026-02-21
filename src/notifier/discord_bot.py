"""
discord_bot.py - Discord通知モジュール（10件・コメント強化版）
"""

import requests
import json
import os
from datetime import datetime
from loguru import logger


class DiscordNotifier:

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url and webhook_url != "YOUR_DISCORD_WEBHOOK_URL")

    def send_daily_report(self, results: list[dict], market_overview: dict = None):
        """日次スクリーニングレポートを送信（上位10件）"""
        if not self.enabled:
            logger.warning("Discord通知: Webhook URLが未設定")
            return

        today = datetime.now().strftime("%Y年%m月%d日")
        embeds = []

        # ① 市場概況
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
                "title": f"📊 市場概況 — {today}",
                "color": 0x2196F3,
                "fields": market_fields,
            })

        # ② 強気買いシグナル（🔥）
        strong_buy = [r for r in results if r.get("action") == "強気買い"][:5]
        if strong_buy:
            fields = []
            for r in strong_buy:
                fields.append(self._build_stock_field(r))
            embeds.append({
                "title": "🔥 強気買いシグナル",
                "color": 0xFF5722,
                "fields": fields,
            })

        # ③ 買いシグナル（📈）
        buy_signals = [r for r in results
                      if r.get("action") in ["買い", "買い（RSI底値圏）", "買い（政策恩恵）"]][:5]
        if buy_signals:
            fields = []
            for r in buy_signals:
                fields.append(self._build_stock_field(r))
            embeds.append({
                "title": "📈 買いシグナル",
                "color": 0xFF9800,
                "fields": fields,
            })

        # ④ 総合TOP10ランキング
        top10 = results[:10]
        ranking_text = ""
        for i, r in enumerate(top10, 1):
            policy = "🏛️" if r.get("policy_sectors") else ""
            ranking_text += (
                f"`{i:2}.` {r['action_emoji']}{policy} **{r['name']}** "
                f"({r['ticker']}) — **{r['total_score']:.0f}点**\n"
                f"　　テク:{r['technical_score']} / ファン:{r['fundamental_score']} / 政策:{r['policy_score']}\n"
            )

        embeds.append({
            "title": "🏆 本日のTOP10ランキング",
            "description": ranking_text,
            "color": 0x9C27B0,
            "footer": {
                "text": "⚠️ 本ツールは参考情報です。投資は自己責任でお願いします。"
            },
            "timestamp": datetime.utcnow().isoformat(),
        })

        self._send_embeds(embeds)
        logger.success("Discord通知送信完了")

    def _build_stock_field(self, r: dict) -> dict:
        """銘柄フィールドを生成（詳細コメント付き）"""
        # 政策セクター
        policy_str = ""
        if r.get("policy_sectors"):
            sectors = r["policy_sectors"][:2]
            policy_str = f"🏛️ {' / '.join(sectors)}\n"

        # データソース
        source = r.get("data_source", "")
        source_str = f"📂 {source}\n" if source else ""

        # AI財務コメント
        ai_comment = r.get("ai_comment", "")
        ai_str = f"🤖 {ai_comment[:60]}...\n" if len(ai_comment) > 60 else (f"🤖 {ai_comment}\n" if ai_comment else "")

        # 基本指標
        per_str = f"PER:{r.get('per', '-'):.1f}" if r.get("per") else "PER:-"
        pbr_str = f"PBR:{r.get('pbr', '-'):.2f}" if r.get("pbr") else "PBR:-"
        roe = r.get("roe")
        roe_pct = roe * 100 if roe and roe < 1 else roe
        roe_str = f"ROE:{roe_pct:.1f}%" if roe_pct else "ROE:-"

        value = (
            f"**総合スコア: {r['total_score']:.0f}点** — {r['action_emoji']} {r['action']}\n"
            f"📊 テク:{r['technical_score']} 💰 ファン:{r['fundamental_score']} 🏛️ 政策:{r['policy_score']}\n"
            f"💴 株価: ¥{r.get('current_price', 0):,.0f} | {per_str} | {pbr_str} | {roe_str}\n"
            f"{policy_str}"
            f"{source_str}"
            f"{ai_str}"
            f"💡 {r.get('comment', '')}"
        )

        return {
            "name": f"{r['action_emoji']} {r['name']} ({r['ticker']})",
            "value": value[:1024],  # Discord制限
            "inline": False,
        }

    def send_error_notification(self, error_msg: str):
        if not self.enabled:
            return
        embed = {
            "title": "⚠️ システムエラー",
            "description": f"```{error_msg[:1000]}```",
            "color": 0xF44336,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._send_embeds([embed])

    def _send_embeds(self, embeds: list):
        payload = {
            "username": "Stock AI Investor 🤖",
            "embeds": embeds[:10],
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
                logger.warning(f"Discord送信ステータス: {response.status_code} / {response.text}")
        except Exception as e:
            logger.error(f"Discord送信エラー: {e}")
