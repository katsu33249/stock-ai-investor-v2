"""Discord通知スクリプト - PHASE 3 バックテスト結果"""
import json
import os
import sys
import requests

with open("data/ml/backtest_result.json", encoding="utf-8") as f:
    data = json.load(f)

r          = data["result"]
s          = data["settings"]
run_number = sys.argv[1] if len(sys.argv) > 1 else "?"

lines = [
    "📈 **ML PHASE 3 - バックテスト完了**",
    "",
    f"**結果** {r['判定']}",
    f"・期間:         {r['period']}",
    f"・年率リターン: {float(r['annual_return']):.1%}  （目標: 15%以上）",
    f"・最大DD:       {float(r['max_drawdown']):.1%}  （目標: -20%以内）",
    f"・シャープレシオ: {r['sharpe_ratio']:.2f}  （目標: 1.0以上）",
    f"・勝率:         {float(r['win_rate']):.1%}",
    f"・PF:           {r['profit_factor']:.2f}",
    f"・総取引数:     {r['total_trades']}",
    "",
    "**設定**",
    f"・初期資金: {s['initial_capital']:,}円",
    f"・最大同時保有: {s['max_positions']}銘柄",
    f"・保有期間: {s['hold_days']}日",
    f"・売買コスト: {s['trade_cost']*100:.1f}%",
    "",
    f"・Artifact: backtest-result-{run_number}",
]

msg     = "\n".join(lines)
webhook = os.environ["DISCORD_WEBHOOK_URL"]
resp    = requests.post(webhook, json={"content": msg})
print(f"Discord通知: {resp.status_code}")
