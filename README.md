# 🤖 Stock AI Investor - 日本株AI投資システム

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-orange.svg)](.github/workflows)

> **政策連動型・AI駆動の日本株スクリーニング & ポートフォリオ管理システム**

---

## 🎯 システム概要

このシステムは、日本の政府予算・政策方針と株式市場を連動させた**AI駆動の投資判断支援ツール**です。
防衛・半導体・GX（グリーントランスフォーメーション）など注目セクターの銘柄を自動スクリーニングし、
テクニカル・ファンダメンタル・政策スコアを統合した総合評価を提供します。

```
┌─────────────────────────────────────────────────────────┐
│                    Stock AI Investor                      │
├──────────────┬──────────────┬──────────────┬────────────┤
│  📊 Screener │  🔍 Analyzer │  💼 Portfolio│  🔔 Alert  │
│  銘柄抽出    │  スコア分析  │  管理・最適化 │  Discord通知│
└──────────────┴──────────────┴──────────────┴────────────┘
```

---

## 🏗️ ディレクトリ構成

```
stock-ai-investor/
├── 📁 src/
│   ├── screener/          # 銘柄スクリーニング
│   │   ├── policy_screener.py    # 政策連動スクリーナー
│   │   └── technical_screener.py # テクニカルスクリーナー
│   ├── analyzer/          # スコア分析エンジン
│   │   ├── scoring_engine.py     # 総合スコアリング
│   │   ├── fundamental.py        # ファンダメンタル分析
│   │   └── technical.py          # テクニカル分析
│   ├── portfolio/         # ポートフォリオ管理
│   │   ├── optimizer.py          # 最適化エンジン
│   │   └── tracker.py            # 保有管理
│   ├── notifier/          # 通知システム
│   │   └── discord_bot.py        # Discord通知
│   └── utils/             # ユーティリティ
│       ├── data_fetcher.py       # データ取得
│       └── logger.py             # ログ管理
├── 📁 config/
│   ├── settings.yaml             # 設定ファイル
│   └── policy_keywords.yaml     # 政策キーワード
├── 📁 data/
│   └── portfolio.csv             # ポートフォリオデータ
├── 📁 .github/workflows/
│   └── daily_scan.yml            # 自動スキャンCI/CD
├── main.py                       # エントリーポイント
└── requirements.txt              # 依存ライブラリ
```

---

## ⚡ クイックスタート

### 1. リポジトリのクローン
```bash
git clone https://github.com/yourusername/stock-ai-investor.git
cd stock-ai-investor
```

### 2. 仮想環境の作成
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3. 依存ライブラリのインストール
```bash
pip install -r requirements.txt
```

### 4. 設定ファイルの編集
```bash
cp config/settings.yaml.example config/settings.yaml
# settings.yamlにDiscord Webhook URLなどを設定
```

### 5. 実行
```bash
# フルスキャン実行
python main.py --mode full

# 政策銘柄のみスキャン
python main.py --mode policy

# ポートフォリオ評価
python main.py --mode portfolio
```

---

## 📊 スコアリング方式

| カテゴリ | ウェイト | 評価項目 |
|---------|---------|---------|
| 🏛️ 政策スコア | 35% | 予算配分、補助金対象、政策キーワード合致度 |
| 📈 テクニカル | 35% | RSI、MACD、移動平均、出来高トレンド |
| 💰 ファンダメンタル | 30% | PER、PBR、ROE、売上成長率、利益率 |

**総合スコア = 政策(35%) + テクニカル(35%) + ファンダメンタル(30%)**

---

## 🎯 注目セクター（2025-2026年度）

| セクター | 予算規模 | 主要銘柄例 |
|---------|---------|-----------|
| 🛡️ 防衛・安全保障 | 約8兆円 | 三菱重工、川崎重工、IHI |
| 💻 半導体・AI | 約4兆円 | ソシオネクスト、東京エレクトロン |
| 🌱 GX・再エネ | 約2兆円 | レノバ、日本風力開発 |
| 🏥 医療・DX | 約1兆円 | フィリップス、PHC |

---

## 🤖 GitHub Actions 自動化

毎日**平日9:00（JST）**に自動スキャンが実行されます：
1. 最新の株価データ取得
2. 政策連動スクリーニング実行
3. スコアリング＆ランキング更新
4. Discord/Slackへ上位銘柄通知
5. `data/results/` へ結果CSV自動保存

---

## ⚠️ 免責事項

本ツールは**投資判断の参考情報**を提供するものです。
実際の投資判断はご自身の責任で行ってください。
