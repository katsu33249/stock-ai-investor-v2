# 🚀 GitHubセットアップガイド

## 初心者向け：GitHubへのアップロード手順

---

## STEP 1: GitHubアカウントの準備

1. https://github.com にアクセスしてアカウント作成（無料）
2. 右上の「+」→「New repository」をクリック
3. Repository name: `stock-ai-investor`
4. Privateを選択（投資情報は非公開推奨）
5. 「Create repository」をクリック

---

## STEP 2: Gitのインストールと初期設定

```bash
# Gitインストール確認
git --version

# ユーザー情報設定（初回のみ）
git config --global user.name "あなたの名前"
git config --global user.email "your@email.com"
```

---

## STEP 3: プロジェクトをGitHubにアップロード

```bash
# プロジェクトフォルダに移動
cd stock-ai-investor

# Gitリポジトリ初期化
git init

# .gitignoreの作成（機密情報を除外）
cat > .gitignore << EOF
# 環境変数・機密情報
.env
config/settings_runtime.yaml
*.pem
*.key

# Python
__pycache__/
*.pyc
*.pyo
venv/
.venv/
*.egg-info/

# データ（ローカル保存のみ）
data/logs/
data/results/

# IDE
.vscode/
.idea/
*.swp
EOF

# 全ファイルをステージング
git add .

# 最初のコミット
git commit -m "🎉 Initial commit: Stock AI Investor"

# メインブランチ設定
git branch -M main

# GitHubリポジトリと接続（URLはGitHubページからコピー）
git remote add origin https://github.com/あなたのユーザー名/stock-ai-investor.git

# GitHubにプッシュ
git push -u origin main
```

---

## STEP 4: Discord Webhook の設定

### GitHub Secretsへの登録方法

1. GitHubのリポジトリページを開く
2. 「Settings」→「Secrets and variables」→「Actions」
3. 「New repository secret」をクリック
4. Name: `DISCORD_WEBHOOK_URL`
5. Secret: `Discordのwebhook URL`を貼り付け
6. 「Add secret」をクリック

### Discord Webhook URLの取得方法

1. Discordサーバー → 通知を送りたいチャンネル → ⚙️設定
2. 「連携サービス」→「ウェブフックを作成」
3. 「ウェブフックURLをコピー」

---

## STEP 5: GitHub Actionsの有効化

1. リポジトリの「Actions」タブをクリック
2. 「I understand my workflows, go ahead and enable them」
3. 左メニュー「Daily Stock Scan」→「Enable workflow」

### 手動実行テスト

1. 「Actions」→「Daily Stock Scan」
2. 「Run workflow」→「Run workflow」ボタン
3. ログを確認して正常動作を確認

---

## STEP 6: ローカルでの実行

```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# ライブラリインストール
pip install -r requirements.txt

# 設定ファイルコピー
cp config/settings.yaml config/settings.yaml
# settings.yamlを開いてDiscord URLなどを設定

# 実行
python main.py --mode full

# 特定銘柄の分析
python main.py --ticker 7011.T
```

---

## 毎日の自動化フロー

```
毎平日 9:00 JST
    ↓
GitHub Actions 起動
    ↓
株価データ自動取得
    ↓
AIスコアリング実行
    ↓
CSVに結果保存
    ↓
Discordに通知 📱
    ↓
GitHubにコミット
```

---

## よくある質問

**Q: 無料で使えますか？**
A: GitHub Actionsは月2,000分まで無料。1回のスキャンは約5分なので、毎日実行しても余裕があります。

**Q: データはどこから取得しますか？**
A: Yahoo Financeから無料で取得。リアルタイムではなく15〜20分遅延データです。

**Q: スコアはどのくらい信頼できますか？**
A: あくまで参考指標です。最終的な投資判断は必ずご自身でお願いします。
