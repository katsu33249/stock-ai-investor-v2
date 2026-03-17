name: 🤖 ML PHASE 1 - 学習データ収集

permissions:
  contents: write

on:
  workflow_dispatch:  # 手動実行のみ

jobs:
  collect-training-data:
    name: 学習データ収集（178銘柄 × 10年）
    runs-on: ubuntu-latest
    timeout-minutes: 120  # 最大2時間

    steps:
      - name: 📥 リポジトリ取得
        uses: actions/checkout@v4

      - name: 🐍 Python 3.11 セットアップ
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
          cache: "pip"

      - name: 📦 ライブラリインストール
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: 📁 出力ディレクトリ作成
        run: |
          mkdir -p data/ml
          mkdir -p data/logs
          mkdir -p data/cache

      - name: 🚀 学習データ収集実行
        env:
          JQUANTS_API_KEY: ${{ secrets.JQUANTS_API_KEY }}
          EDINET_DB_API_KEY: ${{ secrets.EDINET_DB_API_KEY }}
        run: |
          python ml/collect_training_data.py

      - name: 📊 収集結果サマリー
        run: |
          if [ -f data/ml/feature_info.json ]; then
            echo "=== 収集結果 ==="
            cat data/ml/feature_info.json
          else
            echo "feature_info.json が見つかりません"
            exit 1
          fi

      - name: 💾 学習データをアーティファクトとして保存
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: training-data-${{ github.run_number }}
          path: |
            data/ml/training_data.parquet
            data/ml/training_data.csv
            data/ml/feature_info.json
          retention-days: 90

      - name: 📝 メタ情報のみリポジトリにコミット（CSVは大きすぎるためArtifactのみ）
        uses: stefanzweifel/git-auto-commit-action@v5
        if: success()
        with:
          commit_message: "🤖 ML学習データ収集完了 - Run#${{ github.run_number }}"
          file_pattern: "data/ml/feature_info.json"
          commit_user_name: "Stock AI Bot"
          commit_user_email: "bot@github.actions"

      - name: ✅ Discord通知（完了）
        if: success()
        run: |
          RECORDS=$(cat data/ml/feature_info.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['total_records']:,}\")")
          POSITIVE=$(cat data/ml/feature_info.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['positive_rate']:.1%}\")")
          curl -H "Content-Type: application/json" \
            -d "{\"content\": \"🤖 **ML PHASE 1 完了**\n📊 総レコード数: ${RECORDS}\n📈 正例率（上昇）: ${POSITIVE}\n➡️ 次: PHASE 2 モデル学習\"}" \
            ${{ secrets.DISCORD_WEBHOOK_URL }}

      - name: ❌ Discord通知（失敗）
        if: failure()
        run: |
          curl -H "Content-Type: application/json" \
            -d '{"content": "⚠️ **ML PHASE 1 失敗**\nActionsログを確認してください。"}' \
            ${{ secrets.DISCORD_WEBHOOK_URL }}
