"""demo_trades.csv の重複行を削除するワンタイムスクリプト"""
import pandas as pd
from pathlib import Path

path = Path("data/ml/demo_trades.csv")
if not path.exists():
    print("ファイルが見つかりません")
    exit()

df = pd.read_csv(path, dtype=str)
before = len(df)

# entry_date + ticker で重複削除（最初の1件を残す）
df = df.drop_duplicates(subset=["entry_date", "ticker"], keep="first")
after = len(df)

df.to_csv(path, index=False, encoding="utf-8-sig")
print(f"重複削除完了: {before}行 → {after}行 ({before-after}件削除)")
