"""
セクター分析モジュール
=====================
policy_keywords.yaml のセクター定義をもとに
各セクターの強度スコアを計算する
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger


POLICY_YAML_PATH = Path("config/policy_keywords.yaml")


def load_sector_tickers() -> dict:
    """
    policy_keywords.yaml からセクター→銘柄リストを取得
    戻り値: {"防衛": ["7011", "7012", ...], ...}
    """
    if not POLICY_YAML_PATH.exists():
        logger.warning(f"policy_keywords.yaml が見つかりません: {POLICY_YAML_PATH}")
        return {}

    with open(POLICY_YAML_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sector_map = {}
    for sector_name, sector_data in config.get("policy_sectors", {}).items():
        if not isinstance(sector_data, dict):
            continue
        tickers = [
            t.replace(".T", "")
            for t in sector_data.get("ticker_list", [])
        ]
        if tickers:
            sector_map[sector_name] = tickers

    logger.info(f"セクター数: {len(sector_map)}")
    return sector_map


def build_ticker_sector_map(sector_map: dict) -> dict:
    """
    銘柄→セクター の逆引き辞書を作成
    戻り値: {"7011": "防衛", "6501": "半導体", ...}
    """
    ticker_sector = {}
    for sector, tickers in sector_map.items():
        for t in tickers:
            ticker_sector[t] = sector
    return ticker_sector


def calc_sector_scores(
    price_dict: dict,
    sector_map: dict,
    weights: dict = None,
) -> dict:
    """
    セクター別強度スコアを計算
    
    Args:
        price_dict: {ticker: {"close": [...], "volume": [...]}}
        sector_map: {"防衛": ["7011", "7012", ...], ...}
        weights:    {"return_5d": 0.5, "vol_ratio": 0.3, "policy_hit": 0.2}
    
    戻り値:
        {
          "防衛":    {"score": 1.15, "return_5d": 0.05, "vol_ratio": 1.8},
          "半導体":  {"score": 1.02, ...},
          ...
        }
    """
    if weights is None:
        weights = {"return_5d": 0.5, "vol_ratio": 0.3, "policy_hit": 0.2}

    sector_stats = {}

    for sector, tickers in sector_map.items():
        returns   = []
        vol_ratios = []

        for ticker in tickers:
            data = price_dict.get(ticker) or price_dict.get(f"{ticker}.T")
            if data is None:
                continue

            closes  = data.get("close", [])
            volumes = data.get("volume", [])

            if len(closes) < 21:
                continue

            # 5日リターン
            ret5 = (closes[-1] / closes[-6] - 1) if closes[-6] > 0 else 0
            returns.append(ret5)

            # 出来高比率（直近1日 / 20日平均）
            if len(volumes) >= 21 and volumes[-1] > 0:
                vol_ma20 = np.mean(volumes[-21:-1])
                vol_ratio = volumes[-1] / vol_ma20 if vol_ma20 > 0 else 1.0
                vol_ratios.append(vol_ratio)

        if not returns:
            continue

        avg_return    = float(np.mean(returns))
        avg_vol_ratio = float(np.mean(vol_ratios)) if vol_ratios else 1.0

        # policy_hit: そのセクターの銘柄数（正規化）
        policy_hit = min(len(tickers) / 20.0, 1.0)

        sector_stats[sector] = {
            "return_5d":  avg_return,
            "vol_ratio":  avg_vol_ratio,
            "policy_hit": policy_hit,
            "ticker_count": len(tickers),
        }

    if not sector_stats:
        return {}

    # スコアを正規化してブースト係数（0.9〜1.2）に変換
    df = pd.DataFrame(sector_stats).T

    for col in ["return_5d", "vol_ratio", "policy_hit"]:
        if col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            if mx > mn:
                df[f"{col}_norm"] = (df[col] - mn) / (mx - mn)
            else:
                df[f"{col}_norm"] = 0.5

    df["raw_score"] = (
        df.get("return_5d_norm",  pd.Series(0.5, index=df.index)) * weights["return_5d"]  +
        df.get("vol_ratio_norm",  pd.Series(0.5, index=df.index)) * weights["vol_ratio"]  +
        df.get("policy_hit_norm", pd.Series(0.5, index=df.index)) * weights["policy_hit"]
    )

    # ブースト係数: 0.9〜1.2
    mn, mx = df["raw_score"].min(), df["raw_score"].max()
    if mx > mn:
        df["score"] = 0.9 + (df["raw_score"] - mn) / (mx - mn) * 0.3
    else:
        df["score"] = 1.0

    result = {}
    for sector in df.index:
        result[sector] = {
            "score":      round(float(df.loc[sector, "score"]), 4),
            "return_5d":  round(float(df.loc[sector, "return_5d"]), 4),
            "vol_ratio":  round(float(df.loc[sector, "vol_ratio"]), 4),
        }

    # スコア順にソート
    result = dict(sorted(result.items(), key=lambda x: x[1]["score"], reverse=True))
    logger.info(f"セクタースコア計算完了: {len(result)}セクター")
    return result


def get_top_sectors(sector_scores: dict, top_n: int = 5) -> list:
    """上位N セクター名リストを返す"""
    return list(sector_scores.keys())[:top_n]


if __name__ == "__main__":
    # テスト
    sector_map = load_sector_tickers()
    ticker_sector = build_ticker_sector_map(sector_map)
    print(f"セクター数: {len(sector_map)}")
    print(f"銘柄数: {len(ticker_sector)}")
    for s, tickers in list(sector_map.items())[:3]:
        print(f"  {s}: {tickers[:3]}...")
