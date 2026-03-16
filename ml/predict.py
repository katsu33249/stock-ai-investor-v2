"""
PHASE 4: 日次MLシグナル予測スクリプト
=======================================
目的: 毎朝、当日の特徴量を計算してモデルで予測し、
      高確率銘柄をDiscordに通知する

実行タイミング: daily_scan.yml から呼び出される（毎朝6時）

出力:
  - data/ml/today_signals.json（当日のMLシグナル）
  - Discord通知（prob >= SIGNAL_THRESHOLD の銘柄）
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

warnings.filterwarnings("ignore")

# ============================================================
# 設定
# ============================================================
MODEL_PATH       = Path("data/ml/model.pkl")
CACHE_PATH       = Path("data/cache/fundamental_cache.json")
CONFIG_PATH      = Path("config/policy_keywords.yaml")
SIGNAL_PATH      = Path("data/ml/today_signals.json")
PRICE_CACHE_PATH = Path("data/cache/price_cache.pkl")
CACHE_EXPIRE_HOURS = 18  # キャッシュ有効時間
TRADES_PATH      = Path("data/ml/demo_trades.csv")
SIGNAL_THRESHOLD = 0.55   # バックテストで最適化済み
TOP_N            = 5      # Discord通知する上位銘柄数

JQUANTS_API_KEY  = os.environ.get("JQUANTS_API_KEY", "")
DISCORD_WEBHOOK  = os.environ.get("DISCORD_WEBHOOK_URL", "")

def _headers() -> dict:
    return {"x-api-key": JQUANTS_API_KEY}

# ============================================================
# ロガー設定
# ============================================================
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}\n",
    level="INFO"
)
Path("data/logs").mkdir(parents=True, exist_ok=True)
logger.add("data/logs/predict_{time:YYYYMMDD}.log", rotation="1 day", level="DEBUG")


# ============================================================
# 1. 銘柄リスト取得
# ============================================================
def get_tickers() -> list[str]:
    """policy_keywords.yaml から銘柄リストを取得"""
    import yaml
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tickers = []
    sectors = config.get("policy_sectors", {})
    # policy_sectors は辞書形式: {sector_name: {ticker_list: [...]}}
    for sector_name, sector_data in sectors.items():
        if isinstance(sector_data, dict):
            tickers.extend(sector_data.get("ticker_list", []))
    tickers = [t.replace(".T", "") for t in tickers]
    tickers = list(dict.fromkeys(tickers))  # 重複除去
    logger.info(f"銘柄数: {len(tickers)}")
    return tickers


# ============================================================
# 2. J-Quants 銘柄マスタから会社名を取得
# ============================================================
# ★ 銘柄名マスタ（src/utils/data_fetcher.py の STOCK_NAME_MAP と同期）
STOCK_NAME_MAP = {
    "7011": "三菱重工業", "7012": "川崎重工業", "7013": "IHI",
    "6479": "ミネベアミツミ", "7762": "シチズン時計", "6952": "カシオ計算機",
    "4062": "イビデン", "6967": "新光電気工業", "6203": "豊和工業",
    "4274": "細谷火工", "7721": "東京計器", "6208": "石川製作所",
    "7980": "重松製作所", "4275": "カーリットHD",
    "6503": "三菱電機", "6701": "NEC", "6702": "富士通", "7270": "SUBARU",
    "8035": "東京エレクトロン", "6857": "アドバンテスト", "6146": "ディスコ",
    "4063": "信越化学工業", "4523": "エーザイ", "6723": "ルネサスエレクトロニクス",
    "6526": "ソシオネクスト", "6920": "レーザーテック",
    "7735": "SCREENホールディングス", "6758": "ソニーグループ", "6600": "キオクシアHD",
    "9519": "レノバ", "6367": "ダイキン工業", "6501": "日立製作所",
    "5020": "ENEOSホールディングス", "9531": "東京ガス", "8113": "ユニ・チャーム",
    "4208": "UBE", "7203": "トヨタ自動車", "5401": "日本製鉄", "7003": "三井E&S",
    "9432": "NTT", "9433": "KDDI", "9984": "ソフトバンクグループ",
    "4307": "野村総合研究所", "9613": "NTTデータグループ",
    "3769": "GMOペイメントゲートウェイ", "4704": "トレンドマイクロ",
    "9719": "SCSK", "4739": "伊藤忠テクノソリューションズ", "4324": "電通グループ",
    "4502": "武田薬品工業", "4519": "中外製薬", "4021": "日産化学",
    "7741": "HOYA", "6869": "シスメックス", "4543": "テルモ",
    "4578": "大塚ホールディングス", "7733": "オリンパス",
    "1802": "大林組", "1803": "清水建設", "1812": "鹿島建設",
    "1801": "大成建設", "5444": "大和工業", "3407": "旭化成",
    "1811": "前田建設工業", "1861": "熊谷組", "5411": "JFEホールディングス",
    "8306": "三菱UFJ", "8316": "三井住友FG", "8411": "みずほFG",
    "8591": "オリックス", "8001": "伊藤忠商事",
    "7164": "全国保証", "8750": "第一生命HD", "8725": "MS&AD",
    "5707": "東邦亜鉛", "5706": "三井金属鉱業", "5713": "住友金属鉱山",
    "5714": "DOWAホールディングス", "5741": "UACJ", "3436": "SUMCO",
    "4042": "東ソー", "4183": "三井化学", "5019": "出光興産",
    "1662": "石油資源開発", "7746": "岡本硝子", "7485": "岡谷鋼機",
    "5541": "大平洋金属", "4004": "レゾナックHD", "5857": "AREホールディングス",
    "5802": "住友電気工業", "5801": "古河電気工業", "5711": "三菱マテリアル",
    "3692": "FFRIセキュリティ", "3697": "SHIFT", "3915": "テラスカイ",
    "3923": "ラクス", "3984": "ユーザーローカル", "3993": "PKSHA Technology",
    "3994": "マネーフォワード", "4180": "Appier Group", "4194": "ビジョナル",
    "4259": "エクサウィザーズ", "4384": "ラクスル", "4443": "Sansan",
    "4475": "HENNGE", "4480": "メドレー", "4483": "JMDC",
    "5032": "ANYCOLOR", "5253": "カバー", "6027": "弁護士ドットコム",
    "6532": "ベイカレント", "9348": "ispace",
    "3498": "霞ヶ関キャピタル", "3491": "GA technologies",
    "4371": "コアコンセプト・テクノロジー", "4417": "グローバルセキュリティエキスパート",
    "4431": "スマレジ", "4449": "ギフティ", "7033": "マネジメントソリューションズ",
    "9166": "GENDA", "9556": "INTLOOP", "6200": "インソース",
    "6196": "ストライク", "3182": "オイシックス・ラ・大地",
    "4051": "GMOフィナンシャルゲート", "4058": "トヨクモ",
    "4375": "セーフィー", "4592": "サンバイオ", "5038": "eWeLL",
    "5139": "オープンワーク", "5243": "note", "5254": "Arent",
    "6562": "ジーニー", "7388": "FPパートナー", "9279": "ギフトホールディングス",
    "9338": "INFORICH", "9467": "アルファポリス",
    "2980": "SREホールディングス", "2986": "LAホールディングス",
    "2998": "クリアル", "3133": "海帆", "3479": "ティーケーピー",
    "3482": "ロードスターキャピタル", "3496": "アズーム",
    "4165": "ブレイド", "4377": "ワンキャリア", "4393": "バンク・オブ・イノベーション",
    "4413": "ボードルア", "4419": "Finatextホールディングス",
    "4477": "BASE", "4563": "アンジェス", "4565": "ネクセラファーマ",
    "4575": "キャンバス", "4593": "ヘリオス", "4894": "クオリプス",
    "5027": "AnyMind Group", "5246": "ELEMENTS",
    "5842": "インテグラル", "6030": "アドベンチャー",
    "6544": "ジャパンエレベーターサービスHD", "7047": "ポート",
    "7059": "コプロ・ホールディングス", "7095": "Macbee Planet",
    "7157": "ライフネット生命保険", "7172": "ジャパンインベストメントアドバイザー",
    "7352": "TWOSTONE&Sons", "7373": "アイドマ・ホールディングス",
    "7685": "BuySell Technologies", "7806": "MTG",
    "8789": "フィンテック グローバル", "9158": "シーユーシー",
    "9168": "ライズ・コンサルティング・グループ", "9211": "エフ・コード",
    "9552": "クオンツ総研ホールディングス", "3989": "シェアリングテクノロジー",
    "4071": "プレイド", "6111": "旭精機工業", "6479": "ミネベアミツミ",
    "7806": "MTG", "9613": "NTTデータグループ",
}


def fetch_company_names() -> dict[str, str]:
    return STOCK_NAME_MAP


def get_company_name_yf(ticker: str) -> str:
    return STOCK_NAME_MAP.get(ticker, ticker)


# ============================================================
# 3. 価格キャッシュ管理
# ============================================================
def load_price_cache() -> dict | None:
    """キャッシュが有効なら返す（18時間以内）"""
    if not PRICE_CACHE_PATH.exists():
        return None
    try:
        with open(PRICE_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        cached_at = cache.get("_cached_at")
        if cached_at is None:
            return None
        elapsed = (datetime.now() - cached_at).total_seconds() / 3600
        if elapsed < CACHE_EXPIRE_HOURS:
            logger.info(f"価格キャッシュ使用 (経過:{elapsed:.1f}時間 / 有効:{CACHE_EXPIRE_HOURS}時間) 銘柄数:{len(cache)-1}")
            return {k: v for k, v in cache.items() if k != "_cached_at"}
        else:
            logger.info(f"価格キャッシュ期限切れ (経過:{elapsed:.1f}時間) → 再取得")
            return None
    except Exception as e:
        logger.warning(f"キャッシュ読み込みエラー: {e}")
        return None


def save_price_cache(prices: dict):
    """価格データをキャッシュに保存"""
    try:
        PRICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache = {"_cached_at": datetime.now(), **prices}
        with open(PRICE_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
        logger.info(f"価格キャッシュ保存: {len(prices)}銘柄 → {PRICE_CACHE_PATH}")
    except Exception as e:
        logger.warning(f"キャッシュ保存エラー: {e}")


# ============================================================
# 4. J-Quants 価格データ取得（直近90日）
# ============================================================
def fetch_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """各銘柄の直近90日の価格を取得"""
    import requests, time

    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=130)).strftime("%Y-%m-%d")

    prices = {}
    # 日付フォーマットをYYYYMMDDに変換（data_fetcher.pyと同じ）
    start_fmt = start_date.replace("-", "")
    end_fmt   = end_date.replace("-", "")

    for i, ticker in enumerate(tickers):
        try:
            # 4桁コード → 5桁（末尾に0追加）
            code = ticker + "0" if len(ticker) == 4 else ticker
            resp = requests.get(
                "https://api.jquants.com/v2/equities/bars/daily",
                headers=_headers(),
                params={"code": code, "from": start_fmt, "to": end_fmt},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    df = pd.DataFrame(data)
                    # AdjC（調整後終値）またはC（終値）を使用
                    if "AdjC" in df.columns:
                        df = df.rename(columns={"Date": "Date", "AdjC": "AdjustmentClose", "AdjVo": "AdjustmentVolume"})
                    else:
                        df = df.rename(columns={"Date": "Date", "C": "AdjustmentClose", "Vo": "AdjustmentVolume"})
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date").reset_index(drop=True)
                    prices[ticker] = df
            if i % 30 == 0:
                logger.info(f"  価格取得中: {i}/{len(tickers)}")
            time.sleep(0.1)
        except Exception as e:
            logger.warning(f"  {ticker}: {e}")

    logger.info(f"価格取得完了: {len(prices)}/{len(tickers)}銘柄")
    return prices


# ============================================================
# 3. 特徴量計算（学習時と同じロジック）
# ============================================================
def calc_features(ticker: str, df: pd.DataFrame, fund: dict) -> dict | None:
    """1銘柄の当日特徴量を計算"""
    try:
        if len(df) < 75:
            return None

        close = df["AdjustmentClose"].astype(float)
        vol   = df["AdjustmentVolume"].astype(float)

        # リターン
        r = {
            "return_1d":  float(close.pct_change(1).iloc[-1]),
            "return_5d":  float(close.pct_change(5).iloc[-1]),
            "return_20d": float(close.pct_change(20).iloc[-1]),
            "return_60d": float(close.pct_change(60).iloc[-1]),
        }

        # 移動平均乖離
        ma5  = close.rolling(5).mean()
        ma25 = close.rolling(25).mean()
        ma75 = close.rolling(75).mean()
        c    = close.iloc[-1]

        r["ma5_dev"]    = float((c / ma5.iloc[-1] - 1))
        r["ma25_dev"]   = float((c / ma25.iloc[-1] - 1))
        r["ma75_dev"]   = float((c / ma75.iloc[-1] - 1))
        r["above_ma75"] = float(c > ma75.iloc[-1])
        r["gc_25_75"]   = float(ma25.iloc[-1] > ma75.iloc[-1])

        # 5MA関連（フィルター用）
        c_prev        = close.iloc[-2]
        ma5_today     = float(ma5.iloc[-1])
        ma5_prev      = float(ma5.iloc[-2])
        # 5MA上抜け: 前日終値が5MA以下 → 当日終値が5MA以上
        r["ma5_breakout"] = float(c_prev <= ma5_prev and c >= ma5_today)
        # 5MAからの乖離（すでにma5_devで計算済み）
        # 5MA上に位置しているか
        r["above_ma5"] = float(c >= ma5_today)

        # RSI14
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
        r["rsi14"] = float(100 - 100 / (1 + rs))

        # ボリンジャーバンド %B
        m20  = close.rolling(20).mean()
        s20  = close.rolling(20).std()
        r["bb_pct"] = float((c - (m20.iloc[-1] - 2*s20.iloc[-1])) /
                             (4*s20.iloc[-1]) if s20.iloc[-1] != 0 else 0.5)

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9).mean()
        r["macd_hist"]   = float(macd.iloc[-1] - sig.iloc[-1])
        r["macd_golden"] = float(
            (macd.iloc[-1] > sig.iloc[-1]) and (macd.iloc[-2] <= sig.iloc[-2])
        )

        # 出来高比率
        vol_ma = vol.rolling(20).mean()
        r["vol_ratio"] = float(vol.iloc[-1] / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 1.0

        # 52週高値・安値からの乖離
        high52 = close.rolling(min(252, len(close))).max().iloc[-1]
        low52  = close.rolling(min(252, len(close))).min().iloc[-1]
        r["from_high"] = float(c / high52 - 1)
        r["from_low"]  = float(c / low52 - 1)

        # 信用倍率（デフォルト値）
        r["margin_ratio"] = float(fund.get("margin_ratio", 2.0))

        # TOPIX（後で追加）
        r["topix_return_5d"]  = 0.0
        r["topix_return_20d"] = 0.0

        # ファンダメンタル
        r["per"]              = float(fund.get("per", 15.0) or 15.0)
        r["pbr"]              = float(fund.get("pbr", 1.0) or 1.0)
        r["roe"]              = float(fund.get("roe", 5.0) or 5.0)
        r["roa"]              = float(fund.get("roa", 3.0) or 3.0)
        r["operating_margin"] = float(fund.get("operating_margin", 5.0) or 5.0)
        r["revenue_growth"]   = float(fund.get("revenue_growth_rate", 0.0) or 0.0)
        r["equity_ratio"]     = float(fund.get("equity_ratio", 40.0) or 40.0)
        r["debt_to_equity"]   = float(fund.get("debt_to_equity", 1.0) or 1.0)
        r["dividend_yield"]   = float(fund.get("dividend_yield", 2.0) or 2.0)
        r["credit_score"]     = float(fund.get("credit_score", 50.0) or 50.0)

        # 異常値チェック
        for k, v in r.items():
            if not np.isfinite(v):
                r[k] = 0.0

        r["ticker"] = ticker
        r["name"]   = str(fund.get("name", ticker))
        return r

    except Exception as e:
        logger.debug(f"  {ticker} 特徴量計算エラー: {e}")
        return None


# ============================================================
# 4. TOPIX直近リターンを取得して追加
# ============================================================
def fetch_topix_data() -> dict:
    """TOPIXデータ取得（5d/20dリターン + 25MA判定）APIコール1回のみ"""
    import requests
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    result = {"r5": 0.0, "r20": 0.0, "above_ma25": True, "current": 0.0}
    try:
        resp = requests.get(
            "https://api.jquants.com/v2/indices/bars/daily/topix",
            headers=_headers(),
            params={"from": start_date, "to": end_date},
            timeout=10
        )
        body = resp.json()
        data = body.get("data", body.get("indices", body.get("topix", [])))
        if not data:
            return result
        sample    = data[0]
        close_key = next((k for k in ["C", "Close", "close", "AdjustmentClose"] if k in sample), None)
        if not close_key:
            logger.warning(f"TOPIX closeキー不明: {list(sample.keys())}")
            return result
        closes = [float(d[close_key]) for d in data if d.get(close_key)]
        if len(closes) < 6:
            return result
        current   = closes[-1]
        ma25      = sum(closes[-25:]) / min(25, len(closes))
        r5        = current / closes[-6]  - 1 if len(closes) >= 6  else 0.0
        r20       = current / closes[-21] - 1 if len(closes) >= 21 else 0.0
        above_ma25 = current >= ma25
        logger.info(f"TOPIX: {current:.0f} 25MA:{ma25:.0f} {'✅上' if above_ma25 else '⚠️下'} 5d:{r5:+.2%} 20d:{r20:+.2%}")
        result = {"r5": round(r5,4), "r20": round(r20,4), "above_ma25": above_ma25, "current": round(current,2)}
    except Exception as e:
        logger.warning(f"TOPIX取得エラー: {e}")
    return result


def fetch_nikkei_futures() -> dict:
    """日経225先物をstooq.comから取得（APIトークン不要・無料）"""
    import requests
    result = {"price": 0.0, "change_pct": 0.0}
    try:
        # stooq.com: nk225f=日経225先物
        resp = requests.get(
            "https://stooq.com/q/l/?s=nk225f.cmefm&f=sd2t2ohlcv&h&e=csv",
            timeout=10
        )
        lines = resp.text.strip().split("
")
        if len(lines) >= 2:
            cols  = lines[0].split(",")
            vals  = lines[1].split(",")
            row   = dict(zip(cols, vals))
            close = float(row.get("Close", 0))
            open_ = float(row.get("Open", close))
            if close > 0 and open_ > 0:
                chg = (close - open_) / open_
                result = {"price": close, "change_pct": round(chg, 4)}
                logger.info(f"日経先物: {close:,.0f} ({chg:+.2%})")
    except Exception as e:
        logger.warning(f"日経先物取得エラー: {e}")
    return result


# ============================================================
# 5. 予測実行
# ============================================================
def predict(features_list: list[dict], topix: dict) -> pd.DataFrame:
    """モデルで予測確率を計算"""
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model     = saved["model"]
    feat_cols = saved["feat_cols"]

    df = pd.DataFrame(features_list)

    # TOPIX追加
    df["topix_return_5d"]  = topix.get("r5", 0.0)
    df["topix_return_20d"] = topix.get("r20", 0.0)

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    df["pred_prob"] = model.predict(X)
    df = df.sort_values("pred_prob", ascending=False)
    return df


# ============================================================
# 6. Discord通知
# ============================================================
def notify_discord(signals: pd.DataFrame, today_str: str, name_map: dict = {},
                   topix: dict = {}, nikkei_f: dict = {}):
    import requests

    # 市場環境ヘッダー
    topix_mark   = "✅25MA上" if topix.get("above_ma25", True) else "⚠️25MA下"
    nk_price     = nikkei_f.get("price", 0.0)
    nk_chg       = nikkei_f.get("change_pct", 0.0)
    nk_emoji     = "🔴" if nk_chg <= -0.01 else ("🟢" if nk_chg >= 0.01 else "⚪")
    nk_str       = f"{nk_price:,.0f} ({nk_chg:+.1%}) {nk_emoji}" if nk_price > 0 else "取得失敗"
    market_line  = f"📊 TOPIX:{topix_mark}  日経先物:{nk_str}"

    if signals.empty:
        msg = f"🤖 **ML シグナル {today_str}**\n{market_line}\n該当銘柄なし（閾値: {SIGNAL_THRESHOLD}）"
    else:
        lines = [f"🤖 **ML シグナル {today_str}** （閾値: {SIGNAL_THRESHOLD}）", market_line, ""]
        for i, (_, row) in enumerate(signals.iterrows(), 1):
            ticker    = row["ticker"]
            name      = name_map.get(ticker) or get_company_name_yf(ticker)
            prob      = row["pred_prob"]
            ret_1d    = row.get("return_1d", 0)
            rsi       = row.get("rsi14", 0)
            above_ma5 = row.get("above_ma5", 1.0)
            ma5_break = row.get("ma5_breakout", 0.0)
            vol_ratio = row.get("vol_ratio", 1.0)
            ma5_mark  = "🔼5MA抜け " if ma5_break == 1.0 else ("📈5MA上 " if above_ma5 == 1.0 else "")
            vol_mark  = "🔥出来高急増 " if vol_ratio >= 2.0 else ""
            lines.append(
                f"**{i}. {name} ({ticker}.T)**  確率:{prob:.0%}  "
                f"前日:{ret_1d:+.1%}  RSI:{rsi:.0f}  {ma5_mark}{vol_mark}"
            )
        msg = "\n".join(lines)

    resp = requests.post(DISCORD_WEBHOOK, json={"content": msg})
    logger.info(f"Discord通知: {resp.status_code} | {len(signals)}銘柄")


# ============================================================
# デモ取引管理
# ============================================================
HOLD_DAYS_TRADE = 5  # 保有期間（営業日ではなく暦日で近似）

def get_current_price(ticker: str) -> float | None:
    """当日の終値を取得"""
    import requests
    try:
        code = ticker + "0" if len(ticker) == 4 else ticker
        today = datetime.now().strftime("%Y%m%d")
        resp = requests.get(
            "https://api.jquants.com/v2/equities/bars/daily",
            headers=_headers(),
            params={"code": code, "from": today, "to": today},
            timeout=10
        )
        data = resp.json().get("data", [])
        if data:
            row = data[-1]
            return float(row.get("AdjC") or row.get("C") or 0)
    except Exception as e:
        logger.warning(f"{ticker} 価格取得エラー: {e}")
    return None


def record_entry(signals: pd.DataFrame, today_str: str, name_map: dict):
    """新規シグナルをdemo_trades.csvに追記"""
    if signals.empty:
        return

    rows = []
    for _, row in signals.iterrows():
        ticker = row["ticker"]
        price  = get_current_price(ticker)
        if price is None:
            price = 0.0

        exit_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")  # 5営業日≒7暦日
        ma5_signal = "🔼抜け" if row.get("ma5_breakout", 0) == 1.0 else ("📈上" if row.get("above_ma5", 0) == 1.0 else "")

        rows.append({
            "entry_date":   today_str,
            "ticker":       ticker,
            "name":         name_map.get(ticker, ticker),
            "prob":         round(float(row["pred_prob"]), 3),
            "ma5_signal":   ma5_signal,
            "entry_price":  price,
            "exit_date":    exit_date,
            "exit_price":   "",
            "return":       "",
            "win":          "",
        })

    new_df = pd.DataFrame(rows)

    if TRADES_PATH.exists():
        existing = pd.read_csv(TRADES_PATH, dtype=str)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(TRADES_PATH, index=False, encoding="utf-8-sig")
    logger.info(f"デモ取引記録: {len(rows)}件追記 → {TRADES_PATH}")


def update_exits(today_str: str):
    """決済予定日を過ぎた未決済行の損益を自動計算"""
    if not TRADES_PATH.exists():
        return

    df = pd.read_csv(TRADES_PATH, dtype=str)
    updated = 0

    for idx, row in df.iterrows():
        # 未決済かつ決済予定日を過ぎている
        if str(row.get("exit_price", "")).strip() != "":
            continue
        try:
            exit_date = datetime.strptime(str(row["exit_date"]), "%Y-%m-%d")
        except Exception:
            continue
        if datetime.now() < exit_date:
            continue

        ticker      = str(row["ticker"])
        entry_price = float(row["entry_price"]) if str(row["entry_price"]) not in ("", "0.0", "0") else None
        if entry_price is None or entry_price == 0:
            continue

        exit_price = get_current_price(ticker)
        if exit_price is None or exit_price == 0:
            continue

        ret = (exit_price - entry_price) / entry_price
        win = "✅" if ret > 0 else "❌"

        df.at[idx, "exit_price"] = round(exit_price, 2)
        df.at[idx, "return"]     = round(ret, 4)
        df.at[idx, "win"]        = win
        updated += 1
        logger.info(f"決済更新: {ticker} {ret:+.1%} {win}")

    if updated > 0:
        df.to_csv(TRADES_PATH, index=False, encoding="utf-8-sig")
        logger.info(f"決済更新: {updated}件完了")


# ============================================================
# メイン処理
# ============================================================
def main():
    Path("data/ml").mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        logger.error(f"モデルが見つかりません: {MODEL_PATH}")
        logger.error("PHASE 2 を先に実行してください")
        return

    today_str = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"=== ML日次予測 {today_str} ===")

    # 銘柄リスト
    tickers = get_tickers()

    # 会社名マップ取得
    name_map  = fetch_company_names()

    # 価格データ取得（キャッシュ優先）
    prices = load_price_cache()
    if prices is None:
        prices = fetch_prices(tickers)
        save_price_cache(prices)
    else:
        logger.info("→ APIコールをスキップしました")

    # TOPIX
    topix    = fetch_topix_data()
    nikkei_f = fetch_nikkei_futures()

    # ファンダメンタルキャッシュ読み込み
    fund_cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            fund_cache = json.load(f)
    logger.info(f"ファンダメンタルキャッシュ: {len(fund_cache)}銘柄")

    # 特徴量計算
    features_list = []
    for ticker, df in prices.items():
        fund = fund_cache.get(ticker, {})
        feat = calc_features(ticker, df, fund)
        if feat:
            features_list.append(feat)

    logger.info(f"特徴量計算完了: {len(features_list)}銘柄")

    if not features_list:
        logger.error("特徴量が計算できませんでした")
        return

    # 予測
    pred_df = predict(features_list, topix)

    # 前日シグナルを読み込み（重複除外用）
    prev_tickers = set()
    if SIGNAL_PATH.exists():
        try:
            with open(SIGNAL_PATH, encoding="utf-8") as f:
                prev = json.load(f)
            # 前日のデータなら除外対象にする
            if prev.get("date") != today_str:
                prev_tickers = {s["ticker"] for s in prev.get("signals", [])}
                logger.info(f"前日シグナル除外: {prev_tickers}")
        except Exception:
            pass

    # シグナル抽出（前日と重複しない新規銘柄のみ・上位TOP_N件）
    all_signals = pred_df[pred_df["pred_prob"] >= SIGNAL_THRESHOLD]

    # 5MAフィルター: 5MA上に位置している銘柄のみ（上抜け or 上位にある）
    if "above_ma5" in all_signals.columns:
        ma5_filtered = all_signals[all_signals["above_ma5"] == 1.0]
        logger.info(f"5MAフィルター後: {len(ma5_filtered)} / {len(all_signals)}銘柄")
        # フィルター後に5件未満なら緩めてフィルター前に戻す
        if len(ma5_filtered) >= 3:
            all_signals = ma5_filtered

    new_signals = all_signals[~all_signals["ticker"].isin(prev_tickers)]
    # 市場環境フィルター: TOPIX25MA以下なら最大3件に絞る
    max_signals = 3 if not topix.get("above_ma25", True) else TOP_N
    signals = new_signals.head(max_signals)
    logger.info(f"シグナル銘柄数: {len(signals)} 新規 / {len(all_signals)} 全体 (閾値:{SIGNAL_THRESHOLD})")

    # 保存
    output = {
        "date":       today_str,
        "threshold":  SIGNAL_THRESHOLD,
        "total_analyzed": len(pred_df),
        "signals": signals[["ticker", "pred_prob", "return_1d", "rsi14", "ma25_dev"]].to_dict("records") if not signals.empty else [],
        "top20": pred_df.head(20)[["ticker", "pred_prob"]].to_dict("records"),
    }
    with open(SIGNAL_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Discord通知
    if DISCORD_WEBHOOK:
        notify_discord(signals, today_str, name_map, topix, nikkei_f)
    else:
        logger.warning("DISCORD_WEBHOOK_URL が未設定")

    # デモ取引記録
    update_exits(today_str)          # 既存の未決済行を更新
    record_entry(signals, today_str, name_map)  # 新規シグナルを追記

    logger.success(f"完了: {len(signals)}件のMLシグナルを通知")


if __name__ == "__main__":
    main()
