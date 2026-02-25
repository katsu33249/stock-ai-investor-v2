"""
data_fetcher.py - J-Quants API V2対応版

V2変更点：
- 認証：APIキー（x-api-key ヘッダー）
- エンドポイント：/v2/equities/bars/daily
- カラム名：Close→C, Open→O, High→H, Low→L, Volume→Vo
- 銘柄マスタ：dateパラメータ必須
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional


# 銘柄名マップ（J-Quantsで取れない場合のフォールバック）
STOCK_NAME_MAP = {
    # 防衛
    "7011.T": "三菱重工業", "7012.T": "川崎重工業", "7013.T": "IHI",
    "6479.T": "ミネベアミツミ", "7762.T": "シチズン時計", "6952.T": "カシオ計算機",
    "4062.T": "イビデン", "6967.T": "新光電気工業", "6203.T": "豊和工業",
    "4274.T": "細谷火工", "7721.T": "東京計器", "6208.T": "石川製作所",
    "7980.T": "重松製作所", "4275.T": "カーリットHD",
    "6503.T": "三菱電機", "6701.T": "NEC", "6702.T": "富士通", "7270.T": "SUBARU",
    # 半導体
    "8035.T": "東京エレクトロン", "6857.T": "アドバンテスト", "6146.T": "ディスコ",
    "4063.T": "信越化学工業", "4523.T": "エーザイ", "6723.T": "ルネサスエレクトロニクス",
    "6526.T": "ソシオネクスト", "6920.T": "レーザーテック",
    "7735.T": "SCREENホールディングス", "6758.T": "ソニーグループ", "6600.T": "キオクシアHD",
    # GX
    "9519.T": "レノバ", "6367.T": "ダイキン工業", "6501.T": "日立製作所",
    "5020.T": "ENEOSホールディングス", "9531.T": "東京ガス", "8113.T": "ユニ・チャーム",
    "4208.T": "UBE", "7203.T": "トヨタ自動車", "5401.T": "日本製鉄", "7003.T": "三井E&S",
    # AI/DX
    "9432.T": "NTT", "9433.T": "KDDI", "9984.T": "ソフトバンクグループ",
    "4307.T": "野村総合研究所", "9613.T": "NTTデータグループ",
    "3769.T": "GMOペイメントゲートウェイ", "4704.T": "トレンドマイクロ",
    "9719.T": "SCSK", "4739.T": "伊藤忠テクノソリューションズ", "4324.T": "電通グループ",
    # 医療
    "4502.T": "武田薬品工業", "4519.T": "中外製薬", "4021.T": "日産化学",
    "7741.T": "HOYA", "6869.T": "シスメックス", "4543.T": "テルモ",
    "4578.T": "大塚ホールディングス", "7733.T": "オリンパス",
    # インフラ
    "1802.T": "大林組", "1803.T": "清水建設", "1812.T": "鹿島建設",
    "1801.T": "大成建設", "5444.T": "大和工業", "3407.T": "旭化成",
    "1811.T": "前田建設工業", "1861.T": "熊谷組", "5411.T": "JFEホールディングス",
    # 金融
    "8306.T": "三菱UFJ", "8316.T": "三井住友FG", "8411.T": "みずほFG",
    "8591.T": "オリックス", "8001.T": "伊藤忠商事",
    "7164.T": "全国保証", "8750.T": "第一生命HD", "8725.T": "MS&AD",
    # レアアース
    "5707.T": "東邦亜鉛", "5706.T": "三井金属鉱業", "5713.T": "住友金属鉱山",
    "5714.T": "DOWAホールディングス", "5741.T": "UACJ", "3436.T": "SUMCO",
    "4042.T": "東ソー", "4183.T": "三井化学", "5019.T": "出光興産",
    "1662.T": "石油資源開発", "7746.T": "岡本硝子", "7485.T": "岡谷鋼機",
    "5541.T": "大平洋金属", "4004.T": "レゾナックHD", "5857.T": "AREホールディングス",
    "5802.T": "住友電気工業", "5801.T": "古河電気工業", "5711.T": "三菱マテリアル",
    # スタートアップ
    "3692.T": "FFRIセキュリティ", "3697.T": "SHIFT", "3915.T": "テラスカイ",
    "3923.T": "ラクス", "3984.T": "ユーザーローカル", "3993.T": "PKSHA Technology",
    "3994.T": "マネーフォワード", "4180.T": "Appier Group", "4194.T": "ビジョナル",
    "4259.T": "エクサウィザーズ", "4384.T": "ラクスル", "4443.T": "Sansan",
    "4475.T": "HENNGE", "4480.T": "メドレー", "4483.T": "JMDC",
    "5032.T": "ANYCOLOR", "5253.T": "カバー", "6027.T": "弁護士ドットコム",
    "6532.T": "ベイカレント", "9348.T": "ispace",
    "3498.T": "霞ヶ関キャピタル", "3491.T": "GA technologies",
    "4371.T": "コアコンセプト・テクノロジー", "4417.T": "グローバルセキュリティエキスパート",
    "4431.T": "スマレジ", "4449.T": "ギフティ", "7033.T": "マネジメントソリューションズ",
    "9166.T": "GENDA", "9556.T": "INTLOOP", "6200.T": "インソース",
    "6196.T": "ストライク", "3182.T": "オイシックス・ラ・大地",
    "4051.T": "GMOフィナンシャルゲート", "4058.T": "トヨクモ",
    "4375.T": "セーフィー", "4592.T": "サンバイオ", "5038.T": "eWeLL",
    "5139.T": "オープンワーク", "5243.T": "note", "5254.T": "Arent",
    "6562.T": "ジーニー", "7388.T": "FPパートナー", "9279.T": "ギフトホールディングス",
    "9338.T": "INFORICH", "9467.T": "アルファポリス",
    "2980.T": "SREホールディングス", "2986.T": "LAホールディングス",
    "2998.T": "クリアル", "3133.T": "海帆", "3479.T": "ティーケーピー",
    "3482.T": "ロードスターキャピタル", "3496.T": "アズーム",
    "4165.T": "ブレイド", "4377.T": "ワンキャリア", "4393.T": "バンク・オブ・イノベーション",
    "4413.T": "ボードルア", "4419.T": "Finatextホールディングス",
    "4477.T": "BASE", "4563.T": "アンジェス", "4565.T": "ネクセラファーマ",
    "4575.T": "キャンバス", "4593.T": "ヘリオス", "4894.T": "クオリプス",
    "5027.T": "AnyMind Group", "5246.T": "ELEMENTS",
    "5842.T": "インテグラル", "6030.T": "アドベンチャー",
    "6544.T": "ジャパンエレベーターサービスHD", "7047.T": "ポート",
    "7059.T": "コプロ・ホールディングス", "7095.T": "Macbee Planet",
    "7157.T": "ライフネット生命保険", "7172.T": "ジャパンインベストメントアドバイザー",
    "7352.T": "TWOSTONE&Sons", "7373.T": "アイドマ・ホールディングス",
    "7685.T": "BuySell Technologies", "7806.T": "MTG",
    "8789.T": "フィンテック グローバル", "9158.T": "シーユーシー",
    "9168.T": "ライズ・コンサルティング・グループ", "9211.T": "エフ・コード",
    "9552.T": "クオンツ総研ホールディングス", "3668.T": "コロプラ",
    "3989.T": "シェアリングテクノロジー", "6562.T": "ジーニー",
}


class DataFetcher:

    BASE_URL = "https://api.jquants.com"

    def __init__(self, history_days: int = 180):
        self.history_days = history_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=history_days)

        self.api_key = os.environ.get("JQUANTS_API_KEY", "")
        if not self.api_key:
            logger.error("JQUANTS_API_KEY が設定されていません")
        else:
            logger.success("J-Quants API V2 初期化完了")

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key}

    def _to_code(self, ticker: str) -> str:
        """7011.T → 70110 形式に変換"""
        code = ticker.replace(".T", "")
        return code + "0" if len(code) == 4 else code

    def get_price_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """株価履歴を取得（V2 API）"""
        if not self.api_key:
            return None
        try:
            code = self._to_code(ticker)
            res = requests.get(
                f"{self.BASE_URL}/v2/equities/bars/daily",
                headers=self._headers(),
                params={
                    "code": code,
                    "from": self.start_date.strftime("%Y%m%d"),
                    "to": self.end_date.strftime("%Y%m%d"),
                },
                timeout=15
            )
            if res.status_code != 200:
                logger.warning(f"株価取得失敗({ticker}): {res.status_code} {res.text[:100]}")
                return None

            data = res.json().get("data", [])
            if not data:
                logger.warning(f"株価データなし: {ticker}")
                return None

            df = pd.DataFrame(data)

            # V2: 調整後カラムがあれば優先使用
            raw_cols = pd.DataFrame(data).columns.tolist()
            if "AdjC" in raw_cols:
                rename = {
                    "Date": "date", "AdjO": "open", "AdjH": "high",
                    "AdjL": "low", "AdjC": "close", "AdjVo": "volume",
                }
            else:
                rename = {
                    "Date": "date", "O": "open", "H": "high",
                    "L": "low", "C": "close", "Vo": "volume",
                }

            df = df.rename(columns=rename)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            return df

        except Exception as e:
            logger.error(f"株価履歴エラー({ticker}): {e}")
            return None

    def get_company_name(self, ticker: str) -> str:
        """銘柄名を取得（静的マップ優先 → J-Quants API → ticker）"""
        # ① 静的マップから取得（最速・最確実）
        if ticker in STOCK_NAME_MAP:
            return STOCK_NAME_MAP[ticker]

        # ② J-Quants API
        if self.api_key:
            try:
                code = self._to_code(ticker)
                today = datetime.now().strftime("%Y%m%d")
                res = requests.get(
                    f"{self.BASE_URL}/v2/equities/master",
                    headers=self._headers(),
                    params={"code": code, "date": today},
                    timeout=10
                )
                if res.status_code == 200:
                    items = res.json().get("data", [])
                    if items:
                        item = items[0]
                        name = (
                            item.get("CompanyName") or
                            item.get("CompanyNameEnglish") or
                            item.get("Name") or
                            item.get("name")
                        )
                        if name:
                            return name
            except Exception as e:
                logger.warning(f"銘柄名API取得エラー({ticker}): {e}")

        # ③ フォールバック: ticker そのまま
        return ticker

    def get_stock_info(self, ticker: str) -> Optional[dict]:
        """銘柄情報を取得（V2 API）"""
        if not self.api_key:
            return None
        try:
            code = self._to_code(ticker)
            today = datetime.now().strftime("%Y%m%d")

            # 銘柄マスタ（V2・dateパラメータ付き）
            res = requests.get(
                f"{self.BASE_URL}/v2/equities/master",
                headers=self._headers(),
                params={"code": code, "date": today},
                timeout=10
            )
            info = {}
            if res.status_code == 200:
                items = res.json().get("data", [])
                if items:
                    item = items[0]
                    api_name = (
                            item.get("CompanyName") or
                            item.get("CompanyNameEnglish") or
                            item.get("Name") or
                            item.get("name")
                        )
                    info = {
                        "name": api_name or STOCK_NAME_MAP.get(ticker, ticker),
                        "sector": item.get("Sector17CodeName", "不明"),
                        "industry": item.get("Sector33CodeName", "不明"),
                        "market": item.get("MarketCodeName", ""),
                    }

            # 株価履歴取得
            history = self.get_price_history(ticker)
            current_price = 0
            volume = 0
            avg_volume = 0
            week52_high = 0
            week52_low = 0
            if history is not None and not history.empty:
                current_price = float(history["close"].iloc[-1])
                volume = float(history["volume"].iloc[-1])
                avg_volume = float(history["volume"].mean())
                week52_high = float(history["high"].max())
                week52_low = float(history["low"].min())

            return {
                "ticker": ticker,
                "name": info.get("name", ticker),
                "sector": info.get("sector", "不明"),
                "industry": info.get("industry", "不明"),
                "current_price": current_price,
                "market_cap": 0,
                "volume": volume,
                "avg_volume": avg_volume,
                "per": None, "pbr": None, "psr": None,
                "ev_ebitda": None, "roe": None, "roa": None,
                "profit_margin": None, "operating_margin": None,
                "revenue_growth": None, "earnings_growth": None,
                "dividend_yield": None, "debt_to_equity": None,
                "current_ratio": None,
                "week52_high": week52_high,
                "week52_low": week52_low,
            }

        except Exception as e:
            logger.error(f"銘柄情報エラー({ticker}): {e}")
            return None

    def get_margin_trading(self, ticker: str) -> Optional[dict]:
        """信用取引週末残高を取得（V2 API・Standardプラン）

        エンドポイント: /v2/markets/margin-interest
        認証: x-api-key（V2標準）
        フィールド: LongVol（買残）/ ShrtVol（売残）
        """
        if not self.api_key:
            return None
        try:
            code = self._to_code(ticker)
            from_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")

            res = requests.get(
                f"{self.BASE_URL}/v2/markets/margin-interest",
                headers=self._headers(),
                params={"code": code, "from": from_date, "to": to_date},
                timeout=10
            )
            if res.status_code != 200:
                logger.warning(f"信用残取得失敗({ticker}): {res.status_code}")
                return None

            data = res.json().get("data", [])
            if not data:
                return None

            latest = data[-1]
            long_vol  = float(latest.get("LongVol",  0) or 0)
            shrt_vol  = float(latest.get("ShrtVol",  0) or 0)
            margin_ratio = round(long_vol / shrt_vol, 2) if shrt_vol > 0 else None

            return {
                "margin_ratio": margin_ratio,
                "long_margin":  long_vol,
                "short_margin": shrt_vol,
                "date": latest.get("Date", ""),
            }

        except Exception as e:
            logger.warning(f"信用残取得エラー({ticker}): {e}")
            return None

    def get_valid_tse_codes(self) -> set:
        """J-Quants masterから東証上場中の全銘柄コードを取得"""
        if not self.api_key:
            return set()
        try:
            today = datetime.now().strftime("%Y%m%d")
            res = requests.get(
                f"{self.BASE_URL}/v2/equities/master",
                headers=self._headers(),
                params={"date": today},
                timeout=30
            )
            if res.status_code != 200:
                logger.warning(f"masterAPI失敗: {res.status_code}")
                return set()
            items = res.json().get("data", [])
            # 5桁コード（末尾0）をセットで返す
            codes = set()
            for item in items:
                code = str(item.get("Code", ""))
                if code:
                    codes.add(code)
            logger.info(f"東証上場銘柄数: {len(codes)}")
            return codes
        except Exception as e:
            logger.warning(f"masterAPI取得エラー: {e}")
            return set()

    def get_multiple_stocks(self, tickers: list) -> dict:
        """複数銘柄を一括取得（東証銘柄のみフィルタリング）"""
        results = {}

        # 東証上場銘柄リストを事前取得してフィルタリング
        valid_codes = self.get_valid_tse_codes()
        if valid_codes:
            tse_tickers = []
            skipped = []
            for t in tickers:
                code = self._to_code(t)  # 7011 → 70110
                if code in valid_codes:
                    tse_tickers.append(t)
                else:
                    skipped.append(t)
            if skipped:
                logger.warning(f"非東証銘柄をスキップ({len(skipped)}件): {skipped[:5]}{'...' if len(skipped)>5 else ''}")
            tickers = tse_tickers
        else:
            logger.warning("masterAPI取得失敗。フィルタリングなしで実行します")

        total = len(tickers)
        logger.info(f"対象銘柄数（東証のみ）: {total}")

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"データ取得中... ({i}/{total}): {ticker}")

            info = self.get_stock_info(ticker)
            if not info:
                continue

            history = self.get_price_history(ticker)
            if history is None:
                continue

            info["price_history"] = history

            margin = self.get_margin_trading(ticker)
            info["margin_ratio"] = margin["margin_ratio"] if margin else None

            results[ticker] = info
            time.sleep(0.6)

        logger.success(f"取得完了: {len(results)}/{total} 銘柄")
        return results

    def get_market_overview(self) -> dict:
        """市場概況（日経平均・TOPIX）を取得"""
        overview = {}
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        to_date = datetime.now().strftime("%Y%m%d")

        # TOPIX
        try:
            res = requests.get(
                f"{self.BASE_URL}/v2/indices/bars/daily/topix",
                headers=self._headers(),
                params={"from": from_date, "to": to_date},
                timeout=10
            )
            if res.status_code == 200:
                data = res.json().get("data", [])
                if len(data) >= 2:
                    close = float(data[-1].get("C", 0))
                    prev  = float(data[-2].get("C", 1))
                    overview["TOPIX"] = {
                        "price": round(close, 2),
                        "change_pct": round((close - prev) / prev * 100, 2),
                    }
        except Exception as e:
            logger.warning(f"TOPIX取得エラー: {e}")

        # 日経平均
        try:
            res2 = requests.get(
                f"{self.BASE_URL}/v2/indices/bars/daily",
                headers=self._headers(),
                params={"code": "0028", "from": from_date, "to": to_date},
                timeout=10
            )
            if res2.status_code == 200:
                data2 = res2.json().get("data", [])
                if len(data2) >= 2:
                    close2 = float(data2[-1].get("C", 0))
                    prev2  = float(data2[-2].get("C", 1))
                    overview["日経平均"] = {
                        "price": round(close2, 2),
                        "change_pct": round((close2 - prev2) / prev2 * 100, 2),
                    }
        except Exception as e:
            logger.warning(f"日経平均取得エラー: {e}")

        return overview


class MarginScorer:
    """信用倍率スコアリング"""

    def score(self, margin_ratio: float) -> tuple:
        if margin_ratio is None:    return 0,  "信用倍率データなし"
        if margin_ratio <= 1.0:     return 10, f"🟢 信用倍率良好({margin_ratio:.1f}倍）"
        elif margin_ratio <= 2.0:   return 7,  f"🟡 信用倍率普通({margin_ratio:.1f}倍）"
        elif margin_ratio <= 3.0:   return 4,  f"🟠 信用倍率やや過熱({margin_ratio:.1f}倍）"
        else:                        return -5, f"🔴 信用倍率過熱({margin_ratio:.1f}倍）要注意"
