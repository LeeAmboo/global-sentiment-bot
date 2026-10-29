import os
import html
import json
import base64
from pathlib import Path
import requests
import yfinance as yf
import akshare as ak
import pandas as pd
import numpy as np
import fear_greed
from datetime import datetime, timedelta, timezone

try:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import unpad
except ImportError:
    AES = None
    unpad = None

# ================= 配置区域 =================
CRYPTO_URL = "https://api.alternative.me/fng/?limit=80"
JIUQUAN_URL = "https://www.jiucaishuo.com/indicators?tab=fg"
JIUCAI_FG_API = "https://api.jiucaishuo.com/v2/kjtl/history-lines"
JIUCAI_REFERER = "https://app.jiucaishuo.com/pagesE/tool/fear_greed"

US_FALLBACK_URL = "https://feargreedchart.com/api/?action=all"
US_HISTORY_URL = "https://feargreedchart.com/api/?action=history"

LIMIT_LOW = 25
LIMIT_HIGH = 75
DANGER_DAYS_THRESHOLD = 10
MAX_PUSHPLUS_LEN = 19000

FEAR_COLOR = "#16a34a"
GREED_COLOR = "#dc2626"
NEUTRAL_COLOR = "#555555"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}

# A股六个基础因子 + 一个可选杠杆率因子自建指数参数
CN_HISTORY_DAYS = 180
CN_DISPLAY_DAYS = 30
CN_HISTORY_FILE = Path(__file__).resolve().parent / "data" / "cn_sentiment_history.json"
CN_GITHUB_HISTORY_PATH = "data/cn_sentiment_history.json"
CN_BASE_FACTORS = (
    "market_volatility",
    "northbound_flow",
    "turnover",
    "futures_basis",
    "price_strength",
    "hedge_demand",
)
CN_OPTIONAL_FACTORS = ("leverage_ratio",)
CN_CURRENT_METHOD = "seven_factor_optional_leverage"
CN_LAST_FACTOR_DETAIL = {}


# ================= 通用工具 =================
def safe(x):
    return html.escape(str(x), quote=True)


def status_text(value):
    if value < LIMIT_LOW:
        return "极度恐慌"
    if value > LIMIT_HIGH:
        return "极度贪婪"
    return "中性"


def value_color(value):
    if value < LIMIT_LOW:
        return FEAR_COLOR
    if value > LIMIT_HIGH:
        return GREED_COLOR
    return "#333333"


def status_class(value):
    if value < LIMIT_LOW:
        return "fear"
    if value > LIMIT_HIGH:
        return "greed"
    return "neutral"


def status_style(value):
    if value < LIMIT_LOW:
        return f"color:{FEAR_COLOR};font-weight:800"
    if value > LIMIT_HIGH:
        return f"color:{GREED_COLOR};font-weight:800"
    return f"color:{NEUTRAL_COLOR};font-weight:600"


def dedupe_by_date(data):
    if not data:
        return data
    seen, out = set(), []
    for item in data:
        d = item.get("date")
        if d not in seen:
            seen.add(d)
            out.append(item)
    return out


def normalize_score(value):
    score = float(value)
    if not np.isfinite(score) or not 0 <= score <= 100:
        raise ValueError(f"指数分数无效或超出 0～100：{value}")
    return int(round(score))


def normalize_date(value):
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    if "T" in text:
        text = text.split("T")[0]
    if " " in text:
        text = text.split(" ")[0]
    return pd.to_datetime(text).strftime("%Y-%m-%d")


def request_json(url, timeout=20):
    response = requests.get(
        url, headers=REQUEST_HEADERS, timeout=timeout
    )
    response.raise_for_status()
    return response.json()


def _github_history_configured():
    return bool(os.getenv("GITHUB_REPOSITORY") and os.getenv("GITHUB_TOKEN"))


def _github_contents_url():
    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    return f"https://api.github.com/repos/{repo}/contents/{CN_GITHUB_HISTORY_PATH}"


def load_cn_saved_history():
    """读取此前保存的A股自建情绪指数历史；优先读取 GitHub 仓库中的持久化文件。"""
    if _github_history_configured():
        try:
            response = requests.get(
                _github_contents_url(),
                headers={
                    "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=20,
            )
            if response.status_code == 200:
                encoded = response.json().get("content", "")
                raw = base64.b64decode(encoded).decode("utf-8")
                records = json.loads(raw)
                if isinstance(records, list):
                    print(f"📚 已从 GitHub 读取 A 股情绪指数历史记录：{len(records)} 条")
                    return records
            elif response.status_code != 404:
                print(f"⚠️ 读取 GitHub 历史文件失败：HTTP {response.status_code}，尝试读取本地文件")
        except Exception as e:
            print(f"⚠️ 读取 GitHub 历史文件异常：{e}，尝试读取本地文件")

    try:
        if CN_HISTORY_FILE.exists():
            with CN_HISTORY_FILE.open("r", encoding="utf-8") as f:
                records = json.load(f)
            if isinstance(records, list):
                print(f"📚 已从本地读取 A 股情绪指数历史记录：{len(records)} 条")
                return records
    except Exception as e:
        print(f"⚠️ 读取本地 A 股历史文件失败：{e}")
    return []


def save_cn_history(records):
    """保存A股情绪指数历史到本地，并尽可能写回 GitHub 仓库供下一次 Actions 使用。"""
    records = sorted(records, key=lambda x: x.get("date", ""), reverse=True)[:365]
    payload = json.dumps(records, ensure_ascii=False, indent=2) + "\n"

    try:
        CN_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        CN_HISTORY_FILE.write_text(payload, encoding="utf-8")
        print(f"💾 已保存本地 A 股情绪指数历史记录：{len(records)} 条")
    except Exception as e:
        print(f"⚠️ 保存本地 A 股历史文件失败：{e}")

    if not _github_history_configured():
        print("⚠️ 未配置 GITHUB_TOKEN 或 GITHUB_REPOSITORY；本地文件可保存，但 GitHub Actions 下次运行无法自动继承。请在 workflow 中配置 GITHUB_TOKEN 并授予 contents: write。")
        return False

    headers = {
        "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = _github_contents_url()
    try:
        # 获取 sha；更新已有文件时 GitHub API 要求提供 sha。
        existing = requests.get(url, headers=headers, timeout=20)
        body = {
            "message": "Update daily A-share sentiment history",
            "content": base64.b64encode(payload.encode("utf-8")).decode("ascii"),
        }
        if existing.status_code == 200:
            sha = existing.json().get("sha")
            if sha:
                body["sha"] = sha
        elif existing.status_code != 404:
            existing.raise_for_status()

        response = requests.put(url, headers=headers, json=body, timeout=30)
        if response.status_code in (200, 201):
            print("☁️ 已将 A 股情绪指数历史记录写回 GitHub，下一次运行可继续累积")
            return True
        print(f"⚠️ 写回 GitHub 历史文件失败：HTTP {response.status_code}，{response.text[:300]}")
    except Exception as e:
        print(f"⚠️ 写回 GitHub 历史文件异常：{e}")
    return False


def clamp(value, low=0, high=100):
    return max(low, min(high, float(value)))


def linear_score(value, low, high):
    """将 low～high 线性映射为 0～100。"""
    if pd.isna(value) or high <= low:
        return 50.0
    return clamp((float(value) - low) / (high - low) * 100)


# ================= 备用 RSI 计算 =================
def calculate_rsi_history(ticker, period="8mo"):
    """A股数据完全不可用时的备用指标；RSI 不等同于贪恐指数。"""
    try:
        df = yf.download(
            ticker, period=period, progress=False,
            auto_adjust=False, threads=False
        )
        if df.empty:
            return None

        close = df["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        records = []
        for date, value in rsi.dropna().iloc[-70:][::-1].items():
            records.append({
                "date": pd.to_datetime(date).strftime("%Y-%m-%d"),
                "value": normalize_score(value)
            })
        return records or None
    except Exception as e:
        print(f"RSI Calculation Error for {ticker}: {e}")
        return None


# ================= CNN 美股贪恐指数 =================
def get_cnn_us_data():
    print("🇺🇸 正在获取 CNN Fear & Greed Index...")
    idx = fear_greed.get()

    if not isinstance(idx, dict) or "score" not in idx:
        raise ValueError("CNN 返回数据格式异常或缺少 score 字段")

    current_val = normalize_score(idx["score"])
    current_date = normalize_date(
        idx["timestamp"] if idx.get("timestamp")
        else datetime.now(timezone.utc)
    )

    print(
        f"✅ CNN 当前指数：{current_val}，日期：{current_date}，"
        f"情绪：{idx.get('rating', '未知')}"
    )

    history_records = []
    try:
        raw_history = fear_greed.get_history(last="120d")
        for item in raw_history:
            if isinstance(item, dict):
                date_value = item.get("date")
                score_value = item.get("score", item.get("value"))
            else:
                date_value = getattr(item, "date", None)
                score_value = getattr(item, "score", getattr(item, "value", None))

            if date_value is None or score_value is None:
                continue
            try:
                history_records.append({
                    "date": normalize_date(date_value),
                    "value": normalize_score(score_value)
                })
            except Exception:
                continue
    except Exception as e:
        print(f"⚠️ CNN 历史数据获取失败：{e}")

    history_records.insert(0, {"date": current_date, "value": current_val})
    history_records = dedupe_by_date(history_records)
    history_records.sort(key=lambda x: x["date"], reverse=True)
    return history_records[:70]


# ================= FearGreedChart 备用美股指数 =================
def get_feargreedchart_us_data():
    print("🔄 正在切换到 FearGreedChart API...")
    data = request_json(US_FALLBACK_URL)

    score_info = data.get("score")
    if not isinstance(score_info, dict) or "score" not in score_info:
        raise ValueError("FearGreedChart 返回数据缺少 score.score")

    current_val = normalize_score(score_info["score"])
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if data.get("ts") is not None:
        try:
            current_date = datetime.fromtimestamp(
                float(data["ts"]) / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d")
        except (ValueError, TypeError, OSError):
            pass

    def parse_history_items(items):
        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            date_value = item.get("date", item.get("timestamp"))
            score_value = item.get("score", item.get("value"))
            if date_value is None or score_value is None:
                continue
            try:
                if isinstance(date_value, (int, float)):
                    ts = float(date_value)
                    if ts > 1e12:
                        ts /= 1000
                    date_value = datetime.fromtimestamp(ts, tz=timezone.utc)
                records.append({
                    "date": normalize_date(date_value),
                    "value": normalize_score(score_value)
                })
            except Exception:
                continue
        return records

    raw_history = data.get("recent", data.get("history", []))
    if not isinstance(raw_history, list):
        raw_history = []
    history_records = parse_history_items(raw_history)

    if not history_records:
        try:
            history_data = request_json(US_HISTORY_URL)
            if isinstance(history_data, list):
                raw_history = history_data
            elif isinstance(history_data, dict):
                raw_history = history_data.get(
                    "data", history_data.get("history", history_data.get("recent", []))
                )
            else:
                raw_history = []
            history_records = parse_history_items(
                raw_history if isinstance(raw_history, list) else []
            )
        except Exception as e:
            print(f"⚠️ FearGreedChart 历史接口获取失败：{e}")

    history_records.insert(0, {"date": current_date, "value": current_val})
    history_records = dedupe_by_date(history_records)
    history_records.sort(key=lambda x: x["date"], reverse=True)
    print(f"✅ FearGreedChart 当前指数：{current_val}，日期：{current_date}")
    return history_records[:70]


def get_us_data():
    try:
        data = get_cnn_us_data()
        print("✅ 美股数据来源：CNN Fear & Greed Index")
        return data, "CNN Fear & Greed Index"
    except Exception as e:
        print(f"⚠️ CNN 贪恐指数获取失败：{e}")

    try:
        data = get_feargreedchart_us_data()
        print("✅ 美股数据来源：FearGreedChart API")
        return data, "FearGreedChart API"
    except Exception as e:
        print(f"❌ FearGreedChart 获取失败：{e}")

    print("❌ CNN 与 FearGreedChart 均获取失败")
    return None, "获取失败"


# ================= 比特币贪恐指数 =================
def get_crypto_data():
    try:
        res = requests.get(
            CRYPTO_URL, headers=REQUEST_HEADERS, timeout=15
        )
        res.raise_for_status()
        raw = res.json()["data"]

        data = [{
            "date": datetime.fromtimestamp(
                int(d["timestamp"]), tz=timezone.utc
            ).strftime("%Y-%m-%d"),
            "value": normalize_score(d["value"])
        } for d in raw]

        data = dedupe_by_date(data)
        data.sort(key=lambda x: x["date"], reverse=True)
        return data, "Alternative.me"
    except Exception as e:
        print(f"Crypto API Failed: {e}")
        return None, "获取失败"


# ================= A股自建贪恐指数 =================
def get_cn_stock_spot():
    """获取全市场A股行情；东方财富失败时尝试新浪。"""
    errors = []
    providers = [
        ("东方财富", lambda: ak.stock_zh_a_spot_em()),
        ("新浪", lambda: ak.stock_zh_a_spot()),
    ]

    for source, fetcher in providers:
        try:
            print(f"🔄 正在获取 A 股全市场行情：{source}")
            df = fetcher()
            if df is None or df.empty:
                raise ValueError("行情数据为空")

            if "涨跌幅" not in df.columns:
                raise ValueError(f"缺少涨跌幅字段，字段为：{list(df.columns)}")

            df = df.copy()
            df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
            if "成交额" in df.columns:
                df["成交额"] = pd.to_numeric(df["成交额"], errors="coerce")
            df = df.dropna(subset=["涨跌幅"])
            df = df[df["涨跌幅"].between(-30, 30)].copy()

            # 排除常见指数/非个股行（若名称字段存在）
            if "名称" in df.columns:
                df = df[~df["名称"].astype(str).str.contains(
                    r"指数|板块|债券|基金|ETF", regex=True, na=False
                )]

            if len(df) < 100:
                raise ValueError(f"有效股票数量过少：{len(df)}")

            print(f"✅ A 股行情获取成功：{source}，有效股票 {len(df)} 只")
            return df, source
        except Exception as e:
            errors.append(f"{source}: {e}")
            print(f"⚠️ {source} 行情获取失败：{e}")

    raise RuntimeError("所有 A 股行情源均失败；" + "；".join(errors))


def get_cn_index_history():
    """获取上证指数日线；成交额缺失时将成交活跃度按中性处理。"""
    errors = []
    providers = [
        ("东方财富", lambda: ak.stock_zh_index_daily_em(symbol="sh000001")),
        ("腾讯", lambda: ak.stock_zh_index_daily_tx(symbol="sh000001")),
    ]

    end_dt = pd.Timestamp.now().normalize()
    start_dt = end_dt - pd.Timedelta(days=CN_HISTORY_DAYS * 2)

    for source, fetcher in providers:
        try:
            print(f"🔄 正在获取上证指数历史行情：{source}")
            df = fetcher()
            if df is None or df.empty:
                raise ValueError("指数历史行情为空")

            df = df.copy()
            date_col = "date" if "date" in df.columns else (
                "日期" if "日期" in df.columns else None
            )
            close_col = "close" if "close" in df.columns else (
                "收盘" if "收盘" in df.columns else None
            )
            if date_col is None or close_col is None:
                raise ValueError(f"缺少日期或收盘价字段：{list(df.columns)}")

            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
            df["close"] = pd.to_numeric(df[close_col], errors="coerce")

            if "amount" in df.columns:
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            elif "成交额" in df.columns:
                df["amount"] = pd.to_numeric(df["成交额"], errors="coerce")
            else:
                df["amount"] = np.nan

            df = df.dropna(subset=["date", "close"])
            df = df[df["close"] > 0]
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
            df = df.drop_duplicates(subset=["date"], keep="last")
            df = df.sort_values("date").reset_index(drop=True)

            if len(df) < 25:
                raise ValueError(f"有效指数历史数据不足：{len(df)} 条")

            print(f"✅ 上证指数历史行情获取成功：{source}，{len(df)} 个交易日")
            return df, source
        except Exception as e:
            errors.append(f"{source}: {e}")
            print(f"⚠️ {source} 指数行情获取失败：{e}")

    raise RuntimeError("上证指数历史行情获取失败；" + "；".join(errors))


def calculate_cn_current_factors(stock_df, index_df):
    """计算当前A股贪恐分数及分项。"""
    idx = index_df.copy()
    idx["return_5"] = idx["close"].pct_change(5) * 100
    idx["return_20"] = idx["close"].pct_change(20) * 100

    latest = idx.iloc[-1]
    ret5 = float(latest["return_5"])
    ret20 = float(latest["return_20"])
    trend_raw = ret5 * 0.4 + ret20 * 0.6
    trend_score = linear_score(trend_raw, -12, 12)

    changes = stock_df["涨跌幅"].dropna()
    total = len(changes)
    if total == 0:
        raise ValueError("没有有效的个股涨跌幅")

    advancers = int((changes > 0).sum())
    decliners = int((changes < 0).sum())
    unchanged = int((changes == 0).sum())
    advance_ratio = advancers / total
    breadth_score = linear_score(advance_ratio, 0.20, 0.80)

    median_return = float(changes.median())
    strong_ratio = float((changes >= 5).sum()) / total
    weak_ratio = float((changes <= -5).sum()) / total
    money_score = (
        linear_score(median_return, -5, 5) * 0.65
        + linear_score(strong_ratio - weak_ratio, -0.20, 0.20) * 0.35
    )

    idx["daily_return"] = idx["close"].pct_change() * 100
    recent_returns = idx["daily_return"].dropna().iloc[-20:]
    volatility = float(recent_returns.std(ddof=1))
    volatility_score = 100 - linear_score(volatility, 0.5, 3.0)

    amount = idx["amount"].dropna()
    amount_ratio = np.nan
    turnover_score = 50.0
    if len(amount) >= 21:
        # 仅在最后21个有效成交额连续对应交易日时使用；否则不猜测
        amount_all = idx["amount"]
        if amount_all.iloc[-21:].notna().all():
            avg_amount = float(amount_all.iloc[-21:-1].mean())
            last_amount = float(amount_all.iloc[-1])
            if avg_amount > 0:
                amount_ratio = last_amount / avg_amount
                turnover_score = linear_score(amount_ratio, 0.5, 1.8)

    factors = {
        "trend": trend_score,
        "breadth": breadth_score,
        "money": clamp(money_score),
        "volatility": volatility_score,
        "turnover": turnover_score,
    }
    score = normalize_score(sum(factors[k] * CN_WEIGHTS[k] for k in CN_WEIGHTS))

    return {
        "score": score,
        "factors": factors,
        "date": pd.to_datetime(latest["date"]).strftime("%Y-%m-%d"),
        "stock_count": total,
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": unchanged,
        "advance_ratio": advance_ratio,
        "median_return": median_return,
        "volatility": volatility,
        "amount_ratio": amount_ratio,
    }


def build_cn_history_record(result):
    """生成当天A股情绪指数记录，用于跨运行保存。"""
    return {
        "date": result["date"],
        "value": int(result["score"]),
        "factors": {k: round(float(v), 2) for k, v in result["factors"].items()},
        "stock_count": int(result["stock_count"]),
        "advancers": int(result["advancers"]),
        "decliners": int(result["decliners"]),
        "unchanged": int(result["unchanged"]),
        "method": "five_factor_composite",
    }


def _parse_jiucai_date(value):
    """解析韭圈儿接口中的日期字段，兼容日期字符串和 Unix 时间戳。"""
    if value is None:
        raise ValueError("韭圈儿记录缺少日期")
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
        number = float(value)
        # 13 位通常为毫秒时间戳，10 位通常为秒时间戳。
        if number > 1e12:
            number /= 1000.0
        if number > 1e9:
            return datetime.fromtimestamp(number, tz=timezone.utc).strftime("%Y-%m-%d")
    return normalize_date(value)


def _decode_jiucai_response(response_text):
    """解码韭圈儿历史接口：优先解析明文 JSON，必要时按 AES-256-CBC 解密。"""
    text = (response_text or "").strip()
    if not text:
        raise ValueError("韭圈儿接口返回空内容")

    # 先尝试直接 JSON；部分接口版本可能返回明文 JSON。
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (list, dict)):
            return parsed
        if isinstance(parsed, str):
            text = parsed.strip()
    except Exception:
        pass

    if AES is None or unpad is None:
        raise RuntimeError("缺少 pycryptodome 依赖，请在 requirements.txt 中添加 pycryptodome")

    # 当前公开资料记录的韭圈儿网页端 AES 参数；网站改版时可能变化。
    key = ("bieyanjiulexixishuibatoufamei" + "ll1").encode("utf-8")[:32]
    iv = ("nengnongchulainb" + "ll1").encode("utf-8")[:16]
    encrypted = base64.b64decode(text)
    decrypted = AES.new(key, AES.MODE_CBC, iv).decrypt(encrypted)
    decrypted = unpad(decrypted, AES.block_size)
    return json.loads(decrypted.decode("utf-8", errors="replace"))


def fetch_jiucai_fear_greed():
    """获取韭圈儿恐贪指数及历史记录，返回按日期从新到旧排序的标准记录。"""
    print("🇨🇳 正在尝试获取韭圈儿官方展示的恐贪指数...")
    headers = {
        **REQUEST_HEADERS,
        "Referer": JIUCAI_REFERER,
        "Origin": "https://app.jiucaishuo.com",
        "Accept": "application/json, text/plain, */*",
    }
    response = requests.get(JIUCAI_FG_API, headers=headers, timeout=25)
    response.raise_for_status()
    payload = _decode_jiucai_response(response.text)

    # 兼容接口可能出现的列表、data/history/lines 包装结构。
    records_raw = payload
    if isinstance(payload, dict):
        for key in ("data", "history", "lines", "result", "list"):
            if isinstance(payload.get(key), list):
                records_raw = payload[key]
                break
        else:
            raise ValueError(f"无法识别韭圈儿接口数据结构，字段：{list(payload.keys())[:20]}")

    if not isinstance(records_raw, list) or not records_raw:
        raise ValueError("韭圈儿接口没有返回有效历史记录")

    records = []
    for item in records_raw:
        if not isinstance(item, dict):
            continue
        date_value = item.get("date", item.get("trade_date", item.get("timestamp", item.get("time"))))
        score_value = item.get("value", item.get("score", item.get("fear_greed", item.get("fearGreed"))))
        if date_value is None or score_value is None:
            continue
        try:
            records.append({
                "date": _parse_jiucai_date(date_value),
                "value": normalize_score(score_value),
                "method": "jiucai_official",
            })
        except Exception:
            continue

    if not records:
        sample = str(records_raw[0])[:300]
        raise ValueError(f"韭圈儿数据中未找到可识别的日期/分数，样例：{sample}")

    records = dedupe_by_date(sorted(records, key=lambda x: x["date"], reverse=True))
    print(f"✅ 韭圈儿恐贪指数获取成功：最新值 {records[0]['value']}，日期 {records[0]['date']}，历史 {len(records)} 条")
    return records


def _find_col(df, candidates):
    """从数据表中按候选名称寻找字段。"""
    for name in candidates:
        if name in df.columns:
            return name
    # 兼容字段名中带单位/空格的情况
    for col in df.columns:
        text = str(col).replace(" ", "")
        for name in candidates:
            if name.replace(" ", "") in text:
                return col
    return None


def _score_by_history(value, history, inverse=False, min_n=30):
    """按历史均值和标准差评分：均值附近为50，偏离越大分数越极端。"""
    vals = pd.to_numeric(pd.Series(history), errors="coerce").dropna()
    if pd.isna(value):
        return None
    if len(vals) >= min_n:
        mean = float(vals.mean())
        std = float(vals.std(ddof=1))
        if std > 1e-9:
            z = (float(value) - mean) / std
            score = clamp(50 + 20 * z)
            return round(100 - score if inverse else score, 2)
    # 历史样本尚不足时，采用中性50；不伪造历史标准差
    return 50.0


def _fetch_index_close(symbol):
    """获取指定指数日线，统一返回 date/close 两列。"""
    errors = []
    providers = [
        ("东方财富", lambda: ak.stock_zh_index_daily_em(symbol=symbol)),
        ("腾讯", lambda: ak.stock_zh_index_daily_tx(symbol=symbol)),
        ("新浪", lambda: ak.stock_zh_index_daily(symbol=symbol)),
    ]
    for name, fetcher in providers:
        try:
            df = fetcher()
            if df is None or df.empty:
                raise ValueError("返回空数据")
            date_col = _find_col(df, ["date", "日期"])
            close_col = _find_col(df, ["close", "收盘", "收盘价"])
            if date_col is None or close_col is None:
                raise ValueError(f"缺少日期/收盘字段：{list(df.columns)}")
            out = pd.DataFrame({
                "date": pd.to_datetime(df[date_col], errors="coerce"),
                "close": pd.to_numeric(df[close_col], errors="coerce"),
            }).dropna()
            out = out[out["close"] > 0].drop_duplicates("date").sort_values("date")
            if len(out) >= 30:
                print(f"✅ 指数 {symbol} 历史行情获取成功：{name}，{len(out)} 条")
                return out.reset_index(drop=True)
            raise ValueError(f"有效历史数据不足：{len(out)} 条")
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError(f"指数 {symbol} 行情获取失败：" + "；".join(errors))


def _get_qvix_factor():
    """因子1：50ETF期权波动率。波动越高，情绪分越低。"""
    df = ak.index_option_50etf_qvix()
    if df is None or df.empty:
        raise ValueError("QVIX 数据为空")
    date_col = _find_col(df, ["date", "日期"])
    value_col = _find_col(df, ["close", "收盘", "收盘价"])
    if date_col is None or value_col is None:
        raise ValueError(f"QVIX 缺少字段：{list(df.columns)}")
    data = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "raw": pd.to_numeric(df[value_col], errors="coerce"),
    }).dropna().sort_values("date")
    if len(data) < 30:
        raise ValueError("QVIX 有效历史数据不足")
    current = float(data.iloc[-1]["raw"])
    score = _score_by_history(current, data["raw"].iloc[-252:], inverse=True)
    return score, current, data["date"].iloc[-1].strftime("%Y-%m-%d")


def _get_northbound_factor():
    """因子2：陆股通累计净买入偏离20日均线；数据源可能因披露规则变化而不可用。"""
    errors = []
    attempts = [
        lambda: ak.stock_hsgt_hist_em(symbol="北向资金"),
        lambda: ak.stock_hsgt_hist_em(symbol="沪股通"),
        lambda: ak.stock_hsgt_fund_flow_summary_em(),
    ]
    for fetcher in attempts:
        try:
            df = fetcher()
            if df is None or df.empty:
                raise ValueError("返回空数据")
            date_col = _find_col(df, ["日期", "date", "交易日期"])
            cumulative_col = _find_col(df, ["累计净买入", "历史累计净买额", "累计买入成交净额", "累计净额"])
            daily_col = _find_col(df, ["当日成交净买额", "当日净买入", "净买入", "资金净流入", "净流入"])
            if date_col is None:
                raise ValueError(f"缺少日期字段：{list(df.columns)}")
            df = df.copy()
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
            value_col = cumulative_col or daily_col
            if value_col is None:
                raise ValueError(f"缺少净买入字段：{list(df.columns)}")
            df["raw_value"] = pd.to_numeric(
                df[value_col].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )
            df = df.dropna(subset=["date", "raw_value"]).sort_values("date")
            if len(df) < 25:
                raise ValueError(f"有效陆股通数据不足：{len(df)} 条")
            # 如果只有每日净买入额，则累加为累计净买入序列
            series = df["raw_value"] if cumulative_col else df["raw_value"].cumsum()
            deviation = series - series.rolling(20, min_periods=20).mean()
            valid = deviation.dropna()
            if len(valid) < 20:
                raise ValueError("陆股通累计净买入偏离值不足")
            score = _score_by_history(float(valid.iloc[-1]), valid.iloc[-252:])
            return score, float(series.iloc[-1]), float(deviation.iloc[-1])
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError("陆股通数据不可用：" + "；".join(errors[-3:]))


def _get_turnover_factor(stock_df, prior_records):
    """因子3：两市成交额相对20日均值的偏离。新模型积累满20天后可形成自身基准。"""
    amount_col = _find_col(stock_df, ["成交额", "成交金额"])
    if amount_col is None:
        raise ValueError("全市场行情缺少成交额字段")
    amounts = pd.to_numeric(stock_df[amount_col], errors="coerce")
    current_amount = float(amounts.dropna().sum())
    if current_amount <= 0:
        raise ValueError("两市成交额无效")
    old_amounts = []
    for item in sorted(prior_records, key=lambda x: x.get("date", "")):
        raw = item.get("raw", {})
        try:
            val = float(raw.get("total_turnover"))
            if np.isfinite(val) and val > 0:
                old_amounts.append(val)
        except (TypeError, ValueError):
            pass
    if len(old_amounts) >= 20:
        last20 = old_amounts[-20:]
        avg = float(np.mean(last20))
        ratio = current_amount / avg if avg else np.nan
        ratios = []
        combined = old_amounts[-80:] + [current_amount]
        for i in range(20, len(combined)):
            base = float(np.mean(combined[i-20:i]))
            if base > 0:
                ratios.append(combined[i] / base)
        score = _score_by_history(ratio, ratios[-252:])
        return score, current_amount, ratio
    raise RuntimeError(f"成交额历史样本不足（已积累{len(old_amounts)}天，需要至少20天）")


def _get_futures_basis_factor(index_df, prior_records):
    """因子4：沪深300股指期货次月年化升贴水率。"""
    today = pd.Timestamp.now().normalize()
    months = []
    for offset in range(1, 4):
        month = today + pd.DateOffset(months=offset)
        code = f"IF{month.strftime('%y%m')}"
        months.append(code)
    errors = []
    futures_df = None
    for code in months:
        try:
            tmp = ak.futures_zh_spot(symbol=code, market="FF", adjust="0")
            if tmp is not None and not tmp.empty:
                futures_df = tmp
                print(f"📈 使用沪深300股指期货合约：{code}")
                break
        except Exception as e:
            errors.append(f"{code}: {e}")
    if futures_df is None:
        raise RuntimeError("未获取到可用IF次月合约：" + "；".join(errors))
    price_col = _find_col(futures_df, ["current_price", "trade", "最新价", "现价"])
    if price_col is None:
        raise ValueError(f"期货数据缺少最新价：{list(futures_df.columns)}")
    futures_price = float(pd.to_numeric(futures_df[price_col], errors="coerce").dropna().iloc[0])
    hs300_df = _fetch_index_close("sh000300")
    spot_price = float(hs300_df["close"].iloc[-1])
    if futures_price <= 0 or spot_price <= 0:
        raise ValueError("期货或沪深300现货价格无效")
    contract_code = str(futures_df.iloc[0].get("symbol", months[0]))
    digits = "".join(ch for ch in contract_code if ch.isdigit())[-4:]
    try:
        yy, mm = int(digits[:2]), int(digits[2:4])
        expiry_month = pd.Timestamp(year=2000 + yy, month=mm, day=1)
        # 中金所股指期货到期日为合约月第三个星期五
        first_day_weekday = expiry_month.weekday()
        first_friday = 1 + ((4 - first_day_weekday) % 7)
        expiry = expiry_month + pd.Timedelta(days=first_friday + 13)
        days_left = max(1, (expiry.normalize() - today).days)
    except Exception:
        days_left = 30
    basis = (futures_price / spot_price - 1.0) * 365.0 / days_left * 100.0
    old_basis = []
    for item in prior_records:
        try:
            old_basis.append(float(item.get("raw", {}).get("annualized_basis")))
        except (TypeError, ValueError):
            pass
    if len(old_basis) >= 20:
        score = _score_by_history(basis, old_basis[-252:])
    else:
        # 尚未积累足够历史时，用固定的-10%至+10%区间作为启动期映射
        score = linear_score(basis, -10, 10)
    return score, basis, futures_price, spot_price


def _get_high_strength_factor(stock_df, prior_records):
    """因子5：创过去一年新高股票占比。优先用同花顺创新高名单。"""
    total = len(stock_df)
    if total <= 0:
        raise ValueError("全市场股票数量无效")
    errors = []
    high_count = None
    for name in ("stock_rank_cxg_ths", "stock_a_high_low_statistics"):
        try:
            func = getattr(ak, name, None)
            if func is None:
                continue
            if name == "stock_rank_cxg_ths":
                try:
                    df = func(symbol="创一年新高")
                except TypeError:
                    df = func()
            else:
                df = func()
            if df is not None and not df.empty:
                # 创新高榜单通常每行一只股票；统计表则尝试识别创新高数量字段
                count_col = _find_col(df, ["创新高", "创一年新高", "新高家数", "一年新高"])
                if count_col and len(df) == 1:
                    high_count = int(pd.to_numeric(df[count_col], errors="coerce").fillna(0).iloc[0])
                else:
                    high_count = len(df)
                break
        except Exception as e:
            errors.append(f"{name}: {e}")
    if high_count is None:
        raise RuntimeError("未能获取一年新高股票数量：" + "；".join(errors))
    ratio = max(0.0, high_count / total)
    old_ratios = []
    for item in prior_records:
        try:
            old_ratios.append(float(item.get("raw", {}).get("high_ratio")))
        except (TypeError, ValueError):
            pass
    if len(old_ratios) >= 20:
        score = _score_by_history(ratio, old_ratios[-252:])
    else:
        # 启动期：0%新高为恐惧，5%及以上为强贪婪
        score = linear_score(ratio, 0, 0.05)
    return score, high_count, ratio


def _get_hedge_demand_factor():
    """因子6：过去20个交易日沪深300收益率减国债指数收益率。国债指数是公开数据可得的近似代理。"""
    equity = _fetch_index_close("sh000300")
    bond = _fetch_index_close("sh000012")
    merged = equity.merge(bond, on="date", suffixes=("_equity", "_bond")).sort_values("date")
    if len(merged) < 252:
        raise ValueError(f"股债指数重叠历史不足：{len(merged)} 条")
    merged["equity_ret20"] = merged["close_equity"].pct_change(20) * 100
    merged["bond_ret20"] = merged["close_bond"].pct_change(20) * 100
    merged["spread"] = merged["equity_ret20"] - merged["bond_ret20"]
    valid = merged["spread"].dropna()
    if valid.empty:
        raise ValueError("股债收益差无法计算")
    spread = float(valid.iloc[-1])
    score = _score_by_history(spread, valid.iloc[-252:])
    return score, spread, merged["date"].iloc[-1].strftime("%Y-%m-%d")


def _extract_financing_purchase_amount(df, expected_date):
    """从交易所融资融券汇总表中提取指定交易日的融资买入额，统一换算为元。"""
    if df is None or df.empty:
        raise ValueError("融资融券汇总数据为空")
    amount_col = _find_col(df, ["融资买入额", "融资买入金额"])
    if amount_col is None:
        raise ValueError(f"融资融券表缺少融资买入额字段：{list(df.columns)}")
    date_col = _find_col(df, ["日期", "交易日期", "信用交易日期", "date"])
    work = df.copy()
    work["_amount"] = pd.to_numeric(
        work[amount_col].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    if date_col is not None:
        work["_date"] = pd.to_datetime(work[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
        target = pd.Timestamp(expected_date).strftime("%Y-%m-%d")
        exact = work[work["_date"] == target]
        if not exact.empty:
            work = exact
        else:
            valid_dates = work.dropna(subset=["_date"])
            valid_dates = valid_dates[valid_dates["_date"] <= target]
            if valid_dates.empty:
                raise ValueError(f"融资融券数据中没有 {target} 或之前的记录")
            latest_date = valid_dates["_date"].max()
            if latest_date != target:
                raise ValueError(f"融资融券数据最新日期为 {latest_date}，与目标日期 {target} 不一致")
            work = valid_dates[valid_dates["_date"] == latest_date]
    values = work["_amount"].dropna()
    if values.empty:
        raise ValueError("融资买入额没有有效数值")
    amount = float(values.iloc[-1])
    if not np.isfinite(amount) or amount <= 0:
        raise ValueError(f"融资买入额无效：{amount}")
    # 交易所网页部分表格以“万元”为单位，AKShare不同版本的输出单位可能不同。
    # 总融资买入额以元计通常远高于1亿元；小于1亿元的原始数值按万元转换。
    if amount < 1e8:
        amount *= 1e4
    return amount


def _fetch_exchange_financing_amount(exchange, target_date):
    """尝试从上交所/深交所公开融资融券汇总接口获取单日融资买入额。"""
    date_text = pd.Timestamp(target_date).strftime("%Y%m%d")
    if exchange == "SSE":
        fn = getattr(ak, "stock_margin_sse", None)
        if fn is None:
            raise RuntimeError("当前 AKShare 版本没有 stock_margin_sse 接口")
        attempts = [
            lambda: fn(symbol="信用交易汇总", start_date=date_text, end_date=date_text),
            lambda: fn(symbol="信用交易汇总", start_date=(pd.Timestamp(target_date)-pd.Timedelta(days=10)).strftime("%Y%m%d"), end_date=date_text),
            lambda: fn(start_date=(pd.Timestamp(target_date)-pd.Timedelta(days=10)).strftime("%Y%m%d"), end_date=date_text),
        ]
    elif exchange == "SZSE":
        fn = getattr(ak, "stock_margin_szse", None)
        if fn is None:
            raise RuntimeError("当前 AKShare 版本没有 stock_margin_szse 接口")
        attempts = [
            lambda: fn(date=date_text),
            lambda: fn(start_date=date_text, end_date=date_text),
            lambda: fn(),
        ]
    else:
        raise ValueError(f"未知交易所：{exchange}")

    errors = []
    for call in attempts:
        try:
            df = call()
            amount = _extract_financing_purchase_amount(df, target_date)
            print(f" 杠杆率数据：{exchange} 融资买入额={amount:.0f}元，日期={pd.Timestamp(target_date).strftime('%Y-%m-%d')}")
            return amount
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError(f"{exchange} 融资买入额获取失败：{'；'.join(errors[-3:])}")


def _get_leverage_factor(stock_df, prior_records, target_date):
    """可选因子7：沪深交易所融资买入额之和 / A股全市场成交额。"""
    amount_col = _find_col(stock_df, ["成交额", "成交金额"])
    if amount_col is None:
        raise ValueError("全市场行情缺少成交额字段，无法计算融资买入占比")
    amounts = pd.to_numeric(stock_df[amount_col], errors="coerce").dropna()
    total_turnover = float(amounts.sum())
    if not np.isfinite(total_turnover) or total_turnover <= 0:
        raise ValueError("A股全市场成交额无效")

    # 当天若为非交易日或数据尚未更新，向前寻找最近一个双方均有数据的交易日。
    errors = []
    financing_total = None
    actual_date = None
    for offset in range(0, 11):
        day = (pd.Timestamp(target_date) - pd.Timedelta(days=offset)).normalize()
        try:
            sse_amount = _fetch_exchange_financing_amount("SSE", day)
            szse_amount = _fetch_exchange_financing_amount("SZSE", day)
            financing_total = sse_amount + szse_amount
            actual_date = day.strftime("%Y-%m-%d")
            break
        except Exception as exc:
            errors.append(f"{day.strftime('%Y-%m-%d')}: {exc}")
    if financing_total is None:
        raise RuntimeError("未能同时获取沪深两市融资买入额：" + "；".join(errors[-3:]))

    # 两市融资买入额 / 两市股票成交额；异常超过100%通常意味着单位或数据口径错误。
    ratio = financing_total / total_turnover
    if not np.isfinite(ratio) or ratio <= 0 or ratio > 1:
        raise ValueError(
            f"融资买入占成交额比例异常（{ratio:.4f}）；可能存在数据单位或日期口径不一致，跳过杠杆率因子"
        )

    old_ratios = []
    for item in sorted(prior_records, key=lambda x: x.get("date", "")):
        try:
            old_ratio = float(item.get("raw", {}).get("financing_purchase_ratio"))
            if np.isfinite(old_ratio) and 0 < old_ratio <= 1:
                old_ratios.append(old_ratio)
        except (TypeError, ValueError):
            continue
    score = _score_by_history(ratio, old_ratios[-252:])
    return score, ratio, financing_total, total_turnover, actual_date


def get_cn_data():
    """按韭圈儿公开描述的六个非杠杆因子，等权计算A股情绪指数。 注意：这是一套参考其公开逻辑的自建模型，并非韭圈儿官方算法复刻。 个别数据源可能不可用；当前版本对可获取因子等权平均，并在日志中列明缺失项。 """
    global CN_LAST_FACTOR_DETAIL
    CN_LAST_FACTOR_DETAIL = {}
    print("🇨🇳 开始计算 A 股情绪指数（六个基础因子 + 可选杠杆率）...")
    prior_all = load_cn_saved_history()
    prior_records = []
    for item in prior_all:
        if item.get("method") not in ("six_factor_equal_weight", CN_CURRENT_METHOD):
            continue
        try:
            prior_records.append({
                "date": normalize_date(item["date"]),
                "value": normalize_score(item["value"]),
                "factors": item.get("factors", {}),
                "raw": item.get("raw", {}),
                "method": item.get("method", "six_factor_equal_weight"),
            })
        except Exception:
            continue
    prior_records = sorted(prior_records, key=lambda x: x["date"])

    # 获取全市场行情和沪深300历史行情
    stock_df, stock_source = get_cn_stock_spot()
    index_df, index_source = get_cn_index_history()
    current_date = pd.to_datetime(index_df["date"].iloc[-1]).strftime("%Y-%m-%d")

    factor_scores = {}
    raw_values = {}
    missing = []

    # 因子1：50ETF期权波动率（反向）
    try:
        score, raw, source_date = _get_qvix_factor()
        factor_scores["market_volatility"] = score
        raw_values["qvix"] = raw
        print(f" 因子1 市场波动率：QVIX={raw:.2f}，恐贪分={score:.1f}（高波动对应低分）")
    except Exception as e:
        missing.append(f"市场波动率: {e}")

    # 因子2：陆股通累计净买入偏离20日均线
    try:
        score, cumulative, deviation = _get_northbound_factor()
        factor_scores["northbound_flow"] = score
        raw_values["northbound_cumulative"] = cumulative
        raw_values["northbound_deviation"] = deviation
        print(f" 因子2 外资认可度：累计净买入={cumulative:.2f}，偏离20日均线={deviation:.2f}，分数={score:.1f}")
    except Exception as e:
        missing.append(f"外资认可度: {e}")

    # 因子3：两市成交额偏离20日均值
    try:
        score, total_turnover, ratio = _get_turnover_factor(stock_df, prior_records)
        factor_scores["turnover"] = score
        raw_values["total_turnover"] = total_turnover
        raw_values["turnover_ratio"] = ratio
        print(f" 因子3 两市成交额：总成交额={total_turnover:.0f}，相对20日均值={ratio:.3f}，分数={score:.1f}")
    except Exception as e:
        # 仍保存当日总成交额，供后续20日基准逐步形成
        amount_col = _find_col(stock_df, ["成交额", "成交金额"])
        if amount_col:
            amount = float(pd.to_numeric(stock_df[amount_col], errors="coerce").dropna().sum())
            raw_values["total_turnover"] = amount
        missing.append(f"两市成交额: {e}")

    # 因子4：沪深300股指期货次月年化升贴水率
    try:
        score, basis, future_price, spot_price = _get_futures_basis_factor(index_df, prior_records)
        factor_scores["futures_basis"] = score
        raw_values["annualized_basis"] = basis
        print(f" 因子4 股指期货升贴水：年化={basis:.2f}%，期货={future_price:.2f}，现货={spot_price:.2f}，分数={score:.1f}")
    except Exception as e:
        missing.append(f"股指期货升贴水: {e}")

    # 因子5：过去一年新高股票占比
    try:
        score, high_count, high_ratio = _get_high_strength_factor(stock_df, prior_records)
        factor_scores["price_strength"] = score
        raw_values["high_count"] = high_count
        raw_values["high_ratio"] = high_ratio
        print(f" 因子5 股价强度：创新高股票={high_count}只，占比={high_ratio:.3%}，分数={score:.1f}")
    except Exception as e:
        missing.append(f"股价强度: {e}")

    # 因子6：股债收益差（沪深300减国债指数）
    try:
        score, spread, spread_date = _get_hedge_demand_factor()
        factor_scores["hedge_demand"] = score
        raw_values["equity_bond_spread_20d"] = spread
        print(f" 因子6 避险需求：20日股债收益差={spread:.2f}个百分点，分数={score:.1f}，日期={spread_date}")
    except Exception as e:
        missing.append(f"避险需求: {e}")

    # 因子7（可选）：融资买入额 / A股全市场成交额。两市数据必须同时有效才纳入。
    try:
        score, leverage_ratio, financing_total, turnover_total, leverage_date = _get_leverage_factor(
            stock_df, prior_records, current_date
        )
        factor_scores["leverage_ratio"] = score
        raw_values["financing_purchase_ratio"] = leverage_ratio
        raw_values["financing_purchase_amount"] = financing_total
        raw_values["leverage_turnover_denominator"] = turnover_total
        raw_values["leverage_data_date"] = leverage_date
        print(
            f" 因子7 杠杆率：融资买入额/全市场成交额={leverage_ratio:.3%}，"
            f"融资买入额合计={financing_total:.0f}元，数据日期={leverage_date}，分数={score:.1f}"
        )
    except Exception as e:
        missing.append(f"杠杆率（可选）: {e}")
        print(f" 因子7 杠杆率未纳入：{e}")

    if not factor_scores:
        print("❌ 所有因子均获取失败，无法计算 A 股情绪指数。")
        for msg in missing:
            print(f" - {msg}")
        return None, "获取失败（A股情绪因子数据源不可用）"

    # 对成功获取的因子等权；缺失因子不伪造分数，但在日志和来源中明确标注。
    score = normalize_score(np.mean(list(factor_scores.values())))
    record = {
        "date": current_date,
        "value": score,
        "factors": {k: round(float(v), 2) for k, v in factor_scores.items()},
        "raw": {k: round(float(v), 6) for k, v in raw_values.items() if isinstance(v, (int, float)) and np.isfinite(v)},
        "available_factors": list(factor_scores.keys()),
        "factor_count": len(factor_scores),
        "factor_weights": {k: round(1.0 / len(factor_scores), 6) for k in factor_scores},
        "method": CN_CURRENT_METHOD,
    }

    factor_labels = {
        "market_volatility": "市场波动率（QVIX）",
        "northbound_flow": "外资认可度（陆股通资金）",
        "turnover": "成交活跃度（成交额偏离）",
        "futures_basis": "期指升贴水",
        "price_strength": "股价强度（创新高占比）",
        "hedge_demand": "避险需求（股债收益差）",
        "leverage_ratio": "杠杆率（融资买入占比）",
    }
    CN_LAST_FACTOR_DETAIL = {
        "factor_labels": factor_labels,
        "factors": {k: round(float(v), 2) for k, v in factor_scores.items()},
        "weights": record["factor_weights"],
        "missing": list(missing),
        "factor_count": len(factor_scores),
    }

    # 当天重复运行时覆盖同一天记录；只保留本模型的历史，避免混入韭圈儿官方分数或旧模型。
    by_date = {item["date"]: item for item in prior_records}
    by_date[current_date] = record
    saved = sorted(by_date.values(), key=lambda x: x["date"], reverse=True)[:365]
    save_cn_history(saved)

    print(f"✅ A股自建情绪指数：{score}/100；本次成功获取 {len(factor_scores)}/7 个因子（杠杆率为可选项），按实际可用因子等权平均。")
    print(" 实际权重：" + "，".join(f"{name}={weight:.2%}" for name, weight in record["factor_weights"].items()))
    if missing:
        print("⚠️ 以下因子本次不可用，未纳入平均值：")
        for msg in missing:
            print(f" - {msg}")
    print("📌 该指数参考韭圈儿公开的六个基础因子思路，并增加可选杠杆率因子自建，不代表韭圈儿官方指数。")

    display_records = [
        {"date": item["date"], "value": item["value"]}
        for item in saved[:CN_DISPLAY_DAYS]
    ]
    source = f"自建情绪指数（可用{len(factor_scores)}/7因子，含可选杠杆率）"
    return display_records, source


# ================= 统计计算 =================
def calc_stats(data, period_type):
    if not data:
        return None
    current = data[0]

    def count(n):
        sub = data[:n]
        low = sum(1 for x in sub if x["value"] < LIMIT_LOW)
        high = sum(1 for x in sub if x["value"] > LIMIT_HIGH)
        return low, high

    l30, h30 = count(30)
    l60, h60 = count(60)
    return {
        "date": current["date"],
        "val": current["value"],
        "status": status_text(current["value"]),
        "l30": l30,
        "h30": h30,
        "l60": l60,
        "h60": h60,
        "period_type": period_type,
    }


# ================= HTML 视图渲染 =================
def make_svg_chart(history):
    if not history:
        return "<p class='muted'>暂无趋势图数据</p>"

    data = list(reversed(history[:30]))
    w, h = 360, 130
    left, right, top, bottom = 28, 8, 10, 20
    pw, ph = w - left - right, h - top - bottom

    def x_pos(i):
        return left + (pw / (len(data) - 1) * i if len(data) > 1 else pw / 2)

    def y_pos(v):
        return top + (100 - max(0, min(100, float(v)))) / 100 * ph

    pts = [
        f"{x_pos(i):.1f},{y_pos(item['value']):.1f}"
        for i, item in enumerate(data)
    ]
    y25, y75 = y_pos(25), y_pos(75)
    start, end = data[0]["date"][5:], data[-1]["date"][5:]

    return f""" <svg viewBox='0 0 {w} {h}' width='100%' xmlns='http://www.w3.org/2000/svg'> <rect x='0' y='0' width='{w}' height='{h}' rx='8' fill='#fff'/> <line x1='{left}' y1='{top}' x2='{left}' y2='{h-bottom}' stroke='#ddd'/> <line x1='{left}' y1='{h-bottom}' x2='{w-right}' y2='{h-bottom}' stroke='#ddd'/> <line x1='{left}' y1='{y75:.1f}' x2='{w-right}' y2='{y75:.1f}' stroke='{GREED_COLOR}' stroke-dasharray='4 4'/> <line x1='{left}' y1='{y25:.1f}' x2='{w-right}' y2='{y25:.1f}' stroke='{FEAR_COLOR}' stroke-dasharray='4 4'/> <text x='4' y='{y75+3:.1f}' font-size='9' fill='#999'>75</text> <text x='4' y='{y25+3:.1f}' font-size='9' fill='#999'>25</text> <polyline points='{' '.join(pts)}' fill='none' stroke='#2563eb' stroke-width='2.2'/> <text x='{left}' y='{h-5}' font-size='9' fill='#999'>{safe(start)}</text> <text x='{w-right}' y='{h-5}' text-anchor='end' font-size='9' fill='#999'>{safe(end)}</text> </svg> """


def make_history_table(history, period_type, compact=False):
    if not history:
        return "<p class='muted'>暂无明细数据</p>"
    data = history[:30]
    if compact:
        pairs = " ｜ ".join(f"{x['date'][5:]}:{x['value']}" for x in data)
        return f"<p class='seq'>{safe(pairs)}</p>"

    rows = []
    for x in data:
        cls, st = status_class(x["value"]), status_style(x["value"])
        rows.append(
            f"<tr><td>{safe(x['date'])}</td>"
            f"<td class='{cls}' style='{st}'>{x['value']}</td>"
            f"<td class='{cls}' style='{st}'>{safe(status_text(x['value']))}</td></tr>"
        )
    return f""" <table class='hist'> <tr><th>日期</th><th>指数</th><th>状态</th></tr> {''.join(rows)} </table> <p class='note'>股票类为近30个{safe(period_type)}；比特币为近30个自然日。</p> """


def make_cn_factor_detail_html():
    """在A股卡片中显示本次分项得分、实际权重和未纳入原因。"""
    detail = CN_LAST_FACTOR_DETAIL
    if not detail:
        return ""
    rows = []
    for key, value in detail.get("factors", {}).items():
        label = detail.get("factor_labels", {}).get(key, key)
        weight = detail.get("weights", {}).get(key, 0)
        rows.append(
            f"<tr><td>{safe(label)}</td><td>{float(value):.1f}</td><td>{float(weight):.2%}</td></tr>"
        )
    missing_html = ""
    if detail.get("missing"):
        missing_items = "".join(f"<li>{safe(item)}</li>" for item in detail["missing"])
        missing_html = f"<p class='note'>未纳入因子（不参与本次平均）：</p><ul class='cn-missing'>{missing_items}</ul>"
    return f""" <details class='cn-factors'> <summary>查看本次因子得分与实际权重（{detail.get('factor_count', 0)}/7）</summary> <table class='hist'><tr><th>因子</th><th>得分</th><th>实际权重</th></tr>{''.join(rows)}</table> {missing_html} </details> """


def make_card(name, source, stats, history, link=None, compact=False):
    if not stats:
        return f"<div class='card err'>❌ {safe(name)} 数据获取失败</div>"

    v, c, pt = stats["val"], value_color(stats["val"]), stats["period_type"]
    warn = ""
    if stats["h30"] >= DANGER_DAYS_THRESHOLD:
        warn = (
            f"<div class='warn'>⚠️ "
            f"<span style='color:{GREED_COLOR};font-weight:800'>贪婪风险</span>："
            f"近30个{safe(pt)}内已有 "
            f"<span style='color:{GREED_COLOR};font-weight:800'>{stats['h30']}</span>"
            f" 次极度贪婪。</div>"
        )
    link_html = (
        f"<p><a class='btn' href='{safe(link)}'>查看详细走势</a></p>"
        if link else ""
    )

    return f""" <div class='card'> <div class='top'> <div><b>{safe(name)}</b><br><span>{safe(stats['date'])}｜{safe(source)}</span></div> <div class='num' style='color:{c}'>{v}<br><small>{safe(stats['status'])}</small></div> </div> <table class='sum'> <tr><th>周期</th><th style='color:{FEAR_COLOR};font-weight:800'>恐慌&lt;{LIMIT_LOW}</th><th style='color:{GREED_COLOR};font-weight:800'>贪婪&gt;{LIMIT_HIGH}</th></tr> <tr><td>近30个{safe(pt)}</td><td style='color:{FEAR_COLOR};font-weight:800'>{stats['l30']}</td><td style='color:{GREED_COLOR};font-weight:800'>{stats['h30']}</td></tr> <tr><td>近60个{safe(pt)}</td><td style='color:{FEAR_COLOR};font-weight:800'>{stats['l60']}</td><td style='color:{GREED_COLOR};font-weight:800'>{stats['h60']}</td></tr> </table> {warn} {make_cn_factor_detail_html() if name == "CN A股" else ""} <details> <summary>📈 查看近30个{safe(pt)}趋势图</summary> <div class='chart'>{make_svg_chart(history)}</div> {make_history_table(history, pt, compact=compact)} </details> {link_html} </div> """


def build_html(results, beijing_time_str, compact=False):
    cards = "".join(
        make_card(
            r["name"], r["source"], r["stats"], r["history"],
            r.get("link"), compact=compact
        )
        for r in results
    )

    return f""" <html><head><meta charset='utf-8'> <style> body{{margin:0;background:#f4f6f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;color:#333}} .wrap{{max-width:620px;margin:0 auto;padding:12px}} h3{{text-align:center;margin:12px 0 16px}} .card{{background:#fff;border:1px solid #eee;border-radius:12px;padding:14px;margin:0 0 14px;box-shadow:0 2px 8px rgba(0,0,0,.04)}} .top{{display:flex;justify-content:space-between;gap:10px;border-bottom:1px solid #eee;padding-bottom:9px;margin-bottom:9px}} .top span{{font-size:11px;color:#999}}.num{{font-size:28px;font-weight:800;text-align:right;line-height:1}}.num small{{font-size:11px}} table{{width:100%;border-collapse:collapse;text-align:center}}th,td{{padding:6px;border-bottom:1px solid #eee;font-size:12px}}th{{background:#f8fafc}}.sum{{background:#fafafa}}.hist{{margin-top:8px}} .fear{{color:{FEAR_COLOR}!important;font-weight:800}}.greed{{color:{GREED_COLOR}!important;font-weight:800}}.neutral{{color:{NEUTRAL_COLOR};font-weight:600}} .warn{{background:#fff1f2;color:#991b1b;border:1px solid #fecdd3;border-radius:6px;padding:8px;margin-top:8px;font-size:12px}} summary{{cursor:pointer;text-align:center;margin-top:10px;padding:8px;border:1px solid #bfdbfe;background:#eff6ff;color:#075985;border-radius:6px;font-size:13px;font-weight:700}} .cn-factors{{margin-top:8px}}.cn-missing{{font-size:11px;color:#999;padding-left:20px;line-height:1.6}} .chart{{margin-top:8px;border:1px solid #eee;border-radius:8px;overflow:hidden;background:#fff}} .btn{{display:block;text-align:center;background:#e7f5ff;color:#075985;text-decoration:none;border:1px solid #bfdbfe;border-radius:6px;padding:8px;font-weight:700;font-size:13px}} .foot{{background:#e9ecef;border-left:4px solid #2563eb;border-radius:8px;padding:12px;font-size:12px;line-height:1.7}} .note,.muted{{font-size:11px;color:#999;text-align:right}}.seq{{font-size:12px;line-height:1.8;color:#555}}.err{{background:#fee2e2}} </style></head> <body><div class='wrap'> <h3>🌍 全球核心资产情绪监控</h3> {cards} <div class='foot'> <b>📊 策略提示</b><br> <span style='color:{FEAR_COLOR};font-weight:800'>🟢 恐慌参考：指数 &lt; {LIMIT_LOW}。</span><br> <span style='color:{GREED_COLOR};font-weight:800'>🔴 贪婪参考：指数 &gt; {LIMIT_HIGH}。</span><br> ⚠️ 近30天极度贪婪 ≥ {DANGER_DAYS_THRESHOLD} 次：注意风险。<br> <span style='color:#777'>指数为情绪观察工具，不构成投资建议。A股曲线使用已保存的自建情绪指数；历史不足30个交易日时仅展示已有记录。<br> Data Updated: {safe(beijing_time_str)}</span> </div> </div></body></html> """


# ================= PushPlus 推送 =================
def send_push(title, content):
    token = os.getenv("PUSHPLUS_TOKEN")
    topic = os.getenv("PUSHPLUS_TOPIC")
    if not token:
        print("❌ 未检测到 PUSHPLUS_TOKEN，跳过推送")
        return False

    url = "http://www.pushplus.plus/send"
    payload = {
        "token": token,
        "title": title,
        "content": content,
        "template": "html",
        "channel": "wechat",
    }
    if topic:
        payload["topic"] = topic

    try:
        res = requests.post(url, json=payload, timeout=20)
        res.raise_for_status()
        result = res.json()
        if result.get("code") == 200:
            print("✅ 推送成功")
            return True
        print(f"❌ 推送失败：{result}")
        return False
    except Exception as e:
        print(f"❌ 网络异常，推送失败：{e}")
        return False


# ================= 主程序 =================
def main():
    print("🚀 开始分析全球市场情绪...")
    tasks = [
        {
            "name": "US 美股",
            "getter": get_us_data,
            "period_type": "交易日",
            "link": None,
        },
        {
            "name": "₿ 比特币",
            "getter": get_crypto_data,
            "period_type": "自然日",
            "link": None,
        },
        {
            "name": "CN A股",
            "getter": get_cn_data,
            "period_type": "交易日",
            "link": JIUQUAN_URL,
        },
    ]

    results, title_parts = [], []
    for task in tasks:
        try:
            raw_data, source = task["getter"]()
        except Exception as e:
            print(f"❌ {task['name']} 数据获取异常：{e}")
            raw_data, source = None, "获取失败"

        stats = calc_stats(raw_data, task["period_type"])
        results.append({
            "name": task["name"],
            "source": source,
            "stats": stats,
            "history": raw_data or [],
            "link": task["link"],
        })
        if stats:
            title_parts.append(f"{task['name'].split()[-1]}:{stats['val']}")

    beijing_time = datetime.now(timezone.utc) + timedelta(hours=8)
    beijing_time_str = beijing_time.strftime("%Y-%m-%d %H:%M") + "（北京时间）"

    html_content = build_html(results, beijing_time_str, compact=False)
    if len(html_content) > MAX_PUSHPLUS_LEN:
        html_content = build_html(results, beijing_time_str, compact=True)

    if len(html_content) > MAX_PUSHPLUS_LEN:
        summary_lines = []
        for r in results:
            s = r["stats"]
            if s:
                summary_lines.append(
                    f"{r['name']}：{s['val']}，{s['status']}，"
                    f"近30个{s['period_type']}恐慌{s['l30']}次，贪婪{s['h30']}次"
                )
            else:
                summary_lines.append(f"{r['name']}：数据获取失败")
        html_content = "<br>".join(summary_lines) + f"<br><br>更新时间：{safe(beijing_time_str)}"

    if title_parts:
        title = beijing_time.strftime("%m-%d") + " | " + " | ".join(title_parts)
        send_push(title, html_content)
    else:
        print("❌ 所有数据获取失败")


if __name__ == "__main__":
    main()
