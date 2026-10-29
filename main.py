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

# ================= 配置区域 =================
CRYPTO_URL = "https://api.alternative.me/fng/?limit=80"
JIUQUAN_URL = "https://funddb.cn/tool/fear"

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

# A股自建指数参数
CN_HISTORY_DAYS = 180
CN_DISPLAY_DAYS = 30
CN_HISTORY_FILE = Path(__file__).resolve().parent / "data" / "cn_sentiment_history.json"
CN_GITHUB_HISTORY_PATH = "data/cn_sentiment_history.json"
CN_WEIGHTS = {
    "trend": 0.25,
    "breadth": 0.25,
    "money": 0.20,
    "volatility": 0.15,
    "turnover": 0.15,
}


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
    """读取此前保存的五维综合指数历史；优先读取 GitHub 仓库中的持久化文件。"""
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
                    print(f"📚 已从 GitHub 读取 A 股五维历史记录：{len(records)} 条")
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
                print(f"📚 已从本地读取 A 股五维历史记录：{len(records)} 条")
                return records
    except Exception as e:
        print(f"⚠️ 读取本地 A 股历史文件失败：{e}")
    return []


def save_cn_history(records):
    """保存五维指数历史到本地，并尽可能写回 GitHub 仓库供下一次 Actions 使用。"""
    records = sorted(records, key=lambda x: x.get("date", ""), reverse=True)[:365]
    payload = json.dumps(records, ensure_ascii=False, indent=2) + "\n"

    try:
        CN_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        CN_HISTORY_FILE.write_text(payload, encoding="utf-8")
        print(f"💾 已保存本地 A 股五维历史记录：{len(records)} 条")
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
            "message": "Update daily A-share five-factor sentiment history",
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
            print("☁️ 已将 A 股五维历史记录写回 GitHub，下一次运行可继续累积")
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
    """生成一条当天真实五维综合指数记录，用于跨运行保存。"""
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


def get_cn_data():
    """计算当天 A 股五维综合指数，并保存历史；曲线只使用已保存的五维分数。"""
    print("🇨🇳 正在计算 A 股自建贪恐指数...")
    try:
        stock_df, stock_source = get_cn_stock_spot()
        index_df, index_source = get_cn_index_history()
        result = calculate_cn_current_factors(stock_df, index_df)

        history = load_cn_saved_history()
        # 兼容并过滤历史文件：只保留五维综合指数记录，排除旧版代理分数/RSI。
        valid_history = []
        for item in history:
            try:
                if item.get("method") != "five_factor_composite":
                    continue
                date = normalize_date(item["date"])
                value = normalize_score(item["value"])
                valid_history.append({**item, "date": date, "value": value})
            except Exception:
                continue

        current_record = build_cn_history_record(result)
        # 同一天重复运行时覆盖当天记录，避免重复日期。
        valid_history = [r for r in valid_history if r["date"] != current_record["date"]]
        valid_history.append(current_record)
        valid_history = dedupe_by_date(sorted(valid_history, key=lambda x: x["date"], reverse=True))
        valid_history = valid_history[:365]
        save_cn_history(valid_history)

        # 曲线和统计均基于保存的真实五维综合指数，最多展示最近30个交易日。
        records = [
            {"date": item["date"], "value": item["value"]}
            for item in valid_history[:CN_DISPLAY_DAYS]
        ]

        print(
            f"✅ A股五维综合指数：{result['score']}，日期：{result['date']}，"
            f"有效股票数：{result['stock_count']}"
        )
        print(
            f"📊 上涨 {result['advancers']} 只，下跌 {result['decliners']} 只，"
            f"平盘 {result['unchanged']} 只"
        )
        print(
            "📈 五维分数："
            f"趋势 {result['factors']['trend']:.1f}；"
            f"宽度 {result['factors']['breadth']:.1f}；"
            f"赚钱效应 {result['factors']['money']:.1f}；"
            f"波动风险 {result['factors']['volatility']:.1f}；"
            f"成交活跃度 {result['factors']['turnover']:.1f}"
        )
        print(f"📉 当前已累计 {len(records)} 条五维综合指数历史记录（最多展示最近30条）。")
        if len(records) < 30:
            print("ℹ️ 五维历史数据将从本次运行开始逐日积累；历史不足30个交易日时，只显示已保存的记录，不再用代理指标补齐。")
        source = f"A股五维综合指数（{stock_source}；{index_source}）"
        return records, source
    except Exception as e:
        print(f"❌ A股五维综合指数计算或保存失败：{e}")
        # 当天行情失败时，仍展示已经保存的五维历史；绝不把 RSI 或代理分数混入五维曲线。
        try:
            history = load_cn_saved_history()
            valid_history = []
            for item in history:
                try:
                    if item.get("method") != "five_factor_composite":
                        continue
                    valid_history.append({
                        "date": normalize_date(item["date"]),
                        "value": normalize_score(item["value"]),
                    })
                except Exception:
                    continue
            valid_history = dedupe_by_date(
                sorted(valid_history, key=lambda x: x["date"], reverse=True)
            )[:CN_DISPLAY_DAYS]
            if valid_history:
                print("⚠️ 当天更新失败，改为展示已保存的五维综合指数历史；未混入其他口径的数据。")
                return valid_history, "A股五维综合指数（当天更新失败，显示已保存历史）"
        except Exception as history_error:
            print(f"⚠️ 读取已保存五维历史失败：{history_error}")
        print("❌ 当天五维指数计算失败，且没有可用的五维历史记录；为避免口径混杂，不使用 RSI 代替历史曲线。")
        return None, "获取失败（暂无可用五维历史）"


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

    return f""" <div class='card'> <div class='top'> <div><b>{safe(name)}</b><br><span>{safe(stats['date'])}｜{safe(source)}</span></div> <div class='num' style='color:{c}'>{v}<br><small>{safe(stats['status'])}</small></div> </div> <table class='sum'> <tr><th>周期</th><th style='color:{FEAR_COLOR};font-weight:800'>恐慌&lt;{LIMIT_LOW}</th><th style='color:{GREED_COLOR};font-weight:800'>贪婪&gt;{LIMIT_HIGH}</th></tr> <tr><td>近30个{safe(pt)}</td><td style='color:{FEAR_COLOR};font-weight:800'>{stats['l30']}</td><td style='color:{GREED_COLOR};font-weight:800'>{stats['h30']}</td></tr> <tr><td>近60个{safe(pt)}</td><td style='color:{FEAR_COLOR};font-weight:800'>{stats['l60']}</td><td style='color:{GREED_COLOR};font-weight:800'>{stats['h60']}</td></tr> </table> {warn} <details> <summary>📈 查看近30个{safe(pt)}趋势图</summary> <div class='chart'>{make_svg_chart(history)}</div> {make_history_table(history, pt, compact=compact)} </details> {link_html} </div> """


def build_html(results, beijing_time_str, compact=False):
    cards = "".join(
        make_card(
            r["name"], r["source"], r["stats"], r["history"],
            r.get("link"), compact=compact
        )
        for r in results
    )

    return f""" <html><head><meta charset='utf-8'> <style> body{{margin:0;background:#f4f6f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;color:#333}} .wrap{{max-width:620px;margin:0 auto;padding:12px}} h3{{text-align:center;margin:12px 0 16px}} .card{{background:#fff;border:1px solid #eee;border-radius:12px;padding:14px;margin:0 0 14px;box-shadow:0 2px 8px rgba(0,0,0,.04)}} .top{{display:flex;justify-content:space-between;gap:10px;border-bottom:1px solid #eee;padding-bottom:9px;margin-bottom:9px}} .top span{{font-size:11px;color:#999}}.num{{font-size:28px;font-weight:800;text-align:right;line-height:1}}.num small{{font-size:11px}} table{{width:100%;border-collapse:collapse;text-align:center}}th,td{{padding:6px;border-bottom:1px solid #eee;font-size:12px}}th{{background:#f8fafc}}.sum{{background:#fafafa}}.hist{{margin-top:8px}} .fear{{color:{FEAR_COLOR}!important;font-weight:800}}.greed{{color:{GREED_COLOR}!important;font-weight:800}}.neutral{{color:{NEUTRAL_COLOR};font-weight:600}} .warn{{background:#fff1f2;color:#991b1b;border:1px solid #fecdd3;border-radius:6px;padding:8px;margin-top:8px;font-size:12px}} summary{{cursor:pointer;text-align:center;margin-top:10px;padding:8px;border:1px solid #bfdbfe;background:#eff6ff;color:#075985;border-radius:6px;font-size:13px;font-weight:700}} .chart{{margin-top:8px;border:1px solid #eee;border-radius:8px;overflow:hidden;background:#fff}} .btn{{display:block;text-align:center;background:#e7f5ff;color:#075985;text-decoration:none;border:1px solid #bfdbfe;border-radius:6px;padding:8px;font-weight:700;font-size:13px}} .foot{{background:#e9ecef;border-left:4px solid #2563eb;border-radius:8px;padding:12px;font-size:12px;line-height:1.7}} .note,.muted{{font-size:11px;color:#999;text-align:right}}.seq{{font-size:12px;line-height:1.8;color:#555}}.err{{background:#fee2e2}} </style></head> <body><div class='wrap'> <h3>🌍 全球核心资产情绪监控</h3> {cards} <div class='foot'> <b>📊 策略提示</b><br> <span style='color:{FEAR_COLOR};font-weight:800'>🟢 恐慌参考：指数 &lt; {LIMIT_LOW}。</span><br> <span style='color:{GREED_COLOR};font-weight:800'>🔴 贪婪参考：指数 &gt; {LIMIT_HIGH}。</span><br> ⚠️ 近30天极度贪婪 ≥ {DANGER_DAYS_THRESHOLD} 次：注意风险。<br> <span style='color:#777'>指数为情绪观察工具，不构成投资建议。A股曲线使用已保存的五维综合指数；历史不足30个交易日时仅展示已有记录。<br> Data Updated: {safe(beijing_time_str)}</span> </div> </div></body></html> """


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
