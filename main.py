import os
import html
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

# 备用美股贪恐指数 API
US_FALLBACK_URL = "https://feargreedchart.com/api/?action=all"
US_HISTORY_URL = "https://feargreedchart.com/api/?action=history"

LIMIT_LOW = 25
LIMIT_HIGH = 75
DANGER_DAYS_THRESHOLD = 10
MAX_PUSHPLUS_LEN = 19000

FEAR_COLOR = "#16a34a"          # 恐慌：绿色
GREED_COLOR = "#dc2626"         # 贪婪：红色
NEUTRAL_COLOR = "#555555"       # 中性：深灰


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
    """按日期去重，保留每个日期首次出现的记录。"""
    if not data:
        return data

    seen = set()
    out = []

    for item in data:
        d = item.get("date")

        if d not in seen:
            seen.add(d)
            out.append(item)

    return out


def normalize_score(value):
    """校验指数分数，确保是 0～100 之间的有效数字。"""
    score = float(value)

    if not np.isfinite(score):
        raise ValueError(f"指数分数不是有效数字：{value}")

    if not 0 <= score <= 100:
        raise ValueError(f"指数分数超出 0～100 范围：{score}")

    return int(round(score))


def normalize_date(value):
    """将日期统一转换为 YYYY-MM-DD。"""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")

    text = str(value).strip()

    # 兼容 ISO 8601 时间字符串
    if "T" in text:
        text = text.split("T")[0]

    # 兼容带时区的日期时间字符串
    if " " in text:
        text = text.split(" ")[0]

    return pd.to_datetime(text).strftime("%Y-%m-%d")


# ================= 备用 RSI 计算 =================
def calculate_rsi_history(ticker, period="8mo"):
    """
    计算指定指数的 RSI 历史数据。
    仅供 A 股原有降级逻辑使用。
    美股不再使用 RSI 代替贪恐指数。
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            progress=False,
            auto_adjust=False
        )

        if df.empty:
            return None

        close = df["Close"]

        if hasattr(close, "columns"):
            close = close.iloc[:, 0]

        delta = close.diff()

        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        records = []

        for date, value in rsi.dropna().iloc[-70:][::-1].items():
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": normalize_score(value)
            })

        return records

    except Exception as e:
        print(f"RSI Calculation Error for {ticker}: {e}")
        return None


# ================= CNN 美股贪恐指数 =================
def get_cnn_us_data():
    """
    第一优先级：CNN Fear & Greed Index。

    获取内容：
    1. CNN 当前贪恐指数；
    2. CNN 历史贪恐指数；
    3. 统一转换为原程序使用的 date/value 数据结构。

    注意：
    CNN 当前指数和历史指数均来自 CNN，
    不使用标普500 RSI 拼接历史曲线。
    """

    print("🇺🇸 正在获取 CNN Fear & Greed Index...")

    # 1. 获取 CNN 当前指数
    idx = fear_greed.get()

    if not isinstance(idx, dict):
        raise ValueError(
            f"CNN 返回数据格式异常：{type(idx).__name__}"
        )

    if "score" not in idx:
        raise ValueError("CNN 返回数据缺少 score 字段")

    current_val = normalize_score(idx["score"])

    # 优先使用 CNN 返回的时间戳确定当前日期
    if idx.get("timestamp"):
        current_date = normalize_date(idx["timestamp"])
    else:
        current_date = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d"
        )

    print(
        f"✅ CNN 当前指数：{current_val}，"
        f"日期：{current_date}，"
        f"情绪：{idx.get('rating', '未知')}"
    )

    # 2. 获取 CNN 自身的历史指数
    history_records = []

    try:
        # 获取最近约 120 个自然日的数据，
        # 通常足够覆盖约 70 个美股交易日。
        raw_history = fear_greed.get_history(last="120d")

        for item in raw_history:
            # 兼容对象和字典两种返回格式
            if isinstance(item, dict):
                date_value = item.get("date")
                score_value = item.get(
                    "score",
                    item.get("value")
                )
            else:
                date_value = getattr(item, "date", None)
                score_value = getattr(
                    item,
                    "score",
                    getattr(item, "value", None)
                )

            if date_value is None or score_value is None:
                continue

            history_records.append({
                "date": normalize_date(date_value),
                "value": normalize_score(score_value)
            })

    except Exception as e:
        # 当前指数已成功获取时，历史数据失败不影响当前值。
        print(f"⚠️ CNN 历史数据获取失败：{e}")

    # 3. 插入当前值，确保当前显示的指数与最新 API 数据一致
    history_records.insert(0, {
        "date": current_date,
        "value": current_val
    })

    # 去重，优先保留刚刚获取的当前值
    history_records = dedupe_by_date(history_records)

    # 按日期从新到旧排序
    history_records.sort(
        key=lambda x: x["date"],
        reverse=True
    )

    # 最多保留最近 70 条记录
    history_records = history_records[:70]

    if not history_records:
        raise ValueError("CNN 没有返回有效指数数据")

    return history_records


# ================= FearGreedChart 备用美股指数 =================
def get_feargreedchart_us_data():
    """
    第二优先级：FearGreedChart API。

    当前指数：
        data["score"]["score"]

    历史指数：
        优先使用 action=all 返回的 recent；
        若没有有效历史记录，则请求 action=history。

    不使用 RSI 作为备用贪恐指数。
    """

    print("🔄 正在切换到 FearGreedChart API...")

    response = requests.get(
        US_FALLBACK_URL,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    # 1. 解析当前指数
    score_info = data.get("score")

    if not isinstance(score_info, dict):
        raise ValueError(
            "FearGreedChart 返回数据缺少 score 对象"
        )

    if "score" not in score_info:
        raise ValueError(
            "FearGreedChart 返回数据缺少 score.score 字段"
        )

    current_val = normalize_score(score_info["score"])

    # 2. 获取日期
    current_date = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d"
    )

    # 优先使用服务器时间戳
    if data.get("ts") is not None:
        try:
            current_date = datetime.fromtimestamp(
                float(data["ts"]) / 1000,
                tz=timezone.utc
            ).strftime("%Y-%m-%d")
        except (ValueError, TypeError, OSError):
            pass

    print(
        f"✅ FearGreedChart 当前指数：{current_val}，"
        f"日期：{current_date}"
    )

    # 3. 解析历史数据
    history_records = []

    # 优先使用 action=all 返回的 recent 字段
    raw_history = data.get("recent")

    if not isinstance(raw_history, list):
        raw_history = []

    # 兼容可能存在的 history 字段
    if not raw_history:
        candidate = data.get("history")

        if isinstance(candidate, list):
            raw_history = candidate

    for item in raw_history:
        if not isinstance(item, dict):
            continue

        date_value = item.get(
            "date",
            item.get("timestamp")
        )

        score_value = item.get(
            "score",
            item.get("value")
        )

        if date_value is None or score_value is None:
            continue

        try:
            # 时间戳可能是秒，也可能是毫秒
            if isinstance(date_value, (int, float)):
                timestamp = float(date_value)

                if timestamp > 1e12:
                    timestamp /= 1000

                date_value = datetime.fromtimestamp(
                    timestamp,
                    tz=timezone.utc
                )

            history_records.append({
                "date": normalize_date(date_value),
                "value": normalize_score(score_value)
            })

        except (ValueError, TypeError, OverflowError, OSError):
            continue

    # 4. 如果 action=all 没有提供有效历史数据，尝试独立历史接口
    if not history_records:
        try:
            history_response = requests.get(
                US_HISTORY_URL,
                timeout=20
            )

            history_response.raise_for_status()

            history_data = history_response.json()

            # 兼容列表和字典形式的 JSON
            if isinstance(history_data, list):
                raw_history = history_data

            elif isinstance(history_data, dict):
                raw_history = history_data.get(
                    "data",
                    history_data.get(
                        "history",
                        history_data.get("recent", [])
                    )
                )

                if not isinstance(raw_history, list):
                    raw_history = []

            else:
                raw_history = []

            for item in raw_history:
                if not isinstance(item, dict):
                    continue

                date_value = item.get(
                    "date",
                    item.get("timestamp")
                )

                score_value = item.get(
                    "score",
                    item.get("value")
                )

                if date_value is None or score_value is None:
                    continue

                try:
                    if isinstance(date_value, (int, float)):
                        timestamp = float(date_value)

                        if timestamp > 1e12:
                            timestamp /= 1000

                        date_value = datetime.fromtimestamp(
                            timestamp,
                            tz=timezone.utc
                        )

                    history_records.append({
                        "date": normalize_date(date_value),
                        "value": normalize_score(score_value)
                    })

                except (
                    ValueError,
                    TypeError,
                    OverflowError,
                    OSError
                ):
                    continue

        except Exception as e:
            print(
                f"⚠️ FearGreedChart 历史接口获取失败：{e}"
            )

    # 5. 插入最新当前值
    history_records.insert(0, {
        "date": current_date,
        "value": current_val
    })

    # 去重：优先保留当前接口刚刚返回的指数
    history_records = dedupe_by_date(history_records)

    history_records.sort(
        key=lambda x: x["date"],
        reverse=True
    )

    history_records = history_records[:70]

    if not history_records:
        raise ValueError(
            "FearGreedChart 没有返回有效指数数据"
        )

    return history_records


# ================= 三类数据获取 =================
def get_us_data():
    """
    美股指数获取优先级：

    第一优先级：CNN Fear & Greed Index
    第二优先级：FearGreedChart API

    两种指数均获取失败时，返回获取失败。
    不使用标普500 RSI 作为美股贪恐指数替代。
    """

    # 第一优先级：CNN
    try:
        data = get_cnn_us_data()

        print("✅ 美股数据来源：CNN Fear & Greed Index")

        return data, "CNN Fear & Greed Index"

    except Exception as e:
        print(
            f"⚠️ CNN 贪恐指数获取失败：{e}"
        )

    # 第二优先级：FearGreedChart
    try:
        data = get_feargreedchart_us_data()

        print(
            "✅ 美股数据来源：FearGreedChart API"
        )

        return data, "FearGreedChart API"

    except Exception as e:
        print(
            f"❌ FearGreedChart 获取失败：{e}"
        )

    # 两个来源均失败
    print("❌ CNN 与 FearGreedChart 均获取失败")

    return None, "获取失败"


def get_crypto_data():
    """比特币：Alternative.me"""
    try:
        res = requests.get(
            CRYPTO_URL,
            timeout=15
        )

        res.raise_for_status()

        raw = res.json()["data"]

        data = []

        for d in raw:
            data.append({
                "date": datetime.fromtimestamp(
                    int(d["timestamp"]),
                    tz=timezone.utc
                ).strftime("%Y-%m-%d"),
                "value": int(d["value"])
            })

        return dedupe_by_date(data), "Alternative.me"

    except Exception as e:
        print(f"Crypto API Failed: {e}")

        return None, "获取失败"


def get_cn_data():
    """
    A股：新浪源。

    因子：
    RSI (40%) + 125日市场动量 (30%) + 30日成交量情绪 (30%)
    """

    try:
        # 1. 抓取沪深300历史日线
        hs300 = ak.stock_zh_index_daily(
            symbol="sh000300"
        )

        # 统一列名
        hs300.columns = [
            col.lower()
            for col in hs300.columns
        ]

        # 数据预处理
        hs300["date"] = pd.to_datetime(
            hs300["date"]
        )

        hs300.set_index(
            "date",
            inplace=True
        )

        hs300["close"] = pd.to_numeric(
            hs300["close"]
        )

        hs300["volume"] = pd.to_numeric(
            hs300["volume"]
        )

        # 2. 计算 RSI
        delta = hs300["close"].diff()

        gain = delta.clip(
            lower=0
        ).rolling(14).mean()

        loss = -delta.clip(
            upper=0
        ).rolling(14).mean()

        rs = gain / loss

        hs300["RSI"] = 100 - (
            100 / (1 + rs)
        )

        # 3. 市场动量：125日均线乖离率
        hs300["MA125"] = hs300[
            "close"
        ].rolling(
            window=125
        ).mean()

        hs300["BIAS"] = (
            (
                hs300["close"] -
                hs300["MA125"]
            ) /
            hs300["MA125"]
        ) * 100

        # BIAS 限定在 [-15%, +15%]，映射到 [0, 100]
        hs300["Momentum_Score"] = np.clip(
            (hs300["BIAS"] + 15) / 30 * 100,
            0,
            100
        )

        # 4. 成交量情绪：当前成交量 / 30日均量
        hs300["Vol_MA30"] = hs300[
            "volume"
        ].rolling(
            window=30
        ).mean()

        hs300["Vol_Ratio"] = (
            hs300["volume"] /
            hs300["Vol_MA30"]
        )

        # 均量的 0.5 倍到 1.5 倍映射到 [0, 100]
        hs300["Volume_Score"] = np.clip(
            (hs300["Vol_Ratio"] - 0.5) /
            1.0 * 100,
            0,
            100
        )

        # 5. 综合情绪指数
        hs300["Fear_Greed_Score"] = (
            hs300["RSI"] * 0.4 +
            hs300["Momentum_Score"] * 0.3 +
            hs300["Volume_Score"] * 0.3
        )

        # 提取最近 70 个有效交易日
        recent_data = hs300.dropna().iloc[-70:].copy()

        records = []

        for date, row in recent_data.iloc[::-1].iterrows():
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": normalize_score(
                    row["Fear_Greed_Score"]
                )
            })

        if not records:
            raise ValueError("沪深300未产生有效情绪数据")

        return (
            dedupe_by_date(records),
            "AkShare 多因子模型 (新浪源)"
        )

    except Exception as e:
        print(f"A-Share Calculation Error: {e}")

        # 保留原有 A 股 RSI 降级逻辑
        return (
            calculate_rsi_history("000300.SS"),
            "沪深300 RSI (降级)"
        )


# ================= 统计计算 =================
def calc_stats(data, period_type):
    if not data:
        return None

    current = data[0]

    def count(n):
        sub = data[:n]

        low = sum(
            1 for x in sub
            if x["value"] < LIMIT_LOW
        )

        high = sum(
            1 for x in sub
            if x["value"] > LIMIT_HIGH
        )

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
        "period_type": period_type
    }


# ================= HTML 视图渲染 =================
def make_svg_chart(history):
    if not history:
        return "<p class='muted'>暂无趋势图数据</p>"

    data = list(reversed(history[:30]))

    w, h = 360, 130

    left, right, top, bottom = 28, 8, 10, 20

    pw = w - left - right
    ph = h - top - bottom

    def x_pos(i):
        return (
            left +
            (
                pw / (len(data) - 1) * i
                if len(data) > 1
                else pw / 2
            )
        )

    def y_pos(v):
        v = max(0, min(100, float(v)))

        return top + (100 - v) / 100 * ph

    pts = []

    for i, item in enumerate(data):
        pts.append(
            f"{x_pos(i):.1f},{y_pos(item['value']):.1f}"
        )

    y25 = y_pos(25)
    y75 = y_pos(75)

    start = data[0]["date"][5:]
    end = data[-1]["date"][5:]

    return f"""
    <svg viewBox='0 0 {w} {h}' width='100%' xmlns='http://www.w3.org/2000/svg'>
      <rect x='0' y='0' width='{w}' height='{h}' rx='8' fill='#fff'/>
      <line x1='{left}' y1='{top}' x2='{left}' y2='{h-bottom}' stroke='#ddd'/>
      <line x1='{left}' y1='{h-bottom}' x2='{w-right}' y2='{h-bottom}' stroke='#ddd'/>
      <line x1='{left}' y1='{y75:.1f}' x2='{w-right}' y2='{y75:.1f}' stroke='{GREED_COLOR}' stroke-dasharray='4 4'/>
      <line x1='{left}' y1='{y25:.1f}' x2='{w-right}' y2='{y25:.1f}' stroke='{FEAR_COLOR}' stroke-dasharray='4 4'/>
      <text x='4' y='{y75+3:.1f}' font-size='9' fill='#999'>75</text>
      <text x='4' y='{y25+3:.1f}' font-size='9' fill='#999'>25</text>
      <polyline points='{' '.join(pts)}' fill='none' stroke='#2563eb' stroke-width='2.2'/>
      <text x='{left}' y='{h-5}' font-size='9' fill='#999'>{safe(start)}</text>
      <text x='{w-right}' y='{h-5}' text-anchor='end' font-size='9' fill='#999'>{safe(end)}</text>
    </svg>
    """


def make_history_table(history, period_type, compact=False):
    if not history:
        return "<p class='muted'>暂无明细数据</p>"

    data = history[:30]

    if compact:
        pairs = " ｜ ".join([
            f"{x['date'][5:]}:{x['value']}"
            for x in data
        ])

        return f"<p class='seq'>{safe(pairs)}</p>"

    rows = []

    for x in data:
        cls = status_class(x["value"])
        st = status_style(x["value"])

        rows.append(
            f"<tr><td>{safe(x['date'])}</td>"
            f"<td class='{cls}' style='{st}'>{x['value']}</td>"
            f"<td class='{cls}' style='{st}'>{safe(status_text(x['value']))}</td></tr>"
        )

    return f"""
    <table class='hist'>
      <tr><th>日期</th><th>指数</th><th>状态</th></tr>
      {''.join(rows)}
    </table>
    <p class='note'>股票类为近30个{safe(period_type)}；比特币为近30个自然日。</p>
    """


def make_card(name, source, stats, history, link=None, compact=False):
    if not stats:
        return (
            f"<div class='card err'>"
            f"❌ {safe(name)} 数据获取失败"
            f"</div>"
        )

    v = stats["val"]
    c = value_color(v)
    pt = stats["period_type"]

    warn = ""

    if stats["h30"] >= DANGER_DAYS_THRESHOLD:
        warn = (
            f"<div class='warn'>⚠️ "
            f"<span class='greed' style='color:{GREED_COLOR};font-weight:800'>贪婪风险</span>"
            f"：近30个{safe(pt)}内已有 "
            f"<span class='greed' style='color:{GREED_COLOR};font-weight:800'>{stats['h30']}</span>"
            f" 次极度贪婪。</div>"
        )

    link_html = ""

    if link:
        link_html = (
            f"<p><a class='btn' href='{safe(link)}'>"
            f"查看详细走势</a></p>"
        )

    return f"""
    <div class='card'>
      <div class='top'>
        <div><b>{safe(name)}</b><br><span>{safe(stats['date'])}｜{safe(source)}</span></div>
        <div class='num' style='color:{c}'>{v}<br><small>{safe(stats['status'])}</small></div>
      </div>
      <table class='sum'>
        <tr><th>周期</th><th class='fear' style='color:{FEAR_COLOR};font-weight:800'>恐慌&lt;{LIMIT_LOW}</th><th class='greed' style='color:{GREED_COLOR};font-weight:800'>贪婪&gt;{LIMIT_HIGH}</th></tr>
        <tr><td>近30个{safe(pt)}</td><td class='fear' style='color:{FEAR_COLOR};font-weight:800'>{stats['l30']}</td><td class='greed' style='color:{GREED_COLOR};font-weight:800'>{stats['h30']}</td></tr>
        <tr><td>近60个{safe(pt)}</td><td class='fear' style='color:{FEAR_COLOR};font-weight:800'>{stats['l60']}</td><td class='greed' style='color:{GREED_COLOR};font-weight:800'>{stats['h60']}</td></tr>
      </table>
      {warn}
      <details>
        <summary>📈 查看近30个{safe(pt)}趋势图</summary>
        <div class='chart'>{make_svg_chart(history)}</div>
        {make_history_table(history, pt, compact=compact)}
      </details>
      {link_html}
    </div>
    """


def build_html(results, beijing_time_str, compact=False):
    cards = "".join(
        make_card(
            r["name"],
            r["source"],
            r["stats"],
            r["history"],
            r.get("link"),
            compact=compact
        )
        for r in results
    )

    return f"""
    <html>
    <head>
    <meta charset='utf-8'>
    <style>
      body{{margin:0;background:#f4f6f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;color:#333}}
      .wrap{{max-width:620px;margin:0 auto;padding:12px}}
      h3{{text-align:center;margin:12px 0 16px}}.card{{background:#fff;border:1px solid #eee;border-radius:12px;padding:14px;margin:0 0 14px;box-shadow:0 2px 8px rgba(0,0,0,.04)}}
      .top{{display:flex;justify-content:space-between;gap:10px;border-bottom:1px solid #eee;padding-bottom:9px;margin-bottom:9px}}.top span{{font-size:11px;color:#999}}.num{{font-size:28px;font-weight:800;text-align:right;line-height:1}}.num small{{font-size:11px}}
      table{{width:100%;border-collapse:collapse;text-align:center}}th,td{{padding:6px;border-bottom:1px solid #eee;font-size:12px}}th{{background:#f8fafc}}.sum{{background:#fafafa}}.hist{{margin-top:8px}}
      .fear{{color:{FEAR_COLOR}!important;font-weight:800}}.greed{{color:{GREED_COLOR}!important;font-weight:800}}.neutral{{color:{NEUTRAL_COLOR};font-weight:600}}
      .warn{{background:#fff1f2;color:#991b1b;border:1px solid #fecdd3;border-radius:6px;padding:8px;margin-top:8px;font-size:12px}}
      summary{{cursor:pointer;text-align:center;margin-top:10px;padding:8px;border:1px solid #bfdbfe;background:#eff6ff;color:#075985;border-radius:6px;font-size:13px;font-weight:700}}.chart{{margin-top:8px;border:1px solid #eee;border-radius:8px;overflow:hidden;background:#fff}}.btn{{display:block;text-align:center;background:#e7f5ff;color:#075985;text-decoration:none;border:1px solid #bfdbfe;border-radius:6px;padding:8px;font-weight:700;font-size:13px}}.foot{{background:#e9ecef;border-left:4px solid #2563eb;border-radius:8px;padding:12px;font-size:12px;line-height:1.7}}.note,.muted{{font-size:11px;color:#999;text-align:right}}.seq{{font-size:12px;line-height:1.8;color:#555}}.err{{background:#fee2e2}}
    </style>
    </head>
    <body><div class='wrap'>
      <h3>🌍 全球核心资产情绪监控</h3>
      {cards}
      <div class='foot'>
        <b>📊 策略提示</b><br>
        <span class='fear' style='color:{FEAR_COLOR};font-weight:800'>🟢 恐慌机会：指数 &lt; {LIMIT_LOW}，关注定投。</span><br>
        <span class='greed' style='color:{GREED_COLOR};font-weight:800'>🔴 贪婪风险：指数 &gt; {LIMIT_HIGH}，关注止盈。</span><br>
        ⚠️ 近30天<span class='greed' style='color:{GREED_COLOR};font-weight:800'>极度贪婪</span> ≥ {DANGER_DAYS_THRESHOLD} 次：注意高位。<br>
        <span style='color:#999'>Data Updated: {safe(beijing_time_str)}</span>
      </div>
    </div></body></html>
    """


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
        "channel": "wechat"
    }

    if topic:
        payload["topic"] = topic

    try:
        res = requests.post(
            url,
            json=payload,
            timeout=20
        )

        result = res.json()

        if result.get("code") == 200:
            print("✅ 推送成功")
            return True

        print(f"❌ 推送失败：{result}")

        return False

    except Exception as e:
        print(f"❌ 网络异常，推送失败: {e}")

        return False


# ================= 主程序 =================
def main():
    print("🚀 开始分析全球市场情绪...")

    tasks = [
        {
            "name": "US 美股",
            "getter": get_us_data,
            "ticker": None,
            "period_type": "交易日",
            "link": None
        },
        {
            "name": "₿ 比特币",
            "getter": get_crypto_data,
            "ticker": None,
            "period_type": "自然日",
            "link": None
        },
        {
            "name": "CN A股",
            "getter": get_cn_data,
            "ticker": None,
            "period_type": "交易日",
            "link": JIUQUAN_URL
        },
    ]

    results = []
    title_parts = []

    for task in tasks:
        raw_data, source = task["getter"]()

        stats = calc_stats(
            raw_data,
            task["period_type"]
        )

        results.append({
            "name": task["name"],
            "source": source,
            "stats": stats,
            "history": raw_data or [],
            "link": task["link"]
        })

        if stats:
            title_parts.append(
                f"{task['name'].split()[-1]}:{stats['val']}"
            )

    beijing_time = (
        datetime.now(timezone.utc) +
        timedelta(hours=8)
    )

    beijing_time_str = (
        beijing_time.strftime("%Y-%m-%d %H:%M") +
        "（北京时间）"
    )

    html_content = build_html(
        results,
        beijing_time_str,
        compact=False
    )

    if len(html_content) > MAX_PUSHPLUS_LEN:
        html_content = build_html(
            results,
            beijing_time_str,
            compact=True
        )

    if len(html_content) > MAX_PUSHPLUS_LEN:
        summary_lines = []

        for r in results:
            s = r["stats"]

            if s:
                summary_lines.append(
                    f"{r['name']}：{s['val']}，{s['status']}，"
                    f"近30个{s['period_type']}恐慌{s['l30']}次，"
                    f"贪婪{s['h30']}次"
                )
            else:
                summary_lines.append(
                    f"{r['name']}：数据获取失败"
                )

        html_content = (
            "<br>".join(summary_lines) +
            f"<br><br>更新时间：{safe(beijing_time_str)}"
        )

    if title_parts:
        title = (
            beijing_time.strftime("%m-%d") +
            " | " +
            " | ".join(title_parts)
        )

        send_push(title, html_content)

    else:
        print("❌ 所有数据获取失败")


if __name__ == "__main__":
    main()
