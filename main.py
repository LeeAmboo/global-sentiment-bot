import os
import re
import html
import requests
import yfinance as yf
import akshare as ak
import pandas as pd
import numpy as np
import fear_greed
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区域 =================
CRYPTO_URL = "https://api.alternative.me/fng/?limit=80"
JIUQUAN_URL = "https://funddb.cn/tool/fear"

# A股贪恐指数：kaifengmayao/a-share-fear-greed-index
CN_FG_REPO = "https://github.com/kaifengmayao/a-share-fear-greed-index"
CN_FG_API_TREE = (
    "https://api.github.com/repos/"
    "kaifengmayao/a-share-fear-greed-index/git/trees/main?recursive=1"
)
CN_FG_RAW_BASE = (
    "https://raw.githubusercontent.com/"
    "kaifengmayao/a-share-fear-greed-index/main"
)
CN_FG_LATEST_MD = f"{CN_FG_RAW_BASE}/reports/latest.md"

# 备用美股贪恐指数 API
US_FALLBACK_URL = "https://feargreedchart.com/api/?action=all"
US_HISTORY_URL = "https://feargreedchart.com/api/?action=history"

LIMIT_LOW = 25
LIMIT_HIGH = 75
DANGER_DAYS_THRESHOLD = 10
MAX_PUSHPLUS_LEN = 19000
CN_HISTORY_LIMIT = 70

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
    seen, out = set(), []
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


def request_text(url, timeout=20):
    response = requests.get(
        url, headers=REQUEST_HEADERS, timeout=timeout
    )
    response.raise_for_status()
    return response.text


# ================= 备用 RSI 计算 =================
def calculate_rsi_history(ticker, period="8mo"):
    """仅供 A 股开源指数失败时降级使用；不会把 RSI 标成贪恐指数。"""
    try:
        df = yf.download(
            ticker, period=period, progress=False, auto_adjust=False
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
    print("🇺🇸 正在获取 CNN Fear & Greed Index...")
    idx = fear_greed.get()
    if not isinstance(idx, dict):
        raise ValueError(f"CNN 返回数据格式异常：{type(idx).__name__}")
    if "score" not in idx:
        raise ValueError("CNN 返回数据缺少 score 字段")

    current_val = normalize_score(idx["score"])
    current_date = (
        normalize_date(idx["timestamp"])
        if idx.get("timestamp")
        else datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
            history_records.append({
                "date": normalize_date(date_value),
                "value": normalize_score(score_value)
            })
    except Exception as e:
        print(f"⚠️ CNN 历史数据获取失败：{e}")

    history_records.insert(0, {"date": current_date, "value": current_val})
    history_records = dedupe_by_date(history_records)
    history_records.sort(key=lambda x: x["date"], reverse=True)
    history_records = history_records[:70]
    if not history_records:
        raise ValueError("CNN 没有返回有效指数数据")
    return history_records


# ================= FearGreedChart 备用美股指数 =================
def get_feargreedchart_us_data():
    print("🔄 正在切换到 FearGreedChart API...")
    data = request_json(US_FALLBACK_URL)
    score_info = data.get("score")
    if not isinstance(score_info, dict):
        raise ValueError("FearGreedChart 返回数据缺少 score 对象")
    if "score" not in score_info:
        raise ValueError("FearGreedChart 返回数据缺少 score.score 字段")
    current_val = normalize_score(score_info["score"])

    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if data.get("ts") is not None:
        try:
            current_date = datetime.fromtimestamp(
                float(data["ts"]) / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d")
        except (ValueError, TypeError, OSError):
            pass
    print(f"✅ FearGreedChart 当前指数：{current_val}，日期：{current_date}")

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
                    timestamp = float(date_value)
                    if timestamp > 1e12:
                        timestamp /= 1000
                    date_value = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                records.append({
                    "date": normalize_date(date_value),
                    "value": normalize_score(score_value)
                })
            except (ValueError, TypeError, OverflowError, OSError):
                continue
        return records

    raw_history = data.get("recent")
    if not isinstance(raw_history, list):
        raw_history = []
    if not raw_history and isinstance(data.get("history"), list):
        raw_history = data["history"]
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
                if not isinstance(raw_history, list):
                    raw_history = []
            else:
                raw_history = []
            history_records = parse_history_items(raw_history)
        except Exception as e:
            print(f"⚠️ FearGreedChart 历史接口获取失败：{e}")

    history_records.insert(0, {"date": current_date, "value": current_val})
    history_records = dedupe_by_date(history_records)
    history_records.sort(key=lambda x: x["date"], reverse=True)
    history_records = history_records[:70]
    if not history_records:
        raise ValueError("FearGreedChart 没有返回有效指数数据")
    return history_records


def get_us_data():
    """美股指数优先 CNN，其次 FearGreedChart；均失败则返回失败。"""
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
    """比特币：Alternative.me"""
    try:
        res = requests.get(CRYPTO_URL, headers=REQUEST_HEADERS, timeout=15)
        res.raise_for_status()
        raw = res.json()["data"]
        data = []
        for d in raw:
            data.append({
                "date": datetime.fromtimestamp(
                    int(d["timestamp"]), tz=timezone.utc
                ).strftime("%Y-%m-%d"),
                "value": normalize_score(d["value"])
            })
        data = dedupe_by_date(data)
        data.sort(key=lambda x: x["date"], reverse=True)
        return data, "Alternative.me"
    except Exception as e:
        print(f"Crypto API Failed: {e}")
        return None, "获取失败"


# ================= A股开源贪恐指数 =================
def parse_afgi_markdown(markdown_text, fallback_date=None):
    """
    解析 AFGI 项目日报 Markdown 中的日期和最终指数。
    项目报告格式示例：
      - 日期：2026-10-16
      - 指数：48.2 / 100
    """
    date_match = re.search(
        r"(?m)^\s*-\s*日期\s*[:：]\s*(\d{4}-\d{2}-\d{2})",
        markdown_text
    )
    score_match = re.search(
        r"(?m)^\s*-\s*指数\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*100",
        markdown_text
    )
    if not score_match:
        # 兼容少量格式差异
        score_match = re.search(
            r"(?m)^\s*-\s*(?:最终)?(?:恐惧贪婪)?指数\s*[:：]\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:分|/100)",
            markdown_text
        )
    if not score_match:
        raise ValueError("日报 Markdown 中未找到最终指数")

    date_value = date_match.group(1) if date_match else fallback_date
    if not date_value:
        raise ValueError("日报 Markdown 中未找到指数日期")

    return {
        "date": normalize_date(date_value),
        "value": normalize_score(score_match.group(1))
    }


def get_cn_data():
    """
    A股：kaifengmayao/a-share-fear-greed-index。

    从其 reports/YYYY-MM-DD.md 日报读取已计算好的最终指数。
    先读取 latest.md，再从 GitHub 文件树中找最近的日期报告，
    尽量补齐历史数据。报告只含一个当日分数时，仍能正常显示，
    但历史统计只基于成功获取到的报告，不伪造历史分数。
    """
    print("🇨🇳 正在获取 A股 AFGI 开源贪恐指数...")
    records_by_date = {}

    # 1. 先获取 latest.md；若项目未更新 latest.md，后续会比较日期报告
    try:
        latest_text = request_text(CN_FG_LATEST_MD, timeout=20)
        latest_record = parse_afgi_markdown(latest_text)
        records_by_date[latest_record["date"]] = latest_record
        print(
            f"✅ 获取 AFGI latest.md：{latest_record['value']}，"
            f"日期：{latest_record['date']}"
        )
    except Exception as e:
        print(f"⚠️ AFGI latest.md 获取失败：{e}")

    # 2. 查询仓库文件树，找日期格式的日报文件
    try:
        tree_data = request_json(CN_FG_API_TREE, timeout=25)
        tree = tree_data.get("tree", [])
        date_files = []
        for item in tree:
            path = item.get("path", "")
            match = re.fullmatch(r"reports/(\d{4}-\d{2}-\d{2})\.md", path)
            if match:
                date_files.append((match.group(1), path))

        date_files.sort(key=lambda x: x[0], reverse=True)

        # 只抓最近若干天的报告，最多并发 8 个请求，减少运行时间
        candidates = date_files[:CN_HISTORY_LIMIT]
        if candidates:
            print(f"🔄 正在补齐 AFGI 历史报告，候选文件 {len(candidates)} 个...")
            def fetch_report(item):
                date_text, path = item
                url = f"{CN_FG_RAW_BASE}/{path}"
                text = request_text(url, timeout=20)
                return parse_afgi_markdown(text, fallback_date=date_text)

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(fetch_report, item) for item in candidates]
                for future in as_completed(futures):
                    try:
                        record = future.result()
                        records_by_date[record["date"]] = record
                    except Exception as e:
                        print(f"⚠️ 一份 AFGI 历史日报解析失败：{e}")

    except Exception as e:
        print(f"⚠️ AFGI 历史报告列表获取失败：{e}")

    records = list(records_by_date.values())
    records = dedupe_by_date(records)
    records.sort(key=lambda x: x["date"], reverse=True)

    if records:
        records = records[:CN_HISTORY_LIMIT]
        latest = records[0]
        print(
            f"✅ A股 AFGI 获取成功：{latest['value']}，"
            f"日期：{latest['date']}，有效记录：{len(records)} 条"
        )
        return records, "kaifengmayao A股 AFGI"

    # 3. 开源项目不可用时，保留原脚本的沪深300 RSI 降级逻辑
    print("⚠️ AFGI 开源报告不可用，降级尝试沪深300 RSI...")
    fallback_data = calculate_rsi_history("000300.SS")
    if fallback_data:
        print("⚠️ 当前显示的是沪深300 RSI（降级），不是 AFGI 贪恐指数")
        return fallback_data, "沪深300 RSI（降级）"

    print("❌ A股 AFGI 和 RSI 降级数据均获取失败")
    return None, "获取失败"


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
        "period_type": period_type
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
        v = max(0, min(100, float(v)))
        return top + (100 - v) / 100 * ph

    pts = [
        f"{x_pos(i):.1f},{y_pos(item['value']):.1f}"
        for i, item in enumerate(data)
    ]
    y25, y75 = y_pos(25), y_pos(75)
    start, end = data[0]["date"][5:], data[-1]["date"][5:]
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
        pairs = " ｜ ".join([f"{x['date'][5:]}:{x['value']}" for x in data])
        return f"<p class='seq'>{safe(pairs)}</p>"
    rows = []
    for x in data:
        cls, st = status_class(x["value"]), status_style(x["value"])
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
        return f"<div class='card err'>❌ {safe(name)} 数据获取失败</div>"
    v, c, pt = stats["val"], value_color(stats["val"]), stats["period_type"]
    warn = ""
    if stats["h30"] >= DANGER_DAYS_THRESHOLD:
        warn = (
            f"<div class='warn'>⚠️ "
            f"<span class='greed' style='color:{GREED_COLOR};font-weight:800'>贪婪风险</span>"
            f"：近30个{safe(pt)}内已有 "
            f"<span class='greed' style='color:{GREED_COLOR};font-weight:800'>{stats['h30']}</span>"
            f" 次极度贪婪。</div>"
        )
    link_html = (
        f"<p><a class='btn' href='{safe(link)}'>查看详细走势</a></p>"
        if link else ""
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
            r["name"], r["source"], r["stats"], r["history"],
            r.get("link"), compact=compact
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
        res = requests.post(url, json=payload, timeout=20)
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
            "name": "US 美股", "getter": get_us_data,
            "ticker": None, "period_type": "交易日", "link": None
        },
        {
            "name": "₿ 比特币", "getter": get_crypto_data,
            "ticker": None, "period_type": "自然日", "link": None
        },
        {
            "name": "CN A股", "getter": get_cn_data,
            "ticker": None, "period_type": "交易日", "link": JIUQUAN_URL
        },
    ]

    results, title_parts = [], []
    for task in tasks:
        raw_data, source = task["getter"]()
        stats = calc_stats(raw_data, task["period_type"])
        results.append({
            "name": task["name"], "source": source, "stats": stats,
            "history": raw_data or [], "link": task["link"]
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
        html_content = (
            "<br>".join(summary_lines)
            + f"<br><br>更新时间：{safe(beijing_time_str)}"
        )

    if title_parts:
        title = beijing_time.strftime("%m-%d") + " | " + " | ".join(title_parts)
        send_push(title, html_content)
    else:
        print("❌ 所有数据获取失败")


if __name__ == "__main__":
    main()
