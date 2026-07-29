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

LIMIT_LOW = 25
LIMIT_HIGH = 75
DANGER_DAYS_THRESHOLD = 10
MAX_PUSHPLUS_LEN = 19000        # PushPlus 限制约 2 万字，留一点余量

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


# ================= 备用 RSI 计算 =================
def calculate_rsi_history(ticker, period="8mo"):
    """如果主 API 失败，降级使用 yfinance 计算 RSI 作为情绪替代"""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
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
                "value": int(round(float(value)))
            })
        return records
    except Exception as e:
        print(f"RSI Calculation Error for {ticker}: {e}")
        return None


# ================= 交易日过滤 =================
def filter_by_trading_days(data, ticker, period="8mo"):
    if not data:
        return data
    try:
        price_df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        if price_df.empty:
            return data
        trading_dates = set(price_df.index.strftime("%Y-%m-%d"))
        filtered = [d for d in data if d["date"] in trading_dates]
        if len(filtered) < 30:
            return data
        return filtered
    except Exception:
        return data


# ================= 三类数据获取 =================
def get_us_data():
    """美股：使用 fear-greed 开源库，失败则降级使用 S&P500 RSI"""
    try:
        # 获取最新的情绪数据，由于库自身设计，历史数据我们拉取快照
        idx = fear_greed.get()
        # 目前 fear-greed 返回当前时间，为构建历史图表，我们用 yfinance 补全交易日历史作为趋势
        # 实际操作中，如果你需要纯正的 CNN 历史曲线，仍需通过内部 API。这里混合使用。
        history_records = calculate_rsi_history("^GSPC")
        if history_records:
            # 将最新的 fear-greed 官方评分替换今日/昨日数据
            history_records.insert(0, {
                "date": idx.last_update.strftime("%Y-%m-%d") if hasattr(idx, 'last_update') else datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "value": int(round(idx.value))
            })
            return dedupe_by_date(history_records), "fear-greed官方+历史RSI"
        else:
            return None, "获取失败"
    except Exception as e:
        print(f"US Fear-Greed API Failed: {e}, Switching to S&P 500 RSI...")
        return calculate_rsi_history("^GSPC"), "S&P500 RSI替代"


def get_crypto_data():
    """比特币：Alternative.me"""
    try:
        res = requests.get(CRYPTO_URL, timeout=15)
        res.raise_for_status()
        raw = res.json()["data"]
        data = []
        for d in raw:
            data.append({
                "date": datetime.fromtimestamp(int(d["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%d"),
                "value": int(d["value"])
            })
        return dedupe_by_date(data), "Alternative.me"
    except Exception as e:
        print(f"Crypto API Failed: {e}")
        return None, "获取失败"


def get_cn_data():
    """
    A股：通过 AkShare 提取沪深300数据，构建多因子贪恐指数
    因子：RSI (40%) + 125日市场动量 (40%) + 30日成交量情绪 (20%)
    """
    try:
        # 1. 抓取沪深300历史日线 (无需 API Token)
        hs300 = ak.stock_zh_index_daily_em(symbol="sh000300")
        hs300.rename(columns={'date': 'date', 'close': 'close', 'volume': 'volume'}, inplace=True)
        hs300['date'] = pd.to_datetime(hs300['date'])
        hs300.set_index('date', inplace=True)

        # 2. 计算 RSI
        delta = hs300['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        hs300['RSI'] = 100 - (100 / (1 + rs))

        # 3. 市场动量 (Momentum)：125日均线乖离率 (BIAS)
        hs300['MA125'] = hs300['close'].rolling(window=125).mean()
        hs300['BIAS'] = (hs300['close'] - hs300['MA125']) / hs300['MA125'] * 100
        # 归一化：将 BIAS 限定在 [-15%, +15%] 区间，映射到 [0, 100] 分数
        hs300['Momentum_Score'] = np.clip((hs300['BIAS'] + 15) / 30 * 100, 0, 100)

        # 4. 成交量情绪 (Volume)：当前成交量 / 30日均量
        hs300['Vol_MA30'] = hs300['volume'].rolling(window=30).mean()
        hs300['Vol_Ratio'] = hs300['volume'] / hs300['Vol_MA30']
        # 归一化：均量的 0.5倍 到 1.5倍 映射到 [0, 100]
        hs300['Volume_Score'] = np.clip((hs300['Vol_Ratio'] - 0.5) / 1.0 * 100, 0, 100)

        # 5. 合成综合情绪指数
        hs300['Fear_Greed_Score'] = (hs300['RSI'] * 0.4) + (hs300['Momentum_Score'] * 0.4) + (hs300['Volume_Score'] * 0.2)
        
        # 提取最近 70 个交易日
        recent_data = hs300.dropna().iloc[-70:].copy()
        
        records = []
        for date, row in recent_data.iloc[::-1].iterrows():
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": int(round(row['Fear_Greed_Score']))
            })
            
        return dedupe_by_date(records), "AkShare 多因子模型"
    except Exception as e:
        print(f"A-Share Calculation Error: {e}")
        # 如果 AkShare 失败，回退到原有的 yfinance RSI（防止完全断更）
        return calculate_rsi_history("000300.SS"), "沪深300 RSI (降级)"


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
    pw = w - left - right
    ph = h - top - bottom

    def x_pos(i):
        return left + (pw / (len(data) - 1) * i if len(data) > 1 else pw / 2)

    def y_pos(v):
        v = max(0, min(100, float(v)))
        return top + (100 - v) / 100 * ph

    pts = []
    for i, item in enumerate(data):
        pts.append(f"{x_pos(i):.1f},{y_pos(item['value']):.1f}")

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
        pairs = " ｜ ".join([f"{x['date'][5:]}:{x['value']}" for x in data])
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
        return f"<div class='card err'>❌ {safe(name)} 数据获取失败</div>"
    v = stats["val"]
    c = value_color(v)
    pt = stats["period_type"]
    warn = ""
    if stats["h30"] >= DANGER_DAYS_THRESHOLD:
        warn = f"<div class='warn'>⚠️ <span class='greed' style='color:{GREED_COLOR};font-weight:800'>贪婪风险</span>：近30个{safe(pt)}内已有 <span class='greed' style='color:{GREED_COLOR};font-weight:800'>{stats['h30']}</span> 次极度贪婪。</div>"
    link_html = ""
    if link:
        link_html = f"<p><a class='btn' href='{safe(link)}'>查看详细走势</a></p>"
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
        make_card(r["name"], r["source"], r["stats"], r["history"], r.get("link"), compact=compact)
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
        else:
            print(f"❌ 推送失败：{result}")
            return False
    except Exception as e:
        print(f"❌ 网络异常，推送失败: {e}")
        return False


# ================= 主程序 =================
def main():
    print("🚀 开始分析全球市场情绪...")

    tasks = [
        {"name": "US 美股", "getter": get_us_data, "ticker": None, "period_type": "交易日", "link": None},
        {"name": "₿ 比特币", "getter": get_crypto_data, "ticker": None, "period_type": "自然日", "link": None},
        {"name": "CN A股", "getter": get_cn_data, "ticker": None, "period_type": "交易日", "link": JIUQUAN_URL},
    ]

    results = []
    title_parts = []

    for task in tasks:
        raw_data, source = task["getter"]()
        # US/CN 已经在函数内处理了交易日，比特币为自然日，无需调用 filter_by_trading_days
        stats = calc_stats(raw_data, task["period_type"])
        results.append({
            "name": task["name"],
            "source": source,
            "stats": stats,
            "history": raw_data or [],
            "link": task["link"]
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
                summary_lines.append(f"{r['name']}：{s['val']}，{s['status']}，近30个{s['period_type']}恐慌{s['l30']}次，贪婪{s['h30']}次")
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
