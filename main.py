import os
import html
import requests
import yfinance as yf
from datetime import datetime, timedelta, timezone

# ================= 配置区域 =================
CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CRYPTO_URL = "https://api.alternative.me/fng/?limit=80"
ASHARE_CODE = "000300.SS"      # 沪深300
JIUQUAN_URL = "https://funddb.cn/tool/fear"

LIMIT_LOW = 25
LIMIT_HIGH = 75
DANGER_DAYS_THRESHOLD = 10
MAX_PUSHPLUS_LEN = 19000        # PushPlus 限制约 2 万字，留一点余量

FEAR_COLOR = "#16a34a"          # 恐慌：绿色
GREED_COLOR = "#dc2626"         # 贪婪：红色
NEUTRAL_COLOR = "#555555"       # 中性：深灰
LINE_COLOR = "#2563eb"          # 折线：蓝色


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


def pair_style(value):
    """
    横向明细使用：
    恐慌整组标绿，贪婪整组标红，中性保持普通灰色。
    """
    if value < LIMIT_LOW:
        return f"color:{FEAR_COLOR};font-weight:800"
    if value > LIMIT_HIGH:
        return f"color:{GREED_COLOR};font-weight:800"
    return f"color:{NEUTRAL_COLOR};font-weight:500"


def dot_color(value):
    """
    折线图点颜色：
    恐慌绿，贪婪红，中性蓝。
    """
    if value < LIMIT_LOW:
        return FEAR_COLOR
    if value > LIMIT_HIGH:
        return GREED_COLOR
    return LINE_COLOR


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


# ================= RSI 计算：用于 A 股，及美股备用 =================
def calculate_rsi_history(ticker, period="8mo"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)

        if df.empty:
            print(f"❌ {ticker} 行情数据为空")
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
            print(f"⚠️ {ticker} 交易日数据为空，保留原始数据")
            return data

        trading_dates = set(price_df.index.strftime("%Y-%m-%d"))

        filtered = [
            d for d in data
            if d["date"] in trading_dates
        ]

        print(f"✅ {ticker} 交易日过滤完成：原始 {len(data)} 条，过滤后 {len(filtered)} 条")

        if len(filtered) < 30:
            print(f"⚠️ {ticker} 过滤后不足30条，保留原始数据")
            return data

        return filtered

    except Exception as e:
        print(f"Trading day filter failed for {ticker}: {e}")
        return data


# ================= 三类数据获取 =================
def get_us_data():
    """
    美股：优先使用 CNN 官方恐慌贪婪指数，失败则用 S&P 500 RSI 替代。
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.cnn.com/",
            "Origin": "https://www.cnn.com"
        }

        res = requests.get(CNN_URL, headers=headers, timeout=15)
        res.raise_for_status()

        raw = res.json()["fear_and_greed_historical"]["data"]
        raw.sort(key=lambda x: x["x"], reverse=True)

        data = []

        for d in raw:
            data.append({
                "date": datetime.fromtimestamp(d["x"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                "value": int(round(float(d["y"])))
            })

        return dedupe_by_date(data), "CNN官方"

    except Exception as e:
        print(f"CNN API Failed: {e}, Switching to S&P 500 RSI...")
        return calculate_rsi_history("^GSPC"), "S&P500 RSI替代"


def get_crypto_data():
    """
    比特币：Alternative.me 恐慌贪婪指数，按自然日统计。
    """
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
    A股：用沪深300 RSI 作为情绪替代指标。
    """
    return calculate_rsi_history(ASHARE_CODE), "沪深300 RSI"


# ================= 统计 =================
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


# ================= HTML：压缩版，避免 PushPlus 超过 2 万字 =================
def make_svg_chart(history):
    """
    近30日折线图，使用紧凑 SVG，不使用 JS，适合 PushPlus。
    """
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
    circles = []

    for i, item in enumerate(data):
        x = x_pos(i)
        y = y_pos(item["value"])
        value = int(item["value"])

        pts.append(f"{x:.1f},{y:.1f}")

        circles.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='2.2' fill='{dot_color(value)}'>"
            f"<title>{safe(item['date'])}:{value}</title>"
            f"</circle>"
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
      <text x='4' y='{y75+3:.1f}' font-size='9' fill='{GREED_COLOR}'>75</text>
      <text x='4' y='{y25+3:.1f}' font-size='9' fill='{FEAR_COLOR}'>25</text>
      <polyline points='{' '.join(pts)}' fill='none' stroke='{LINE_COLOR}' stroke-width='2.2'/>
      {''.join(circles)}
      <text x='{left}' y='{h-5}' font-size='9' fill='#999'>{safe(start)}</text>
      <text x='{w-right}' y='{h-5}' text-anchor='end' font-size='9' fill='#999'>{safe(end)}</text>
    </svg>
    """


def make_history_table(history, period_type, compact=False):
    """
    近30个统计周期明细。
    保持原来的横向排布方式：05-29:68 ｜ 05-28:71 ｜ ...
    其中：恐慌 < 25 标绿，贪婪 > 75 标红。
    """
    if not history:
        return "<p class='muted'>暂无明细数据</p>"

    data = history[:30]

    items = []

    for x in data:
        value = int(x["value"])
        text = f"{x['date'][5:]}:{value}"

        items.append(
            f"<span class='{status_class(value)}' style='{pair_style(value)}'>{safe(text)}</span>"
        )

    return f"""
    <p class='seq'>{' <span class="sep">｜</span> '.join(items)}</p>
    <p class='note'>股票类为近30个{safe(period_type)}；比特币为近30个自然日。<br>
    <span class='fear' style='color:{FEAR_COLOR};font-weight:800'>绿色=恐慌&lt;{LIMIT_LOW}</span>，
    <span class='greed' style='color:{GREED_COLOR};font-weight:800'>红色=贪婪&gt;{LIMIT_HIGH}</span>。</p>
    """


def make_card(name, source, stats, history, link=None, compact=False):
    if not stats:
        return f"<div class='card err'>❌ {safe(name)} 数据获取失败</div>"

    v = stats["val"]
    c = value_color(v)
    pt = stats["period_type"]

    warn = ""

    if stats["h30"] >= DANGER_DAYS_THRESHOLD:
        warn = f"<div class='warn'>⚠️ <span class='greed' style='color:{GREED_COLOR};font-weight:800'>贪婪风险</span>：近30个{safe(pt)}内已有 <span class='greed' style='color:{GREED_COLOR};font-weight:800'>{stats['h30']}</span> 次极度贪婪，注意高位风险。</div>"

    link_html = ""

    if link:
        link_html = f"<p><a class='btn' href='{safe(link)}'>查看韭圈儿详情</a></p>"

    return f"""
    <div class='card'>
      <div class='top'>
        <div><b>{safe(name)}</b><br><span>{safe(stats['date'])}｜{safe(source)}</span></div>
        <div class='num' style='color:{c}'>{v}<br><small>{safe(stats['status'])}</small></div>
      </div>

      <table class='sum'>
        <tr>
          <th>周期</th>
          <th class='fear' style='color:{FEAR_COLOR};font-weight:800'>恐慌&lt;{LIMIT_LOW}</th>
          <th class='greed' style='color:{GREED_COLOR};font-weight:800'>贪婪&gt;{LIMIT_HIGH}</th>
        </tr>
        <tr>
          <td>近30个{safe(pt)}</td>
          <td class='fear' style='color:{FEAR_COLOR};font-weight:800'>{stats['l30']}</td>
          <td class='greed' style='color:{GREED_COLOR};font-weight:800'>{stats['h30']}</td>
        </tr>
        <tr>
          <td>近60个{safe(pt)}</td>
          <td class='fear' style='color:{FEAR_COLOR};font-weight:800'>{stats['l60']}</td>
          <td class='greed' style='color:{GREED_COLOR};font-weight:800'>{stats['h60']}</td>
        </tr>
      </table>

      {warn}

      <details>
        <summary>📈 查看近30个{safe(pt)}明细与趋势图</summary>
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
      h3{{text-align:center;margin:12px 0 16px}}
      .card{{background:#fff;border:1px solid #eee;border-radius:12px;padding:14px;margin:0 0 14px;box-shadow:0 2px 8px rgba(0,0,0,.04)}}
      .top{{display:flex;justify-content:space-between;gap:10px;border-bottom:1px solid #eee;padding-bottom:9px;margin-bottom:9px}}
      .top span{{font-size:11px;color:#999}}
      .num{{font-size:28px;font-weight:800;text-align:right;line-height:1}}
      .num small{{font-size:11px}}
      table{{width:100%;border-collapse:collapse;text-align:center}}
      th,td{{padding:6px;border-bottom:1px solid #eee;font-size:12px}}
      th{{background:#f8fafc}}
      .sum{{background:#fafafa}}
      .fear{{color:{FEAR_COLOR}!important;font-weight:800}}
      .greed{{color:{GREED_COLOR}!important;font-weight:800}}
      .neutral{{color:{NEUTRAL_COLOR};font-weight:600}}
      .warn{{background:#fff1f2;color:#991b1b;border:1px solid #fecdd3;border-radius:6px;padding:8px;margin-top:8px;font-size:12px}}
      summary{{cursor:pointer;text-align:center;margin-top:10px;padding:8px;border:1px solid #bfdbfe;background:#eff6ff;color:#075985;border-radius:6px;font-size:13px;font-weight:700}}
      .chart{{margin-top:8px;border:1px solid #eee;border-radius:8px;overflow:hidden;background:#fff}}
      .btn{{display:block;text-align:center;background:#e7f5ff;color:#075985;text-decoration:none;border:1px solid #bfdbfe;border-radius:6px;padding:8px;font-weight:700;font-size:13px}}
      .foot{{background:#e9ecef;border-left:4px solid #2563eb;border-radius:8px;padding:12px;font-size:12px;line-height:1.7}}
      .note,.muted{{font-size:11px;color:#999;text-align:right}}
      .seq{{font-size:12px;line-height:1.9;color:#555;margin:10px 0 4px;word-break:break-word}}
      .sep{{color:#94a3b8;font-weight:400}}
      .err{{background:#fee2e2}}
    </style>
    </head>

    <body>
      <div class='wrap'>
        <h3>🌍 全球核心资产情绪监控</h3>

        {cards}

        <div class='foot'>
          <b>📊 策略提示</b><br>
          <span class='fear' style='color:{FEAR_COLOR};font-weight:800'>🟢 恐慌机会：指数 &lt; {LIMIT_LOW}，可关注分批定投机会。</span><br>
          <span class='greed' style='color:{GREED_COLOR};font-weight:800'>🔴 贪婪风险：指数 &gt; {LIMIT_HIGH}，可关注分批止盈风险。</span><br>
          ⚠️ 近30个统计周期内<span class='greed' style='color:{GREED_COLOR};font-weight:800'>极度贪婪</span>次数 ≥ {DANGER_DAYS_THRESHOLD}：注意高位风险。<br>
          <span style='color:#999'>Data Updated: {safe(beijing_time_str)}</span>
        </div>
      </div>
    </body>
    </html>
    """


# ================= PushPlus 推送 =================
def send_push(title, content):
    token = os.getenv("PUSHPLUS_TOKEN")
    topic = os.getenv("PUSHPLUS_TOPIC")

    if not token:
        print("❌ 未检测到 PUSHPLUS_TOKEN，跳过推送")
        return False

    print(f"📏 推送内容长度：{len(content)} 字符")

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
        print(f"📡 准备推送到群组：{topic}")
    else:
        print("📡 准备推送到个人微信")

    try:
        res = requests.post(url, json=payload, timeout=20)

        print(f"PushPlus Status Code: {res.status_code}")
        print(f"PushPlus Response: {res.text}")

        try:
            result = res.json()
        except Exception:
            print("❌ PushPlus 返回不是 JSON")
            return False

        if result.get("code") == 200:
            print("✅ PushPlus 返回 code=200，推送成功")
            return True
        else:
            print(
                f"❌ PushPlus 推送失败："
                f"code={result.get('code')}，"
                f"msg={result.get('msg')}，"
                f"data={result.get('data')}"
            )
            return False

    except Exception as e:
        print(f"❌ 推送发送失败: {e}")
        return False


# ================= 主程序 =================
def main():
    print("🚀 开始分析全球市场情绪...")

    tasks = [
        {
            "name": "US 美股",
            "getter": get_us_data,
            "ticker": "^GSPC",
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
            "ticker": ASHARE_CODE,
            "period_type": "交易日",
            "link": JIUQUAN_URL
        },
    ]

    results = []
    title_parts = []

    for task in tasks:
        print(f"\n========== {task['name']} ==========")

        raw_data, source = task["getter"]()

        if raw_data:
            print(f"{task['name']} 原始数据条数：{len(raw_data)}")

        if task["ticker"]:
            raw_data = filter_by_trading_days(raw_data, task["ticker"])

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
            print(
                f"{task['name']} 当前值：{stats['val']}，"
                f"近30个{task['period_type']}贪婪次数：{stats['h30']}，"
                f"恐慌次数：{stats['l30']}"
            )

    beijing_time = datetime.now(timezone.utc) + timedelta(hours=8)
    beijing_time_str = beijing_time.strftime("%Y-%m-%d %H:%M") + "（北京时间）"

    html_content = build_html(results, beijing_time_str, compact=False)

    # 如果内容仍然过长，自动切换成 compact=True 重新生成。
    # 这里 compact=True 也保持原来的横向排布，不再改成纵向。
    if len(html_content) > MAX_PUSHPLUS_LEN:
        print(f"⚠️ HTML 长度 {len(html_content)} 超过安全阈值，切换为极简横向明细版本")
        html_content = build_html(results, beijing_time_str, compact=True)

    # 如果仍超限，只推送摘要，避免任务白跑
    if len(html_content) > MAX_PUSHPLUS_LEN:
        print(f"⚠️ 极简版仍过长：{len(html_content)}，改为只发送摘要")

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
        print("❌ 所有数据获取失败，未发送推送")


if __name__ == "__main__":
    main()
