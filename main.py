import requests
import os
import html
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone

# ================= 配置区域 =================
# CNN 恐慌贪婪指数接口
CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

# 比特币恐慌贪婪指数接口
CRYPTO_URL = "https://api.alternative.me/fng/?limit=80"

# A股指数：沪深300
ASHARE_CODE = "000300.SS"

# 韭圈儿链接
JIUQUAN_URL = "https://funddb.cn/tool/fear"

# 阈值设定
LIMIT_LOW = 25
LIMIT_HIGH = 75
DANGER_DAYS_THRESHOLD = 10


# ================= 核心逻辑：RSI 计算 =================
def calculate_rsi_history(ticker, period="6mo"):
    """
    根据指数行情计算 RSI。
    yfinance 返回的行情数据一般只包含交易日，因此该函数生成的数据天然接近交易日序列。
    返回格式：[{"date": "YYYY-MM-DD", "value": int}, ...]
    且按日期倒序排列，最新日期在最前。
    """
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)

        if df.empty:
            print(f"❌ {ticker} 行情数据为空")
            return None

        close = df["Close"]

        # 适配 yfinance 新版可能出现的多级列结构
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        history = []

        # 最近 65 个有效 RSI 交易日，倒序排列
        rsi_data = rsi.dropna().iloc[-65:][::-1]

        for date, value in rsi_data.items():
            history.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": int(round(float(value)))
            })

        return history

    except Exception as e:
        print(f"RSI Calculation Error for {ticker}: {e}")
        return None


# ================= 交易日过滤 =================
def filter_by_trading_days(data, ticker, period="8mo"):
    """
    用 yfinance 获取某个指数的真实交易日期，
    再把恐慌贪婪指数数据过滤为交易日数据。

    适用于：
    - 美股：^GSPC
    - A股：000300.SS

    不适用于：
    - 比特币，因为比特币是连续交易，不需要过滤周末。
    """
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

        # 如果过滤后数据太少，说明两个数据源日期可能不匹配，保留原数据以避免统计失败
        if len(filtered) < 30:
            print(f"⚠️ {ticker} 过滤后不足30条，保留原始数据")
            return data

        return filtered

    except Exception as e:
        print(f"Trading day filter failed for {ticker}: {e}")
        return data


# ================= 数据获取接口 =================
def get_us_data():
    """
    获取美股恐慌贪婪指数。
    优先使用 CNN 官方数据，失败时用 S&P 500 RSI 替代。
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Referer": "https://www.cnn.com/",
            "Origin": "https://www.cnn.com"
        }

        res = requests.get(CNN_URL, headers=headers, timeout=10)
        res.raise_for_status()

        data = res.json()["fear_and_greed_historical"]["data"]

        # 时间倒序，最新日期排在最前面
        data.sort(key=lambda x: x["x"], reverse=True)

        formatted = [{
            "date": datetime.fromtimestamp(d["x"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
            "value": int(round(float(d["y"])))
        } for d in data]

        return formatted, "CNN 官方数据"

    except Exception as e:
        print(f"CNN API Failed: {e}, Switching to S&P 500 RSI...")
        rsi = calculate_rsi_history("^GSPC")
        return rsi, "S&P 500 RSI 替代"


def get_crypto_data():
    """
    获取比特币恐慌贪婪指数。
    比特币为连续交易，因此按自然日统计。
    """
    try:
        res = requests.get(CRYPTO_URL, timeout=10)
        res.raise_for_status()

        data = res.json()["data"]

        return [{
            "date": datetime.fromtimestamp(int(d["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%d"),
            "value": int(d["value"])
        } for d in data], "Alternative.me"

    except Exception as e:
        print(f"Crypto API Failed: {e}")
        return None, "获取失败"


def get_cn_data():
    """
    获取A股数据。
    当前用沪深300 RSI 作为A股情绪替代指标。
    """
    data = calculate_rsi_history(ASHARE_CODE)
    return data, "沪深300 RSI"


# ================= 统计分析 =================
def calc_stats(data, period_type="有效观测日"):
    """
    统计近30个、近60个有效数据中的恐慌和贪婪次数。

    对股票类资产：
    - data 已提前过滤为交易日
    - period_type = 交易日

    对比特币：
    - data 保留自然日
    - period_type = 自然日
    """
    if not data:
        return None

    current_val = data[0]["value"]
    current_date = data[0]["date"]

    def count(limit_days):
        sub_data = data[:limit_days]
        low_count = sum(1 for d in sub_data if d["value"] < LIMIT_LOW)
        high_count = sum(1 for d in sub_data if d["value"] > LIMIT_HIGH)
        return low_count, high_count

    l30, h30 = count(30)
    l60, h60 = count(60)

    status_text = get_status_text(current_val)

    return {
        "val": current_val,
        "date": current_date,
        "status": status_text,
        "l30": l30,
        "h30": h30,
        "l60": l60,
        "h60": h60,
        "period_type": period_type
    }


# ================= HTML 工具函数 =================
def get_color(value):
    """
    根据数值返回颜色。
    绿色表示恐慌机会区，红色表示贪婪风险区。
    """
    if value < LIMIT_LOW:
        return "#28a745"
    if value > LIMIT_HIGH:
        return "#dc3545"
    return "#333333"


def get_status_text(value):
    if value < LIMIT_LOW:
        return "极度恐慌（机会）"
    if value > LIMIT_HIGH:
        return "极度贪婪（风险）"
    return "中性震荡"


def fmt_mmdd(date_str):
    """把 YYYY-MM-DD 简化为 MM-DD，用于图表横轴。"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%m-%d")
    except Exception:
        return date_str


def safe_html(text):
    return html.escape(str(text), quote=True)


def generate_svg_line_chart(history_data, period_type):
    """
    生成近30个统计周期的 SVG 折线图。
    不依赖 JS、Chart.js、ECharts，适合 GitHub Actions 自动运行后通过 PushPlus 推送。
    history_data 默认是倒序：最新在前；图中会改为从左到右按时间递增。
    """
    if not history_data:
        return "<div style='font-size:12px;color:#999;text-align:center;'>暂无趋势图数据</div>"

    data = history_data[:30]
    if not data:
        return "<div style='font-size:12px;color:#999;text-align:center;'>暂无趋势图数据</div>"

    # 图表从左到右按时间递增
    data = list(reversed(data))

    width = 560
    height = 220
    left = 42
    right = 16
    top = 18
    bottom = 38
    plot_w = width - left - right
    plot_h = height - top - bottom

    def clamp(v, lo=0, hi=100):
        return max(lo, min(hi, float(v)))

    def x_pos(i):
        if len(data) == 1:
            return left + plot_w / 2
        return left + i * plot_w / (len(data) - 1)

    def y_pos(v):
        v = clamp(v)
        return top + (100 - v) / 100 * plot_h

    points = []
    circles = []

    for i, item in enumerate(data):
        x = x_pos(i)
        y = y_pos(item["value"])
        points.append(f"{x:.1f},{y:.1f}")
        color = get_color(item["value"])
        date = safe_html(item["date"])
        value = safe_html(item["value"])
        status = safe_html(get_status_text(item["value"]))
        circles.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.2' fill='{color}'>"
            f"<title>{date}：{value}，{status}</title>"
            f"</circle>"
        )

    # 横向参考线：0、25、50、75、100
    grid_lines = []
    for v in [0, 25, 50, 75, 100]:
        y = y_pos(v)
        dash = "4 4" if v in [25, 75] else ""
        line_color = "#28a745" if v == 25 else "#dc3545" if v == 75 else "#dddddd"
        grid_lines.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{width - right}' y2='{y:.1f}' "
            f"stroke='{line_color}' stroke-width='1' stroke-dasharray='{dash}' opacity='0.75'/>"
            f"<text x='{left - 8}' y='{y + 4:.1f}' text-anchor='end' font-size='10' fill='#888'>{v}</text>"
        )

    # 横轴日期标签：首日、中间、末日
    date_labels = []
    label_indices = sorted(set([0, len(data) // 2, len(data) - 1]))
    for i in label_indices:
        x = x_pos(i)
        label = safe_html(fmt_mmdd(data[i]["date"]))
        anchor = "middle"
        if i == 0:
            anchor = "start"
        elif i == len(data) - 1:
            anchor = "end"
        date_labels.append(
            f"<text x='{x:.1f}' y='{height - 12}' text-anchor='{anchor}' font-size='10' fill='#888'>{label}</text>"
        )

    start_date = safe_html(data[0]["date"])
    end_date = safe_html(data[-1]["date"])

    svg = f"""
    <div style="margin-top:12px; background:#fbfbfb; border:1px solid #eeeeee; border-radius:8px; padding:8px; overflow-x:auto;">
        <div style="font-size:12px; color:#666; margin-bottom:6px; text-align:center;">
            近30个{safe_html(period_type)}趋势图（{start_date} 至 {end_date}）
        </div>
        <svg width="100%" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="近30个{safe_html(period_type)}恐慌贪婪指数折线图">
            <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" rx="8"/>
            {''.join(grid_lines)}
            <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#dddddd" stroke-width="1"/>
            <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#dddddd" stroke-width="1"/>
            <polyline points="{' '.join(points)}" fill="none" stroke="#007bff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            {''.join(circles)}
            {''.join(date_labels)}
            <text x="{width - right}" y="14" text-anchor="end" font-size="10" fill="#999">75 贪婪线 / 25 恐慌线</text>
        </svg>
    </div>
    """
    return svg


def generate_history_table_html(history_data, period_type):
    """
    生成近30个统计周期每日明细表。
    表格按倒序展示：最新日期在最上方。
    """
    if not history_data:
        return "<div style='font-size:12px;color:#999;text-align:center;margin-top:8px;'>暂无明细数据</div>"

    rows = []
    for item in history_data[:30]:
        value = item["value"]
        color = get_color(value)
        status = get_status_text(value)
        rows.append(f"""
        <tr style="border-bottom:1px solid #eeeeee;">
            <td style="padding:6px; text-align:center;">{safe_html(item['date'])}</td>
            <td style="padding:6px; text-align:center; font-weight:bold; color:{color};">{safe_html(value)}</td>
            <td style="padding:6px; text-align:center; color:{color};">{safe_html(status)}</td>
        </tr>
        """)

    return f"""
    <div style="margin-top:10px; max-height:360px; overflow:auto; border:1px solid #eeeeee; border-radius:8px;">
        <table style="width:100%; font-size:12px; border-collapse:collapse; background:#ffffff; color:#555;">
            <thead>
                <tr style="background:#f6f8fa; border-bottom:1px solid #eeeeee;">
                    <th style="padding:7px; text-align:center;">日期</th>
                    <th style="padding:7px; text-align:center;">指数</th>
                    <th style="padding:7px; text-align:center;">状态</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    <div style="margin-top:6px; font-size:11px; color:#999; text-align:right;">
        注：股票类为近30个{safe_html(period_type)}；比特币为近30个自然日。
    </div>
    """


def generate_history_detail_html(history_data, period_type):
    chart_html = generate_svg_line_chart(history_data, period_type)
    table_html = generate_history_table_html(history_data, period_type)
    return f"""
    <details style="margin-top:12px;">
        <summary style="cursor:pointer; list-style:none; text-align:center; padding:9px 0; background-color:#f1f8ff; color:#0056b3; border:1px solid #b8daff; border-radius:6px; font-size:13px; font-weight:bold;">
            📈 查看近30个{safe_html(period_type)}明细与趋势图
        </summary>
        <div style="margin-top:10px;">
            {chart_html}
            {table_html}
        </div>
    </details>
    """


# ================= HTML 生成器 =================
def generate_card_html(name, source, stats, history_data=None, link=None):
    if not stats:
        return (
            f"<div style='padding:15px; background:#f8d7da; "
            f"border-radius:8px; margin-bottom:15px;'>"
            f"❌ {safe_html(name)} 数据获取失败</div>"
        )

    color = get_color(stats["val"])
    period_type = stats["period_type"]

    warning_html = ""
    if stats["h30"] >= DANGER_DAYS_THRESHOLD:
        warning_html = f"""
        <div style="margin-top:8px; padding:8px; background-color:#fff3cd; color:#856404; border-radius:4px; font-size:12px; border:1px solid #ffeeba;">
            ⚠️ <b>高危预警</b>：近30个{safe_html(period_type)}内已有 {stats['h30']} 次处于极度贪婪区，建议关注高位风险。
        </div>
        """

    history_html = generate_history_detail_html(history_data or [], period_type)

    link_html = ""
    if link:
        link_html = f"""
        <div style="margin-top:12px; text-align:center;">
            <a href="{safe_html(link)}" style="display:inline-block; width:90%; padding:8px 0; background-color:#e7f5ff; color:#0056b3; text-decoration:none; border-radius:4px; font-size:13px; font-weight:bold; border:1px solid #b8daff;">
                👉 点击查看 [韭圈儿] 详情
            </a>
        </div>
        """

    return f"""
    <div style="margin-bottom:15px; padding:15px; background:#fff; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.05); border:1px solid #eee;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #f0f0f0; padding-bottom:10px; margin-bottom:10px;">
            <div>
                <div style="font-size:16px; font-weight:bold; color:#333;">{safe_html(name)}</div>
                <div style="font-size:11px; color:#999;">{safe_html(stats['date'])} | {safe_html(source)}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:26px; font-weight:bold; color:{color}; line-height:1;">{safe_html(stats['val'])}</div>
                <div style="font-size:11px; color:{color}; margin-top:3px;">{safe_html(stats['status'])}</div>
            </div>
        </div>

        <table style="width:100%; font-size:12px; text-align:center; border-collapse:collapse; color:#555; background-color:#f9f9f9; border-radius:6px;">
            <tr style="border-bottom:1px solid #eee;">
                <th style="padding:6px;">周期</th>
                <th style="color:#28a745;">恐慌次数 (&lt;{LIMIT_LOW})</th>
                <th style="color:#dc3545;">贪婪次数 (&gt;{LIMIT_HIGH})</th>
            </tr>
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:6px;">近30个{safe_html(period_type)}</td>
                <td><b>{safe_html(stats['l30'])}</b></td>
                <td><b>{safe_html(stats['h30'])}</b></td>
            </tr>
            <tr>
                <td style="padding:6px;">近60个{safe_html(period_type)}</td>
                <td><b>{safe_html(stats['l60'])}</b></td>
                <td><b>{safe_html(stats['h60'])}</b></td>
            </tr>
        </table>

        {warning_html}
        {history_html}
        {link_html}
    </div>
    """


# ================= 推送发送 =================
def send_push(title, content):
    token = os.getenv("PUSHPLUS_TOKEN")
    topic = os.getenv("PUSHPLUS_TOPIC")

    if not token:
        print("❌ 未检测到 PUSHPLUS_TOKEN，跳过推送")
        return

    url = "http://www.pushplus.plus/send"

    data = {
        "token": token,
        "title": title,
        "content": content,
        "template": "html"
    }

    # topic 为空时，不传 topic，避免部分场景下推送异常
    if topic:
        data["topic"] = topic

    print(f"📡 准备推送到群组: {topic if topic else '无，单人推送'}")

    try:
        res = requests.post(url, json=data, timeout=10)
        print(f"PushPlus Status Code: {res.status_code}")
        print(f"PushPlus Response: {res.text}")
        print("✅ 推送请求已发送")
    except Exception as e:
        print(f"❌ 推送发送失败: {e}")


# ================= 主程序 =================
if __name__ == "__main__":
    print("🚀 开始分析全球市场情绪...")

    parts = []
    html_cards = ""

    tasks = [
        {
            "name": "🇺🇸 美股",
            "getter": get_us_data,
            "link": None,
            "ticker": "^GSPC",
            "period_type": "交易日"
        },
        {
            "name": "₿ 比特币",
            "getter": get_crypto_data,
            "link": None,
            "ticker": None,
            "period_type": "自然日"
        },
        {
            "name": "🇨🇳 A股",
            "getter": get_cn_data,
            "link": JIUQUAN_URL,
            "ticker": ASHARE_CODE,
            "period_type": "交易日"
        }
    ]

    for task in tasks:
        name = task["name"]
        getter = task["getter"]
        link = task["link"]
        ticker = task["ticker"]
        period_type = task["period_type"]

        print(f"\n========== {name} ==========")

        raw_data, source_name = getter()

        if raw_data:
            print(f"{name} 原始数据条数：{len(raw_data)}")

        # 股票类资产按真实交易日过滤；比特币不需要过滤
        if ticker:
            raw_data = filter_by_trading_days(raw_data, ticker)

        stats = calc_stats(raw_data, period_type=period_type)

        html_cards += generate_card_html(
            name=name,
            source=source_name,
            stats=stats,
            history_data=raw_data,
            link=link
        )

        if stats:
            asset_name = name.split(" ", 1)[1] if " " in name else name
            parts.append(f"{asset_name}:{stats['val']}")

            print(
                f"{name} 当前值：{stats['val']}，"
                f"近30个{period_type}贪婪次数：{stats['h30']}，"
                f"恐慌次数：{stats['l30']}"
            )

    # 计算北京时间
    utc_now = datetime.now(timezone.utc)
    beijing_time = utc_now + timedelta(hours=8)
    beijing_time_str = beijing_time.strftime("%Y-%m-%d %H:%M") + "（北京时间）"

    strategy_footer = f"""
    <div style="margin-top:20px; padding:15px; background-color:#e9ecef; border-radius:8px; font-size:12px; color:#555; border-left:4px solid #007bff;">
        <h4 style="margin:0 0 8px 0; color:#333;">📊 自动化定投/止盈策略提示</h4>
        <ul style="padding-left:15px; margin:0; line-height:1.6;">
            <li><span style="color:#28a745; font-weight:bold;">🟢 买入机会</span>：指数 <b>&lt; {LIMIT_LOW}</b> 时，可关注分批定投机会。</li>
            <li><span style="color:#dc3545; font-weight:bold;">🔴 止盈警示</span>：指数 <b>&gt; {LIMIT_HIGH}</b> 时，可关注分批止盈风险。</li>
            <li><span style="background:#fff3cd; padding:2px 4px; border-radius:2px;">⚠️ <b>高危信号</b></span>：股票类资产按交易日统计，比特币按自然日统计；若近30个统计周期内大于 {LIMIT_HIGH} 的次数超过 <b>{DANGER_DAYS_THRESHOLD} 次</b>，建议关注高位风险。</li>
            <li>每张卡片中的“查看近30个统计周期明细与趋势图”可展开查看每日指数和折线图。</li>
        </ul>
        <div style="margin-top:8px; text-align:right; font-size:11px; color:#999;">
            Data Updated: {safe_html(beijing_time_str)}
        </div>
    </div>
    """

    full_html = f"""
    <html>
    <body style="background-color:#f4f6f9; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
        <div style="max-width:600px; margin:0 auto;">
            <h3 style="text-align:center; color:#333; margin:20px 0;">🌍 全球核心资产情绪监控</h3>
            {html_cards}
            {strategy_footer}
        </div>
    </body>
    </html>
    """

    if parts:
        title_date = beijing_time.strftime("%m-%d")
        title = f"{title_date} | " + " | ".join(parts)
        send_push(title, full_html)
    else:
        print("❌ 所有数据获取失败，未发送推送")
