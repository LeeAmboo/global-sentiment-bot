import requests
import os
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
                "value": int(value)
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
            "date": datetime.fromtimestamp(d["x"] / 1000).strftime("%Y-%m-%d"),
            "value": int(d["y"])
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
            "date": datetime.fromtimestamp(int(d["timestamp"])).strftime("%Y-%m-%d"),
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

    status_text = "中性震荡"
    if current_val < LIMIT_LOW:
        status_text = "极度恐慌（机会）"
    elif current_val > LIMIT_HIGH:
        status_text = "极度贪婪（风险）"

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


# ================= HTML 生成器 =================
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


def generate_card_html(name, source, stats, link=None):
    if not stats:
        return (
            f"<div style='padding:15px; background:#f8d7da; "
            f"border-radius:8px; margin-bottom:15px;'>"
            f"❌ {name} 数据获取失败</div>"
        )

    color = get_color(stats["val"])
    period_type = stats["period_type"]

    warning_html = ""
    if stats["h30"] >= DANGER_DAYS_THRESHOLD:
        warning_html = f"""
        <div style="margin-top:8px; padding:8px; background-color:#fff3cd; color:#856404; border-radius:4px; font-size:12px; border:1px solid #ffeeba;">
            ⚠️ <b>高危预警</b>：近30个{period_type}内已有 {stats['h30']} 次处于极度贪婪区，建议关注高位风险。
        </div>
        """

    link_html = ""
    if link:
        link_html = f"""
        <div style="margin-top:12px; text-align:center;">
            <a href="{link}" style="display:inline-block; width:90%; padding:8px 0; background-color:#e7f5ff; color:#0056b3; text-decoration:none; border-radius:4px; font-size:13px; font-weight:bold; border:1px solid #b8daff;">
                👉 点击查看 [韭圈儿] 详情
            </a>
        </div>
        """

    return f"""
    <div style="margin-bottom:15px; padding:15px; background:#fff; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.05); border:1px solid #eee;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #f0f0f0; padding-bottom:10px; margin-bottom:10px;">
            <div>
                <div style="font-size:16px; font-weight:bold; color:#333;">{name}</div>
                <div style="font-size:11px; color:#999;">{stats['date']} | {source}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:26px; font-weight:bold; color:{color}; line-height:1;">{stats['val']}</div>
                <div style="font-size:11px; color:{color}; margin-top:3px;">{stats['status']}</div>
            </div>
        </div>

        <table style="width:100%; font-size:12px; text-align:center; border-collapse:collapse; color:#555; background-color:#f9f9f9; border-radius:6px;">
            <tr style="border-bottom:1px solid #eee;">
                <th style="padding:6px;">周期</th>
                <th style="color:#28a745;">恐慌次数 (&lt;{LIMIT_LOW})</th>
                <th style="color:#dc3545;">贪婪次数 (&gt;{LIMIT_HIGH})</th>
            </tr>
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:6px;">近30个{period_type}</td>
                <td><b>{stats['l30']}</b></td>
                <td><b>{stats['h30']}</b></td>
            </tr>
            <tr>
                <td style="padding:6px;">近60个{period_type}</td>
                <td><b>{stats['l60']}</b></td>
                <td><b>{stats['h60']}</b></td>
            </tr>
        </table>

        {warning_html}

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

        # 股票类资产按真实交易日过滤
        if ticker:
            raw_data = filter_by_trading_days(raw_data, ticker)

        stats = calc_stats(raw_data, period_type=period_type)

        html_cards += generate_card_html(name, source_name, stats, link)

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
        </ul>
        <div style="margin-top:8px; text-align:right; font-size:11px; color:#999;">
            Data Updated: {beijing_time_str}
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
