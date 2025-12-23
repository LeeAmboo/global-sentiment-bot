import requests
import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# ================= 配置区域 =================
CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CRYPTO_URL = "https://api.alternative.me/fng/?limit=80"
ASHARE_CODE = "000300.SS"
# 韭圈儿链接
JIUQUAN_URL = "https://funddb.cn/tool/fear"

# 阈值设定
LIMIT_LOW = 25  # 恐慌/买入线
LIMIT_HIGH = 75 # 贪婪/卖出线
DANGER_DAYS_THRESHOLD = 10 # 30天内超过多少天贪婪算高危

# ================= 核心逻辑：RSI 计算 =================
def calculate_rsi_history(ticker, period="5mo"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty: return None

        # 处理多级索引 (适配 yfinance 新版)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        history = []
        # 取最近 65 天并反转
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

# ================= 数据获取接口 =================
def get_us_data():
    """获取美股数据 (优先CNN, 失败切RSI)"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.cnn.com/",
            "Origin": "https://www.cnn.com"
        }
        res = requests.get(CNN_URL, headers=headers, timeout=10)
        data = res.json()["fear_and_greed_historical"]["data"]
        data.sort(key=lambda x: x["x"], reverse=True)
        formatted = [{
            "date": datetime.fromtimestamp(d["x"] / 1000).strftime("%Y-%m-%d"),
            "value": int(d["y"])
        } for d in data]
        return formatted, "CNN 官方数据"
    except:
        print("Switching to S&P 500 RSI...")
        rsi = calculate_rsi_history("^GSPC")
        return rsi, "S&P 500 RSI (替代)"

def get_crypto_data():
    """获取加密货币数据"""
    try:
        res = requests.get(CRYPTO_URL, timeout=10)
        data = res.json()["data"]
        return [{
            "date": datetime.fromtimestamp(int(d["timestamp"])).strftime("%Y-%m-%d"),
            "value": int(d["value"])
        } for d in data], "Alternative.me"
    except:
        return None, "获取失败"

def get_cn_data():
    """获取A股数据"""
    data = calculate_rsi_history(ASHARE_CODE)
    return data, "沪深300 RSI (Yahoo)"

# ================= 统计分析 =================
def calc_stats(data):
    if not data: return None
    
    current_val = data[0]["value"]
    current_date = data[0]["date"]

    def count(limit_days):
        sub_data = data[:limit_days]
        low_count = sum(1 for d in sub_data if d["value"] < LIMIT_LOW)
        high_count = sum(1 for d in sub_data if d["value"] > LIMIT_HIGH)
        return low_count, high_count

    l30, h30 = count(30)
    l60, h60 = count(60)

    # 判断当前状态文案
    status_text = "中性震荡"
    if current_val < LIMIT_LOW: status_text = "极度恐慌 (机会)"
    elif current_val > LIMIT_HIGH: status_text = "极度贪婪 (风险)"

    return {
        "val": current_val,
        "date": current_date,
        "status": status_text,
        "l30": l30, "h30": h30,
        "l60": l60, "h60": h60
    }

# ================= HTML 生成器 (UI优化核心) =================
def get_color(value):
    """根据数值返回颜色 (绿买红卖逻辑)"""
    if value < LIMIT_LOW: return "#28a745" # 绿色 (机会)
    if value > LIMIT_HIGH: return "#dc3545" # 红色 (风险)
    return "#333333" # 黑色 (中性)

def generate_card_html(name, source, stats, link=None):
    if not stats:
        return f"<div style='padding:15px; background:#f8d7da; border-radius:8px; margin-bottom:15px;'>❌ {name} 数据获取失败</div>"

    color = get_color(stats['val'])
    
    # 风险提示逻辑
    warning_html = ""
    if stats['h30'] >= DANGER_DAYS_THRESHOLD:
        warning_html = f"""
        <div style="margin-top:8px; padding:8px; background-color:#fff3cd; color:#856404; border-radius:4px; font-size:12px; border:1px solid #ffeeba;">
            ⚠️ <b>高危预警</b>：近30天内已有 {stats['h30']} 天处于极度贪婪区，建议止盈！
        </div>
        """

    # 链接按钮逻辑
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
    <div style="margin-bottom:15px; padding:15px; background:#fff; border-radius:12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border:1px solid #eee;">
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
                <th style="color:#28a745;">恐慌天数 (<{LIMIT_LOW})</th>
                <th style="color:#dc3545;">贪婪天数 (>{LIMIT_HIGH})</th>
            </tr>
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:6px;">近30天</td>
                <td><b>{stats['l30']}</b></td>
                <td><b>{stats['h30']}</b></td>
            </tr>
            <tr>
                <td style="padding:6px;">近60天</td>
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
    if not token: 
        print("❌ 未检测到 Token，跳过推送")
        return
    
    url = "http://www.pushplus.plus/send"
    data = {
        "token": token,
        "title": title,
        "content": content,
        "template": "html"
    }
    try:
        requests.post(url, json=data, timeout=10)
        print("✅ 推送已发送")
    except Exception as e:
        print(f"❌ 推送发送失败: {e}")

# ================= 主程序 =================
if __name__ == "__main__":
    print("🚀 开始分析全球市场情绪...")
    
    parts = []
    html_cards = ""

    # 定义任务列表
    tasks = [
        ("🇺🇸 美股", get_us_data, None),
        ("₿ 比特币", get_crypto_data, None),
        ("🇨🇳 A股", get_cn_data, JIUQUAN_URL)
    ]

    for name, getter, link in tasks:
        # 获取数据
        raw_data, source_name = getter()
        # 计算统计
        stats = calc_stats(raw_data)
        
        # 生成卡片 HTML
        html_cards += generate_card_html(name, source_name, stats, link)
        
        # 如果成功获取，添加到标题
        if stats:
            parts.append(f"{name.split(' ')[1]}:{stats['val']}")

    # 生成策略提示脚部 (Footer)
    strategy_footer = f"""
    <div style="margin-top:20px; padding:15px; background-color:#e9ecef; border-radius:8px; font-size:12px; color:#555; border-left: 4px solid #007bff;">
        <h4 style="margin:0 0 8px 0; color:#333;">📊 自动化定投/止盈策略提示</h4>
        <ul style="padding-left:15px; margin:0; line-height:1.6;">
            <li><span style="color:#28a745; font-weight:bold;">🟢 买入机会</span>：指数 <b>&lt; {LIMIT_LOW}</b> 时，建议开启分批定投。</li>
            <li><span style="color:#dc3545; font-weight:bold;">🔴 止盈警示</span>：指数 <b>&gt; {LIMIT_HIGH}</b> 时，建议分批止盈。</li>
            <li><span style="background:#fff3cd; padding:2px 4px; border-radius:2px;">⚠️ <b>高危信号</b></span>：若近30个交易日内，大于{LIMIT_HIGH}的天数超过 <b>{DANGER_DAYS_THRESHOLD}天</b>，建议大幅减仓止盈。</li>
        </ul>
        <div style="margin-top:8px; text-align:right; font-size:11px; color:#999;">
            Data Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>
    """

    # 组合最终 HTML
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
        today_str = datetime.now().strftime("%m-%d")
        title = f"{today_str} | " + " | ".join(parts)
        send_push(title, full_html)
    else:
        print("❌ 所有数据获取失败，未发送推送")
