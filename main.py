import requests
import os
import time
import yfinance as yf
import pandas as pd
from datetime import datetime

# === 配置区域 ===
# 1. 美股
CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
# 2. 比特币
CRYPTO_URL = "https://api.alternative.me/fng/?limit=60"
# 3. A股 (Yahoo Finance 代码: 沪深300)
ASHARE_CODE = "000300.SS"
# 4. A股跳转链接 (韭圈儿)
JIUQUAN_URL = "https://funddb.cn/tool/fear"

# === 辅助工具：计算 RSI ===
def calculate_rsi_history(ticker, period="4mo"):
    """
    通用函数：下载行情并计算 RSI 历史数据
    返回格式：[{'date': 'YYYY-MM-DD', 'value': 55}, ...]
    """
    try:
        # 下载数据
        df = yf.download(ticker, period=period, progress=False)
        if df.empty: return None
        
        # 处理多级索引问题 (yfinance 新版特性)
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
            
        # 计算 RSI (14天标准)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 格式化输出 (取最近 65 天，反转为最新在前)
        history = []
        # 翻转数据
        recent_rsi = rsi.iloc[-65:].iloc[::-1]
        
        for date, value in recent_rsi.items():
            if pd.isna(value): continue
            history.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': int(value)
            })
        return history
    except Exception as e:
        print(f"RSI计算错误 ({ticker}): {e}")
        return None

# === 核心数据获取 ===

def get_us_data():
    """美股：优先 CNN API，失败则自动切换 SPX RSI"""
    print("正在获取美股数据...")
    
    # --- 方案 A: CNN 官方 API ---
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.cnn.com/",
            "Origin": "https://www.cnn.com"
        }
        res = requests.get(CNN_URL, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()['fear_and_greed_historical']['data']
            data.sort(key=lambda x: x['x'], reverse=True)
            formatted = []
            for item in data:
                formatted.append({
                    'date': datetime.fromtimestamp(item['x'] / 1000).strftime('%Y-%m-%d'),
                    'value': int(item['y'])
                })
            print("✅ 美股 (CNN API) 获取成功")
            return formatted, "CNN 官方指数"
    except Exception as e:
        print(f"⚠️ CNN 接口访问失败: {e}，正在切换备用方案...")

    # --- 方案 B: S&P 500 RSI (备用) ---
    print("🔄 启动备用方案: 计算 S&P 500 RSI...")
    rsi_data = calculate_rsi_history("^GSPC") # S&P 500 代码
    if rsi_data:
        print("✅ 美股 (S&P 500 RSI) 计算成功")
        return rsi_data, "S&P 500 RSI (替代)"
    
    return None, "获取失败"

def get_crypto_data():
    """比特币：Alternative.me API"""
    try:
        res = requests.get(CRYPTO_URL, timeout=15)
        data = res.json()['data']
        formatted = []
        for item in data:
            formatted.append({
                'date': datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d'),
                'value': int(item['value'])
            })
        return formatted, "Crypto Fear & Greed"
    except Exception as e:
        print(f"BTC 获取失败: {e}")
        return None, "获取失败"

def get_cn_data():
    """A股：沪深300 RSI"""
    data = calculate_rsi_history(ASHARE_CODE)
    if data:
        return data, "沪深300 RSI"
    return None, "获取失败"

# === 统计与报告生成 ===

def calculate_stats(history_data, market_name, source_name, link=None):
    if not history_data: return None
    
    current = history_data[0]
    
    # 阈值判断 (RSI 和 恐慌指数 通用 <30/25 为机会)
    # 为了统一体验，我们设定：
    # 恐慌/超卖: < 25
    # 贪婪/超买: > 75
    LIMIT_LOW = 25
    LIMIT_HIGH = 75
    
    def count_days(limit):
        target = history_data[:limit]
        low = sum(1 for d in target if d['value'] < LIMIT_LOW)
        high = sum(1 for d in target if d['value'] > LIMIT_HIGH)
        return low, high

    l30, h30 = count_days(30)
    l60, h60 = count_days(60)
    
    return {
        "name": market_name,
        "source": source_name,
        "val": current['value'],
        "date": current['date'],
        "L30": l30, "H30": h30,
        "L60": l60, "H60": h60,
        "link": link
    }

def get_color(value):
    if value < 25: return "#28a745" # 绿
    if value > 75: return "#dc3545" # 红
    return "black"

def generate_html_card(stats):
    if not stats: return "<div style='color:red'>❌ 数据获取失败</div>"
    
    color = get_color(stats['val'])
    
    # 额外链接按钮
    link_html = ""
    if stats.get('link'):
        link_html = f"""
        <div style="margin-top:10px; text-align:center;">
            <a href="{stats['link']}" style="display:inline-block; padding:8px 15px; background-color:#e7f5ff; color:#0056b3; text-decoration:none; border-radius:4px; font-size:12px; border:1px solid #b8daff;">
                👉 点击查看 [韭圈儿] 详情
            </a>
        </div>
        """
    
    return f"""
    <div style="margin-bottom:15px; padding:15px; background:#fff; border-radius:10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); border:1px solid #eee;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #f0f0f0; padding-bottom:8px; margin-bottom:8px;">
            <div>
                <b style="font-size:16px; color:#333;">{stats['name']}</b>
                <div style="font-size:11px; color:#999;">{stats['date']} | {stats['source']}</div>
            </div>
            <span style="font-weight:bold; font-size:24px; color:{color}">{stats['val']}</span>
        </div>
        
        <table style="width:100%; font-size:12px; text-align:center; border-collapse:collapse; color:#555;">
            <tr style="background:#f8f9fa;">
                <th style="padding:5px;">周期</th>
                <th>恐慌 (<25)</th>
                <th>贪婪 (>75)</th>
            </tr>
            <tr><td style="padding:5px;">近30天</td><td><b>{stats['L30']}</b> 天</td><td><b>{stats['H30']}</b> 天</td></tr>
            <tr><td style="padding:5px;">近60天</td><td><b>{stats['L60']}</b> 天</td><td><b>{stats['H60']}</b> 天</td></tr>
        </table>
        {link_html}
    </div>
    """

def send_push(title, content):
    token = os.environ.get("PUSHPLUS_TOKEN")
    if not token: return
    url = "http://www.pushplus.plus/send"
    data = {"token": token, "title": title, "content": content, "template": "html"}
    requests.post(url, json=data)

if __name__ == "__main__":
    print("🚀 启动全球市场扫描...")
    
    # 1. 获取数据
    # 美股
    us_data, us_src = get_us_data()
    us_stats = calculate_stats(us_data, "🇺🇸 美股", us_src)
    
    # BTC
    btc_data, btc_src = get_crypto_data()
    btc_stats = calculate_stats(btc_data, "₿ 比特币", btc_src)
    
    # A股 (带链接)
    cn_data, cn_src = get_cn_data()
    cn_stats = calculate_stats(cn_data, "🇨🇳 A股", cn_src, link=JIUQUAN_URL)
    
    # 2. 准备推送
    parts = []
    html_body = ""
    
    if us_stats: 
        parts.append(f"美:{us_stats['val']}")
        html_body += generate_html_card(us_stats)
    else: html_body += "<div>❌ 美股获取失败</div>"
        
    if btc_stats: 
        parts.append(f"币:{btc_stats['val']}")
        html_body += generate_html_card(btc_stats)
        
    if cn_stats: 
        parts.append(f"A:{cn_stats['val']}")
        html_body += generate_html_card(cn_stats)
    else: html_body += "<div>❌ A股获取失败</div>"
    
    # 标题加上日期
    today_str = datetime.now().strftime('%m-%d')
    title = f"{today_str} | " + " | ".join(parts)
    
    full_html = f"""
    <html>
    <body style="background-color:#f4f6f9; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
        <div style="max-width:600px; margin:0 auto;">
            <h3 style="text-align:center; color:#333; margin-top:20px;">🌍 全球核心资产情绪监控</h3>
            {html_body}
            <div style="text-align:center; font-size:12px; color:#aaa; margin-bottom:20px;">
                策略提示：绿色分批定投，红色分批止盈
            </div>
        </div>
    </body>
    </html>
    """
    
    # 只有当至少有一个数据成功时才推送
    if parts:
        send_push(title, full_html)
        print("✅ 推送完成")
    else:
        print("❌ 所有数据获取失败，取消推送")
