import requests
import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# === 配置区域 ===
US_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CRYPTO_URL = "https://api.alternative.me/fng/?limit=60"
# A股代码: 沪深300 (Yahoo Finance 代码为 000300.SS)
ASHARE_CODE = "000300.SS"

# === 1. 美股 & 比特币 (保持原样，因为 API 很稳定) ===

def get_us_data():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.cnn.com/",
        "Origin": "https://www.cnn.com"
    }
    try:
        res = requests.get(US_URL, headers=headers, timeout=15)
        res.raise_for_status()
        data = res.json()['fear_and_greed_historical']['data']
        data.sort(key=lambda x: x['x'], reverse=True)
        formatted = []
        for item in data:
            formatted.append({
                'date': datetime.fromtimestamp(item['x'] / 1000).strftime('%Y-%m-%d'),
                'value': int(item['y'])
            })
        return formatted
    except Exception as e:
        print(f"美股获取错误: {e}")
        return None

def get_crypto_data():
    try:
        res = requests.get(CRYPTO_URL, timeout=15)
        data = res.json()['data']
        formatted = []
        for item in data:
            formatted.append({
                'date': datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d'),
                'value': int(item['value'])
            })
        return formatted
    except Exception as e:
        print(f"BTC获取错误: {e}")
        return None

# === 2. A股 (自主计算 RSI 情绪指标) ===

def calculate_rsi(series, period=14):
    """计算 RSI 指标"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_ashare_sentiment():
    """通过 Yahoo Finance 获取数据并计算情绪"""
    print("正在通过 Yahoo Finance 计算 A 股情绪...")
    try:
        # 获取过去 4 个月的数据(保证有足够的计算窗口)
        # 沪深300
        df = yf.download(ASHARE_CODE, period="4mo", progress=False)
        
        if df.empty:
            print("A股数据下载为空")
            return None
            
        # 计算 RSI (14天)
        # 注意：yfinance 返回的 Close 可能是多级索引，确保取值正确
        close_price = df['Close']
        if isinstance(close_price, pd.DataFrame):
            close_price = close_price.iloc[:, 0]
            
        rsi = calculate_rsi(close_price)
        
        # 截取最近 60 天的数据
        # 将 Series 转换为我们要的 list 格式
        history = []
        # 按时间降序 (最新在前)
        recent_rsi = rsi.iloc[-65:].iloc[::-1] 
        
        for date, value in recent_rsi.items():
            if pd.isna(value): continue
            history.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': int(value) # RSI 也是 0-100
            })
            
        print(f"✅ A股(RSI)计算成功，当前值: {history[0]['value']}")
        return history
    except Exception as e:
        print(f"A股计算失败: {e}")
        return None

# === 3. 统计与推送 ===

def calculate_stats(history_data, market_name, is_rsi=False):
    if not history_data: return None
    current = history_data[0]
    
    # 阈值设定
    # 对于恐慌指数: <25 恐慌, >75 贪婪
    # 对于 RSI (A股): <30 超卖(恐慌), >70 超买(贪婪) 是标准定义，这里为了统一体验，我们依然沿用 30/70 或 25/75
    # 建议 A股 RSI 使用 30/70 作为界限更准确，或者您可以手动调整下方数字
    
    limit_low = 30 if is_rsi else 25
    limit_high = 70 if is_rsi else 75
    
    def count_days(limit):
        target = history_data[:limit]
        low = sum(1 for d in target if d['value'] < limit_low)
        high = sum(1 for d in target if d['value'] > limit_high)
        return low, high

    l30, h30 = count_days(30)
    l60, h60 = count_days(60)
    
    return {
        "name": market_name, "val": current['value'], "date": current['date'],
        "L30": l30, "H30": h30, "L60": l60, "H60": h60,
        "limit_low": limit_low, "limit_high": limit_high,
        "desc": "RSI指标" if is_rsi else "恐慌指数"
    }

def get_color(value, is_rsi=False):
    low = 30 if is_rsi else 25
    high = 70 if is_rsi else 75
    if value < low: return "#28a745" # 绿
    if value > high: return "#dc3545" # 红
    return "black"

def generate_html_block(stats):
    if not stats: return ""
    color = get_color(stats['val'], stats['desc'] == "RSI指标")
    
    return f"""
    <div style="margin-bottom:15px; padding:12px; background:#f8f9fa; border-radius:8px; border:1px solid #ddd;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #eee; padding-bottom:5px; margin-bottom:5px;">
            <span style="font-weight:bold; font-size:15px;">{stats['name']}</span>
            <span style="font-weight:bold; font-size:22px; color:{color}">{stats['val']}</span>
        </div>
        <div style="font-size:12px; color:#666; margin-bottom:5px;">
            指标: {stats['desc']} | 更新: {stats['date']}
        </div>
        <table style="width:100%; font-size:12px; text-align:center; border-collapse:collapse;">
            <tr style="background:#eee;">
                <th>统计周期</th>
                <th>恐慌 (<{stats['limit_low']})</th>
                <th>贪婪 (>{stats['limit_high']})</th>
            </tr>
            <tr><td>近30天</td><td><b>{stats['L30']}</b> 天</td><td><b>{stats['H30']}</b> 天</td></tr>
            <tr><td>近60天</td><td><b>{stats['L60']}</b> 天</td><td><b>{stats['H60']}</b> 天</td></tr>
        </table>
    </div>
    """

def send_push(title, content):
    token = os.environ.get("PUSHPLUS_TOKEN")
    if not token: return
    url = "http://www.pushplus.plus/send"
    data = {"token": token, "title": title, "content": content, "template": "html"}
    requests.post(url, json=data)

if __name__ == "__main__":
    print("开始执行...")
    
    us = calculate_stats(get_us_data(), "🇺🇸 美股 (CNN)")
    btc = calculate_stats(get_crypto_data(), "₿ 比特币 (BTC)")
    # A股使用 RSI 模式
    cn = calculate_stats(get_ashare_sentiment(), "🇨🇳 A股 (沪深300)", is_rsi=True)
    
    parts = []
    html_body = ""
    
    if us: 
        parts.append(f"美:{us['val']}")
        html_body += generate_html_block(us)
    else: html_body += "<div>❌ 美股获取失败</div>"
        
    if btc: 
        parts.append(f"币:{btc['val']}")
        html_body += generate_html_block(btc)
    else: html_body += "<div>❌ BTC获取失败</div>"

    if cn: 
        parts.append(f"A:{cn['val']}")
        html_body += generate_html_block(cn)
    else: html_body += "<div>❌ A股获取失败 (Yahoo连接错误)</div>"
    
    title = " | ".join(parts) + " [全球情绪日报]"
    
    full_html = f"""
    <html><body>
    <h3 style="text-align:center;">🌍 全球核心资产情绪监控</h3>
    <p style="text-align:center;color:gray;font-size:12px">{datetime.now().strftime('%Y-%m-%d')}</p>
    <hr>
    {html_body}
    <div style="font-size:12px; color:gray; margin-top:20px; padding:10px; background:#eee;">
    <b>指标说明：</b><br>
    1. 美股/BTC 使用官方恐慌指数。<br>
    2. <b>A股使用 RSI 技术指标</b> (因官方IP封锁)：<br>
       基于沪深300指数真实交易数据计算。<br>
       • RSI < 30: 极度超卖 (恐慌/机会)<br>
       • RSI > 70: 极度超买 (贪婪/风险)
    </div>
    </body></html>
    """
    
    if parts: # 至少有一个成功才推送
        send_push(title, full_html)
    print("完成")
