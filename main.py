import requests
import os
from datetime import datetime

# === 1. 数据源配置 ===
# 美股 (CNN 官方接口)
US_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
# 比特币 (Alternative.me 行业标准接口)
CRYPTO_URL = "https://api.alternative.me/fng/?limit=60"
# A股 (且慢-沪深300温度接口，能稳定提供历史数据用于统计)
CN_URL = "https://qieman.com/pmdd/data-service/idx-eval/daily-eval?idxCode=000300"

# === 2. 数据获取函数 ===

def get_us_data():
    """获取美股数据"""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(US_URL, headers=headers, timeout=15)
        res.raise_for_status()
        data = res.json()['fear_and_greed_historical']['data']
        # 排序：最新日期在前
        data.sort(key=lambda x: x['x'], reverse=True)
        formatted = []
        for item in data:
            formatted.append({
                'date': datetime.fromtimestamp(item['x'] / 1000).strftime('%Y-%m-%d'),
                'value': int(item['y'])
            })
        return formatted
    except Exception as e:
        print(f"美股获取失败: {e}")
        return None

def get_crypto_data():
    """获取比特币数据"""
    try:
        res = requests.get(CRYPTO_URL, timeout=15)
        res.raise_for_status()
        data = res.json()['data']
        formatted = []
        for item in data:
            formatted.append({
                'date': datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d'),
                'value': int(item['value'])
            })
        return formatted
    except Exception as e:
        print(f"BTC获取失败: {e}")
        return None

def get_cn_data():
    """获取A股数据(且慢温度)"""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(CN_URL, headers=headers, timeout=15)
        res.raise_for_status()
        data = res.json() 
        # 且慢API返回按日期升序，只取最近65天即可
        history = data[-65:] 
        history.reverse() # 反转为最新在前
        
        formatted = []
        for item in history:
            # 转换时间戳
            d_str = datetime.fromtimestamp(item['date'] / 1000).strftime('%Y-%m-%d')
            # 转换数值 (API返回的是小数如0.15，转换为15)
            val = int(item['pePercentile'] * 100)
            formatted.append({'date': d_str, 'value': val})
        return formatted
    except Exception as e:
        print(f"A股获取失败: {e}")
        return None

# === 3. 统计计算函数 ===

def calculate_stats(history_data, market_name):
    if not history_data:
        return None
    
    current = history_data[0]
    
    def count_days(limit):
        target = history_data[:limit]
        # 统计规则：小于25 或 大于75
        low = sum(1 for d in target if d['value'] < 25)
        high = sum(1 for d in target if d['value'] > 75)
        return low, high

    l30, h30 = count_days(30)
    l60, h60 = count_days(60)
    
    return {
        "name": market_name,
        "val": current['value'],
        "date": current['date'],
        "L30": l30, "H30": h30,
        "L60": l60, "H60": h60
    }

def get_color(value):
    if value < 25: return "#28a745" # 绿色 (买入机会)
    if value > 75: return "#dc3545" # 红色 (卖出风险)
    return "black"

# === 4. 生成推送内容 ===

def generate_html_block(stats):
    if not stats: return "<div>数据获取失败</div>"
    color = get_color(stats['val'])
    
    return f"""
    <div style="margin-bottom:15px; padding:12px; background:#f8f9fa; border-radius:8px; border:1px solid #ddd;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #eee; padding-bottom:5px; margin-bottom:5px;">
            <span style="font-weight:bold; font-size:15px;">{stats['name']}</span>
            <span style="font-weight:bold; font-size:22px; color:{color}">{stats['val']}</span>
        </div>
        
        <table style="width:100%; font-size:12px; text-align:center; border-collapse:collapse;">
            <tr style="background:#eee;"><th>范围</th><th><25 (恐慌)</th><th>>75 (贪婪)</th></tr>
            <tr><td>近30天</td><td><b>{stats['L30']}</b> 天</td><td><b>{stats['H30']}</b> 天</td></tr>
            <tr><td>近60天</td><td><b>{stats['L60']}</b> 天</td><td><b>{stats['H60']}</b> 天</td></tr>
        </table>
    </div>
    """

def send_push(title, content):
    token = os.environ.get("PUSHPLUS_TOKEN")
    if not token:
        print("未配置Token")
        return
    url = "http://www.pushplus.plus/send"
    data = {"token": token, "title": title, "content": content, "template": "html"}
    requests.post(url, json=data)

if __name__ == "__main__":
    print("开始执行分析...")
    
    # 获取三大市场
    us = calculate_stats(get_us_data(), "🇺🇸 美股 (CNN)")
    btc = calculate_stats(get_crypto_data(), "₿ 比特币 (BTC)")
    cn = calculate_stats(get_cn_data(), "🇨🇳 A股 (沪深300温度)")
    
    # 拼装标题
    parts = []
    if us: parts.append(f"美:{us['val']}")
    if btc: parts.append(f"币:{btc['val']}")
    if cn: parts.append(f"A:{cn['val']}")
    title = " | ".join(parts) + " [全球情绪日报]"
    
    # 拼装正文
    html = f"""
    <html><body>
    <h3 style="text-align:center;">🌍 全球核心资产情绪监控</h3>
    <p style="text-align:center;color:gray;font-size:12px">{datetime.now().strftime('%Y-%m-%d')}</p>
    {generate_html_block(us)}
    {generate_html_block(btc)}
    {generate_html_block(cn)}
    <p style="font-size:12px; color:gray; text-align:center;">
    A股使用且慢市场温度(0-100)，原理同恐慌指数<br>
    <span style="color:#28a745">绿色 < 25</span> | <span style="color:#dc3545">红色 > 75</span>
    </p>
    </body></html>
    """
    
    send_push(title, html)
    print("推送完成")
