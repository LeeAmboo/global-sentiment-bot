import requests
import os
import time
from datetime import datetime

# === 配置区域 ===
# 美股 (CNN)
US_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
# 比特币
CRYPTO_URL = "https://api.alternative.me/fng/?limit=60"
# A股 (且慢)
CN_URL = "https://qieman.com/pmdd/data-service/idx-eval/daily-eval?idxCode=000300"

# === 通用伪装头 (假装是 Chrome 浏览器) ===
COMMON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "Connection": "keep-alive"
}

def get_us_data():
    """获取美股数据"""
    print("正在获取美股数据...")
    headers = COMMON_HEADERS.copy()
    headers["Referer"] = "https://www.cnn.com/"
    headers["Origin"] = "https://www.cnn.com"
    
    try:
        # CNN 有时候会因为网络波动超时，重试一次
        try:
            res = requests.get(US_URL, headers=headers, timeout=20)
        except:
            time.sleep(2)
            res = requests.get(US_URL, headers=headers, timeout=20)
            
        if res.status_code != 200:
            print(f"❌ 美股请求被拦截: Status {res.status_code}")
            return None
            
        data = res.json()['fear_and_greed_historical']['data']
        data.sort(key=lambda x: x['x'], reverse=True)
        formatted = []
        for item in data:
            formatted.append({
                'date': datetime.fromtimestamp(item['x'] / 1000).strftime('%Y-%m-%d'),
                'value': int(item['y'])
            })
        print(f"✅ 美股获取成功，最新值: {formatted[0]['value']}")
        return formatted
    except Exception as e:
        print(f"❌ 美股获取报错: {e}")
        return None

def get_crypto_data():
    """获取比特币数据"""
    print("正在获取BTC数据...")
    try:
        res = requests.get(CRYPTO_URL, headers=COMMON_HEADERS, timeout=20)
        data = res.json()['data']
        formatted = []
        for item in data:
            formatted.append({
                'date': datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d'),
                'value': int(item['value'])
            })
        print(f"✅ BTC获取成功，最新值: {formatted[0]['value']}")
        return formatted
    except Exception as e:
        print(f"❌ BTC获取报错: {e}")
        return None

def get_cn_data():
    """获取A股数据"""
    print("正在获取A股数据...")
    headers = COMMON_HEADERS.copy()
    # A股必须要有且慢的 Referer，否则会被认为是盗链
    headers["Referer"] = "https://qieman.com/idx"
    headers["Host"] = "qieman.com"
    
    try:
        res = requests.get(CN_URL, headers=headers, timeout=20)
        
        if res.status_code != 200:
            print(f"❌ A股请求被拦截: Status {res.status_code}")
            # 如果被拦截，尝试打印一点内容看看是不是验证码
            # print(res.text[:100]) 
            return None

        data = res.json()
        history = data[-65:]
        history.reverse()
        
        formatted = []
        for item in history:
            d_str = datetime.fromtimestamp(item['date'] / 1000).strftime('%Y-%m-%d')
            val = int(item['pePercentile'] * 100)
            formatted.append({'date': d_str, 'value': val})
            
        print(f"✅ A股获取成功，最新值: {formatted[0]['value']}")
        return formatted
    except Exception as e:
        print(f"❌ A股获取报错: {e}")
        return None

def calculate_stats(history_data, market_name):
    if not history_data:
        return None
    current = history_data[0]
    
    def count_days(limit):
        target = history_data[:limit]
        low = sum(1 for d in target if d['value'] < 25)
        high = sum(1 for d in target if d['value'] > 75)
        return low, high

    l30, h30 = count_days(30)
    l60, h60 = count_days(60)
    
    return {
        "name": market_name, "val": current['value'], "date": current['date'],
        "L30": l30, "H30": h30, "L60": l60, "H60": h60
    }

def get_color(value):
    if value < 25: return "#28a745"
    if value > 75: return "#dc3545"
    return "black"

def generate_html_block(stats):
    if not stats: return "" # 如果数据为空，则不显示该模块，或者显示报错
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
    if not token: return
    url = "http://www.pushplus.plus/send"
    data = {"token": token, "title": title, "content": content, "template": "html"}
    requests.post(url, json=data)

if __name__ == "__main__":
    # 获取数据
    us_stats = calculate_stats(get_us_data(), "🇺🇸 美股 (CNN)")
    crypto_stats = calculate_stats(get_crypto_data(), "₿ 比特币 (BTC)")
    cn_stats = calculate_stats(get_cn_data(), "🇨🇳 A股 (沪深300)")
    
    # 错误处理：如果三个都失败了
    if not us_stats and not crypto_stats and not cn_stats:
        print("全部获取失败，不推送")
        exit()

    # 拼装
    parts = []
    html_body = ""
    
    if us_stats: 
        parts.append(f"美:{us_stats['val']}")
        html_body += generate_html_block(us_stats)
    else:
        html_body += "<div style='color:red'>❌ 美股数据获取失败 (Check Logs)</div>"

    if crypto_stats: 
        parts.append(f"币:{crypto_stats['val']}")
        html_body += generate_html_block(crypto_stats)
        
    if cn_stats: 
        parts.append(f"A:{cn_stats['val']}")
        html_body += generate_html_block(cn_stats)
    else:
        html_body += "<div style='color:red'>❌ A股数据获取失败 (Check Logs)</div>"
        
    title = " | ".join(parts) + " [情绪日报]"
    
    full_html = f"""
    <html><body>
    <h3 style="text-align:center;">🌍 全球核心资产情绪监控</h3>
    <p style="text-align:center;color:gray;font-size:12px">{datetime.now().strftime('%Y-%m-%d')}</p>
    {html_body}
    </body></html>
    """
    
    send_push(title, full_html)
