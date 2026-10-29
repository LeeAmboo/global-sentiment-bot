import os
import html
import json
import base64
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

US_FALLBACK_URL = "https://feargreedchart.com/api/?action=all"
US_HISTORY_URL = "https://feargreedchart.com/api/?action=history"

LIMIT_LOW = 25
LIMIT_HIGH = 75
DANGER_DAYS_THRESHOLD = 10
MAX_PUSHPLUS_LEN = 19000

FEAR_COLOR = "#16a34a"
GREED_COLOR = "#dc2626"
NEUTRAL_COLOR = "#555555"


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

    seen = set()
    out = []

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


# ================= 备用 RSI 计算 =================
def calculate_rsi_history(ticker, period="8mo"):
    """计算 RSI 历史数据，仅供 A 股降级逻辑使用。"""
    try:
        df = yf.download(
            ticker,
            period=period,
            progress=False,
            auto_adjust=False
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
    """获取 CNN 当前及历史贪恐指数。"""
    print("🇺🇸 正在获取 CNN Fear & Greed Index...")

    idx = fear_greed.get()

    if not isinstance(idx, dict):
        raise ValueError(
            f"CNN 返回数据格式异常：{type(idx).__name__}"
        )

    if "score" not in idx:
        raise ValueError("CNN 返回数据缺少 score 字段")

    current_val = normalize_score(idx["score"])

    if idx.get("timestamp"):
        current_date = normalize_date(idx["timestamp"])
    else:
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(
        f"✅ CNN 当前指数：{current_val}，"
        f"日期：{current_date}，"
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
                score_value = getattr(
                    item, "score", getattr(item, "value", None)
                )

            if date_value is None or score_value is None:
                continue

            try:
                history_records.append({
                    "date": normalize_date(date_value),
                    "value": normalize_score(score_value)
                })
            except (ValueError, TypeError, OverflowError):
                continue

    except Exception as e:
        print(f"⚠️ CNN 历史数据获取失败：{e}")

    # 当前值优先，避免历史接口覆盖最新数据
    history_records.insert(0, {
        "date": current_date,
        "value": current_val
    })

    history_records = dedupe_by_date(history_records)
    history_records.sort(key=lambda x: x["date"], reverse=True)
    history_records = history_records[:70]

    if not history_records:
        raise ValueError("CNN 没有返回有效指数数据")

    return history_records


# ================= FearGreedChart 备用美股指数 =================
def get_feargreedchart_us_data():
    """使用 FearGreedChart 作为 CNN 的备用数据源。"""
    print("🔄 正在切换到 FearGreedChart API...")

    response = requests.get(US_FALLBACK_URL, timeout=20)
    response.raise_for_status()
    data = response.json()

    score_info = data.get("score")

    if not isinstance(score_info, dict):
        raise ValueError("FearGreedChart 返回数据缺少 score 对象")

    if "score" not in score_info:
        raise ValueError(
            "FearGreedChart 返回数据缺少 score.score 字段"
        )

    current_val = normalize_score(score_info["score"])

    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if data.get("ts") is not None:
        try:
            current_date = datetime.fromtimestamp(
                float(data["ts"]) / 1000,
                tz=timezone.utc
            ).strftime("%Y-%m-%d")
        except (ValueError, TypeError, OSError):
            pass

    print(
        f"✅ FearGreedChart 当前指数：{current_val}，"
        f"日期：{current_date}"
    )

    history_records = []
    raw_history = data.get("recent")

    if not isinstance(raw_history, list):
        raw_history = []

    if not raw_history and isinstance(data.get("history"), list):
        raw_history = data["history"]

    for item in raw_history:
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

                date_value = datetime.fromtimestamp(
                    timestamp, tz=timezone.utc
                )

            history_records.append({
                "date": normalize_date(date_value),
                "value": normalize_score(score_value)
            })
        except (ValueError, TypeError, OverflowError, OSError):
            continue

    if not history_records:
        try:
            history_response = requests.get(
                US_HISTORY_URL,
                timeout=20
            )
            history_response.raise_for_status()
            history_data = history_response.json()

            if isinstance(history_data, list):
                raw_history = history_data
            elif isinstance(history_data, dict):
                raw_history = history_data.get(
                    "data",
                    history_data.get(
                        "history",
                        history_data.get("recent", [])
                    )
                )
                if not isinstance(raw_history, list):
                    raw_history = []
            else:
                raw_history = []

            for item in raw_history:
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

                        date_value = datetime.fromtimestamp(
                            timestamp, tz=timezone.utc
                        )

                    history_records.append({
                        "date": normalize_date(date_value),
                        "value": normalize_score(score_value)
                    })
                except (
                    ValueError, TypeError, OverflowError, OSError
                ):
                    continue

        except Exception as e:
            print(f"⚠️ FearGreedChart 历史接口获取失败：{e}")

    history_records.insert(0, {
        "date": current_date,
        "value": current_val
    })

    history_records = dedupe_by_date(history_records)
    history_records.sort(key=lambda x: x["date"], reverse=True)
    history_records = history_records[:70]

    if not history_records:
        raise ValueError("FearGreedChart 没有返回有效指数数据")

    return history_records


# ================= 三类数据获取 =================
def get_us_data():
    """优先 CNN，失败后使用 FearGreedChart，不用 RSI 替代。"""
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


def get_crypto_data():
    """比特币：Alternative.me。"""
    try:
        res = requests.get(CRYPTO_URL, timeout=15)
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


# ================= A股通用数据工具 =================
def _to_numeric_column(df, candidates):
    """按候选字段名查找数值列，兼容 AKShare 不同版本的列名。"""
    if df is None or df.empty:
        return None

    normalized = {
        str(c).strip().lower().replace(" ", ""): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "")
        if key in normalized:
            return pd.to_numeric(
                df[normalized[key]], errors="coerce"
            )

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "")
        for normalized_name, original_name in normalized.items():
            if key in normalized_name:
                return pd.to_numeric(
                    df[original_name], errors="coerce"
                )

    return None


def _rolling_percentile_score(series, window=750, min_periods=125):
    """将指标映射为自身滚动历史分位数（0—100）。"""
    clean = pd.to_numeric(series, errors="coerce")

    def percentile_of_last(values):
        vals = pd.Series(values).dropna()
        if vals.empty:
            return np.nan
        return float(vals.rank(pct=True).iloc[-1] * 100)

    return clean.rolling(
        window=window, min_periods=min_periods
    ).apply(percentile_of_last, raw=False)


def _build_cn_backfill_records(hs300, amount_history, days=60):
    """历史回填仅使用可重建的 RSI、趋势和成交活跃度。"""
    if hs300 is None or hs300.empty:
        return []

    frame = hs300.copy()

    if (
        amount_history is not None
        and not amount_history.empty
        and "Turnover_Score" in amount_history.columns
    ):
        frame = frame.join(
            amount_history[["Turnover_Score"]], how="left"
        )
    else:
        frame["Turnover_Score"] = np.nan

    frame = frame.sort_index().tail(max(days, 1))

    base_weights = {
        "RSI": 0.20,
        "趋势": 0.20,
        "成交活跃度": 0.20
    }

    records = []

    for date, row in frame.iterrows():
        factor_values = {}

        for name, column in (
            ("RSI", "RSI"),
            ("趋势", "Trend_Score"),
            ("成交活跃度", "Turnover_Score")
        ):
            value = row.get(column)
            if pd.notna(value) and np.isfinite(float(value)):
                factor_values[name] = float(np.clip(value, 0, 100))

        if not factor_values:
            continue

        weight_sum = sum(
            base_weights[name] for name in factor_values
        )

        normalized = {
            name: base_weights[name] / weight_sum
            for name in factor_values
        }

        score = sum(
            factor_values[name] * normalized[name]
            for name in factor_values
        )

        records.append({
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "value": normalize_score(score),
            "model": "five_factor_v1",
            "factor_count": len(factor_values),
            "factors": {
                name: round(value, 2)
                for name, value in factor_values.items()
            },
            "weights": {
                name: round(value, 6)
                for name, value in normalized.items()
            },
            "backfilled": True,
            "backfill_note": (
                "历史回填仅使用可回溯的 RSI、趋势、成交活跃度；"
                "历史市场宽度/融资因子未伪造"
            )
        })

    return records


# ================= GitHub 历史记录 =================
def _github_history_request(method, url, **kwargs):
    """使用 GitHub Contents API 保存每日模型分数。"""
    token = os.getenv("GITHUB_TOKEN")

    if not token:
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    headers.update(kwargs.pop("headers", {}))

    return requests.request(
        method, url, headers=headers, timeout=15, **kwargs
    )


def _load_cn_five_factor_history():
    """优先读取 GitHub 仓库文件，兼容本地文件。"""
    repo = os.getenv("GITHUB_REPOSITORY")
    branch = os.getenv("GITHUB_REF_NAME", "main")
    path = "data/cn_five_factor_history.json"
    token = os.getenv("GITHUB_TOKEN")

    if repo and token:
        url = f"https://api.github.com/repos/{repo}/contents/{path}"

        try:
            res = _github_history_request(
                "GET", url, params={"ref": branch}
            )

            if res is not None and res.status_code == 200:
                payload = res.json()
                content = base64.b64decode(
                    payload.get("content", "")
                ).decode("utf-8")
                data = json.loads(content)

                if isinstance(data, list):
                    return data, payload.get("sha"), url, branch, path

            elif res is not None and res.status_code not in (404,):
                print(
                    f"⚠️ 读取五因子历史文件失败："
                    f"HTTP {res.status_code} {res.text[:200]}"
                )
            else:
                return [], None, url, branch, path

        except Exception as e:
            print(f"⚠️ GitHub 历史文件读取失败：{e}")

    local_path = os.path.join("data", "cn_five_factor_history.json")

    try:
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data, None, None, branch, path

    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass

    return [], None, None, branch, path


def _save_cn_five_factor_history(records, sha, url, branch, path):
    """保存历史记录；GitHub Actions 环境优先使用 Contents API。"""
    local_path = os.path.join("data", "cn_five_factor_history.json")

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"⚠️ 本地历史文件保存失败：{e}")

    if not url or not os.getenv("GITHUB_TOKEN"):
        print("ℹ️ 未配置 GitHub 历史写入环境，五因子历史仅保存到当前运行环境。")
        return

    try:
        content = base64.b64encode(
            json.dumps(
                records, ensure_ascii=False, indent=2
            ).encode("utf-8")
        ).decode("ascii")

        payload = {
            "message": "Update A-share five-factor sentiment history",
            "content": content,
            "branch": branch,
        }

        if sha:
            payload["sha"] = sha

        res = _github_history_request("PUT", url, json=payload)

        if res is not None and res.status_code in (200, 201):
            print("✅ 五因子历史已写入 GitHub 仓库")
        else:
            status = res.status_code if res is not None else "无响应"
            detail = res.text[:300] if res is not None else ""
            print(f"⚠️ 五因子历史写入 GitHub 失败：{status} {detail}")

    except Exception as e:
        print(f"⚠️ 五因子历史写入异常：{e}")


# ================= 融资买入额：已修复 =================
def _fetch_margin_buy_amount(trade_date):
    """
    获取指定交易日沪深两市融资买入额。

    上交所 stock_margin_sse 使用 start_date/end_date；
    深交所 stock_margin_szse 使用 date。

    任一交易所获取失败时返回 None，不拼接不完整的两市数据。
    """
    target_date = pd.Timestamp(trade_date).normalize()
    date_str = target_date.strftime("%Y%m%d")
    values = []

    fetchers = [
        (
            "SSE",
            "stock_margin_sse",
            {"start_date": date_str, "end_date": date_str}
        ),
        (
            "SZSE",
            "stock_margin_szse",
            {"date": date_str}
        )
    ]

    for market, func_name, kwargs in fetchers:
        func = getattr(ak, func_name, None)

        if func is None:
            print(f"⚠️ 杠杆因子：AKShare 不提供 {func_name}")
            return None

        try:
            # 关键修复：按不同交易所的真实接口参数分别调用
            df = func(**kwargs)

            if df is None or df.empty:
                print(
                    f"⚠️ 杠杆因子：{market} {date_str} 无数据"
                )
                return None

            # 兼容不同版本可能出现的日期字段
            date_col = None
            date_names = {
                "信用交易日期",
                "日期",
                "交易日期",
                "date",
                "trade_date"
            }

            for col_name in df.columns:
                if str(col_name).strip().lower() in date_names:
                    date_col = col_name
                    break

            if date_col is not None:
                raw_dates = (
                    df[date_col]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\.0$", "", regex=True)
                )

                parsed_dates = pd.to_datetime(
                    raw_dates, format="%Y%m%d", errors="coerce"
                )

                # 兼容 YYYY-MM-DD 等其他日期格式
                missing = parsed_dates.isna()

                if missing.any():
                    parsed_dates.loc[missing] = pd.to_datetime(
                        raw_dates.loc[missing],
                        errors="coerce"
                    )

                matched = df.loc[
                    parsed_dates.dt.normalize() == target_date
                ]

                if matched.empty:
                    print(
                        f"⚠️ 杠杆因子：{market} 未返回目标日期 "
                        f"{date_str}，接口返回日期字段为 {date_col}"
                    )
                    return None

                df = matched

            amount_col = _to_numeric_column(
                df,
                [
                    "融资买入额",
                    "融资买入额(元)",
                    "融资买入金额"
                ]
            )

            if amount_col is None:
                print(
                    f"⚠️ 杠杆因子：{market} 找不到融资买入额字段，"
                    f"实际字段={list(df.columns)}"
                )
                return None

            valid = amount_col.dropna()

            if valid.empty:
                print(
                    f"⚠️ 杠杆因子：{market} 融资买入额全部为空"
                )
                return None

            amount = float(valid.sum())

            if not np.isfinite(amount) or amount <= 0:
                print(
                    f"⚠️ 杠杆因子：{market} 融资买入额无效："
                    f"{amount}"
                )
                return None

            values.append(amount)

            print(
                f"✅ 杠杆因子：{market} {date_str} "
                f"融资买入额={amount:,.0f} 元"
            )

        except Exception as e:
            print(
                f"⚠️ 杠杆因子：读取 {market} {date_str} 失败："
                f"{type(e).__name__}: {e}"
            )
            return None

    if len(values) == 2:
        total = sum(values)
        print(f"✅ 沪深两市融资买入额合计：{total:,.0f} 元")
        return total

    return None


# ================= A股五因子模型 =================
def get_cn_data():
    """
    A股五因子情绪模型：
    RSI 20% + 125日趋势分位数 20% + 成交活跃度 20%
    + 市场宽度 25% + 融资买入额占成交额比例 15%。

    缺失因子不填 0，按有效因子的原始权重重新归一化。
    """
    try:
        # ---------- 因子1、2：沪深300价格动能与长期趋势 ----------
        hs300 = ak.stock_zh_index_daily(symbol="sh000300")

        if hs300 is None or hs300.empty:
            raise ValueError("沪深300历史行情为空")

        hs300.columns = [
            str(c).lower().strip() for c in hs300.columns
        ]

        required = {"date", "close"}

        if not required.issubset(set(hs300.columns)):
            raise ValueError(
                f"沪深300行情缺少必要字段：{list(hs300.columns)}"
            )

        hs300["date"] = pd.to_datetime(
            hs300["date"], errors="coerce"
        )
        hs300["close"] = pd.to_numeric(
            hs300["close"], errors="coerce"
        )

        hs300 = hs300.dropna(
            subset=["date", "close"]
        ).sort_values("date")

        hs300 = hs300.drop_duplicates(
            subset=["date"], keep="last"
        ).set_index("date")

        delta = hs300["close"].diff()

        gain = delta.clip(lower=0).ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean()

        loss = -delta.clip(upper=0).ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False
        ).mean()

        rs = gain / loss.replace(0, np.nan)
        hs300["RSI"] = 100 - (100 / (1 + rs))

        hs300.loc[(loss == 0) & (gain > 0), "RSI"] = 100
        hs300.loc[(gain == 0) & (loss > 0), "RSI"] = 0
        hs300.loc[(gain == 0) & (loss == 0), "RSI"] = 50

        hs300["MA125"] = hs300["close"].rolling(
            125, min_periods=125
        ).mean()

        hs300["BIAS125"] = (
            hs300["close"] / hs300["MA125"] - 1
        ) * 100

        hs300["Trend_Score"] = _rolling_percentile_score(
            hs300["BIAS125"],
            window=750,
            min_periods=125
        )

        # ---------- 因子3：全市场成交额相对20日均值 ----------
        amount_history = None

        for symbol in ("sh000001", "sz399001"):
            try:
                df = None
                fn = getattr(ak, "stock_zh_index_daily_em", None)

                if fn is not None:
                    df = fn(symbol=symbol)

                if (
                    df is None
                    or df.empty
                    or _to_numeric_column(
                        df, ["amount", "成交额"]
                    ) is None
                ):
                    df = ak.stock_zh_index_daily(symbol=symbol)

                if df is None or df.empty:
                    raise ValueError("指数历史数据为空")

                df.columns = [
                    str(c).strip().lower() for c in df.columns
                ]

                if "date" not in df.columns:
                    raise ValueError(
                        f"缺少日期字段：{list(df.columns)}"
                    )

                amount_col = _to_numeric_column(
                    df, ["amount", "成交额"]
                )

                if amount_col is None:
                    raise ValueError(
                        f"{symbol} 历史接口未提供成交额字段"
                    )

                one = pd.DataFrame({
                    "date": pd.to_datetime(
                        df["date"], errors="coerce"
                    ),
                    "amount": amount_col
                }).dropna()

                one = one[one["amount"] > 0]
                one = one.drop_duplicates("date", keep="last")
                one = one.set_index("date").sort_index()
                one = one.rename(columns={"amount": symbol})

                if amount_history is None:
                    amount_history = one
                else:
                    amount_history = amount_history.join(
                        one, how="outer"
                    )

            except Exception as e:
                print(
                    f"⚠️ 成交额历史接口 {symbol} 获取失败：{e}"
                )

        if (
            amount_history is not None
            and "sh000001" in amount_history.columns
            and "sz399001" in amount_history.columns
        ):
            amount_history["Market_Amount"] = (
                amount_history["sh000001"].fillna(0)
                + amount_history["sz399001"].fillna(0)
            )

            amount_history.loc[
                amount_history["Market_Amount"] <= 0,
                "Market_Amount"
            ] = np.nan

            amount_history["Amount_MA20"] = (
                amount_history["Market_Amount"].rolling(
                    20, min_periods=20
                ).mean()
            )

            amount_history["Amount_Ratio"] = (
                amount_history["Market_Amount"]
                / amount_history["Amount_MA20"]
            )

            amount_history["Turnover_Score"] = (
                _rolling_percentile_score(
                    amount_history["Amount_Ratio"],
                    window=750,
                    min_periods=60
                )
            )

            # 统一日期索引，后续按交易日匹配
            amount_history.index = pd.to_datetime(
                amount_history.index, errors="coerce"
            ).normalize()

            amount_history = amount_history[
                ~amount_history.index.isna()
            ]

            amount_history = amount_history[
                ~amount_history.index.duplicated(keep="last")
            ].sort_index()

        else:
            amount_history = None
            print(
                "⚠️ 全市场历史成交额因子不可用："
                "无法取得沪深两市同口径成交额数据"
            )

        # ---------- 因子4：当日市场宽度 ----------
        breadth_score = None
        spot = None

        for getter_name in (
            "stock_zh_a_spot_em",
            "stock_zh_a_spot"
        ):
            getter = getattr(ak, getter_name, None)

            if getter is None:
                continue

            try:
                spot = getter()
                if spot is not None and not spot.empty:
                    break
            except Exception as e:
                print(
                    f"⚠️ 全市场股票行情 {getter_name} 获取失败：{e}"
                )

        if spot is not None and not spot.empty:
            change_col = _to_numeric_column(
                spot, ["涨跌幅", "涨跌幅(%)", "pct_chg"]
            )

            if change_col is not None:
                changes = change_col.replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()

                advancing = int((changes > 0).sum())
                declining = int((changes < 0).sum())
                denominator = advancing + declining

                if denominator >= 100:
                    breadth_score = (
                        advancing / denominator * 100
                    )
                    print(
                        f"📊 市场宽度：上涨{advancing}只、"
                        f"下跌{declining}只，"
                        f"得分{breadth_score:.2f}"
                    )
                else:
                    print(
                        "⚠️ 市场宽度因子跳过："
                        "有效涨跌股票数量不足"
                    )
            else:
                print(
                    "⚠️ 市场宽度因子跳过："
                    f"找不到涨跌幅字段，字段={list(spot.columns)}"
                )
        else:
            print("⚠️ 市场宽度因子跳过：无法获取全市场股票行情")

        # ---------- 当前交易日及有效因子 ----------
        current_date = hs300.index.max()
        current_date = pd.Timestamp(current_date).normalize()
        current_date_str = current_date.strftime("%Y-%m-%d")
        latest = hs300.loc[hs300.index.normalize() == current_date].iloc[-1]

        scores = {}

        weights = {
            "RSI": 0.20,
            "趋势": 0.20,
            "成交活跃度": 0.20,
            "市场宽度": 0.25,
            "杠杆率": 0.15,
        }

        if pd.notna(latest.get("RSI")):
            scores["RSI"] = float(
                np.clip(latest["RSI"], 0, 100)
            )

        if pd.notna(latest.get("Trend_Score")):
            scores["趋势"] = float(
                np.clip(latest["Trend_Score"], 0, 100)
            )

        if amount_history is not None:
            available_turnover = amount_history[
                amount_history["Turnover_Score"].notna()
            ]

            if not available_turnover.empty:
                turnover_date = available_turnover.index.max()
                turnover_value = available_turnover.loc[
                    turnover_date, "Turnover_Score"
                ]

                if isinstance(turnover_value, pd.Series):
                    turnover_value = turnover_value.iloc[-1]

                if pd.notna(turnover_value):
                    scores["成交活跃度"] = float(
                        np.clip(turnover_value, 0, 100)
                    )

                    if turnover_date != current_date:
                        print(
                            f"⚠️ 成交活跃度数据日期为 "
                            f"{turnover_date:%Y-%m-%d}，"
                            f"沪深300最新日期为 {current_date_str}"
                        )

        if breadth_score is not None:
            scores["市场宽度"] = float(
                np.clip(breadth_score, 0, 100)
            )

        # ---------- 因子5：融资买入额 / 全市场成交额 ----------
        leverage_score = None
        market_amount_today = None

        if spot is not None and not spot.empty:
            amount_col_spot = _to_numeric_column(
                spot, ["成交额", "成交额(元)", "amount"]
            )

            if amount_col_spot is not None:
                amount_values = amount_col_spot.dropna()

                if not amount_values.empty:
                    market_amount_today = float(
                        amount_values.clip(lower=0).sum()
                    )

                    print(
                        f"📊 实时股票快照成交额合计："
                        f"{market_amount_today:,.0f} 元"
                    )

        margin_amount = None
        margin_date = None

        # 最近10个自然日内寻找沪深两市均有融资数据的日期
        for offset in range(0, 11):
            candidate_date = (
                current_date - pd.Timedelta(days=offset)
            ).normalize()

            candidate_amount = _fetch_margin_buy_amount(
                candidate_date
            )

            if candidate_amount is not None:
                margin_amount = candidate_amount
                margin_date = candidate_date
                break

        market_amount_for_margin = None

        if margin_amount is not None and margin_date is not None:
            # 修复点：统一日期格式，避免 Timestamp 索引带时间部分而匹配失败
            if amount_history is not None:
                normalized_index = pd.to_datetime(
                    amount_history.index, errors="coerce"
                ).normalize()

                same_day_rows = amount_history.loc[
                    normalized_index == margin_date.normalize()
                ]

                if not same_day_rows.empty:
                    candidate_values = pd.to_numeric(
                        same_day_rows["Market_Amount"],
                        errors="coerce"
                    ).dropna()

                    if not candidate_values.empty:
                        candidate_amount = float(
                            candidate_values.iloc[-1]
                        )

                        if (
                            np.isfinite(candidate_amount)
                            and candidate_amount > 0
                        ):
                            market_amount_for_margin = candidate_amount
                            print(
                                f"✅ 成交额日期匹配成功："
                                f"{margin_date:%Y-%m-%d}，"
                                f"全市场成交额={candidate_amount:,.0f} 元"
                            )

            # 只有融资数据日期与最新交易日一致时，才允许使用实时快照成交额
            if (
                market_amount_for_margin is None
                and margin_date.date() == current_date.date()
                and market_amount_today is not None
                and market_amount_today > 0
            ):
                market_amount_for_margin = market_amount_today
                print(
                    "ℹ️ 使用实时股票快照成交额计算当日杠杆率"
                )

        if (
            margin_amount is not None
            and market_amount_for_margin is not None
            and market_amount_for_margin > 0
        ):
            leverage_ratio = (
                margin_amount / market_amount_for_margin
            )

            if (
                np.isfinite(leverage_ratio)
                and 0 <= leverage_ratio <= 0.25
            ):
                leverage_score = float(
                    np.clip(
                        (leverage_ratio - 0.02)
                        / (0.12 - 0.02) * 100,
                        0,
                        100
                    )
                )

                scores["杠杆率"] = leverage_score

                print(
                    f"📈 杠杆率：融资买入额/成交额="
                    f"{leverage_ratio:.2%}，"
                    f"得分={leverage_score:.2f}，"
                    f"数据日期={margin_date:%Y-%m-%d}"
                )
            else:
                print(
                    f"⚠️ 杠杆率因子跳过：比例异常 "
                    f"{leverage_ratio:.4f}，"
                    "请检查融资额和成交额的单位是否一致"
                )
        else:
            print(
                "⚠️ 杠杆率因子跳过："
                "两市融资买入额或同一交易日的全市场成交额不可用"
            )

        # ---------- 计算最终指数 ----------
        active = {
            name: score
            for name, score in scores.items()
            if np.isfinite(score)
        }

        if not active:
            raise ValueError("五因子模型没有任何有效因子")

        active_weight_sum = sum(
            weights[name] for name in active
        )

        normalized_weights = {
            name: weights[name] / active_weight_sum
            for name in active
        }

        final_score = sum(
            active[name] * normalized_weights[name]
            for name in active
        )

        print(
            "📊 A股五因子得分："
            + "；".join(
                f"{name}={active[name]:.2f}"
                f"（实际权重{normalized_weights[name]:.1%}）"
                for name in active
            )
        )

        print(
            f"🇨🇳 A股五因子贪恐指数：{final_score:.2f}，"
            f"有效因子 {len(active)}/5"
        )

        # ---------- 首次运行/历史不足时回填最近60个交易日 ----------
        history, sha, api_url, branch, path = (
            _load_cn_five_factor_history()
        )

        history = [
            item for item in history
            if isinstance(item, dict)
            and item.get("date")
            and item.get("model") == "five_factor_v1"
        ]

        existing_dates = {
            str(item.get("date")) for item in history
        }

        backfill_records = _build_cn_backfill_records(
            hs300, amount_history, days=60
        )

        added = 0

        for backfill_record in backfill_records:
            if backfill_record["date"] not in existing_dates:
                history.append(backfill_record)
                existing_dates.add(backfill_record["date"])
                added += 1

        if added:
            print(
                f"🗓️ 历史回填：新增 {added} 个交易日记录；"
                "回填仅使用可回溯的价格/成交活跃度因子"
            )
        else:
            print(
                "🗓️ 历史回填：现有记录覆盖了最近交易日，"
                "或本次未能构建额外记录"
            )

        record = {
            "date": current_date_str,
            "value": normalize_score(final_score),
            "model": "five_factor_v1",
            "factor_count": len(active),
            "factors": {
                name: round(value, 2)
                for name, value in active.items()
            },
            "weights": {
                name: round(value, 6)
                for name, value in normalized_weights.items()
            },
        }

        # 同一天重跑时更新，不重复累积
        history = [
            item for item in history
            if item.get("date") != current_date_str
        ]

        history.append(record)
        history.sort(
            key=lambda item: item["date"], reverse=True
        )
        history = history[:250]

        _save_cn_five_factor_history(
            history, sha, api_url, branch, path
        )

        records = [
            {
                "date": item["date"],
                "value": normalize_score(item["value"])
            }
            for item in history[:70]
        ]

        if not records:
            records = [{
                "date": current_date_str,
                "value": normalize_score(final_score)
            }]

        return (
            dedupe_by_date(records),
            f"A股五因子模型（有效{len(active)}/5）"
        )

    except Exception as e:
        print(f"A-Share Five-Factor Calculation Error: {e}")
        print("⚠️ 五因子主模型失败，尝试原有沪深300 RSI 降级逻辑")

        fallback = calculate_rsi_history("000300.SS")

        if fallback:
            return fallback, "沪深300 RSI（降级，非五因子）"

        return None, "获取失败"


# ================= 统计计算 =================
def calc_stats(data, period_type):
    if not data:
        return None

    current = data[0]

    def count(n):
        sub = data[:n]

        low = sum(
            1 for x in sub
            if x["value"] < LIMIT_LOW
        )

        high = sum(
            1 for x in sub
            if x["value"] > LIMIT_HIGH
        )

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
        return (
            left
            + (
                pw / (len(data) - 1) * i
                if len(data) > 1
                else pw / 2
            )
        )

    def y_pos(v):
        v = max(0, min(100, float(v)))
        return top + (100 - v) / 100 * ph

    pts = [
        f"{x_pos(i):.1f},{y_pos(item['value']):.1f}"
        for i, item in enumerate(data)
    ]

    y25 = y_pos(25)
    y75 = y_pos(75)
    start = data[0]["date"][5:]
    end = data[-1]["date"][5:]

    return f"""
    <svg viewBox='0 0 {w} {h}' width='100%'
         xmlns='http://www.w3.org/2000/svg'>
      <rect x='0' y='0' width='{w}' height='{h}'
            rx='8' fill='#fff'/>
      <line x1='{left}' y1='{top}' x2='{left}' y2='{h-bottom}'
            stroke='#ddd'/>
      <line x1='{left}' y1='{h-bottom}' x2='{w-right}'
            y2='{h-bottom}' stroke='#ddd'/>
      <line x1='{left}' y1='{y75:.1f}' x2='{w-right}'
            y2='{y75:.1f}' stroke='{GREED_COLOR}'
            stroke-dasharray='4 4'/>
      <line x1='{left}' y1='{y25:.1f}' x2='{w-right}'
            y2='{y25:.1f}' stroke='{FEAR_COLOR}'
            stroke-dasharray='4 4'/>
      <text x='4' y='{y75+3:.1f}' font-size='9'
            fill='#999'>75</text>
      <text x='4' y='{y25+3:.1f}' font-size='9'
            fill='#999'>25</text>
      <polyline points='{' '.join(pts)}' fill='none'
                stroke='#2563eb' stroke-width='2.2'/>
      <text x='{left}' y='{h-5}' font-size='9'
            fill='#999'>{safe(start)}</text>
      <text x='{w-right}' y='{h-5}' text-anchor='end'
            font-size='9' fill='#999'>{safe(end)}</text>
    </svg>
    """


def make_history_table(history, period_type, compact=False):
    if not history:
        return "<p class='muted'>暂无明细数据</p>"

    data = history[:30]

    if compact:
        pairs = " ｜ ".join(
            f"{x['date'][5:]}:{x['value']}" for x in data
        )
        return f"<p class='seq'>{safe(pairs)}</p>"

    rows = []

    for x in data:
        cls = status_class(x["value"])
        st = status_style(x["value"])

        rows.append(
            f"<tr><td>{safe(x['date'])}</td>"
            f"<td class='{cls}' style='{st}'>{x['value']}</td>"
            f"<td class='{cls}' style='{st}'>"
            f"{safe(status_text(x['value']))}</td></tr>"
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
        return (
            f"<div class='card err'>"
            f"❌ {safe(name)} 数据获取失败</div>"
        )

    v = stats["val"]
    c = value_color(v)
    pt = stats["period_type"]
    warn = ""

    if stats["h30"] >= DANGER_DAYS_THRESHOLD:
        warn = (
            f"<div class='warn'>⚠️ "
            f"<span class='greed' style='color:{GREED_COLOR};font-weight:800'>"
            f"贪婪风险</span>：近30个{safe(pt)}内已有 "
            f"<span class='greed' style='color:{GREED_COLOR};font-weight:800'>"
            f"{stats['h30']}</span> 次极度贪婪。</div>"
        )

    link_html = ""

    if link:
        link_html = (
            f"<p><a class='btn' href='{safe(link)}'>"
            f"查看详细走势</a></p>"
        )

    return f"""
    <div class='card'>
      <div class='top'>
        <div>
          <b>{safe(name)}</b><br>
          <span>{safe(stats['date'])}｜{safe(source)}</span>
        </div>
        <div class='num' style='color:{c}'>
          {v}<br><small>{safe(stats['status'])}</small>
        </div>
      </div>
      <table class='sum'>
        <tr>
          <th>周期</th>
          <th class='fear' style='color:{FEAR_COLOR};font-weight:800'>
            恐慌&lt;{LIMIT_LOW}
          </th>
          <th class='greed' style='color:{GREED_COLOR};font-weight:800'>
            贪婪&gt;{LIMIT_HIGH}
          </th>
        </tr>
        <tr>
          <td>近30个{safe(pt)}</td>
          <td class='fear' style='color:{FEAR_COLOR};font-weight:800'>
            {stats['l30']}
          </td>
          <td class='greed' style='color:{GREED_COLOR};font-weight:800'>
            {stats['h30']}
          </td>
        </tr>
        <tr>
          <td>近60个{safe(pt)}</td>
          <td class='fear' style='color:{FEAR_COLOR};font-weight:800'>
            {stats['l60']}
          </td>
          <td class='greed' style='color:{GREED_COLOR};font-weight:800'>
            {stats['h60']}
          </td>
        </tr>
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
        .hist{{margin-top:8px}}
        .fear{{color:{FEAR_COLOR}!important;font-weight:800}}
        .greed{{color:{GREED_COLOR}!important;font-weight:800}}
        .neutral{{color:{NEUTRAL_COLOR};font-weight:600}}
        .warn{{background:#fff1f2;color:#991b1b;border:1px solid #fecdd3;border-radius:6px;padding:8px;margin-top:8px;font-size:12px}}
        summary{{cursor:pointer;text-align:center;margin-top:10px;padding:8px;border:1px solid #bfdbfe;background:#eff6ff;color:#075985;border-radius:6px;font-size:13px;font-weight:700}}
        .chart{{margin-top:8px;border:1px solid #eee;border-radius:8px;overflow:hidden;background:#fff}}
        .btn{{display:block;text-align:center;background:#e7f5ff;color:#075985;text-decoration:none;border:1px solid #bfdbfe;border-radius:6px;padding:8px;font-weight:700;font-size:13px}}
        .foot{{background:#e9ecef;border-left:4px solid #2563eb;border-radius:8px;padding:12px;font-size:12px;line-height:1.7}}
        .note,.muted{{font-size:11px;color:#999;text-align:right}}
        .seq{{font-size:12px;line-height:1.8;color:#555}}
        .err{{background:#fee2e2}}
      </style>
    </head>
    <body>
      <div class='wrap'>
        <h3>🌍 全球核心资产情绪监控</h3>
        {cards}
        <div class='foot'>
          <b>📊 策略提示</b><br>
          <span class='fear' style='color:{FEAR_COLOR};font-weight:800'>
            🟢 恐慌机会：指数 &lt; {LIMIT_LOW}，关注定投。
          </span><br>
          <span class='greed' style='color:{GREED_COLOR};font-weight:800'>
            🔴 贪婪风险：指数 &gt; {LIMIT_HIGH}，关注止盈。
          </span><br>
          ⚠️ 近30天
          <span class='greed' style='color:{GREED_COLOR};font-weight:800'>
            极度贪婪
          </span>
          ≥ {DANGER_DAYS_THRESHOLD} 次：注意高位。<br>
          <span style='color:#999'>
            Data Updated: {safe(beijing_time_str)}
          </span>
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
        res.raise_for_status()
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
            "name": "US 美股",
            "getter": get_us_data,
            "period_type": "交易日",
            "link": None
        },
        {
            "name": "₿ 比特币",
            "getter": get_crypto_data,
            "period_type": "自然日",
            "link": None
        },
        {
            "name": "CN A股",
            "getter": get_cn_data,
            "period_type": "交易日",
            "link": JIUQUAN_URL
        },
    ]

    results = []
    title_parts = []

    for task in tasks:
        raw_data, source = task["getter"]()

        # 确保最新记录排在首位
        if raw_data:
            raw_data = sorted(
                raw_data,
                key=lambda x: x.get("date", ""),
                reverse=True
            )

        stats = calc_stats(raw_data, task["period_type"])

        results.append({
            "name": task["name"],
            "source": source,
            "stats": stats,
            "history": raw_data or [],
            "link": task["link"]
        })

        if stats:
            title_parts.append(
                f"{task['name'].split()[-1]}:{stats['val']}"
            )

    beijing_time = (
        datetime.now(timezone.utc) + timedelta(hours=8)
    )

    beijing_time_str = (
        beijing_time.strftime("%Y-%m-%d %H:%M")
        + "（北京时间）"
    )

    html_content = build_html(
        results, beijing_time_str, compact=False
    )

    if len(html_content) > MAX_PUSHPLUS_LEN:
        html_content = build_html(
            results, beijing_time_str, compact=True
        )

    if len(html_content) > MAX_PUSHPLUS_LEN:
        summary_lines = []

        for r in results:
            s = r["stats"]

            if s:
                summary_lines.append(
                    f"{r['name']}：{s['val']}，{s['status']}，"
                    f"近30个{s['period_type']}恐慌{s['l30']}次，"
                    f"贪婪{s['h30']}次"
                )
            else:
                summary_lines.append(
                    f"{r['name']}：数据获取失败"
                )

        html_content = (
            "<br>".join(summary_lines)
            + f"<br><br>更新时间：{safe(beijing_time_str)}"
        )

    if title_parts:
        title = (
            beijing_time.strftime("%m-%d")
            + " | "
            + " | ".join(title_parts)
        )

        send_push(title, html_content)
    else:
        print("❌ 所有数据获取失败")


if __name__ == "__main__":
    main()
