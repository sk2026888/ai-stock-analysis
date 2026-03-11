import pandas_datareader.data as web
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt

# 避免图表中文乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def fetch_stock_data(ticker, start_date, end_date):
    """
    使用 stooq 数据源获取股票数据（GitHub Actions 环境稳定可用）
    支持标的：
    - 美股：AAPL, MSFT, TSLA
    - A股：600000.SS（浦发银行）、000001.SZ（平安银行）
    """
    try:
        # 使用 stooq 数据源，海外访问稳定
        df = web.DataReader(ticker, 'stooq', start_date, end_date)
        if df.empty:
            raise ValueError(f"未获取到 {ticker} 的数据，请检查标的代码")
        # 按时间正序排列
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"【数据获取失败】{ticker} - {str(e)}")
        print("👉 建议：1. 更换美股标的（如AAPL） 2. 检查标的代码格式")
        raise

def ai_trend_analysis(df):
    """AI趋势分析（线性回归模型）"""
    if len(df) < 5:
        raise ValueError(f"有效数据不足（仅{len(df)}条），至少需要5个交易日")
    
    df['Date_Index'] = np.arange(len(df))
    X = df[['Date_Index']]
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    trend_slope = model.coef_[0]
    latest_price = round(df['Close'].iloc[-1], 2)
    price_change_pct = round((latest_price - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100, 2)
    pct_changes = df['Close'].pct_change().dropna()
    volatility_pct = round(pct_changes.std() * np.sqrt(252) * 100, 2) if len(pct_changes) > 0 else 0
    next_day_pred = round(model.predict([[len(df)]])[0], 2)

    return {
        'trend': '上涨' if trend_slope > 0 else '下跌',
        'trend_strength': round(abs(trend_slope), 4),
        'latest_price': latest_price,
        'price_change_pct': price_change_pct,
        'volatility_pct': volatility_pct,
        'predicted_next_day': next_day_pred
    }

def generate_report(analysis_result, ticker, start_date, end_date):
    """生成分析报告"""
    ticker_name_map = {
        "AAPL": "苹果公司",
        "MSFT": "微软公司",
        "600000.SS": "浦发银行",
        "000001.SZ": "平安银行"
    }
    ticker_name = ticker_name_map.get(ticker, ticker)

    report = f"""
# AI 股票分析报告
**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析标的**: {ticker}（{ticker_name}）
**分析周期**: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}

## 📊 核心分析结果
| 指标 | 数值 |
|------|------|
| 最新收盘价 | {analysis_result['latest_price']} |
| 周期涨跌幅 | {analysis_result['price_change_pct']}% |
| 年化波动率 | {analysis_result['volatility_pct']}% |
| 趋势方向 | {analysis_result['trend']} |
| 趋势强度 | {analysis_result['trend_strength']} |
| 次日预测价格 | {analysis_result['predicted_next_day']} |

## 📝 分析结论
{ticker_name}（{ticker}）在分析周期内呈现 **{analysis_result['trend']}** 趋势。
- 周期内价格变动 {analysis_result['price_change_pct']}%，波动率 {analysis_result['volatility_pct']}%
- 模型预测次日收盘价约为 {analysis_result['predicted_next_day']}

> ⚠️ 重要提示：此分析仅为技术演示，不构成任何投资建议！
    """
    with open('analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 报告生成成功：analysis_report.md（标的：{ticker_name}）")

def main():
    """主函数（默认分析美股AAPL，GitHub Actions最稳定）"""
    TICKER = "AAPL"  # 优先用美股，GitHub Actions环境访问最稳定
    DAYS_BACK = 90

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=DAYS_BACK)

    print(f"📈 开始分析 {TICKER}（{start_date} 至 {end_date}）")
    stock_data = fetch_stock_data(TICKER, start_date, end_date)
    print(f"📊 成功获取 {len(stock_data)} 条有效交易数据")
    
    trend_result = ai_trend_analysis(stock_data)
    generate_report(trend_result, TICKER, start_date, end_date)

if __name__ == "__main__":
    main()
