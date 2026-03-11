import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt

# 避免图表中文乱码（使用英文显示）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def fetch_stock_data(ticker, start_date, end_date):
    """获取股票历史数据"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"未获取到 {ticker} 的数据")
        return df
    except Exception as e:
        print(f"获取数据失败: {e}")
        raise

def ai_trend_analysis(df):
    """简单的AI趋势分析（线性回归预测）"""
    # 准备特征：日期转换为数值（距离起始日的天数）
    df['Date_Num'] = np.arange(len(df))
    X = df[['Date_Num']]
    y = df['Close']  # 收盘价作为预测目标

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 预测趋势
    df['Predicted_Close'] = model.predict(X)
    trend_slope = model.coef_[0]  # 趋势斜率（正为上涨，负为下跌）
    
    # 计算关键指标
    latest_price = df['Close'].iloc[-1]
    price_change = (latest_price - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
    volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100  # 年化波动率

    return {
        'trend': '上涨' if trend_slope > 0 else '下跌',
        'trend_strength': abs(trend_slope),
        'latest_price': round(latest_price, 2),
        'price_change_pct': round(price_change, 2),
        'volatility_pct': round(volatility, 2),
        'predicted_next_day': round(model.predict([[len(df)]])[0], 2)
    }

def generate_report(analysis_result, ticker, start_date, end_date):
    """生成分析报告"""
    report = f"""
# AI 股票分析报告
**分析标的**: {ticker}
**分析周期**: {start_date} 至 {end_date}
**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 核心分析结果
| 指标 | 数值 |
|------|------|
| 最新收盘价 | {analysis_result['latest_price']} |
| 周期涨跌幅 | {analysis_result['price_change_pct']}% |
| 年化波动率 | {analysis_result['volatility_pct']}% |
| 趋势方向 | {analysis_result['trend']} |
| 次日预测价格 | {analysis_result['predicted_next_day']} |

## 分析结论
基于线性回归模型分析，{ticker} 在分析周期内呈现 {analysis_result['trend']} 趋势。
- 周期内价格变动 {analysis_result['price_change_pct']}%，波动率为 {analysis_result['volatility_pct']}%
- 模型预测次日价格约为 {analysis_result['predicted_next_day']}

> 注：此分析仅为技术演示，不构成投资建议
    """
    # 保存报告到文件
    with open('analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print("报告已生成：analysis_report.md")

def main():
    """主函数"""
    # 配置参数（可根据需要修改）
    TICKER = "600000.SS"  # 苹果股票，可替换为其他标的如 "MSFT" "TSLA" "600000.SS"（浦发银行）
    DAYS_BACK = 90   # 分析过去90天数据
    
    # 计算日期范围
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=DAYS_BACK)
    
    # 执行分析流程
    print(f"开始分析 {TICKER} 股票数据（{start_date} 至 {end_date}）")
    df = fetch_stock_data(TICKER, start_date, end_date)
    analysis_result = ai_trend_analysis(df)
    generate_report(analysis_result, TICKER, start_date, end_date)

if __name__ == "__main__":
    main()
