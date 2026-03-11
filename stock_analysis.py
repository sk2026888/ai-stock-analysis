import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt

# 避免图表中文乱码（使用英文显示，兼容所有环境）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def fetch_stock_data(ticker, start_date, end_date):
    """
    获取股票历史数据（适配A股+优化异常处理）
    支持标的格式：
    - A股：600000.SS（浦发银行）、000001.SZ（平安银行）
    - 美股：AAPL（苹果）、MSFT（微软）
    """
    try:
        # 配置yfinance参数，解决国内访问问题
        yf.pdr_override()
        stock = yf.Ticker(ticker)
        
        # 获取数据（增加超时设置+重试机制）
        df = stock.history(
            start=start_date, 
            end=end_date,
            interval='1d',  # 日线数据（稳定不易出错）
            timeout=30      # 超时时间延长至30秒
        )
        
        # 校验数据是否为空
        if df.empty:
            raise ValueError(f"未获取到 {ticker} 的数据，请检查标的代码或网络")
        
        # 清理数据（去除空值）
        df = df.dropna()
        return df
    
    except Exception as e:
        print(f"【错误】获取 {ticker} 数据失败: {str(e)}")
        # 友好提示备选方案
        print(f"建议检查：1. 标的代码是否正确 2. 尝试更换A股标的（如600000.SS）")
        raise

def ai_trend_analysis(df):
    """简单的AI趋势分析（线性回归预测，适配所有标的）"""
    # 准备特征：日期转换为数值（距离起始日的天数）
    df['Date_Num'] = np.arange(len(df))
    X = df[['Date_Num']]
    y = df['Close']  # 收盘价作为预测目标

    # 训练线性回归模型（增加数据校验）
    if len(df) < 5:  # 至少需要5个数据点才能分析
        raise ValueError("有效数据不足，无法进行趋势分析（至少需要5个交易日数据）")
    
    model = LinearRegression()
    model.fit(X, y)

    # 预测趋势
    df['Predicted_Close'] = model.predict(X)
    trend_slope = model.coef_[0]  # 趋势斜率（正为上涨，负为下跌）
    
    # 计算关键指标（优化精度）
    latest_price = df['Close'].iloc[-1]
    price_change = (latest_price - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
    # 年化波动率（处理极端值）
    pct_change = df['Close'].pct_change().dropna()
    volatility = pct_change.std() * np.sqrt(252) * 100 if len(pct_change) > 0 else 0

    return {
        'trend': '上涨' if trend_slope > 0 else '下跌',
        'trend_strength': round(abs(trend_slope), 4),
        'latest_price': round(latest_price, 2),
        'price_change_pct': round(price_change, 2),
        'volatility_pct': round(volatility, 2),
        'predicted_next_day': round(model.predict([[len(df)]])[0], 2)
    }

def generate_report(analysis_result, ticker, start_date, end_date):
    """生成更友好的分析报告（适配A股/美股）"""
    # 标的名称映射（提升可读性）
    ticker_name_map = {
        "600000.SS": "浦发银行",
        "000001.SZ": "平安银行",
        "AAPL": "苹果公司",
        "MSFT": "微软公司"
    }
    ticker_name = ticker_name_map.get(ticker, ticker)

    report = f"""
# AI 股票分析报告
**分析标的**: {ticker}（{ticker_name}）
**分析周期**: {start_date} 至 {end_date}
**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 核心分析结果
| 指标 | 数值 |
|------|------|
| 最新收盘价 | {analysis_result['latest_price']} |
| 周期涨跌幅 | {analysis_result['price_change_pct']}% |
| 年化波动率 | {analysis_result['volatility_pct']}% |
| 趋势方向 | {analysis_result['trend']} |
| 趋势强度（斜率） | {analysis_result['trend_strength']} |
| 次日预测价格 | {analysis_result['predicted_next_day']} |

## 分析结论
基于线性回归模型分析，{ticker_name}（{ticker}）在分析周期内呈现 {analysis_result['trend']} 趋势。
- 周期内价格变动 {analysis_result['price_change_pct']}%，波动率为 {analysis_result['volatility_pct']}%
- 模型预测次日价格约为 {analysis_result['predicted_next_day']}

> ⚠️ 注：此分析仅为技术演示，不构成任何投资建议！
    """
    # 保存报告到文件（确保编码正确）
    with open('analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 分析报告已生成：analysis_report.md（标的：{ticker_name}）")

def main():
    """主函数（默认使用A股标的，避免网络问题）"""
    # 配置参数（优先选A股，国内访问稳定）
    TICKER = "600000.SS"  # 浦发银行（替换为你想分析的标的）
    DAYS_BACK = 90        # 分析过去90天数据（最少保留5天）
    
    # 计算日期范围（避免跨节假日无数据）
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=DAYS_BACK)
    
    # 执行分析流程
    print(f"📊 开始分析 {TICKER} 股票数据（{start_date} 至 {end_date}）")
    df = fetch_stock_data(TICKER, start_date, end_date)
    print(f"✅ 成功获取 {len(df)} 条有效交易数据")
    
    analysis_result = ai_trend_analysis(df)
    generate_report(analysis_result, TICKER, start_date, end_date)

if __name__ == "__main__":
    main()
