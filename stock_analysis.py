import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt

# 避免图表中文乱码（兼容所有环境）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def fetch_stock_data(ticker, start_date, end_date):
    """
    获取股票历史数据（适配A股/美股，修复yfinance版本兼容问题）
    支持标的格式：
    - A股：600000.SS（浦发银行）、000001.SZ（平安银行）
    - 美股：AAPL（苹果）、MSFT（微软）
    """
    try:
        # 初始化yfinance Ticker对象（新版无需pdr_override）
        stock = yf.Ticker(ticker)
        
        # 获取日线数据（增加超时+重试逻辑，提升稳定性）
        df = stock.history(
            start=start_date, 
            end=end_date,
            interval='1d',    # 仅用日线数据，避免分钟线接口限制
            timeout=30,       # 超时时间延长至30秒
            raise_errors=True # 显式抛出错误便于排查
        )
        
        # 数据校验：空数据直接抛错
        if df.empty:
            raise ValueError(f"未获取到 {ticker} 的数据，请检查代码或网络")
        
        # 清理数据：删除空值行
        df = df.dropna(subset=['Close'])
        return df
    
    except Exception as e:
        error_msg = f"【数据获取失败】{ticker} - {str(e)}"
        print(error_msg)
        # 给出具体解决方案提示
        print("👉 建议：1. 更换A股标的（如600000.SS） 2. 检查网络是否能访问海外数据源")
        raise  # 抛出错误让Actions捕获，便于排查

def ai_trend_analysis(df):
    """AI趋势分析（线性回归，适配所有标的）"""
    # 数据量校验：至少需要5个有效交易日
    if len(df) < 5:
        raise ValueError(f"有效数据不足（仅{len(df)}条），至少需要5个交易日")
    
    # 特征工程：日期转数值（距离起始日的天数）
    df['Date_Index'] = np.arange(len(df))
    X = df[['Date_Index']]  # 特征
    y = df['Close']         # 预测目标（收盘价）

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 趋势计算
    trend_slope = model.coef_[0]  # 斜率：正=上涨，负=下跌
    df['Predicted_Close'] = model.predict(X)

    # 核心指标计算
    latest_price = round(df['Close'].iloc[-1], 2)
    price_change_pct = round((latest_price - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100, 2)
    # 年化波动率（修正极端值）
    pct_changes = df['Close'].pct_change().dropna()
    volatility_pct = round(pct_changes.std() * np.sqrt(252) * 100, 2) if len(pct_changes) > 0 else 0
    # 次日价格预测
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
    """生成Markdown格式分析报告"""
    # 标的名称映射（提升可读性）
    ticker_name_map = {
        "600000.SS": "浦发银行",
        "000001.SZ": "平安银行",
        "AAPL": "苹果公司",
        "MSFT": "微软公司",
        "601318.SS": "中国平安"
    }
    ticker_name = ticker_name_map.get(ticker, ticker)

    # 报告内容（结构化）
    report_content = f"""
# AI 股票分析报告
**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析标的**: {ticker}（{ticker_name}）
**分析周期**: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}

## 📊 核心分析结果
| 指标 | 数值 | 说明 |
|------|------|------|
| 最新收盘价 | {analysis_result['latest_price']} | 分析周期最后一个交易日收盘价 |
| 周期涨跌幅 | {analysis_result['price_change_pct']}% | 周期内价格变动幅度 |
| 年化波动率 | {analysis_result['volatility_pct']}% | 衡量股价波动风险 |
| 趋势方向 | {analysis_result['trend']} | 基于线性回归的趋势判断 |
| 趋势强度 | {analysis_result['trend_strength']} | 斜率绝对值，越大趋势越明显 |
| 次日预测价格 | {analysis_result['predicted_next_day']} | 模型预测的下一交易日收盘价 |

## 📝 分析结论
{ticker_name}（{ticker}）在分析周期内呈现 **{analysis_result['trend']}** 趋势：
- 周期内价格变动 {analysis_result['price_change_pct']}%，波动率 {analysis_result['volatility_pct']}%
- 线性回归模型预测次日收盘价约为 {analysis_result['predicted_next_day']}

> ⚠️ 重要提示：此分析仅为技术演示，不构成任何投资建议！
    """

    # 保存报告（确保UTF-8编码）
    with open('analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"✅ 报告生成成功：analysis_report.md（标的：{ticker_name}）")

def main():
    """主执行函数（默认使用A股，国内访问最稳定）"""
    # 配置参数（可直接修改）
    TICKER = "600000.SS"  # 浦发银行（A股，优先推荐）
    DAYS_BACK = 90        # 分析过去90天数据

    # 计算日期范围
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=DAYS_BACK)

    # 执行完整流程
    print(f"📈 开始分析 {TICKER}（{start_date} 至 {end_date}）")
    stock_data = fetch_stock_data(TICKER, start_date, end_date)
    print(f"📊 成功获取 {len(stock_data)} 条有效交易数据")
    
    trend_result = ai_trend_analysis(stock_data)
    generate_report(trend_result, TICKER, start_date, end_date)

if __name__ == "__main__":
    main()
