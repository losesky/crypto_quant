#!/usr/bin/env python
"""
基本使用示例脚本，展示如何使用量化交易框架
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_quant.data.sources.binance_source import BinanceDataSource
from crypto_quant.strategies.technical.macd_strategy import MACDStrategy
from crypto_quant.strategies.ml_based.lstm_strategy import LSTMStrategy
from crypto_quant.backtesting.engine.backtest_engine import BacktestEngine
from crypto_quant.backtesting.visualization.performance_visualizer import PerformanceVisualizer
from crypto_quant.utils.logger import logger
from crypto_quant.utils.font_helper import get_font_helper
from crypto_quant.utils.output_helper import get_image_path, get_report_path

# 获取字体助手实例
font_helper = get_font_helper()

def run_strategy_backtest(data_source, strategy, symbol, interval, start_date, end_date=None):
    """
    运行策略回测

    Args:
        data_source: 数据源对象
        strategy: 策略对象
        symbol (str): 交易对
        interval (str): K线间隔
        start_date (str): 开始日期
        end_date (str, optional): 结束日期

    Returns:
        tuple: (backtest_engine, results, performance)
    """
    # 获取历史数据
    df = data_source.get_historical_data(symbol, interval, start=start_date, end=end_date)
    
    logger.info(f"获取到{len(df)}行历史数据，时间范围: {df.index.min()} 至 {df.index.max()}")
    
    # 创建回测引擎
    backtest_engine = BacktestEngine(df, strategy)
    
    # 运行回测
    backtest_engine.run()
    
    # 打印性能报告
    backtest_engine.print_performance_report()
    
    return backtest_engine, backtest_engine.results, backtest_engine.performance


def main():
    """
    主函数
    """
    # 设置参数
    symbol = "BTC/USDT"
    interval = "1d"
    
    # 修改开始日期，确保获取足够的数据
    # 对于LSTM模型，我们需要更多的历史数据来训练
    start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")  # 使用两年的数据
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 初始化数据源
    logger.info("Initializing Binance data source...")
    data_source = BinanceDataSource()
    
    # ========================
    # 1. 运行MACD策略回测
    # ========================
    logger.info("Running MACD strategy backtest...")
    macd_strategy = MACDStrategy(fast=12, slow=26, signal=9)
    macd_engine, macd_results, macd_performance = run_strategy_backtest(
        data_source, macd_strategy, symbol, interval, start_date, end_date
    )
    
    # 获取的数据长度检查
    data_length = len(macd_results)
    logger.info(f"获取到{data_length}条数据进行回测")
    
    # 绘制MACD策略回测结果
    macd_fig = macd_engine.plot_results()
    
    # 应用中文字体
    font_helper.apply_font_to_figure(macd_fig)
    
    plt.tight_layout()
    # 使用输出助手保存图像
    macd_output_path = get_image_path("macd_backtest_results.png")
    plt.savefig(macd_output_path)
    logger.info(f"MACD策略回测结果已保存至: {macd_output_path}")
    
    # 检查数据量是否足够进行LSTM模型训练
    min_data_for_lstm = 60  # LSTM至少需要60条数据才能有效训练
    
    if data_length < min_data_for_lstm:
        logger.warning(f"数据量({data_length})不足以有效训练LSTM模型(至少需要{min_data_for_lstm}条)，跳过LSTM策略回测")
        # 创建比较图表 - 只显示MACD策略
        fig = plt.figure(figsize=(12, 6))
        plt.plot(macd_results.index, macd_results['cumulative_strategy_returns'], label=f'MACD Strategy ({macd_performance["annual_return"]:.2%})')
        plt.plot(macd_results.index, macd_results['cumulative_returns'], label='Buy & Hold', linestyle='--')
        
        # 使用中文标题和标签
        if font_helper.has_chinese_font:
            title = '策略性能'
            xlabel = '日期'
            ylabel = '累计收益率'
        else:
            title = 'Strategy Performance'
            xlabel = 'Date'
            ylabel = 'Cumulative Return'
        
        font_helper.set_chinese_title(plt.gca(), title)
        font_helper.set_chinese_label(plt.gca(), xlabel=xlabel, ylabel=ylabel)
        font_helper.set_chinese_legend(plt.gca())
        
        plt.grid(True)
        plt.tight_layout()
        
        # 使用输出助手保存图像
        comparison_path = get_image_path("strategy_comparison.png")
        plt.savefig(comparison_path)
        logger.info(f"策略比较图表已保存至: {comparison_path}")
        
        logger.info("由于数据量不足，示例仅展示了MACD策略回测结果。请检查数据源设置或使用更长的时间范围。")
        return
    
    # ========================
    # 2. 运行LSTM策略回测
    # ========================
    logger.info("Running LSTM strategy backtest...")
    # 减小序列长度以适应小样本数据
    sequence_length = min(20, max(5, data_length // 10))  # 根据数据量动态调整序列长度
    lstm_strategy = LSTMStrategy(sequence_length=sequence_length, prediction_threshold=0.01)
    lstm_engine, lstm_results, lstm_performance = run_strategy_backtest(
        data_source, lstm_strategy, symbol, interval, start_date, end_date
    )
    
    # 可视化LSTM策略回测
    visualizer = PerformanceVisualizer(lstm_results, lstm_performance)
    
    # 绘制资金曲线
    equity_fig = visualizer.plot_equity_curve()
    font_helper.apply_font_to_figure(equity_fig)
    plt.tight_layout()
    equity_path = get_image_path("lstm_equity_curve.png")
    plt.savefig(equity_path)
    logger.info(f"LSTM策略资金曲线已保存至: {equity_path}")
    
    # 绘制回撤图
    underwater_fig = visualizer.plot_underwater_chart()
    font_helper.apply_font_to_figure(underwater_fig)
    plt.tight_layout()
    underwater_path = get_image_path("lstm_underwater_chart.png")
    plt.savefig(underwater_path)
    logger.info(f"LSTM策略回撤图已保存至: {underwater_path}")
    
    # 绘制月度收益热力图
    heatmap_fig = visualizer.plot_monthly_returns_heatmap()
    font_helper.apply_font_to_figure(heatmap_fig)
    plt.tight_layout()
    heatmap_path = get_image_path("lstm_monthly_returns.png")
    plt.savefig(heatmap_path)
    logger.info(f"LSTM策略月度收益热力图已保存至: {heatmap_path}")
    
    # 保存交互式仪表盘
    report_path = get_report_path("lstm_strategy_report.html")
    visualizer.save_report(str(report_path))
    logger.info(f"LSTM策略交互式报告已保存至: {report_path}")
    
    # ========================
    # 3. 策略比较
    # ========================
    logger.info("Comparing strategy performance...")
    
    # 创建比较图表
    fig = plt.figure(figsize=(12, 6))
    plt.plot(macd_results.index, macd_results['cumulative_strategy_returns'], label=f'MACD Strategy ({macd_performance["annual_return"]:.2%})')
    plt.plot(lstm_results.index, lstm_results['cumulative_strategy_returns'], label=f'LSTM Strategy ({lstm_performance["annual_return"]:.2%})')
    plt.plot(macd_results.index, macd_results['cumulative_returns'], label='Buy & Hold', linestyle='--')
    
    # 使用中文标题和标签
    if font_helper.has_chinese_font:
        title = '策略性能比较'
        xlabel = '日期'
        ylabel = '累计收益率'
    else:
        title = 'Strategy Performance Comparison'
        xlabel = 'Date'
        ylabel = 'Cumulative Return'
    
    font_helper.set_chinese_title(plt.gca(), title)
    font_helper.set_chinese_label(plt.gca(), xlabel=xlabel, ylabel=ylabel)
    font_helper.set_chinese_legend(plt.gca())
    
    plt.grid(True)
    plt.tight_layout()
    
    # 使用输出助手保存图像
    comparison_path = get_image_path("strategy_comparison.png")
    plt.savefig(comparison_path)
    logger.info(f"策略比较图表已保存至: {comparison_path}")
    
    # 预测下一天价格
    next_day_price = lstm_strategy.predict_next_day(lstm_results, n_steps=1)
    if next_day_price:
        current_price = lstm_results['close'].iloc[-1]
        price_change = (next_day_price[0] - current_price) / current_price
        
        logger.info(f"Current {symbol} price: {current_price:.2f}")
        logger.info(f"LSTM prediction for tomorrow: {next_day_price[0]:.2f} ({price_change:.2%})")
        
        if price_change > lstm_strategy.prediction_threshold:
            logger.info("Signal: BUY")
        elif price_change < -lstm_strategy.prediction_threshold:
            logger.info("Signal: SELL")
        else:
            logger.info("Signal: HOLD")
    
    logger.info("Example completed. Please check the generated charts and reports.")


if __name__ == "__main__":
    main() 