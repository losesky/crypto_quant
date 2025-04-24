#!/usr/bin/env python
"""
增强型LSTM策略示例脚本，展示如何使用增强的LSTM模型进行价格预测和交易
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_quant.data.sources.binance_source import BinanceDataSource
from crypto_quant.strategies.ml_based.enhanced_lstm_strategy import EnhancedLSTMStrategy
from crypto_quant.strategies.technical.macd_strategy import MACDStrategy
from crypto_quant.backtesting.engine.backtest_engine import BacktestEngine
from crypto_quant.backtesting.visualization.performance_visualizer import PerformanceVisualizer
from crypto_quant.utils.logger import logger
from crypto_quant.utils.font_helper import get_font_helper
from crypto_quant.utils.output_helper import get_image_path, get_report_path

# 获取字体助手实例
font_helper = get_font_helper()

def run_enhanced_lstm_strategy(data_source, symbol, interval, start_date, end_date=None, 
                               sequence_length=20, hidden_dim=128, feature_engineering=True,
                               use_attention=True, stop_loss_pct=0.03):
    """
    运行增强型LSTM策略

    Args:
        data_source: 数据源对象
        symbol (str): 交易对
        interval (str): K线间隔
        start_date (str): 开始日期
        end_date (str, optional): 结束日期
        sequence_length (int): 序列长度
        hidden_dim (int): 隐藏层维度
        feature_engineering (bool): 是否启用特征工程
        use_attention (bool): 是否使用注意力机制
        stop_loss_pct (float, optional): 止损百分比

    Returns:
        tuple: (backtest_engine, results, performance)
    """
    # 获取历史数据
    df = data_source.get_historical_data(symbol, interval, start=start_date, end=end_date)
    
    data_length = len(df)
    logger.info(f"获取到{data_length}行历史数据，时间范围: {df.index.min()} 至 {df.index.max()}")
    
    # 创建增强型LSTM策略
    lstm_strategy = EnhancedLSTMStrategy(
        sequence_length=sequence_length,
        prediction_threshold=0.01,
        hidden_dim=hidden_dim,
        num_layers=3,
        feature_engineering=feature_engineering,
        use_attention=use_attention,
        stop_loss_pct=stop_loss_pct
    )
    
    # 创建回测引擎
    backtest_engine = BacktestEngine(df, lstm_strategy)
    
    # 运行回测
    backtest_engine.run()
    
    # 打印性能报告
    backtest_engine.print_performance_report()
    
    return backtest_engine, backtest_engine.results, backtest_engine.performance

def run_optimized_macd_strategy(data_source, symbol, interval, start_date, end_date=None):
    """
    运行优化后的MACD策略，用于比较

    Args:
        data_source: 数据源对象
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
    
    # 创建优化后的MACD策略（使用之前优化过的参数）
    macd_strategy = MACDStrategy(
        fast_period=8,
        slow_period=30,
        signal_period=9,
        stop_loss_pct=0.03
    )
    
    # 创建回测引擎
    backtest_engine = BacktestEngine(df, macd_strategy)
    
    # 运行回测
    backtest_engine.run()
    
    # 打印性能报告
    backtest_engine.print_performance_report()
    
    return backtest_engine, backtest_engine.results, backtest_engine.performance

def compare_lstm_variations(data_source, symbol, interval, start_date, end_date=None):
    """
    比较不同LSTM策略变体的性能

    Args:
        data_source: 数据源对象
        symbol (str): 交易对
        interval (str): K线间隔
        start_date (str): 开始日期
        end_date (str, optional): 结束日期
    """
    # 获取历史数据
    df = data_source.get_historical_data(symbol, interval, start=start_date, end=end_date)
    
    logger.info(f"比较不同LSTM变体: 数据量={len(df)}行, 时间范围: {df.index.min()} 至 {df.index.max()}")
    
    # 定义不同的LSTM变体
    variations = [
        {
            "name": "基础LSTM",
            "params": {
                "sequence_length": 10,
                "hidden_dim": 64,
                "feature_engineering": False,
                "use_attention": False,
                "stop_loss_pct": None
            }
        },
        {
            "name": "特征增强LSTM",
            "params": {
                "sequence_length": 20,
                "hidden_dim": 64,
                "feature_engineering": True,
                "use_attention": False,
                "stop_loss_pct": None
            }
        },
        {
            "name": "注意力LSTM",
            "params": {
                "sequence_length": 20,
                "hidden_dim": 64,
                "feature_engineering": False,
                "use_attention": True,
                "stop_loss_pct": None
            }
        },
        {
            "name": "完全增强LSTM",
            "params": {
                "sequence_length": 20,
                "hidden_dim": 128,
                "feature_engineering": True,
                "use_attention": True,
                "stop_loss_pct": 0.03
            }
        }
    ]
    
    # 存储结果
    results = []
    
    # 创建对比图表
    plt.figure(figsize=(14, 7))
    
    # 添加Buy & Hold基准线
    returns = df['close'].pct_change().fillna(0)
    cumulative_returns = (1 + returns).cumprod()
    plt.plot(df.index, cumulative_returns, label='Buy & Hold', linestyle='--', color='gray')
    
    # 运行每种LSTM变体
    for variant in variations:
        logger.info(f"测试LSTM变体: {variant['name']}")
        
        strategy = EnhancedLSTMStrategy(
            sequence_length=variant['params']['sequence_length'],
            hidden_dim=variant['params']['hidden_dim'],
            feature_engineering=variant['params']['feature_engineering'],
            use_attention=variant['params']['use_attention'],
            stop_loss_pct=variant['params']['stop_loss_pct']
        )
        
        engine = BacktestEngine(df.copy(), strategy)
        engine.run()
        
        # 打印性能
        print(f"\n===== {variant['name']} 策略表现 =====")
        engine.print_performance_report()
        
        # 存储结果
        results.append({
            'name': variant['name'],
            'engine': engine,
            'performance': engine.performance,
            'results': engine.results
        })
        
        # 绘制权益曲线
        plt.plot(
            engine.results.index, 
            engine.results['cumulative_strategy_returns'], 
            label=f"{variant['name']} ({engine.performance['annual_return']:.2%})"
        )
    
    # 设置图表
    title = font_helper.get_label('LSTM策略变体对比', 'LSTM Strategy Variants Comparison')
    xlabel = font_helper.get_label('日期', 'Date')
    ylabel = font_helper.get_label('累计收益率', 'Cumulative Return')
    
    font_helper.set_chinese_title(plt.gca(), title)
    font_helper.set_chinese_label(plt.gca(), xlabel=xlabel, ylabel=ylabel)
    font_helper.set_chinese_legend(plt.gca())
    
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表
    comparison_path = get_image_path("lstm_variants_comparison.png")
    plt.savefig(comparison_path)
    logger.info(f"LSTM变体对比图表已保存至: {comparison_path}")
    
    # 输出性能比较表格
    print("\n===== LSTM变体性能对比 =====")
    print("策略名称               年化收益率      夏普比率      卡尔玛比率      最大回撤        胜率")
    print("-" * 100)
    
    for result in results:
        perf = result['performance']
        print(f"{result['name']:<20} {perf['annual_return']:>12.2%} {perf['sharpe_ratio']:>12.2f} "
              f"{perf['calmar_ratio']:>12.2f} {perf['max_drawdown']:>12.2%} {perf['win_rate']:>12.2%}")
    
    return results

def main():
    """
    主函数，运行增强型LSTM策略示例
    """
    # 设置参数
    symbol = "BTC/USDT"
    interval = "1d"
    
    # 使用三年数据
    start_date = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 初始化数据源
    logger.info("初始化Binance数据源...")
    data_source = BinanceDataSource()
    
    # 运行不同LSTM变体的对比
    variation_results = compare_lstm_variations(data_source, symbol, interval, start_date, end_date)
    
    # 选择最佳变体，运行详细的回测
    best_variant = variation_results[-1]  # 完全增强LSTM
    logger.info(f"选择最佳LSTM变体进行详细回测: {best_variant['name']}")
    
    # 运行增强型LSTM策略
    lstm_engine, lstm_results, lstm_performance = run_enhanced_lstm_strategy(
        data_source, 
        symbol, 
        interval, 
        start_date, 
        end_date,
        sequence_length=20,
        hidden_dim=128,
        feature_engineering=True,
        use_attention=True,
        stop_loss_pct=0.03
    )
    
    # 运行优化后的MACD策略进行比较
    macd_engine, macd_results, macd_performance = run_optimized_macd_strategy(
        data_source, 
        symbol, 
        interval, 
        start_date, 
        end_date
    )
    
    # 创建可视化工具
    visualizer = PerformanceVisualizer(lstm_results, lstm_performance)
    
    # 绘制资金曲线
    equity_fig = visualizer.plot_equity_curve()
    font_helper.apply_font_to_figure(equity_fig)
    plt.tight_layout()
    equity_path = get_image_path("enhanced_lstm_equity_curve.png")
    plt.savefig(equity_path)
    logger.info(f"增强型LSTM策略资金曲线已保存至: {equity_path}")
    
    # 绘制回撤图
    underwater_fig = visualizer.plot_underwater_chart()
    font_helper.apply_font_to_figure(underwater_fig)
    plt.tight_layout()
    underwater_path = get_image_path("enhanced_lstm_underwater_chart.png")
    plt.savefig(underwater_path)
    logger.info(f"增强型LSTM策略回撤图已保存至: {underwater_path}")
    
    # 绘制月度收益热力图
    heatmap_fig = visualizer.plot_monthly_returns_heatmap()
    font_helper.apply_font_to_figure(heatmap_fig)
    plt.tight_layout()
    heatmap_path = get_image_path("enhanced_lstm_monthly_returns.png")
    plt.savefig(heatmap_path)
    logger.info(f"增强型LSTM策略月度收益热力图已保存至: {heatmap_path}")
    
    # 创建策略对比图表
    plt.figure(figsize=(12, 6))
    plt.plot(macd_results.index, macd_results['cumulative_returns'], label='Buy & Hold', linestyle='--')
    plt.plot(
        macd_results.index, 
        macd_results['cumulative_strategy_returns'], 
        label=f'优化MACD ({macd_performance["annual_return"]:.2%})'
    )
    plt.plot(
        lstm_results.index, 
        lstm_results['cumulative_strategy_returns'], 
        label=f'增强LSTM ({lstm_performance["annual_return"]:.2%})'
    )
    
    # 设置图表
    title = font_helper.get_label('增强LSTM vs 优化MACD 策略对比', 'Enhanced LSTM vs Optimized MACD Strategy Comparison')
    xlabel = font_helper.get_label('日期', 'Date')
    ylabel = font_helper.get_label('累计收益率', 'Cumulative Return')
    
    font_helper.set_chinese_title(plt.gca(), title)
    font_helper.set_chinese_label(plt.gca(), xlabel=xlabel, ylabel=ylabel)
    font_helper.set_chinese_legend(plt.gca())
    
    plt.grid(True)
    plt.tight_layout()
    
    # 保存对比图表
    comparison_path = get_image_path("lstm_vs_macd_comparison.png")
    plt.savefig(comparison_path)
    logger.info(f"策略对比图表已保存至: {comparison_path}")
    
    # 保存交互式仪表盘
    report_path = get_report_path("enhanced_lstm_strategy_report.html")
    visualizer.save_report(str(report_path))
    logger.info(f"增强型LSTM策略交互式报告已保存至: {report_path}")
    
    # 预测未来价格
    future_days = 7
    last_data = lstm_results.iloc[-100:]  # 使用最近100天的数据
    predictions = best_variant['engine'].strategy.predict_next_day(last_data, n_steps=future_days)
    
    if predictions:
        current_price = lstm_results['close'].iloc[-1]
        logger.info(f"当前 {symbol} 价格: {current_price:.2f}")
        logger.info(f"未来{future_days}天预测价格:")
        
        for i, price in enumerate(predictions):
            change = (price - current_price) / current_price
            date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
            logger.info(f"  {date}: {price:.2f} ({change:.2%})")
        
        # 绘制预测图表
        dates = [datetime.now() + timedelta(days=i+1) for i in range(future_days)]
        plt.figure(figsize=(10, 6))
        
        # 绘制历史价格
        plt.plot(last_data.index, last_data['close'], label='历史价格')
        
        # 绘制预测价格
        plt.plot(dates, predictions, 'r--o', label='预测价格')
        plt.axvline(x=datetime.now(), color='green', linestyle='-', alpha=0.3, label='当前')
        
        title = font_helper.get_label(f'{symbol} 价格预测', f'{symbol} Price Prediction')
        xlabel = font_helper.get_label('日期', 'Date')
        ylabel = font_helper.get_label('价格 (USDT)', 'Price (USDT)')
        
        font_helper.set_chinese_title(plt.gca(), title)
        font_helper.set_chinese_label(plt.gca(), xlabel=xlabel, ylabel=ylabel)
        font_helper.set_chinese_legend(plt.gca())
        
        plt.grid(True)
        plt.tight_layout()
        
        prediction_path = get_image_path("btc_price_prediction.png")
        plt.savefig(prediction_path)
        logger.info(f"价格预测图表已保存至: {prediction_path}")
    
    logger.info("增强型LSTM策略示例运行完成")

if __name__ == "__main__":
    main() 