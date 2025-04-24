#!/usr/bin/env python
"""
参数优化示例脚本，展示如何优化策略参数
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timedelta
import subprocess
import platform

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入字体助手
from crypto_quant.utils.font_helper import get_font_helper
from crypto_quant.data.sources.binance_source import BinanceDataSource
from crypto_quant.strategies.technical.macd_strategy import MACDStrategy
from crypto_quant.backtesting.engine.backtest_engine import BacktestEngine
from crypto_quant.optimization.parameter_optimizer import ParameterOptimizer
from crypto_quant.utils.logger import logger
from crypto_quant.utils.output_helper import get_image_path

# 获取字体助手
font_helper = get_font_helper()

def optimize_macd_parameters():
    """
    优化MACD策略参数
    """
    # 设置参数
    symbol = "BTC/USDT"
    interval = "1d"
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 初始化数据源
    logger.info("初始化数据源...")
    data_source = BinanceDataSource()
    
    # 获取历史数据
    logger.info(f"获取{symbol}历史数据，周期: {interval}...")
    df = data_source.get_historical_data(symbol, interval, start=start_date, end=end_date)
    
    logger.info(f"获取到{len(df)}行历史数据，时间范围: {df.index.min()} 至 {df.index.max()}")
    
    # 定义MACD参数网格
    param_grid = {
        'fast_period': range(8, 16, 2),     # [8, 10, 12, 14] 快线周期
        'slow_period': range(20, 32, 2),    # [20, 22, 24, 26, 28, 30] 慢线周期
        'signal_period': range(7, 12),      # [7, 8, 9, 10, 11] 信号线周期
        'stop_loss_pct': [0.5, 1.0, 1.5, 2.0]  # 止损百分比
    }
    
    # 创建参数优化器
    logger.info("创建参数优化器...")
    optimizer = ParameterOptimizer(
        strategy_class=MACDStrategy,
        param_grid=param_grid,
        data=df,
        initial_capital=10000,
        commission=0.001,
        metric='calmar_ratio',  # 使用卡尔玛比率作为优化目标
        method='grid_search'    # 使用网格搜索方法
    )
    
    # 运行优化
    logger.info("开始参数优化...")
    best_params = optimizer.run()
    
    # 保存最佳参数
    logger.info(f"最佳MACD参数: {best_params}")
    
    # 可视化优化结果
    logger.info("可视化优化结果...")
    optimizer.plot_results(top_n=10)
    
    # 使用最佳参数进行回测
    logger.info("使用最佳参数进行回测...")
    best_strategy = MACDStrategy(**best_params)
    
    engine = BacktestEngine(
        data=df,
        strategy=best_strategy,
        initial_capital=10000,
        commission=0.001
    )
    
    engine.run()
    engine.print_performance_report()
    
    # 绘制回测结果
    fig = engine.plot_results()
    
    # 应用中文字体
    font_helper.apply_font_to_figure(fig)
    
    plt.tight_layout()
    
    # 使用输出助手保存图像
    output_path = get_image_path("optimized_macd_backtest.png")
    plt.savefig(output_path)
    logger.info(f"优化后的MACD回测结果已保存至: {output_path}")
    
    return best_params, optimizer


def compare_strategies(df, original_params, optimized_params):
    """
    比较原始策略和优化后的策略
    
    Args:
        df: 回测数据
        original_params: 原始参数
        optimized_params: 优化后的参数
    """
    logger.info("比较原始策略和优化后的策略...")
    
    # 创建原始策略和优化后的策略
    original_strategy = MACDStrategy(**original_params)
    optimized_strategy = MACDStrategy(**optimized_params)
    
    # 回测原始策略
    original_engine = BacktestEngine(
        data=df.copy(),
        strategy=original_strategy,
        initial_capital=10000,
        commission=0.001
    )
    original_engine.run()
    
    # 回测优化后的策略
    optimized_engine = BacktestEngine(
        data=df.copy(),
        strategy=optimized_strategy,
        initial_capital=10000,
        commission=0.001
    )
    optimized_engine.run()
    
    # 获取结果
    original_results = original_engine.results
    optimized_results = optimized_engine.results
    
    # 打印性能比较
    print("\n===== 策略性能比较 =====")
    print("指标               原始策略        优化后策略")
    print("-" * 50)
    
    metrics = [
        ('年化收益率', 'annual_return', '{:.2%}'),
        ('夏普比率', 'sharpe_ratio', '{:.2f}'),
        ('卡尔玛比率', 'calmar_ratio', '{:.2f}'),
        ('最大回撤', 'max_drawdown', '{:.2%}'),
        ('总收益率', 'total_return', '{:.2%}'),
        ('胜率', 'win_rate', '{:.2%}'),
        ('盈亏比', 'profit_factor', '{:.2f}'),
        ('交易次数', 'trades_count', '{:.0f}')
    ]
    
    for name, key, fmt in metrics:
        orig_val = original_engine.performance.get(key, 0)
        opt_val = optimized_engine.performance.get(key, 0)
        print(f"{name:15} {fmt.format(orig_val):15} {fmt.format(opt_val):15}")
    
    # 绘制权益曲线比较
    fig = plt.figure(figsize=(12, 6))
    plt.plot(
        original_results.index,
        original_results['cumulative_returns'],
        label='Buy & Hold',
        linestyle='--',
        alpha=0.7
    )
    
    # 准备中英文标签
    original_label_zh = f'原始MACD ({original_params["fast_period"]},{original_params["slow_period"]},{original_params["signal_period"]})'
    original_label_en = f'Original MACD ({original_params["fast_period"]},{original_params["slow_period"]},{original_params["signal_period"]})'
    
    optimized_label_zh = f'优化MACD ({optimized_params["fast_period"]},{optimized_params["slow_period"]},{optimized_params["signal_period"]})'
    optimized_label_en = f'Optimized MACD ({optimized_params["fast_period"]},{optimized_params["slow_period"]},{optimized_params["signal_period"]})'
    
    # 使用get_label方法自动选择标签
    original_label = font_helper.get_label(original_label_zh, original_label_en)
    optimized_label = font_helper.get_label(optimized_label_zh, optimized_label_en)
    title = font_helper.get_label('MACD策略优化对比', 'MACD Strategy Optimization Comparison')
    xlabel = font_helper.get_label('日期', 'Date')
    ylabel = font_helper.get_label('累计收益率', 'Cumulative Return')
    
    plt.plot(
        original_results.index,
        original_results['cumulative_strategy_returns'],
        label=original_label,
        alpha=0.8
    )
    plt.plot(
        optimized_results.index,
        optimized_results['cumulative_strategy_returns'],
        label=optimized_label,
        alpha=0.8
    )
    
    # 设置标题和标签
    font_helper.set_chinese_title(plt.gca(), title)
    font_helper.set_chinese_label(plt.gca(), xlabel=xlabel, ylabel=ylabel)
    font_helper.set_chinese_legend(plt.gca())
    
    plt.grid(True)
    plt.tight_layout()
    
    # 使用输出助手保存图像
    comparison_path = get_image_path("macd_strategy_comparison.png")
    plt.savefig(comparison_path)
    logger.info(f"策略对比图表已保存至: {comparison_path}")


def main():
    """
    主函数
    """
    # 原始MACD参数
    original_params = {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9,
        'stop_loss_pct': 0
    }
    
    # 优化MACD参数
    optimized_params, optimizer = optimize_macd_parameters()
    
    # 获取数据用于比较
    symbol = "BTC/USDT"
    interval = "1d"
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 初始化数据源并获取数据
    data_source = BinanceDataSource()
    df = data_source.get_historical_data(symbol, interval, start=start_date, end=end_date)
    
    # 比较原始策略和优化后的策略
    compare_strategies(df, original_params, optimized_params)
    
    logger.info("参数优化示例运行完成")


if __name__ == "__main__":
    main() 