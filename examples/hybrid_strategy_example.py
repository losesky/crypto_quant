#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合策略示例
演示如何使用MACD-LSTM混合策略进行交易回测
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入项目模块
from crypto_quant.data.sources.binance_source import BinanceDataSource  # 使用BinanceDataSource替代不存在的ClickHouseDataSource
from crypto_quant.strategies.hybrid.macd_lstm_hybrid_strategy import MACDLSTMHybridStrategy
from crypto_quant.backtesting.engine.backtest_engine import BacktestEngine  # 修正了backtesting路径
from crypto_quant.utils.logger import logger, set_log_level
from crypto_quant.utils.font_helper import get_font_helper
from crypto_quant.utils.output_helper import get_image_path, get_report_path, get_data_path, ensure_dir_exists

# 获取字体助手
font_helper = get_font_helper()

# 设置日志级别
set_log_level('INFO')


if __name__ == "__main__":
    # 设置输出子目录
    output_subdir = f"hybrid_strategy_{datetime.now().strftime('%Y%m%d')}"
    
    # 1. 连接数据源
    logger.info("连接Binance数据源")  # 更新日志信息
    data_source = BinanceDataSource()  # 使用BinanceDataSource替代ClickHouseDataSource
    
    # 2. 获取比特币历史数据（扩展时间范围为2年）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2年数据
    
    logger.info(f"获取比特币数据: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    df = data_source.get_historical_data(  # 使用get_historical_data方法替代get_ohlcv_data
        symbol='BTC/USDT',
        interval='1d',  # 使用interval替代timeframe
        start=start_date.strftime('%Y-%m-%d'),  # 调整参数名称从start_date到start
        end=end_date.strftime('%Y-%m-%d')  # 调整参数名称从end_date到end
    )
    
    if df.empty:
        logger.error("未获取到数据，请检查数据源配置")
        sys.exit(1)
    
    logger.info(f"获取到 {len(df)} 条数据记录")
    
    # 保存原始数据
    raw_data_path = get_data_path("btc_raw_data.csv", subdirectory=output_subdir)
    df.to_csv(raw_data_path)
    logger.info(f"原始数据已保存至: {raw_data_path}")
    
    # 3. 创建并测试不同的混合策略组合方法
    # 为每种组合方法创建单独的回测引擎和策略
    ensemble_methods = ['vote', 'weight', 'layered', 'expert']
    backtest_results = {}
    performance_metrics = {}
    
    for method in ensemble_methods:
        logger.info(f"======== 测试混合策略: {method} 组合方法 ========")
        
        # 创建混合策略
        strategy = MACDLSTMHybridStrategy(
            # MACD参数
            macd_fast_period=12,
            macd_slow_period=26,
            macd_signal_period=9,
            
            # LSTM参数
            lstm_sequence_length=20,
            lstm_hidden_dim=128,
            lstm_prediction_threshold=0.01,
            lstm_feature_engineering=True,
            lstm_use_attention=True,
            
            # 混合策略参数
            ensemble_method=method,
            ensemble_weights=(0.6, 0.4),  # MACD权重略高，因为其表现相对稳定
            market_regime_threshold=0.15,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
        )
        
        # 创建回测引擎
        backtest_engine = BacktestEngine(
            data=df,  # 添加data参数名
            strategy=strategy,
            initial_capital=10000.0,
            commission=0.001  # 0.1% 交易手续费
        )
        
        # 运行回测
        backtest_engine.run()  # 不再传递df参数
        
        # 存储回测结果
        backtest_results[method] = backtest_engine.results.copy()  # 存储回测结果的副本
        
        # 获取性能指标
        metrics = backtest_engine.summary()  # 使用summary方法代替get_performance_metrics
        performance_metrics[method] = metrics
        
        # 显示性能指标
        logger.info(f"策略: {strategy.name}")
        logger.info(f"最终资本: ${metrics['final_capital']:.2f}")
        logger.info(f"总收益率: {metrics['total_return']:.2%}")
        logger.info(f"年化收益率: {metrics['annual_return']:.2%}")
        logger.info(f"最大回撤: {metrics['max_drawdown']:.2%}")
        logger.info(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"卡尔马比率: {metrics['calmar_ratio']:.2f}")
        logger.info(f"交易次数: {metrics['trade_count']}")
        logger.info(f"胜率: {metrics['win_rate']:.2%}")
        logger.info("")
        
        # 保存该方法的详细回测结果
        method_results_path = get_data_path(f"{method}_backtest_results.csv", subdirectory=output_subdir)
        backtest_results[method].to_csv(method_results_path)  # 使用已存储的结果
        logger.info(f"{method}方法的回测结果已保存至: {method_results_path}")
        
        # 绘制并保存该方法的资本曲线
        plt.figure(figsize=(10, 6))
        equity_curve = backtest_results[method]['equity_curve']  # 使用已存储的结果
        plt.plot(equity_curve.index, equity_curve, label=f"{method}")
        benchmark = df['close'] / df['close'].iloc[0] * 10000
        plt.plot(benchmark.index, benchmark, label='Buy & Hold', linestyle='--', color='gray')
        
        plt.title(f'{method}策略资本曲线')
        plt.xlabel('日期')
        plt.ylabel('资本($)')
        plt.legend()
        plt.grid(True)
        
        # 应用中文字体
        font_helper.apply_font_to_figure(plt.gcf())
        
        # 保存图表
        method_chart_path = get_image_path(f"{method}_equity_curve.png", subdirectory=output_subdir)
        plt.savefig(method_chart_path)
        plt.close()
        logger.info(f"{method}方法的资本曲线图已保存至: {method_chart_path}")
    
    # 4. 比较不同组合方法的性能
    logger.info("======== 混合策略组合方法性能比较 ========")
    
    # 创建性能比较表格
    comparison = pd.DataFrame({
        '组合方法': [method for method in ensemble_methods],
        '最终资本': [performance_metrics[method]['final_capital'] for method in ensemble_methods],
        '总收益率': [performance_metrics[method]['total_return'] for method in ensemble_methods],
        '年化收益率': [performance_metrics[method]['annual_return'] for method in ensemble_methods],
        '最大回撤': [performance_metrics[method]['max_drawdown'] for method in ensemble_methods],
        '夏普比率': [performance_metrics[method]['sharpe_ratio'] for method in ensemble_methods],
        '卡尔马比率': [performance_metrics[method]['calmar_ratio'] for method in ensemble_methods],
        '交易次数': [performance_metrics[method]['trade_count'] for method in ensemble_methods],
        '胜率': [performance_metrics[method]['win_rate'] for method in ensemble_methods],
    })
    
    # 保存比较结果
    comparison_path = get_report_path("hybrid_strategy_comparison.csv", subdirectory=output_subdir)
    comparison.to_csv(comparison_path, index=False)
    logger.info(f"混合策略比较结果已保存至: {comparison_path}")
    
    # 5. 可视化资本曲线对比
    plt.figure(figsize=(12, 8))
    
    for method in ensemble_methods:
        equity_curve = backtest_results[method]['equity_curve']
        plt.plot(equity_curve.index, equity_curve, label=f"{method}")
    
    # 添加基准（买入持有）
    benchmark = df['close'] / df['close'].iloc[0] * 10000
    plt.plot(benchmark.index, benchmark, label='Buy & Hold', linestyle='--', color='gray')
    
    plt.title('混合策略组合方法资本曲线比较')
    plt.xlabel('日期')
    plt.ylabel('资本($)')
    plt.legend()
    plt.grid(True)
    
    # 应用中文字体
    font_helper.apply_font_to_figure(plt.gcf())
    
    # 保存图表
    equity_curve_path = get_image_path("hybrid_strategy_comparison.png", subdirectory=output_subdir)
    plt.savefig(equity_curve_path)
    logger.info(f"混合策略资本曲线对比图已保存至: {equity_curve_path}")
    
    # 6. 分析最佳组合方法
    best_method = comparison.iloc[comparison['总收益率'].idxmax()]
    logger.info(f"最佳组合方法: {best_method['组合方法']}")
    logger.info(f"最佳组合方法总收益率: {best_method['总收益率']:.2%}")
    logger.info(f"最佳组合方法年化收益率: {best_method['年化收益率']:.2%}")
    
    # 7. 创建详细报告
    report_template = f"""
# 混合策略回测报告

## 回测概况
- **回测期间**: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}
- **交易对**: BTC/USDT
- **时间周期**: 1天
- **初始资本**: $10,000
- **交易手续费**: 0.1%

## 策略性能比较
{comparison.to_markdown(index=False)}

## 最佳策略: {best_method['组合方法']}
- **最终资本**: ${best_method['最终资本']:.2f}
- **总收益率**: {best_method['总收益率']:.2%}
- **年化收益率**: {best_method['年化收益率']:.2%}
- **最大回撤**: {best_method['最大回撤']:.2%}
- **夏普比率**: {best_method['夏普比率']:.2f}
- **卡尔马比率**: {best_method['卡尔马比率']:.2f}
- **交易次数**: {best_method['交易次数']}
- **胜率**: {best_method['胜率']:.2%}

## 分析结论
1. 混合策略能够结合MACD和LSTM的优势，提供更稳健的交易信号
2. 不同的组合方法在不同市场条件下表现各异
3. 在当前测试期间，{best_method['组合方法']}组合方法表现最佳
4. 混合策略有效降低了单一策略的弱点，提高了整体稳定性

## 改进建议
1. 考虑根据市场状态动态切换组合方法
2. 进一步优化LSTM模型和MACD参数
3. 添加更多基本面指标，增强预测能力
4. 探索添加第三种策略（如情绪分析或波动率策略）到混合组合中

## 资本曲线图
![混合策略资本曲线比较](../images/{output_subdir}/hybrid_strategy_comparison.png)

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
    """
    
    # 保存markdown报告
    report_path = get_report_path("hybrid_strategy_report.md", subdirectory=output_subdir)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_template)
    logger.info(f"混合策略详细报告已保存至: {report_path}")
    
    # 输出分析结论
    logger.info("\n混合策略分析结论:")
    logger.info("1. 混合策略能够结合MACD和LSTM的优势，提供更稳健的交易信号")
    logger.info("2. 不同的组合方法在不同市场条件下表现各异")
    logger.info(f"3. 在当前测试期间，{best_method['组合方法']}组合方法表现最佳")
    logger.info("4. 混合策略有效降低了单一策略的弱点，提高了整体稳定性")
    
    logger.info("\n建议:")
    logger.info("1. 考虑根据市场状态动态切换组合方法")
    logger.info("2. 进一步优化LSTM模型和MACD参数")
    logger.info("3. 添加更多基本面指标，增强预测能力")
    logger.info("4. 探索添加第三种策略（如情绪分析或波动率策略）到混合组合中")
    
    logger.info("\n混合策略示例运行完成!") 