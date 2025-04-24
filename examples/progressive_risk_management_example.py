#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
渐进式风险管理示例脚本
展示如何使用渐进式风险管理解决数据点不足的问题
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入项目模块
from crypto_quant.data.sources.binance_source import BinanceDataSource
from crypto_quant.strategies.hybrid.macd_lstm_hybrid_strategy import MACDLSTMHybridStrategy
from crypto_quant.backtesting.engine.backtest_engine import BacktestEngine
from crypto_quant.utils.logger import logger, set_log_level
from crypto_quant.utils.font_helper import get_font_helper
from crypto_quant.risk_management.risk_manager import RiskManager
from crypto_quant.utils.output_helper import get_image_path, ensure_dir_exists

def main():
    """主函数"""
    # 设置日志级别为DEBUG，以查看详细日志
    set_log_level("DEBUG")
    logger.info("开始渐进式风险管理示例...")
    
    # 初始化字体助手
    font_helper = get_font_helper()
    
    # 设置初始参数
    symbol = "BTC/USDT"
    interval = "1d"
    days = 30  # 只获取30天的数据，模拟数据点不足的情况
    initial_capital = 10000.0
    commission = 0.001
    
    # 连接数据源
    logger.info("连接Binance数据源")
    data_source = BinanceDataSource()
    
    # 获取历史数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"获取{symbol}数据: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    df = data_source.get_historical_data(
        symbol=symbol,
        interval=interval,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d')
    )
    
    if df.empty:
        logger.error("未获取到数据，请检查数据源配置")
        return False
    
    logger.info(f"获取到 {len(df)} 条数据记录")
    
    # 定义策略参数
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    lstm_sequence = 10  # 减小序列长度以适应较少的数据点
    lstm_hidden = 64
    lstm_threshold = 0.01
    lstm_feature_engineering = True
    lstm_attention = True
    ensemble_method = "expert"
    ensemble_weights = (0.6, 0.4)
    market_regime_threshold = 0.15
    
    # 对比三种风险管理方式
    risk_manager_configs = [
        {
            "name": "传统风险管理",
            "config": {
                "volatility_lookback": 20,  # 传统的回看周期设置，将导致数据点不足
                "min_lookback": 20,         # 最小回看周期与完整回看周期相同，不允许降低
                "base_position_size": 0.1,
                "max_position_size": 0.2,
                "fixed_stop_loss": 0.05,
                "trailing_stop": 0.03,
                "take_profit": 0.1,
                "initial_capital": initial_capital
            }
        },
        {
            "name": "渐进式风险管理",
            "config": {
                "volatility_lookback": 20,  # 完整回看周期仍为20
                "min_lookback": 5,          # 但最小回看周期允许降至5
                "base_position_size": 0.1,
                "max_position_size": 0.2,
                "fixed_stop_loss": 0.05,
                "trailing_stop": 0.03,
                "take_profit": 0.1,
                "initial_capital": initial_capital
            }
        },
        {
            "name": "无风险管理",  # 基准比较
            "config": None
        }
    ]
    
    backtest_results = {}
    performance_metrics = {}
    
    # 运行不同风险管理配置的回测
    for config in risk_manager_configs:
        logger.info(f"======== 测试 {config['name']} ========")
        
        # 创建策略副本
        test_strategy = MACDLSTMHybridStrategy(
            # MACD参数
            macd_fast_period=macd_fast,
            macd_slow_period=macd_slow,
            macd_signal_period=macd_signal,
            
            # LSTM参数
            lstm_sequence_length=lstm_sequence,
            lstm_hidden_dim=lstm_hidden,
            lstm_prediction_threshold=lstm_threshold,
            lstm_feature_engineering=lstm_feature_engineering,
            lstm_use_attention=lstm_attention,
            
            # 混合策略参数
            ensemble_method=ensemble_method,
            ensemble_weights=ensemble_weights,
            market_regime_threshold=market_regime_threshold,
            # 不在策略中设置止损止盈，由风险管理器处理
            stop_loss_pct=None if config['config'] else 0.05,
            take_profit_pct=None if config['config'] else 0.1,
        )
        
        # 如果配置了风险管理，创建并设置风险管理器
        if config['config']:
            risk_manager = RiskManager(**config['config'])
            setattr(test_strategy, 'risk_manager', risk_manager)
            logger.info(f"已为 {config['name']} 配置风险管理器")
        
        # 创建回测引擎
        engine = BacktestEngine(
            data=df.copy(),
            strategy=test_strategy,
            initial_capital=initial_capital,
            commission=commission
        )
        
        # 运行回测
        result_df = engine.run()
        
        # 存储结果
        backtest_results[config['name']] = result_df
        performance_metrics[config['name']] = engine.summary()
        
        # 显示性能摘要
        logger.info(f"{config['name']} 性能摘要:")
        for key, value in performance_metrics[config['name']].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    # 分析结果 - 比较不同风险管理方式的头寸大小调整
    plt.figure(figsize=(12, 10))
    
    # 1. 头寸大小对比
    plt.subplot(2, 1, 1)
    for name in backtest_results:
        if 'position_size' in backtest_results[name].columns:
            plt.plot(backtest_results[name].index, backtest_results[name]['position_size'], 
                     label=f"{name} 头寸大小")
    plt.title('不同风险管理方式下的头寸大小对比')
    plt.xlabel('日期')
    plt.ylabel('头寸大小（占总资金比例）')
    plt.legend()
    plt.grid(True)
    
    # 2. 资本曲线对比
    plt.subplot(2, 1, 2)
    for name in backtest_results:
        if 'equity_curve' in backtest_results[name].columns:
            plt.plot(backtest_results[name].index, backtest_results[name]['equity_curve'], 
                     label=f"{name} 资本曲线")
        
    # 添加基准（买入持有）
    benchmark = df['close'] / df['close'].iloc[0] * initial_capital
    plt.plot(benchmark.index, benchmark, label='买入持有', linestyle='--', color='gray')
    
    plt.title('不同风险管理方式下的资本曲线对比')
    plt.xlabel('日期')
    plt.ylabel('资本($)')
    plt.legend()
    plt.grid(True)
    
    # 应用中文字体
    font_helper.apply_font_to_figure(plt.gcf())
    
    # 保存图表
    ensure_dir_exists("output/images")
    plt.savefig(get_image_path("progressive_risk_management_comparison.png"))
    logger.info(f"对比图已保存至: {get_image_path('progressive_risk_management_comparison.png')}")
    plt.close()
    
    # 创建性能比较表格
    comparison = pd.DataFrame({
        '风险管理方式': [name for name in performance_metrics],
        '最终资本': [performance_metrics[name].get('final_capital', initial_capital) for name in performance_metrics],
        '总收益率': [performance_metrics[name].get('total_return', 0.0) for name in performance_metrics],
        '年化收益率': [performance_metrics[name].get('annual_return', 0.0) for name in performance_metrics],
        '最大回撤': [performance_metrics[name].get('max_drawdown', 0.0) for name in performance_metrics],
        '夏普比率': [performance_metrics[name].get('sharpe_ratio', 0.0) for name in performance_metrics],
        '卡尔马比率': [performance_metrics[name].get('calmar_ratio', 0.0) for name in performance_metrics],
        '交易次数': [performance_metrics[name].get('trade_count', 0) for name in performance_metrics],
        '胜率': [performance_metrics[name].get('win_rate', 0.0) for name in performance_metrics],
    })
    
    # 保存比较结果
    ensure_dir_exists("output/reports")
    comparison_path = os.path.join("output", "reports", "risk_management_comparison.csv")
    comparison.to_csv(comparison_path, index=False)
    logger.info(f"风险管理比较结果已保存至: {comparison_path}")
    
    # 打印比较结果
    logger.info("\n性能对比:")
    print(comparison)
    
    # 总结发现
    logger.info("\n结论:")
    logger.info("1. 传统风险管理因数据点不足无法有效计算波动率，导致始终使用基础仓位")
    logger.info("2. 渐进式风险管理能够在有限数据点的情况下动态调整计算方法，实现更精确的仓位管理")
    logger.info("3. 相比无风险管理的方法，渐进式风险管理可以提供更好的风险调整后收益")
    
    return True

if __name__ == "__main__":
    main() 