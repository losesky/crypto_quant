#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market Regime Classifier Integration Test with Hybrid Strategy

This script tests the integration of the new MarketRegimeClassifier
with the MACDLSTMHybridStrategy, demonstrating how different market
regimes affect trading decisions.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.font_manager import FontProperties

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from crypto_quant package
from crypto_quant.data.sources.binance_source import BinanceDataSource
from crypto_quant.analysis.market_regime_classifier import MarketRegimeClassifier
from crypto_quant.strategies.hybrid.macd_lstm_hybrid_strategy import MACDLSTMHybridStrategy
from crypto_quant.utils.logger import logger
from crypto_quant.utils.font_helper import get_font_helper
from crypto_quant.utils.output_helper import get_image_path, get_report_path

# Get font helper instance
font_helper = get_font_helper()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """
    Main function to test the integration of MarketRegimeClassifier with HybridStrategy
    """
    # Set parameters
    symbol = "BTC/USDT"
    interval = "1d"
    
    # Load 1 year of daily data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Initialize data source
    logger.info("Initializing Binance data source...")
    data_source = BinanceDataSource()
    
    # Get historical data
    logger.info("Loading Bitcoin historical data...")
    df = data_source.get_historical_data(
        symbol=symbol,
        interval=interval,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d')
    )
    
    if df is None or len(df) < 100:
        logger.error("Failed to load sufficient data or data loading returned None")
        return
    
    logger.info(f"Loaded {len(df)} data points from {df.index.min()} to {df.index.max()}")
    
    # Print data range
    print(f"\nData Range: {df.index.min()} to {df.index.max()}")
    
    # First, run standalone market regime classifier
    logger.info("Running standalone market regime classifier...")
    
    classifier = MarketRegimeClassifier(
        adx_threshold=25.0,
        volatility_threshold=0.03,
        visualization_enabled=True
    )
    
    # Classify all market states
    df_classified = classifier.classify_all(df)
    
    # Create visualization
    fig = classifier.visualize(df_classified, title="Bitcoin Market Regimes")
    
    # Apply Chinese font if available
    if font_helper.has_chinese_font:
        plt.suptitle("比特币市场状态分类", fontsize=16)
    else:
        plt.suptitle("Bitcoin Market Regime Classification", fontsize=16)
    
    # Save the visualization
    market_regime_plot_path = get_image_path('market_regimes_standalone.png')
    plt.savefig(market_regime_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Market regime visualization saved to {market_regime_plot_path}")
    
    # Print state statistics
    state_stats = classifier.get_state_statistics(df_classified)
    print("\nMarket State Statistics:")
    for state, stats in state_stats.items():
        print(f"\n{state.upper()}:")
        print(f"  Count: {stats['count']} ({stats['percent']:.2f}% of data)")
        print(f"  Avg Return: {stats['avg_return']:.4f}%")
        print(f"  Positive Return Ratio: {stats['pos_return_ratio']:.2f}%")
    
    # Now, set up the hybrid strategy with different ensemble methods
    ensemble_methods = ['vote', 'weight', 'layered', 'expert']
    results = {}
    
    for method in ensemble_methods:
        logger.info(f"Running hybrid strategy with {method} ensemble method...")
        
        # Initialize hybrid strategy
        strategy = MACDLSTMHybridStrategy(
            # MACD parameters
            macd_fast_period=12,
            macd_slow_period=26,
            macd_signal_period=9,
            
            # LSTM parameters
            lstm_sequence_length=20,
            lstm_hidden_dim=128,
            lstm_feature_engineering=True,
            
            # Hybrid parameters
            ensemble_method=method,
            market_regime_threshold=0.03,  # Match classifier's volatility threshold
            output_dir=""
        )
        
        # 兼容性修复：确保_layered_ensemble方法能够获取正确的MACD柱状图列
        if method == 'layered':
            original_layered_ensemble = strategy._layered_ensemble
            
            def layered_ensemble_wrapper(macd_signal, lstm_signal, row_index, df):
                # 首先检查macd_hist列是否存在
                if 'macd_hist' not in df.columns:
                    # 如果histogram列存在，使用它
                    if 'histogram' in df.columns:
                        logger.info("Using histogram column instead of macd_hist")
                        df['macd_hist'] = df['histogram']
                    else:
                        # 如果都不存在，使用0值创建macd_hist列
                        logger.warning("Cannot find MACD histogram data, creating zero-filled macd_hist column")
                        df['macd_hist'] = 0
                
                return original_layered_ensemble(macd_signal, lstm_signal, row_index, df)
            
            strategy._layered_ensemble = layered_ensemble_wrapper
            logger.info("Added compatibility layer for layered ensemble method")
        
        # 同时修复generate_signals方法，确保必要的列名一致性
        original_generate_signals = strategy.generate_signals
        
        def generate_signals_wrapper(df=None):
            result = original_generate_signals(df)
            if result is not None:
                # 确保macd_hist列存在
                if 'histogram' in result.columns and 'macd_hist' not in result.columns:
                    result['macd_hist'] = result['histogram']
                
                # 确保其他必要的MACD列名也正确映射
                if 'signal_line' in result.columns and 'macd_signal' not in result.columns:
                    result['macd_signal'] = result['signal_line']
            
            return result
        
        strategy.generate_signals = generate_signals_wrapper
        logger.info("Added column name consistency wrapper for generate_signals method")
        
        # Create backtest engine instance
        from crypto_quant.backtesting.engine.backtest_engine import BacktestEngine
        
        # Prepare strategy with the data
        strategy.prepare(df)
        
        # Create backtest engine
        backtest_engine = BacktestEngine(df, strategy)
        
        # Run backtest
        backtest_engine.run()
        
        # Get results and performance
        signals_df = backtest_engine.results
        performance = backtest_engine.performance
        
        # Print performance report
        backtest_engine.print_performance_report()
        
        # Store results
        results[method] = {
            'signals_df': signals_df,
            'performance': performance
        }
        
        # 打印性能指标，用于调试
        print(f"\nPerformance metrics for {method} method:")
        for key, value in performance.items():
            print(f"  {key}: {value}")
        
        # 检查signals_df是否为None
        if signals_df is None:
            logger.warning(f"Results DataFrame for {method} method is None")
        else:
            print(f"signals_df columns: {list(signals_df.columns)}")
        
        # Visualize results
        backtest_fig = backtest_engine.plot_results()
        
        # Apply Chinese title if available
        if font_helper.has_chinese_font:
            plt.gcf().suptitle(f"混合策略 - {method.upper()}", fontsize=16)
        else:
            plt.gcf().suptitle(f"Hybrid Strategy - {method.upper()} Ensemble Method", fontsize=16)
        
        # Save signals visualization
        strategy_plot_path = get_image_path(f'hybrid_strategy_{method}.png')
        plt.savefig(strategy_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Strategy visualization saved to {strategy_plot_path}")
        
        # Close plot to avoid memory issues
        plt.close()
    
    # Compare performance across ensemble methods
    print("\nPerformance Comparison Across Ensemble Methods:")
    
    # 调整对比指标，确保与performance字典中的键名一致
    comparison_metrics = ['annual_return', 'total_return', 'max_drawdown', 
                         'sharpe_ratio', 'calmar_ratio', 'trade_count', 'win_rate']
    
    comparison_df = pd.DataFrame(index=comparison_metrics)
    
    for method in ensemble_methods:
        # 检查这个方法是否有有效的性能数据
        if method not in results or 'performance' not in results[method] or results[method]['performance'] is None:
            logger.warning(f"No valid performance data for {method} method, skipping in comparison")
            # 为这个方法添加NaN值
            comparison_df[method] = [np.nan] * len(comparison_metrics)
            continue
            
        metrics = results[method]['performance']
        
        # 创建结果数组，对于缺失的度量使用NaN
        method_values = []
        for metric in comparison_metrics:
            if metric in metrics:
                # 对百分比值进行调整
                if metric in ['annual_return', 'total_return', 'max_drawdown', 'win_rate']:
                    value = metrics[metric] * 100
                else:
                    value = metrics[metric]
                method_values.append(value)
            else:
                # 改进处理方式 - 使用更明确的警告信息并尝试为常见指标提供合理替代值
                if metric == 'trade_count' and 'winning_trades' in metrics and 'losing_trades' in metrics:
                    # 如果有胜负交易数，可以计算总交易数
                    value = metrics['winning_trades'] + metrics['losing_trades']
                    logger.info(f"为{method}方法计算trade_count: {value} (从winning_trades和losing_trades)")
                    method_values.append(value)
                else:
                    logger.warning(f"指标 '{metric}' 在 {method} 方法的性能数据中不存在，将使用NaN值")
                    method_values.append(np.nan)
                
        comparison_df[method] = method_values
    
    # Format comparison DataFrame for better readability
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(comparison_df)
    
    # Compare signals by market regime
    print("\nSignal Generation by Market Regime:")
    
    # Merge market state data with signals
    for method in ensemble_methods:
        # 检查这个方法是否有有效的信号数据
        if method not in results or 'signals_df' not in results[method] or results[method]['signals_df'] is None:
            logger.warning(f"No valid signals data for {method} method, skipping in signal analysis")
            continue
            
        signals_df = results[method]['signals_df']
        
        # 检查必要的列是否存在
        if 'signal' not in signals_df.columns:
            logger.warning(f"'signal' column not found in results for {method} method")
            continue
            
        # 添加市场状态信息
        try:
            if 'market_state' not in signals_df.columns:
                # Add market state from classifier
                signals_df = pd.merge(
                    signals_df,
                    df_classified[['market_state']],
                    left_index=True,
                    right_index=True,
                    how='left'
                )
            
            # Count signal types by market state
            signal_by_state = pd.crosstab(
                signals_df['market_state'],
                signals_df['signal'].apply(lambda x: 'buy' if x > 0 else ('sell' if x < 0 else 'hold')),
                normalize='index'
            ) * 100
            
            print(f"\n{method.upper()} Ensemble Method - Signal Distribution by Market State (%):")
            print(signal_by_state)
        except Exception as e:
            logger.error(f"Error generating signal distribution for {method} method: {str(e)}")
        
        # Calculate strategy performance by market state
        try:
            returns_by_state = {}
            for state in signals_df['market_state'].unique():
                # Skip unknown state
                if state == 'unknown':
                    continue
                    
                # Get data for this state
                state_df = signals_df[signals_df['market_state'] == state]
                
                # Calculate returns for this state
                if len(state_df) > 1 and 'close' in state_df.columns and 'position' in state_df.columns:
                    # Simple return calculation (position * price_change)
                    returns = state_df['position'].shift(1) * state_df['close'].pct_change()
                    
                    returns_by_state[state] = {
                        'avg_return': returns.mean() * 100,
                        'win_rate': (returns > 0).mean() * 100 if len(returns.dropna()) > 0 else 0,
                        'trade_count': (state_df['position'] != state_df['position'].shift(1)).sum()
                    }
            
            if returns_by_state:
                print(f"\n{method.upper()} Ensemble Method - Performance by Market State:")
                for state, metrics in returns_by_state.items():
                    print(f"  {state}: Avg Return = {metrics['avg_return']:.4f}%, "
                          f"Win Rate = {metrics['win_rate']:.2f}%, "
                          f"Trades = {metrics['trade_count']}")
            else:
                logger.warning(f"No performance by market state data available for {method} method")
        except Exception as e:
            logger.error(f"Error calculating performance by market state for {method} method: {str(e)}")
    
    # Create visualization comparing performance across methods
    try:
        plt.figure(figsize=(12, 8))
        
        # 跟踪是否有任何有效的数据被绘制
        has_valid_data = False
        
        # 确定共同的初始资金
        initial_capital = 10000
        
        # 检测异常情况
        has_invalid_equity = False
        
        # Extract equity curves
        for method in ensemble_methods:
            if method not in results or 'signals_df' not in results[method] or results[method]['signals_df'] is None:
                logger.warning(f"No valid data for {method} method, skipping in visualization")
                continue
                
            signals_df = results[method]['signals_df']
            
            if 'equity_curve' in signals_df.columns:
                # 检查equity_curve是否有效
                equity_curve = signals_df['equity_curve'].copy()
                
                # 检查equity_curve是否全部为0或接近0
                if equity_curve.max() < 0.1:
                    logger.warning(f"{method} strategy equity curve appears to be near zero or invalid")
                    has_invalid_equity = True
                    continue
                
                # 检查equity_curve是否包含NaN值
                if equity_curve.isna().any():
                    logger.warning(f"{method} strategy equity curve contains NaN values, filling with forward fill")
                    equity_curve = equity_curve.fillna(method='ffill').fillna(1.0)
                
                # 重新计算曲线 - 从共同的初始资金开始
                normalized_equity = equity_curve / equity_curve.iloc[0] * initial_capital
                
                # 绘制标准化后的曲线
                plt.plot(signals_df.index, normalized_equity, label=f"{method.upper()} Strategy")
                has_valid_data = True
                
                logger.info(f"Plotted {method} strategy with initial value {equity_curve.iloc[0]:.4f} and final value {equity_curve.iloc[-1]:.4f}")
        
        # Add buy and hold baseline
        if 'close' in df.columns:
            baseline = df['close'] / df['close'].iloc[0] * initial_capital
            plt.plot(df.index, baseline, '--', label='Buy & Hold', color='gray')
            has_valid_data = True
        
        # 如果检测到异常情况，尝试使用累积收益率绘制
        if has_invalid_equity:
            logger.warning("Detected invalid equity curves, trying to use cumulative returns instead")
            for method in ensemble_methods:
                if method not in results or 'signals_df' not in results[method] or results[method]['signals_df'] is None:
                    continue
                    
                signals_df = results[method]['signals_df']
                
                # 尝试使用累积收益率计算资金曲线
                if 'cumulative_strategy_returns' in signals_df.columns:
                    cum_returns = signals_df['cumulative_strategy_returns'].copy()
                    
                    # 检查有效性
                    if cum_returns.max() > 0.1 and not cum_returns.isna().all():
                        # 填充NaN值
                        cum_returns = cum_returns.fillna(method='ffill').fillna(1.0)
                        # 计算资金曲线
                        alt_equity = cum_returns * initial_capital
                        plt.plot(signals_df.index, alt_equity, label=f"{method.upper()} (alt)", linestyle=':')
                        has_valid_data = True
                        logger.info(f"Plotted alternative {method} strategy curve")
                        
                # 如果没有cumulative_returns，尝试直接计算
                elif 'strategy_returns_after_commission' in signals_df.columns:
                    # 计算累积收益率
                    returns = signals_df['strategy_returns_after_commission'].fillna(0)
                    cum_returns = (1 + returns).cumprod()
                    alt_equity = cum_returns * initial_capital
                    plt.plot(signals_df.index, alt_equity, label=f"{method.upper()} (calc)", linestyle='-.')
                    has_valid_data = True
                    logger.info(f"Plotted calculated {method} strategy curve")
        
        # 只有在有有效数据的情况下才保存图表
        if has_valid_data:
            # Apply Chinese font if available
            if font_helper.has_chinese_font:
                # 使用字体助手设置中文标题和标签
                zh_title = '不同策略组合方法的性能比较'
                zh_xlabel = '日期'
                zh_ylabel = '账户价值'
                
                # 直接使用 FontProperties 对象设置标题
                try:
                    plt.title(zh_title, fontproperties=FontProperties(fname=font_helper.chinese_font))
                    plt.xlabel(zh_xlabel, fontproperties=FontProperties(fname=font_helper.chinese_font))
                    plt.ylabel(zh_ylabel, fontproperties=FontProperties(fname=font_helper.chinese_font))
                except Exception as e:
                    logger.warning(f"设置图表中文标题失败，使用英文标题: {str(e)}")
                    plt.title('Performance Comparison of Different Ensemble Methods')
                    plt.xlabel('Date')
                    plt.ylabel('Equity')
            else:
                plt.title('Performance Comparison of Different Ensemble Methods')
                plt.xlabel('Date')
                plt.ylabel('Equity')
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 添加y轴范围控制，避免因为异常值导致图表不可读
            plt.ylim(bottom=initial_capital * 0.5, top=plt.ylim()[1] * 1.1)
            
            # 保存前应用字体
            font_helper.apply_font_to_figure(plt.gcf())
            
            # Save comparison chart
            comparison_path = get_image_path('hybrid_strategy_comparison.png')
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            logger.info(f"Ensemble methods comparison chart saved to {comparison_path}")
        else:
            logger.warning("No valid data to create comparison visualization")
        
        plt.close()
    except Exception as e:
        logger.error(f"Error creating comparison visualization: {str(e)}")
    
    logger.info("Integration test completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in main function: {str(e)}") 