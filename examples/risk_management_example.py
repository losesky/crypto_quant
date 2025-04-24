#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风险管理示例脚本
展示如何使用高级风险管理功能并测试其效果
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

from crypto_quant.data.sources.binance_source import BinanceDataSource
from crypto_quant.strategies.hybrid.macd_lstm_hybrid_strategy import MACDLSTMHybridStrategy
from crypto_quant.backtesting.engine.backtest_engine import BacktestEngine
from crypto_quant.risk_management.risk_manager import RiskManager
from crypto_quant.utils.logger import logger, set_log_level
from crypto_quant.utils.font_helper import get_font_helper


def main():
    """主函数"""
    # 设置日志级别
    set_log_level("INFO")
    logger.info("开始风险管理示例脚本")
    
    # 初始化字体助手
    font_helper = get_font_helper()
    
    # 数据参数
    symbol = "BTC/USDT"
    interval = "1d"
    days = 730  # 2年数据
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 策略参数
    initial_capital = 10000.0
    commission = 0.001
    
    # 获取数据
    logger.info(f"获取{symbol}历史数据 ({start_date} 至 {end_date})")
    data_source = BinanceDataSource()
    df = data_source.get_historical_data(
        symbol=symbol,
        interval=interval,
        start=start_date,
        end=end_date
    )
    
    if df.empty:
        logger.error("获取数据失败")
        return
    
    logger.info(f"获取到 {len(df)} 条数据记录")
    
    # 创建基本混合策略
    base_strategy = MACDLSTMHybridStrategy(
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
        ensemble_method="weight",
        ensemble_weights=(0.6, 0.4),
        market_regime_threshold=0.15,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )
    
    # 创建带有高级风险管理的策略
    advanced_strategy = MACDLSTMHybridStrategy(
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
        ensemble_method="weight",
        ensemble_weights=(0.6, 0.4),
        market_regime_threshold=0.15,
        # 不在策略中设置止损止盈，由风险管理器处理
        stop_loss_pct=None,
        take_profit_pct=None,
    )
    
    # 创建风险管理器
    risk_manager = RiskManager(
        max_drawdown=0.15,              # 最大允许回撤比例
        max_position_size=0.2,          # 最大头寸比例（占总资金）
        base_position_size=0.1,         # 基础头寸比例
        fixed_stop_loss=0.05,           # 固定止损比例
        trailing_stop=0.03,             # 追踪止损比例
        take_profit=0.10,               # 止盈比例
        max_trades_per_day=3,           # 每日最大交易次数
        time_stop_bars=10,              # 时间止损(K线数)
        consecutive_losses=3,           # 允许连续亏损次数
        volatility_lookback=20,         # 波动率计算回看周期
        volatility_scale_factor=3.0,    # 波动率调整系数
        use_atr_for_stops=True,         # 是否使用ATR动态调整止损
        atr_stop_multiplier=2.0,        # ATR止损乘数
        drawdown_position_reduce=True,  # 回撤时是否降低仓位
        initial_capital=initial_capital # 初始资金
    )
    
    logger.info("进行基本策略回测（不使用高级风险管理）...")
    
    # 创建基本策略回测引擎
    base_engine = BacktestEngine(
        data=df.copy(),
        strategy=base_strategy,
        initial_capital=initial_capital,
        commission=commission
    )
    
    # 运行基本策略回测
    base_engine.run()
    
    # 打印基本策略性能报告
    logger.info("\n=== 基本策略性能报告 ===")
    base_engine.print_performance_report()
    
    logger.info("\n进行高级风险管理策略回测...")
    
    # 创建高级风险管理策略回测引擎
    advanced_engine = BacktestEngine(
        data=df.copy(),
        strategy=advanced_strategy,
        initial_capital=initial_capital,
        commission=commission,
        risk_manager=risk_manager
    )
    
    # 运行高级风险管理策略回测
    advanced_engine.run()
    
    # 打印高级风险管理策略性能报告
    logger.info("\n=== 高级风险管理策略性能报告 ===")
    advanced_engine.print_performance_report()
    
    # 比较两种策略的性能
    logger.info("\n=== 策略性能比较 ===")
    
    # 获取性能摘要
    base_performance = base_engine.summary()
    advanced_performance = advanced_engine.summary()
    
    # 提取关键指标
    comparison = pd.DataFrame({
        '指标': ['最终资本', '总收益率', '年化收益率', '最大回撤', '夏普比率', '卡尔玛比率', '交易次数', '胜率'],
        '基本策略': [
            f"${base_performance.get('final_capital', initial_capital):.2f}",
            f"{base_performance.get('total_return', 0):.2%}",
            f"{base_performance.get('annual_return', 0):.2%}",
            f"{base_performance.get('max_drawdown', 0):.2%}",
            f"{base_performance.get('sharpe_ratio', 0):.2f}",
            f"{base_performance.get('calmar_ratio', 0):.2f}",
            f"{base_performance.get('trade_count', 0)}",
            f"{base_performance.get('win_rate', 0):.2%}"
        ],
        '高级风险管理策略': [
            f"${advanced_performance.get('final_capital', initial_capital):.2f}",
            f"{advanced_performance.get('total_return', 0):.2%}",
            f"{advanced_performance.get('annual_return', 0):.2%}",
            f"{advanced_performance.get('max_drawdown', 0):.2%}",
            f"{advanced_performance.get('sharpe_ratio', 0):.2f}",
            f"{advanced_performance.get('calmar_ratio', 0):.2f}",
            f"{advanced_performance.get('trade_count', 0)}",
            f"{advanced_performance.get('win_rate', 0):.2%}"
        ]
    })
    
    # 打印比较表格
    print("\n策略性能比较:")
    print(comparison.to_string(index=False))
    
    # 绘制资本曲线对比图
    plt.figure(figsize=(12, 8))
    
    if not base_engine.results.empty and 'equity_curve' in base_engine.results.columns:
        plt.plot(base_engine.results.index, base_engine.results['equity_curve'], 
                label='基本策略', color='blue')
    
    if not advanced_engine.results.empty and 'equity_curve' in advanced_engine.results.columns:
        plt.plot(advanced_engine.results.index, advanced_engine.results['equity_curve'], 
                label='高级风险管理策略', color='green')
    
    # 添加基准（买入持有）
    benchmark = df['close'] / df['close'].iloc[0] * initial_capital
    plt.plot(benchmark.index, benchmark, label='买入持有', linestyle='--', color='gray')
    
    plt.title('风险管理策略对比')
    plt.xlabel('日期')
    plt.ylabel('资本($)')
    plt.legend()
    plt.grid(True)
    
    # 应用中文字体
    font_helper.apply_font_to_figure(plt.gcf())
    
    # 保存图表
    plt.savefig('risk_management_comparison.png')
    logger.info("资本曲线对比图已保存到 risk_management_comparison.png")
    
    # 绘制回撤对比图
    plt.figure(figsize=(12, 6))
    
    if not base_engine.results.empty and 'drawdown' in base_engine.results.columns:
        plt.plot(base_engine.results.index, base_engine.results['drawdown'] * 100, 
                label='基本策略', color='blue')
    
    if not advanced_engine.results.empty and 'drawdown' in advanced_engine.results.columns:
        plt.plot(advanced_engine.results.index, advanced_engine.results['drawdown'] * 100, 
                label='高级风险管理策略', color='green')
    
    plt.title('回撤对比')
    plt.xlabel('日期')
    plt.ylabel('回撤(%)')
    plt.legend()
    plt.grid(True)
    
    # 应用中文字体
    font_helper.apply_font_to_figure(plt.gcf())
    
    # 保存图表
    plt.savefig('drawdown_comparison.png')
    logger.info("回撤对比图已保存到 drawdown_comparison.png")
    
    # 绘制仓位对比图
    plt.figure(figsize=(12, 6))
    
    if not base_engine.results.empty and 'position' in base_engine.results.columns:
        plt.plot(base_engine.results.index, base_engine.results['position'], 
                label='基本策略', color='blue')
    
    if not advanced_engine.results.empty and 'position' in advanced_engine.results.columns:
        plt.plot(advanced_engine.results.index, advanced_engine.results['position'], 
                label='高级风险管理策略', color='green')
    
    # 如果高级策略有头寸大小数据，添加到图中
    if not advanced_engine.results.empty and 'position_size' in advanced_engine.results.columns:
        ax2 = plt.gca().twinx()
        ax2.plot(advanced_engine.results.index, advanced_engine.results['position_size'], 
                label='头寸比例', color='red', linestyle='--')
        ax2.set_ylabel('头寸比例')
        ax2.set_ylim(0, 0.25)
        
        # 合并图例
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        plt.legend()
    
    plt.title('仓位对比')
    plt.xlabel('日期')
    plt.ylabel('仓位')
    plt.grid(True)
    
    # 应用中文字体
    font_helper.apply_font_to_figure(plt.gcf())
    
    # 保存图表
    plt.savefig('position_comparison.png')
    logger.info("仓位对比图已保存到 position_comparison.png")
    
    # 输出分析结论
    logger.info("\n风险管理分析结论:")
    
    # 计算改进百分比
    base_return = base_performance.get('total_return', 0)
    adv_return = advanced_performance.get('total_return', 0)
    return_improvement = ((1 + adv_return) / (1 + base_return) - 1) * 100 if base_return > -1 else 0
    
    base_drawdown = abs(base_performance.get('max_drawdown', 0))
    adv_drawdown = abs(advanced_performance.get('max_drawdown', 0))
    drawdown_improvement = ((base_drawdown - adv_drawdown) / base_drawdown) * 100 if base_drawdown > 0 else 0
    
    base_calmar = base_performance.get('calmar_ratio', 0)
    adv_calmar = advanced_performance.get('calmar_ratio', 0)
    calmar_improvement = ((adv_calmar - base_calmar) / base_calmar) * 100 if base_calmar > 0 else 0
    
    logger.info(f"1. 收益率变化: {'提高' if return_improvement > 0 else '降低'} {abs(return_improvement):.1f}%")
    logger.info(f"2. 最大回撤改善: {'减少' if drawdown_improvement > 0 else '增加'} {abs(drawdown_improvement):.1f}%")
    logger.info(f"3. 卡尔玛比率变化: {'提高' if calmar_improvement > 0 else '降低'} {abs(calmar_improvement):.1f}%")
    
    if adv_calmar > base_calmar:
        logger.info("4. 高级风险管理显著提高了风险调整后收益，推荐在实际交易中使用")
    elif adv_drawdown < base_drawdown and adv_return >= base_return * 0.9:
        logger.info("4. 高级风险管理有效降低了回撤，同时基本保持了收益水平，建议在实际交易中使用")
    elif adv_drawdown > base_drawdown and adv_return > base_return:
        logger.info("4. 高级风险管理提高了收益但增加了回撤，适合风险承受能力较强的投资者")
    else:
        logger.info("4. 在当前测试条件下，高级风险管理效果不明显，需要进一步优化参数")

    logger.info("风险管理示例完成")

if __name__ == "__main__":
    main() 