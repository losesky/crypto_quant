#!/usr/bin/env python
"""
运行混合策略回测并生成报告的脚本
这个脚本在回测后立即生成报告，确保报告被正确生成
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
from datetime import datetime, timedelta
import json
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crypto_quant.data.sources.binance_source import BinanceDataSource
from crypto_quant.strategies.hybrid.macd_lstm_hybrid_strategy import MACDLSTMHybridStrategy
from crypto_quant.backtesting.engine.backtest_engine import BacktestEngine
from crypto_quant.backtesting.visualization.performance_visualizer import PerformanceVisualizer
from crypto_quant.utils.logger import logger, set_log_level
from crypto_quant.utils.font_helper import get_font_helper
from crypto_quant.utils.output_helper import get_image_path, get_report_path, get_data_path, ensure_dir_exists

# 获取字体助手
font_helper = get_font_helper()

# 设置日志级别
set_log_level('DEBUG')

def try_format_row(row):
    """安全地格式化报告表格的一行数据，处理可能出现的异常情况"""
    try:
        # 检查所有需要的键是否存在
        required_keys = ['组合方法', '最终资本', '总收益率', '年化收益率', '最大回撤', '夏普比率', '胜率']
        for key in required_keys:
            if key not in row or pd.isna(row[key]):
                logging.warning(f"行数据缺少键'{key}'或值为NaN: {row}")
                # 如果缺少关键数据，返回一个带有可用数据的行，缺失数据显示为N/A
                safe_row = {}
                for k in required_keys:
                    if k in row and not pd.isna(row[k]):
                        safe_row[k] = row[k]
                    else:
                        safe_row[k] = "N/A"
                
                return f"<tr><td>{safe_row['组合方法']}</td><td>{safe_row['最终资本'] if safe_row['最终资本'] != 'N/A' else 'N/A'}</td><td>{safe_row['总收益率'] if safe_row['总收益率'] != 'N/A' else 'N/A'}</td><td>{safe_row['年化收益率'] if safe_row['年化收益率'] != 'N/A' else 'N/A'}</td><td>{safe_row['最大回撤'] if safe_row['最大回撤'] != 'N/A' else 'N/A'}</td><td>{safe_row['夏普比率'] if safe_row['夏普比率'] != 'N/A' else 'N/A'}</td><td>{safe_row['胜率'] if safe_row['胜率'] != 'N/A' else 'N/A'}</td></tr>"
        
        # 安全地格式化每个数值字段
        formatted_values = {
            '组合方法': row['组合方法'],
            '最终资本': f"${row['最终资本']:.2f}" if isinstance(row['最终资本'], (int, float)) and not pd.isna(row['最终资本']) else "N/A",
            '总收益率': f"{row['总收益率']:.2%}" if isinstance(row['总收益率'], (int, float)) and not pd.isna(row['总收益率']) else "N/A",
            '年化收益率': f"{row['年化收益率']:.2%}" if isinstance(row['年化收益率'], (int, float)) and not pd.isna(row['年化收益率']) else "N/A",
            '最大回撤': f"{row['最大回撤']:.2%}" if isinstance(row['最大回撤'], (int, float)) and not pd.isna(row['最大回撤']) else "N/A",
            '夏普比率': f"{row['夏普比率']:.2f}" if isinstance(row['夏普比率'], (int, float)) and not pd.isna(row['夏普比率']) else "N/A",
            '胜率': f"{row['胜率']:.2%}" if isinstance(row['胜率'], (int, float)) and not pd.isna(row['胜率']) else "N/A"
        }
        
        # 生成格式化的HTML行
        return f"<tr><td>{formatted_values['组合方法']}</td><td>{formatted_values['最终资本']}</td><td>{formatted_values['总收益率']}</td><td>{formatted_values['年化收益率']}</td><td>{formatted_values['最大回撤']}</td><td>{formatted_values['夏普比率']}</td><td>{formatted_values['胜率']}</td></tr>"
    except Exception as e:
        logging.error(f"格式化表格行时出错: {str(e)}")
        # 返回带有错误信息的行
        method_name = row.get('组合方法', 'Unknown')
        return f"<tr><td>{method_name}</td><td colspan='6'>数据处理错误: {str(e)}</td></tr>"

def run_hybrid_backtest():
    """运行混合策略回测并生成报告"""
    try:
        # 设置参数
        symbol = "BTC/USDT"
        interval = "1d"
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        # 创建输出子目录
        output_subdir = f"hybrid_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 初始化数据源
        logger.info("初始化Binance数据源...")
        data_source = BinanceDataSource()
        
        # 获取历史数据
        logger.info(f"获取{symbol}历史数据...")
        df = data_source.get_historical_data(symbol, interval, start=start_date, end=end_date)
        
        logger.info(f"获取到{len(df)}行历史数据，时间范围: {df.index.min()} 至 {df.index.max()}")
        
        # 保存原始数据
        raw_data_path = get_data_path(f"{symbol.replace('/', '_').lower()}_raw_data.csv", subdirectory=output_subdir)
        df.to_csv(raw_data_path)
        logger.info(f"原始数据已保存至: {raw_data_path}")
        
        # 要测试的组合方法
        ensemble_methods = ['vote', 'weight', 'layered', 'expert']
        
        # 存储回测结果和性能指标
        backtest_results = {}
        performance_metrics = {}
        
        for method in ensemble_methods:
            logger.info(f"======== 测试混合策略: {method} 组合方法 ========")
            
            try:
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
                    data=df,
                    strategy=strategy,
                    initial_capital=10000.0,
                    commission=0.001  # 0.1% 交易手续费
                )
                
                # 运行回测
                backtest_engine.run()
                
                # 保存详细回测结果
                backtest_results[method] = backtest_engine.results.copy()
                
                # 获取性能指标
                metrics = backtest_engine.summary()
                performance_metrics[method] = metrics
                
                # 显示性能指标，添加错误处理
                try:
                    logger.info(f"策略: {strategy.name}")
                    logger.info(f"最终资本: ${metrics.get('final_capital', 10000.0):.2f}")
                    logger.info(f"总收益率: {metrics.get('total_return', 0.0):.2%}")
                    logger.info(f"年化收益率: {metrics.get('annual_return', 0.0):.2%}")
                    logger.info(f"最大回撤: {metrics.get('max_drawdown', 0.0):.2%}")
                    logger.info(f"夏普比率: {metrics.get('sharpe_ratio', 0.0):.2f}")
                    logger.info(f"卡尔马比率: {metrics.get('calmar_ratio', 0.0):.2f}")
                    logger.info(f"交易次数: {metrics.get('trade_count', 0)}")
                    logger.info(f"胜率: {metrics.get('win_rate', 0.0):.2%}")
                    logger.info("")
                except Exception as e:
                    logger.error(f"显示性能指标时出错: {str(e)}")
                
                # 检查结果是否为空
                if backtest_results[method].empty:
                    logger.warning(f"{method}方法的回测结果为空，跳过保存和可视化步骤")
                    continue
                
                # 保存该方法的详细回测结果
                try:
                    method_results_path = get_data_path(f"{method}_backtest_results.csv", subdirectory=output_subdir)
                    backtest_results[method].to_csv(method_results_path)
                    logger.info(f"{method}方法的回测结果已保存至: {method_results_path}")
                except Exception as e:
                    logger.error(f"保存{method}方法的回测结果时出错: {str(e)}")
                    continue
                
                # 创建可视化器
                try:
                    visualizer = PerformanceVisualizer(backtest_results[method], metrics)
                    
                    # 绘制并保存该方法的资本曲线
                    equity_fig = visualizer.plot_equity_curve()
                    if equity_fig is not None:
                        font_helper.apply_font_to_figure(equity_fig)
                        plt.tight_layout()
                        equity_path = get_image_path(f"{method}_equity_curve.png", subdirectory=output_subdir)
                        plt.savefig(equity_path)
                        plt.close()
                        logger.info(f"{method}方法的资本曲线图已保存至: {equity_path}")
                    else:
                        logger.warning(f"无法生成{method}方法的资本曲线图")
                    
                    # 绘制回撤图
                    underwater_fig = visualizer.plot_underwater_chart()
                    if underwater_fig is not None:
                        font_helper.apply_font_to_figure(underwater_fig)
                        plt.tight_layout()
                        underwater_path = get_image_path(f"{method}_underwater_chart.png", subdirectory=output_subdir)
                        plt.savefig(underwater_path)
                        plt.close()
                        logger.info(f"{method}方法的回撤图已保存至: {underwater_path}")
                    else:
                        logger.warning(f"无法生成{method}方法的回撤图")
                    
                    # 保存交互式报告
                    report_path = get_report_path(f"{method}_strategy_report.html", subdirectory=output_subdir)
                    try:
                        logger.info(f"生成{method}方法的交互式报告...")
                        success = visualizer.save_report(report_path)
                        if success:
                            logger.info(f"{method}方法的交互式报告已保存至: {report_path}")
                        else:
                            logger.error(f"保存{method}方法的交互式报告失败!")
                    except Exception as e:
                        logger.error(f"生成{method}方法的交互式报告时出错: {str(e)}")
                        logger.error(traceback.format_exc())
                except Exception as e:
                    logger.error(f"创建{method}方法的可视化时出错: {str(e)}")
                    logger.error(traceback.format_exc())
            except Exception as e:
                logger.error(f"{method}方法回测过程中出错: {str(e)}")
                logger.error(traceback.format_exc())
                logger.info(f"跳过{method}方法，继续执行下一个方法")
                # 创建空的结果和指标，避免后续处理出错
                backtest_results[method] = pd.DataFrame()
                performance_metrics[method] = {
                    'final_capital': 10000.0,
                    'total_return': 0.0,
                    'annual_return': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'calmar_ratio': 0.0,
                    'trade_count': 0,
                    'win_rate': 0.0
                }
        
        # 比较不同组合方法的性能
        logger.info("======== 混合策略组合方法性能比较 ========")
        
        # 过滤出成功执行的方法
        successful_methods = []
        for method in ensemble_methods:
            # 检查DataFrame是否存在且不为空
            if method in backtest_results and isinstance(backtest_results[method], pd.DataFrame):
                if len(backtest_results[method]) > 0:
                    # 至少包含基本的回测结果列
                    if 'close' in backtest_results[method].columns:
                        successful_methods.append(method)
                        logger.info(f"方法 {method} 成功执行，产生了 {len(backtest_results[method])} 行回测结果")
                    else:
                        logger.warning(f"方法 {method} 缺少必要的数据列")
                else:
                    logger.warning(f"方法 {method} 的回测结果为空")
            else:
                logger.warning(f"方法 {method} 没有生成有效的回测结果DataFrame")
        
        # 检查是否至少有一个策略成功
        if not successful_methods:
            logger.error("没有成功执行的策略，无法进行比较")
            # 尝试创建简单的失败报告
            failure_report_path = get_report_path("hybrid_strategy_failure_report.html", subdirectory=output_subdir)
            try:
                with open(failure_report_path, 'w', encoding='utf-8') as f:
                    f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>回测失败报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        .error {{ color: red; }}
        pre {{ background-color: #f5f5f5; padding: 10px; }}
    </style>
</head>
<body>
    <h1>混合策略回测失败</h1>
    <p>回测期间: {start_date} 至 {end_date}</p>
    <p>交易对: {symbol}</p>
    
    <h2 class="error">执行结果: 所有策略执行均失败</h2>
    
    <h3>尝试的策略组合方法:</h3>
    <ul>
        {' '.join([f'<li>{method}</li>' for method in ensemble_methods])}
    </ul>
    
    <h3>可能的原因:</h3>
    <ol>
        <li>数据问题：数据可能存在缺失或异常值</li>
        <li>策略参数：当前市场条件下参数可能不适用</li>
        <li>边界条件：可能遇到了未处理的边界情况</li>
    </ol>
    
    <h3>建议操作:</h3>
    <ol>
        <li>检查原始数据质量</li>
        <li>调整策略参数</li>
        <li>减少回测时间范围</li>
        <li>每次只测试一个策略组合方法</li>
    </ol>
    
    <p><i>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i></p>
</body>
</html>
                    """)
                logger.info(f"已创建失败报告: {failure_report_path}")
                # 即使没有成功策略，也返回True表示脚本正常执行
                return True
            except Exception as e:
                logger.error(f"创建失败报告时出错: {str(e)}")
                logger.error(traceback.format_exc())
            
            return False
            
        logger.info(f"成功执行的策略: {successful_methods}")
        
        # 创建性能比较表格
        comparison = pd.DataFrame({
            '组合方法': [method for method in successful_methods],
            '最终资本': [performance_metrics[method]['final_capital'] for method in successful_methods],
            '总收益率': [performance_metrics[method]['total_return'] for method in successful_methods],
            '年化收益率': [performance_metrics[method]['annual_return'] for method in successful_methods],
            '最大回撤': [performance_metrics[method]['max_drawdown'] for method in successful_methods],
            '夏普比率': [performance_metrics[method]['sharpe_ratio'] for method in successful_methods],
            '卡尔马比率': [performance_metrics[method]['calmar_ratio'] for method in successful_methods],
            '交易次数': [performance_metrics[method]['trade_count'] for method in successful_methods],
            '胜率': [performance_metrics[method]['win_rate'] for method in successful_methods],
        })
        
        # 保存比较结果
        comparison_path = get_data_path("hybrid_strategy_comparison.csv", subdirectory=output_subdir)
        comparison.to_csv(comparison_path, index=False)
        logger.info(f"混合策略比较结果已保存至: {comparison_path}")
        
        # 可视化资本曲线对比
        try:
            plt.figure(figsize=(12, 8))
            
            valid_methods = []
            for method in successful_methods:
                if 'equity_curve' in backtest_results[method].columns:
                    equity_curve = backtest_results[method]['equity_curve']
                    plt.plot(equity_curve.index, equity_curve, label=f"{method}")
                    valid_methods.append(method)
                else:
                    logger.warning(f"方法 {method} 缺少equity_curve列，无法在比较图中显示")
            
            # 至少有一个有效策略才添加基准
            if valid_methods:
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
            else:
                logger.warning("没有可用的策略资本曲线，无法生成比较图")
            
            # 关闭图表
            plt.close()
        except Exception as e:
            logger.error(f"生成资本曲线比较图时出错: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 找到最佳方法
        try:
            if len(comparison) > 0:
                # 安全地获取最佳方法
                try:
                    best_method_idx = comparison['总收益率'].idxmax()
                    best_method = comparison.iloc[best_method_idx]
                    best_method_name = best_method.get('组合方法', '未知方法')
                    logger.info(f"最佳组合方法: {best_method_name}")
                    
                    # 安全地记录最佳方法的指标
                    total_return = best_method.get('总收益率', float('nan'))
                    annual_return = best_method.get('年化收益率', float('nan'))
                    
                    if not pd.isna(total_return):
                        logger.info(f"最佳组合方法总收益率: {total_return:.2%}")
                    else:
                        logger.warning("最佳组合方法总收益率数据缺失")
                        
                    if not pd.isna(annual_return):
                        logger.info(f"最佳组合方法年化收益率: {annual_return:.2%}")
                    else:
                        logger.warning("最佳组合方法年化收益率数据缺失")
                except Exception as e:
                    logger.error(f"确定最佳方法时出错: {str(e)}")
                    # 创建一个空的最佳方法字典作为后备
                    best_method = pd.Series({
                        '组合方法': '数据处理错误',
                        '最终资本': 10000.0,
                        '总收益率': 0.0,
                        '年化收益率': 0.0,
                        '最大回撤': 0.0,
                        '夏普比率': 0.0,
                        '胜率': 0.0
                    })
            else:
                logger.warning("比较表为空，无法确定最佳方法")
                # 创建一个空的最佳方法字典作为后备
                best_method = pd.Series({
                    '组合方法': '没有有效数据',
                    '最终资本': 10000.0,
                    '总收益率': 0.0,
                    '年化收益率': 0.0,
                    '最大回撤': 0.0,
                    '夏普比率': 0.0,
                    '胜率': 0.0
                })
        except Exception as e:
            logger.error(f"处理最佳方法数据时出错: {str(e)}")
            # 创建一个空的最佳方法字典作为后备
            best_method = pd.Series({
                '组合方法': '数据处理错误',
                '最终资本': 10000.0,
                '总收益率': 0.0,
                '年化收益率': 0.0,
                '最大回撤': 0.0,
                '夏普比率': 0.0,
                '胜率': 0.0
            })
        
        # 创建一个安全的格式化函数
        def safe_format_for_report(value, format_spec=""):
            """安全格式化任何值，防止格式说明符错误"""
            if not isinstance(value, (int, float)) or pd.isna(value):
                return "N/A"
            try:
                if format_spec == ".2%":
                    return f"{value:.2%}"
                elif format_spec == ".2f":
                    return f"{value:.2f}"
                else:
                    return str(value)
            except Exception as e:
                logger.warning(f"格式化值'{value}'时出错: {str(e)}")
                return "格式错误"
        
        # 生成比较表格的Markdown格式
        try:
            comparison_markdown = comparison.to_markdown(index=False)
            logger.info("成功生成策略比较表格的Markdown格式")
        except Exception as e:
            logger.error(f"生成策略比较表格的Markdown格式时出错: {str(e)}")
            # 创建备用表格
            comparison_markdown = """
| 组合方法 | 最终资本 | 总收益率 | 年化收益率 | 最大回撤 | 夏普比率 | 卡尔马比率 | 交易次数 | 胜率 |
|---------|---------|---------|-----------|---------|---------|-----------|---------|------|
"""
            # 手动添加每一行
            for _, row in comparison.iterrows():
                try:
                    method = row.get('组合方法', 'N/A')
                    final_capital = safe_format_for_report(row.get('最终资本', None), ".2f")
                    total_return = safe_format_for_report(row.get('总收益率', None), ".2%")
                    annual_return = safe_format_for_report(row.get('年化收益率', None), ".2%")
                    max_drawdown = safe_format_for_report(row.get('最大回撤', None), ".2%")
                    sharpe_ratio = safe_format_for_report(row.get('夏普比率', None), ".2f")
                    calmar_ratio = safe_format_for_report(row.get('卡尔马比率', None), ".2f")
                    trade_count = str(row.get('交易次数', 'N/A'))
                    win_rate = safe_format_for_report(row.get('胜率', None), ".2%")
                    
                    comparison_markdown += f"| {method} | {final_capital} | {total_return} | {annual_return} | {max_drawdown} | {sharpe_ratio} | {calmar_ratio} | {trade_count} | {win_rate} |\n"
                except Exception as sub_e:
                    logger.error(f"格式化表格行时出错: {str(sub_e)}")
                    comparison_markdown += f"| 格式化错误 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n"
            
            logger.info("已创建备用策略比较表格的Markdown格式")
        
        # 预先安全格式化所有需要在模板中使用的值
        best_method_name = best_method.get('组合方法', 'N/A')
        best_final_capital = safe_format_for_report(best_method.get('最终资本', None), ".2f")
        best_total_return = safe_format_for_report(best_method.get('总收益率', None), ".2%")
        best_annual_return = safe_format_for_report(best_method.get('年化收益率', None), ".2%")
        best_max_drawdown = safe_format_for_report(best_method.get('最大回撤', None), ".2%")
        best_sharpe_ratio = safe_format_for_report(best_method.get('夏普比率', None), ".2f")
        best_calmar_ratio = safe_format_for_report(best_method.get('卡尔马比率', None), ".2f")
        best_trade_count = str(best_method.get('交易次数', 'N/A'))
        best_win_rate = safe_format_for_report(best_method.get('胜率', None), ".2%")
        
        # 创建报告内容
        report_template = f"""# 混合策略回测报告

## 回测概览
- **回测周期**: {start_date} 至 {end_date}
- **交易对**: {symbol}
- **起始资金**: $10,000
- **手续费率**: 0.1%

## 策略性能比较
{comparison_markdown}

## 最佳策略: {best_method_name}
- **最终资本**: ${best_final_capital}
- **总收益率**: {best_total_return}
- **年化收益率**: {best_annual_return}
- **最大回撤**: {best_max_drawdown}
- **夏普比率**: {best_sharpe_ratio}
- **卡尔马比率**: {best_calmar_ratio}
- **交易次数**: {best_trade_count}
- **胜率**: {best_win_rate}

## 分析与结论
- 在{start_date}至{end_date}期间，{best_method_name}策略表现最佳
- 总收益率为{best_total_return}，年化收益率为{best_annual_return}
- 最大回撤为{best_max_drawdown}，表明策略在市场下跌时的风险控制能力
- 夏普比率为{best_sharpe_ratio}，显示了策略的风险调整后收益情况
- 交易次数为{best_trade_count}次，胜率为{best_win_rate}

## 改进建议
- 考虑增加止损策略以进一步控制风险
- 可以尝试调整MACD参数以适应不同市场条件
- 探索其他特征工程方法以提高LSTM模型的预测准确性
- 考虑加入市场情绪指标以增强策略的适应性

![资金曲线对比](images/{output_subdir}/hybrid_strategy_comparison.png)

*报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存markdown报告
        report_path = get_report_path("hybrid_strategy_report.md", subdirectory=output_subdir)
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_template)
            logger.info(f"混合策略详细报告已保存至: {report_path}")
        except Exception as e:
            logger.error(f"保存markdown报告时出错: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 尝试保存简单HTML报告
        html_report_path = get_report_path("hybrid_strategy_summary.html", subdirectory=output_subdir)
        try:
            # HTML报告内容
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>混合策略回测报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
        .container {{ width: 90%; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .error {{ color: red; }}
        .warning {{ color: orange; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>混合策略回测报告</h1>
        <p>回测期间: {start_date} 至 {end_date}</p>
        <p>交易对: {symbol}</p>
        <p>时间周期: {interval}</p>
        
        <h2>策略性能比较</h2>
        <table>
            <tr>
                <th>组合方法</th>
                <th>最终资本</th>
                <th>总收益率</th>
                <th>年化收益率</th>
                <th>最大回撤</th>
                <th>夏普比率</th>
                <th>胜率</th>
            </tr>
            {"".join([
                try_format_row(row) for _, row in comparison.iterrows()
            ])}
        </table>
        
        <h2>最佳策略: {best_method.get('组合方法', 'N/A')}</h2>
        <p>最终资本: ${best_final_capital}</p>
        <p>总收益率: {best_total_return}</p>
        <p>年化收益率: {best_annual_return}</p>
        <p>最大回撤: {best_max_drawdown}</p>
        <p>夏普比率: {best_sharpe_ratio}</p>
        <p>胜率: {best_win_rate}</p>
        
        <h2>资本曲线图</h2>
        <img src="../images/{output_subdir}/hybrid_strategy_comparison.png" style="width:100%; max-width:800px;">
        
        <p><i>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i></p>
    </div>
</body>
</html>
            """
            
            with open(html_report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"混合策略HTML摘要报告已保存至: {html_report_path}")
        except Exception as e:
            logger.error(f"保存HTML摘要报告时出错: {str(e)}")
            logger.error(traceback.format_exc())
        
        logger.info(f"\n混合策略回测完成! 所有输出已保存至 {output_subdir} 目录")
        return True
    except Exception as e:
        logger.error(f"运行混合策略回测时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("开始运行混合策略回测...")
    success = run_hybrid_backtest()
    if success:
        logger.info("混合策略回测成功完成!")
    else:
        logger.error("混合策略回测失败!") 