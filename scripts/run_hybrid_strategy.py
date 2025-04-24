#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合策略回测命令行工具
用于执行混合策略回测，支持命令行参数设置
"""
import sys
import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
import traceback

# 强制matplotlib使用Agg后端，确保在无GUI环境中也能正确保存图片
import matplotlib
matplotlib.use('Agg')

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 在导入项目模块前先设置项目路径
from crypto_quant.data.sources.binance_source import BinanceDataSource
from crypto_quant.strategies.hybrid.macd_lstm_hybrid_strategy import MACDLSTMHybridStrategy
from crypto_quant.backtesting.engine.backtest_engine import BacktestEngine
from crypto_quant.utils.logger import logger, set_log_level
from crypto_quant.utils.font_helper import get_font_helper
from crypto_quant.utils.output_helper import (
    get_image_path, get_report_path, get_data_path, 
    ensure_dir_exists, clean_old_outputs
)
# 导入新的风险管理模块
from crypto_quant.risk_management.risk_manager import RiskManager
# 导入新的图像调试模块
from crypto_quant.utils.image_debug import image_logger, trace_image_save, debug_figure, setup_image_debug_logger

# 添加一个自定义函数来获取与报告相同目录的图片路径
def get_image_in_report_dir(filename, subdirectory=None):
    """
    获取与报告相同目录的图片路径
    
    Args:
        filename (str): 文件名
        subdirectory (str, optional): 子目录名
        
    Returns:
        str: 完整的文件路径
    """
    try:
        from crypto_quant.utils.output_helper import DEFAULT_REPORTS_DIR, ensure_dir_exists
        
        if subdirectory is None:
            subdirectory = datetime.now().strftime('%Y-%m-%d')
        
        # 确保有.png扩展名
        if not filename.endswith('.png'):
            filename = f"{filename}.png"
        
        # 获取绝对路径
        report_dir = os.path.abspath(os.path.join(DEFAULT_REPORTS_DIR, subdirectory))
        logger.info(f"报告目录绝对路径: {report_dir}")
        
        # 确保目录存在
        if not os.path.exists(report_dir):
            logger.info(f"创建报告目录: {report_dir}")
            os.makedirs(report_dir, exist_ok=True)
        
        full_path = os.path.join(report_dir, filename)
        logger.info(f"完整的图片绝对路径: {full_path}")
        
        # 检查目录是否可写
        if not os.access(report_dir, os.W_OK):
            logger.error(f"目录不可写: {report_dir}")
            # 尝试修改权限
            try:
                logger.info(f"尝试修改目录权限: {report_dir}")
                os.chmod(report_dir, 0o755)
            except Exception as perm_error:
                logger.error(f"无法修改目录权限: {str(perm_error)}")
        
        return full_path
    except Exception as e:
        logger.error(f"获取图片路径时出错: {str(e)}")
        logger.error(traceback.format_exc())
        # 返回一个默认路径
        return os.path.join(os.getcwd(), filename)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="混合策略回测工具")
    
    # 数据参数
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="交易对符号")
    parser.add_argument("--interval", type=str, default="1d", help="时间周期")
    parser.add_argument("--days", type=int, default=730, help="回测天数")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="初始资本")
    parser.add_argument("--commission", type=float, default=0.001, help="交易手续费")
    
    # MACD参数
    parser.add_argument("--macd-fast", type=int, default=12, help="MACD快线周期")
    parser.add_argument("--macd-slow", type=int, default=26, help="MACD慢线周期")
    parser.add_argument("--macd-signal", type=int, default=9, help="MACD信号线周期")
    
    # LSTM参数
    parser.add_argument("--lstm-sequence", type=int, default=20, help="LSTM序列长度")
    parser.add_argument("--lstm-hidden", type=int, default=128, help="LSTM隐藏层维度")
    parser.add_argument("--lstm-threshold", type=float, default=0.01, help="LSTM预测阈值")
    parser.add_argument("--lstm-attention", action="store_true", help="是否使用注意力机制")
    
    # 混合策略参数
    parser.add_argument("--ensemble-methods", type=str, default="vote,weight,layered,expert", 
                        help="要测试的混合方法,以逗号分隔")
    parser.add_argument("--macd-weight", type=float, default=0.6, help="MACD权重")
    
    # 风险管理参数 - 新增
    parser.add_argument("--risk-management", action="store_true", help="是否启用高级风险管理")
    parser.add_argument("--max-drawdown", type=float, default=0.15, help="最大允许回撤比例")
    parser.add_argument("--max-position-size", type=float, default=0.2, help="最大头寸比例")
    parser.add_argument("--base-position-size", type=float, default=0.1, help="基础头寸比例")
    parser.add_argument("--fixed-stop-loss", type=float, default=0.05, help="固定止损比例")
    parser.add_argument("--trailing-stop", type=float, default=0.03, help="追踪止损比例")
    parser.add_argument("--take-profit", type=float, default=0.10, help="止盈比例")
    parser.add_argument("--volatility-scale", action="store_true", help="是否根据波动率调整仓位")
    parser.add_argument("--use-atr-stops", action="store_true", help="是否使用ATR动态止损")
    parser.add_argument("--max-trades-day", type=int, default=3, help="每日最大交易次数")
    parser.add_argument("--time-stop-bars", type=int, default=10, help="时间止损K线数")
    parser.add_argument("--volatility-lookback", type=int, default=20, help="波动率计算回看周期")
    parser.add_argument("--min-lookback", type=int, default=5, help="最小回看周期，用于渐进式风险管理")
    
    # 输出参数
    parser.add_argument("--output-dir", type=str, help="输出目录名称")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="日志级别")
    parser.add_argument("--clean-days", type=int, default=0, 
                        help="清理多少天前的输出文件，0表示不清理")
    
    # 图片保存测试
    parser.add_argument("--test-image-save", action="store_true", help="测试图片保存功能")
    
    return parser.parse_args()

def run_hybrid_strategy(args):
    """运行混合策略回测"""
    # 设置日志级别
    set_log_level(args.log_level)
    
    # 测试matplotlib图片保存功能
    if args.test_image_save:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            logger.info("开始测试matplotlib图片保存功能...")
            
            # 创建一个简单的测试图
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            
            plt.figure(figsize=(5, 5))
            plt.plot(x, y)
            plt.title("测试图片")
            
            # 保存到当前目录
            test_path = os.path.join(os.getcwd(), "test_matplotlib.png")
            logger.info(f"尝试保存图片到当前目录: {test_path}")
            plt.savefig(test_path)
            
            if os.path.exists(test_path):
                logger.info(f"图片保存成功！文件大小: {os.path.getsize(test_path)} 字节")
            else:
                logger.error("图片保存失败，文件不存在")
            
            # 保存到报告目录
            if args.output_dir:
                report_dir = os.path.join(parent_dir, 'output', 'reports', args.output_dir)
                if not os.path.exists(report_dir):
                    os.makedirs(report_dir, exist_ok=True)
                
                report_path = os.path.join(report_dir, "test_report.png")
                logger.info(f"尝试保存图片到报告目录: {report_path}")
                plt.savefig(report_path)
                
                if os.path.exists(report_path):
                    logger.info(f"保存到报告目录成功！文件大小: {os.path.getsize(report_path)} 字节")
                else:
                    logger.error("保存到报告目录失败，文件不存在")
            
            plt.close()
            logger.info("matplotlib图片保存测试完成")
            return True
        except Exception as e:
            logger.error(f"测试matplotlib图片保存功能时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    # 设置输出目录
    if args.output_dir:
        output_subdir = args.output_dir
    else:
        output_subdir = f"hybrid_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"输出目录设置为: {output_subdir}")
    
    # 清理旧输出
    if args.clean_days > 0:
        logger.info(f"清理{args.clean_days}天前的输出文件")
        clean_old_outputs(days=args.clean_days)
    
    # 初始化字体助手
    font_helper = get_font_helper()
    
    # 连接数据源
    logger.info("连接Binance数据源")
    data_source = BinanceDataSource()
    
    # 获取历史数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    logger.info(f"获取{args.symbol}数据: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    df = data_source.get_historical_data(
        symbol=args.symbol,
        interval=args.interval,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d')
    )
    
    if df.empty:
        logger.error("未获取到数据，请检查数据源配置")
        return False
    
    logger.info(f"获取到 {len(df)} 条数据记录")
    
    # 保存原始数据
    raw_data_path = get_data_path(f"{args.symbol.replace('/', '_')}_raw_data.csv", subdirectory=output_subdir)
    df.to_csv(raw_data_path)
    logger.info(f"原始数据已保存至: {raw_data_path}")
    
    # 解析混合方法列表
    ensemble_methods = [method.strip() for method in args.ensemble_methods.split(',')]
    logger.info(f"将测试以下混合方法: {ensemble_methods}")
    
    # 初始化风险管理器(如果启用)
    risk_manager = None
    if args.risk_management:
        logger.info("初始化高级风险管理器...")
        risk_manager = RiskManager(
            max_drawdown=args.max_drawdown,
            max_position_size=args.max_position_size,
            base_position_size=args.base_position_size,
            fixed_stop_loss=args.fixed_stop_loss,
            trailing_stop=args.trailing_stop,
            take_profit=args.take_profit,
            max_trades_per_day=args.max_trades_day if args.max_trades_day > 0 else None,
            time_stop_bars=args.time_stop_bars if args.time_stop_bars > 0 else None,
            volatility_lookback=args.volatility_lookback,
            min_lookback=args.min_lookback,
            volatility_scale_factor=3.0 if args.volatility_scale else 0.0,
            use_atr_for_stops=args.use_atr_stops,
            initial_capital=args.initial_capital
        )
        logger.info(f"风险管理器初始化完成，最大回撤限制: {args.max_drawdown:.2%}, 最大头寸比例: {args.max_position_size:.2%}, "
                   f"使用渐进式风险管理，最小回看周期: {args.min_lookback}")
    
    # 创建并测试不同的混合策略组合方法
    backtest_results = {}
    performance_metrics = {}
    
    for method in ensemble_methods:
        logger.info(f"======== 测试混合策略: {method} 组合方法 ========")
        
        try:
            # 创建混合策略
            strategy = MACDLSTMHybridStrategy(
                # MACD参数
                macd_fast_period=args.macd_fast,
                macd_slow_period=args.macd_slow,
                macd_signal_period=args.macd_signal,
                
                # LSTM参数
                lstm_sequence_length=args.lstm_sequence,
                lstm_hidden_dim=args.lstm_hidden,
                lstm_prediction_threshold=args.lstm_threshold,
                lstm_feature_engineering=True,
                lstm_use_attention=args.lstm_attention,
                
                # 混合策略参数
                ensemble_method=method,
                ensemble_weights=(args.macd_weight, 1.0 - args.macd_weight),
                market_regime_threshold=0.15,
                # 如果使用高级风险管理，这里设置为None
                stop_loss_pct=None if args.risk_management else args.fixed_stop_loss,
                take_profit_pct=None if args.risk_management else args.take_profit,
            )
            
            # 如果使用风险管理，设置到策略对象中
            if args.risk_management and risk_manager is not None:
                setattr(strategy, 'risk_manager', risk_manager)
                logger.info("已将风险管理器设置到策略对象中")
            
            # 兼容性修复：确保_layered_ensemble方法能够获取正确的MACD直方图列
            # 修改generate_signals方法逻辑，将'histogram'映射到'macd_hist'
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
            logger.info("已添加兼容层确保MACD柱状图列名一致性")
            
            # 添加额外的安全层，为layered方法的缺失列提供默认值
            if method == 'layered':
                original_layered_ensemble = strategy._layered_ensemble
                
                def layered_ensemble_wrapper(macd_signal, lstm_signal, row_index, df):
                    # 首先检查macd_hist列是否存在
                    if 'macd_hist' not in df.columns:
                        # 如果histogram列存在，使用它
                        if 'histogram' in df.columns:
                            logger.info("在layered方法中使用histogram列代替macd_hist")
                            df['macd_hist'] = df['histogram']
                        else:
                            # 如果都不存在，使用0值创建macd_hist列
                            logger.warning("无法找到MACD柱状图数据，创建全0的macd_hist列以避免错误")
                            df['macd_hist'] = 0
                    
                    return original_layered_ensemble(macd_signal, lstm_signal, row_index, df)
                
                strategy._layered_ensemble = layered_ensemble_wrapper
                logger.info("为layered方法添加安全层以处理缺失的MACD柱状图数据")
            
            # 创建回测引擎
            backtest_engine = BacktestEngine(
                data=df,
                strategy=strategy,
                initial_capital=args.initial_capital,
                commission=args.commission
            )
            
            # 运行回测
            backtest_engine.run()
            
            # 存储结果的副本
            backtest_results[method] = backtest_engine.results.copy()
            
            # 获取性能指标
            metrics = backtest_engine.summary()
            performance_metrics[method] = metrics
            
            # 显示性能指标
            logger.info(f"策略: {strategy.name}")
            logger.info(f"最终资本: ${metrics.get('final_capital', args.initial_capital):.2f}")
            logger.info(f"总收益率: {metrics.get('total_return', 0.0):.2%}")
            logger.info(f"年化收益率: {metrics.get('annual_return', 0.0):.2%}")
            logger.info(f"最大回撤: {metrics.get('max_drawdown', 0.0):.2%}")
            logger.info(f"夏普比率: {metrics.get('sharpe_ratio', 0.0):.2f}")
            logger.info(f"卡尔马比率: {metrics.get('calmar_ratio', 0.0):.2f}")
            logger.info(f"交易次数: {metrics.get('trade_count', 0)}")
            logger.info(f"胜率: {metrics.get('win_rate', 0.0):.2%}")
            logger.info("")
            
            # 保存该方法的详细回测结果
            try:
                method_results_path = get_data_path(f"{method}_backtest_results.csv", subdirectory=output_subdir)
                backtest_results[method].to_csv(method_results_path)
                logger.info(f"{method}方法的回测结果已保存至: {method_results_path}")
            except Exception as e:
                logger.error(f"保存{method}方法的回测结果时出错: {str(e)}")
                logger.error(traceback.format_exc())
            
            # 绘制并保存该方法的资本曲线
            try:
                import matplotlib.pyplot as plt
                
                # 首先检查必要的列是否存在
                if 'equity_curve' not in backtest_results[method].columns:
                    image_logger.error(f"{method}方法的回测结果中缺少equity_curve列，跳过图表生成")
                    logger.warning(f"{method}方法的回测结果中缺少equity_curve列，跳过图表生成")
                    continue
                
                image_logger.info(f"开始为 {method} 方法创建资本曲线图...")
                
                plt.figure(figsize=(12, 8))
                
                # 绘制资本曲线
                equity_curve = backtest_results[method]['equity_curve']
                # 标准化曲线 - 计算收益率而非绝对值
                normalized_curve = (equity_curve / equity_curve.iloc[0] - 1) * 100
                plt.plot(equity_curve.index, normalized_curve, label=f"{method} 收益率")
                
                # 添加基准（买入持有）- 也标准化为收益率
                benchmark = df['close'] / df['close'].iloc[0] * args.initial_capital
                normalized_benchmark = (benchmark / benchmark.iloc[0] - 1) * 100
                plt.plot(benchmark.index, normalized_benchmark, label='Buy & Hold 收益率', linestyle='--', color='gray')
                
                # 设置图表属性
                plt.title(f'{method}策略收益率曲线' + (' (含高级风险管理)' if args.risk_management else ''))
                plt.xlabel('日期')
                plt.ylabel('收益率(%)')
                
                # 合并图例
                lines1, labels1 = plt.gca().get_legend_handles_labels()
                
                # 记录图表信息到调试日志
                debug_figure(plt.gcf(), f"{method}策略收益率曲线")
                
                # 应用中文字体
                font_helper.apply_font_to_figure()
                
                # 添加调试日志
                image_logger.info(f"准备保存 {method} 方法的收益率曲线图...")
                
                # 保存图表 - 使用新的图像调试工具
                try:
                    # 直接使用绝对路径构建目标文件路径
                    report_dir = os.path.abspath(os.path.join(parent_dir, 'output', 'reports', output_subdir))
                    image_logger.info(f"报告目录绝对路径: {report_dir}")
                    
                    # 确保目录存在
                    if not os.path.exists(report_dir):
                        os.makedirs(report_dir, exist_ok=True)
                        image_logger.info(f"创建了报告目录: {report_dir}")
                    
                    # 报告目录权限检查
                    if not os.access(report_dir, os.W_OK):
                        image_logger.error(f"报告目录不可写: {report_dir}")
                        try:
                            # 尝试修改权限
                            os.chmod(report_dir, 0o755)
                            image_logger.info(f"已修改报告目录权限: {report_dir}")
                        except Exception as perm_e:
                            image_logger.error(f"修改权限失败: {str(perm_e)}")
                    
                    # 生成目标文件路径
                    image_filename = f"{method}_equity_curve.png"
                    method_chart_path = os.path.join(report_dir, image_filename)
                    
                    # 使用高级跟踪函数保存图像
                    success = trace_image_save(plt.gcf(), method_chart_path, dpi=100)
                    
                    if success:
                        image_logger.info(f"{method}方法的收益率曲线图已成功保存至: {method_chart_path}")
                        # 还尝试保存到images目录以供后续查看
                        try:
                            images_dir = os.path.abspath(os.path.join(parent_dir, 'output', 'images', datetime.now().strftime('%Y-%m-%d')))
                            if not os.path.exists(images_dir):
                                os.makedirs(images_dir, exist_ok=True)
                            
                            image_backup_path = os.path.join(images_dir, image_filename)
                            trace_image_save(plt.gcf(), image_backup_path, dpi=100)
                            image_logger.info(f"已创建备份图像: {image_backup_path}")
                        except Exception as backup_e:
                            image_logger.warning(f"创建备份图像失败: {str(backup_e)}")
                    else:
                        # 尝试保存到临时目录
                        image_logger.warning("无法保存到报告目录，尝试保存到临时目录...")
                        temp_dir = '/tmp'
                        temp_path = os.path.join(temp_dir, image_filename)
                        
                        if trace_image_save(plt.gcf(), temp_path):
                            image_logger.info(f"成功保存到临时目录: {temp_path}")
                            # 尝试复制到目标位置
                            import shutil
                            try:
                                shutil.copy(temp_path, method_chart_path)
                                image_logger.info(f"已从临时目录复制到目标位置: {method_chart_path}")
                            except Exception as cp_err:
                                image_logger.error(f"从临时目录复制失败: {str(cp_err)}")
                        else:
                            image_logger.error("临时目录保存也失败")
                except Exception as e:
                    image_logger.error(f"保存图表时出错: {str(e)}")
                    image_logger.error(traceback.format_exc())
                    logger.error(f"保存{method}方法的收益率曲线图时出错: {str(e)}")
                
                plt.close()
                logger.info(f"{method}方法的收益率曲线图已保存或尝试保存")
            except Exception as e:
                image_logger.error(f"生成{method}方法的收益率曲线图时出错: {str(e)}")
                image_logger.error(traceback.format_exc())
                logger.error(f"生成{method}方法的收益率曲线图时出错: {str(e)}")
                logger.error(traceback.format_exc())
        
        except Exception as e:
            logger.error(f"{method}策略运行时出错: {str(e)}")
            logger.error(traceback.format_exc())
            # 跳过错误的策略，继续执行下一个
            logger.info(f"跳过{method}策略，继续执行下一个")
            # 创建包含基本列的空DataFrame，以防后续代码引用
            empty_df = pd.DataFrame(index=df.index)
            empty_df['close'] = df['close']
            empty_df['equity_curve'] = df['close'] / df['close'].iloc[0] * args.initial_capital
            empty_df['position'] = 0
            empty_df['signal'] = 0
            empty_df['drawdown'] = 0
            backtest_results[method] = empty_df
            
            performance_metrics[method] = {
                'final_capital': args.initial_capital,
                'total_return': 0.0,
                'annual_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'calmar_ratio': 0.0,
                'trade_count': 0,
                'win_rate': 0.0
            }
    
    # 过滤出成功执行的方法
    successful_methods = []
    for method in ensemble_methods:
        if method in backtest_results and not backtest_results[method].empty:
            if 'equity_curve' in backtest_results[method].columns:
                successful_methods.append(method)
                logger.info(f"方法 {method} 成功执行")
            else:
                logger.warning(f"方法 {method} 缺少equity_curve列，不算作成功执行")
        else:
            logger.warning(f"方法 {method} 回测结果为空")
    
    if not successful_methods:
        logger.error("没有成功执行的策略，无法生成报告")
        return False
        
    logger.info(f"成功执行的策略: {successful_methods}")
    
    # 比较不同组合方法的性能
    try:
        logger.info("======== 混合策略组合方法性能比较 ========")
        
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
        comparison_path = get_report_path("hybrid_strategy_comparison.csv", subdirectory=output_subdir)
        comparison.to_csv(comparison_path, index=False)
        logger.info(f"混合策略比较结果已保存至: {comparison_path}")
        
        # 可视化资本曲线对比
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            image_logger.info("开始创建混合策略组合方法收益率比较图...")
            
            valid_methods = []
            for method in successful_methods:
                if 'equity_curve' not in backtest_results[method].columns:
                    image_logger.warning(f"{method}方法的回测结果中缺少equity_curve列，将在比较图中跳过该方法")
                    logger.warning(f"{method}方法的回测结果中缺少equity_curve列，将在比较图中跳过该方法")
                    continue
                
                # 标准化为收益率百分比
                equity_curve = backtest_results[method]['equity_curve']
                normalized_curve = (equity_curve / equity_curve.iloc[0] - 1) * 100
                plt.plot(equity_curve.index, normalized_curve, label=f"{method}")
                image_logger.debug(f"添加{method}方法到对比图: 收益率范围[{normalized_curve.min():.2f}%-{normalized_curve.max():.2f}%]")
                valid_methods.append(method)
            
            # 只有在至少有一个有效方法时才继续绘图
            if valid_methods:
                image_logger.info(f"有效的策略方法: {', '.join(valid_methods)}")
                
                # 添加基准（买入持有）- 也标准化为收益率
                benchmark = df['close'] / df['close'].iloc[0] * args.initial_capital
                normalized_benchmark = (benchmark / benchmark.iloc[0] - 1) * 100
                plt.plot(benchmark.index, normalized_benchmark, label='Buy & Hold', linestyle='--', color='gray')
                image_logger.debug(f"添加基准数据: 收益率范围[{normalized_benchmark.min():.2f}%-{normalized_benchmark.max():.2f}%]")
                
                plt.title('混合策略组合方法收益率比较' + (' (含高级风险管理)' if args.risk_management else ''))
                plt.xlabel('日期')
                plt.ylabel('收益率(%)')
                plt.legend()
                plt.grid(True)
                
                # 记录图表信息到调试日志
                debug_figure(plt.gcf(), "混合策略组合方法收益率比较")
                
                # 应用中文字体
                font_helper.apply_font_to_figure()
                
                # 添加调试日志
                image_logger.info(f"准备保存混合策略收益率对比图...")
                
                # 保存图表 - 使用图像调试工具
                try:
                    # 直接使用绝对路径构建目标文件路径
                    report_dir = os.path.abspath(os.path.join(parent_dir, 'output', 'reports', output_subdir))
                    image_logger.info(f"报告目录绝对路径: {report_dir}")
                    
                    # 确保目录存在
                    if not os.path.exists(report_dir):
                        os.makedirs(report_dir, exist_ok=True)
                        image_logger.info(f"创建了报告目录: {report_dir}")
                    
                    # 报告目录权限检查
                    if not os.access(report_dir, os.W_OK):
                        image_logger.error(f"报告目录不可写: {report_dir}")
                        try:
                            # 尝试修改权限
                            os.chmod(report_dir, 0o755)
                            image_logger.info(f"已修改报告目录权限: {report_dir}")
                        except Exception as perm_e:
                            image_logger.error(f"修改权限失败: {str(perm_e)}")
                            
                    # 生成目标文件路径
                    image_filename = "hybrid_strategy_comparison.png"
                    equity_curve_path = os.path.join(report_dir, image_filename)
                    
                    # 使用高级跟踪函数保存图像
                    success = trace_image_save(plt.gcf(), equity_curve_path, dpi=100)
                    
                    if success:
                        image_logger.info(f"混合策略收益率对比图已成功保存至: {equity_curve_path}")
                        # 还尝试保存到images目录以供后续查看
                        try:
                            images_dir = os.path.abspath(os.path.join(parent_dir, 'output', 'images', datetime.now().strftime('%Y-%m-%d')))
                            if not os.path.exists(images_dir):
                                os.makedirs(images_dir, exist_ok=True)
                            
                            image_backup_path = os.path.join(images_dir, image_filename)
                            trace_image_save(plt.gcf(), image_backup_path, dpi=100)
                            image_logger.info(f"已创建备份对比图像: {image_backup_path}")
                        except Exception as backup_e:
                            image_logger.warning(f"创建备份对比图像失败: {str(backup_e)}")
                    else:
                        # 尝试保存到临时目录
                        image_logger.warning("无法保存到报告目录，尝试保存到临时目录...")
                        temp_dir = '/tmp'
                        temp_path = os.path.join(temp_dir, image_filename)
                        
                        if trace_image_save(plt.gcf(), temp_path):
                            image_logger.info(f"成功保存到临时目录: {temp_path}")
                            # 尝试复制到目标位置
                            import shutil
                            try:
                                shutil.copy(temp_path, equity_curve_path)
                                image_logger.info(f"已从临时目录复制到目标位置: {equity_curve_path}")
                            except Exception as cp_err:
                                image_logger.error(f"从临时目录复制失败: {str(cp_err)}")
                        else:
                            image_logger.error("临时目录保存也失败")
                except Exception as e:
                    image_logger.error(f"保存对比图时出错: {str(e)}")
                    image_logger.error(traceback.format_exc())
                    logger.error(f"保存混合策略收益率对比图时出错: {str(e)}")
                
                plt.close()
                logger.info(f"混合策略收益率对比图已保存或尝试保存")
            else:
                image_logger.warning("没有有效的策略收益率数据，无法生成比较图")
                logger.warning("没有有效的策略收益率数据，无法生成比较图")
                plt.close()  # 关闭空图
        except Exception as e:
            image_logger.error(f"生成收益率对比图时出错: {str(e)}")
            image_logger.error(traceback.format_exc())
            logger.error(f"生成收益率对比图时出错: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 分析最佳组合方法
        try:
            # 尝试根据卡尔马比率找出最佳方法
            if len(comparison) > 0 and '卡尔马比率' in comparison.columns:
                # 检查是否有有效的卡尔马比率
                valid_rows = ~comparison['卡尔马比率'].isna()
                if valid_rows.any():
                    best_index = comparison.loc[valid_rows, '卡尔马比率'].idxmax()
                    best_method = comparison.iloc[best_index]
                    logger.info(f"最佳组合方法: {best_method['组合方法']}")
                    logger.info(f"最佳组合方法总收益率: {best_method['总收益率']:.2%}")
                    logger.info(f"最佳组合方法年化收益率: {best_method['年化收益率']:.2%}")
                    logger.info(f"最佳组合方法最大回撤: {best_method['最大回撤']:.2%}")
                    logger.info(f"最佳组合方法卡尔马比率: {best_method['卡尔马比率']:.2f}")
                else:
                    logger.warning("所有方法的卡尔马比率都是无效值，无法确定最佳方法")
                    # 创建一个默认值的最佳方法
                    best_method = comparison.iloc[0] if len(comparison) > 0 else pd.Series({
                        '组合方法': successful_methods[0],
                        '最终资本': args.initial_capital,
                        '总收益率': 0.0,
                        '年化收益率': 0.0,
                        '最大回撤': 0.0,
                        '夏普比率': 0.0,
                        '卡尔马比率': 0.0,
                        '交易次数': 0,
                        '胜率': 0.0
                    })
            else:
                logger.warning("比较结果为空或缺少卡尔马比率列，无法确定最佳方法")
                # 创建一个默认值的最佳方法
                best_method = comparison.iloc[0] if len(comparison) > 0 else pd.Series({
                    '组合方法': successful_methods[0],
                    '最终资本': args.initial_capital,
                    '总收益率': 0.0,
                    '年化收益率': 0.0,
                    '最大回撤': 0.0,
                    '夏普比率': 0.0,
                    '卡尔马比率': 0.0,
                    '交易次数': 0,
                    '胜率': 0.0
                })
        except Exception as e:
            logger.error(f"确定最佳方法时出错: {str(e)}")
            logger.error(traceback.format_exc())
            # 创建一个默认值的最佳方法
            best_method = pd.Series({
                '组合方法': successful_methods[0],
                '最终资本': args.initial_capital,
                '总收益率': 0.0,
                '年化收益率': 0.0,
                '最大回撤': 0.0,
                '夏普比率': 0.0,
                '卡尔马比率': 0.0,
                '交易次数': 0,
                '胜率': 0.0
            })
        
        # 创建策略参数摘要
        strategy_params = {
            'symbol': args.symbol,
            'interval': args.interval,
            'days': args.days,
            'initial_capital': args.initial_capital,
            'commission': args.commission,
            'macd_fast': args.macd_fast,
            'macd_slow': args.macd_slow,
            'macd_signal': args.macd_signal,
            'lstm_sequence': args.lstm_sequence,
            'lstm_hidden': args.lstm_hidden,
            'lstm_threshold': args.lstm_threshold,
            'lstm_attention': args.lstm_attention,
            'macd_weight': args.macd_weight,
            'use_risk_management': args.risk_management
        }
        
        # 如果使用了风险管理，添加风险管理参数
        if args.risk_management:
            risk_params = {
                'max_drawdown': args.max_drawdown,
                'max_position_size': args.max_position_size,
                'base_position_size': args.base_position_size,
                'fixed_stop_loss': args.fixed_stop_loss,
                'trailing_stop': args.trailing_stop,
                'take_profit': args.take_profit,
                'volatility_scale': args.volatility_scale,
                'use_atr_stops': args.use_atr_stops,
                'max_trades_day': args.max_trades_day,
                'time_stop_bars': args.time_stop_bars,
                'volatility_lookback': args.volatility_lookback,
                'min_lookback': args.min_lookback
            }
            strategy_params.update(risk_params)
        
        # 保存参数
        params_path = get_data_path("strategy_parameters.json", subdirectory=output_subdir)
        import json
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(strategy_params, f, indent=4, ensure_ascii=False)
        logger.info(f"策略参数已保存至: {params_path}")
        
        # 计算基准收益率(买入持有)
        benchmark_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1.0
        logger.info(f"买入持有基准收益率: {benchmark_return:.2%}")
        
        # 创建回测报告
        try:
            logger.info("======== 创建混合策略回测报告 ========")
            
            # 安全格式化函数
            def safe_format(value, is_percentage=False, decimal_places=2):
                if isinstance(value, (int, float)):
                    if is_percentage:
                        return f"{value:.{decimal_places}%}"
                    else:
                        return f"{value:.{decimal_places}f}"
                else:
                    return str(value)
                
            # 创建比较表的Markdown格式
            try:
                comparison_markdown = comparison.to_markdown(index=False)
            except Exception as e:
                logger.error(f"生成Markdown表格时出错: {str(e)}")
                
                # 手动创建Markdown表格
                comparison_markdown = "| 组合方法 | 最终资本 | 总收益率 | 年化收益率 | 最大回撤 | 夏普比率 | 卡尔马比率 | 交易次数 | 胜率 |\n"
                comparison_markdown += "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
                
                # 添加每个方法的行
                for method in successful_methods:
                    try:
                        metrics = performance_metrics[method]
                        row = f"| {method} | ${safe_format(metrics['final_capital'])} | "
                        row += f"{safe_format(metrics['total_return'], True)} | "
                        row += f"{safe_format(metrics['annual_return'], True)} | "
                        row += f"{safe_format(metrics['max_drawdown'], True)} | "
                        row += f"{safe_format(metrics['sharpe_ratio'])} | "
                        row += f"{safe_format(metrics['calmar_ratio'])} | "
                        row += f"{metrics['trade_count']} | "
                        row += f"{safe_format(metrics['win_rate'], True)} |\n"
                        comparison_markdown += row
                    except Exception as e:
                        logger.error(f"格式化{method}方法的表格行时出错: {str(e)}")
            
            # 创建报告内容
            best_method_name = str(best_method['组合方法'])
            
            report_content = f"""# MACD-LSTM混合策略回测报告

## 概要

- **回测周期**: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}
- **初始资本**: ${args.initial_capital}
- **手续费率**: {args.commission * 100}%
- **交易对**: {args.symbol}

## 策略参数

### MACD参数
- 快速均线周期: {args.macd_fast}
- 慢速均线周期: {args.macd_slow}
- 信号线周期: {args.macd_signal}
- MACD权重: {args.macd_weight}

### LSTM参数
- 序列长度: {args.lstm_sequence}
- 隐藏层维度: {args.lstm_hidden}
- 预测阈值: {args.lstm_threshold}
- 使用注意力机制: {'是' if args.lstm_attention else '否'}

### 混合策略参数
- 集成方法: {', '.join(ensemble_methods)}
- 市场机制阈值: 0.15
- 使用高级风险管理: {'是' if args.risk_management else '否'}

## 性能比较

以下是不同集成方法的性能比较:

{comparison_markdown}

## 最佳组合方法: {best_method_name}

- **最终资本**: ${safe_format(best_method['最终资本'])}
- **总收益率**: {safe_format(best_method['总收益率'], True)}
- **年化收益率**: {safe_format(best_method['年化收益率'], True)}
- **最大回撤**: {safe_format(best_method['最大回撤'], True)}
- **夏普比率**: {safe_format(best_method['夏普比率'])}
- **卡尔马比率**: {safe_format(best_method['卡尔马比率'])}
- **交易次数**: {best_method['交易次数']}
- **胜率**: {safe_format(best_method['胜率'], True)}

## 分析结论

### 策略表现
- 最佳组合方法({best_method_name})在回测期间{'' if best_method['总收益率'] > 0 else '没有'}取得了正收益
- {'策略相比买入持有表现更好' if best_method['总收益率'] > benchmark_return else '策略没有跑赢买入持有'}
- {'策略具有较好的风险调整收益' if best_method['夏普比率'] > 1 else '策略的风险调整收益不理想'}
- {'策略符合交易规范要求' if best_method['卡尔马比率'] >= 2.5 and best_method['最大回撤'] <= 0.15 else '策略未达到交易规范的卡尔马比率(≥2.5)和最大回撤(≤15%)要求'}

### 改进建议
- {'可以考虑调整MACD和LSTM的权重以提高策略表现' if best_method['夏普比率'] < 1 else '当前的MACD和LSTM权重组合表现良好'}
- {'可以尝试调整止损和止盈参数以提高风险管理效果' if best_method['最大回撤'] > 0.15 else '当前的风险管理参数设置合理'}
- {'需要优化策略以提高卡尔马比率至规范要求(≥2.5)' if best_method['卡尔马比率'] < 2.5 else '当前策略的风险回报特性符合交易规范'}
- {'可以考虑增加更多技术指标或基本面数据以提高预测准确性' if best_method['胜率'] < 0.5 else '当前的预测模型表现良好'}

## 附图

- [收益率对比图](hybrid_strategy_comparison.png)
- [最佳方法收益率曲线]({best_method_name}_equity_curve.png)

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

            # 验证图片文件是否存在
            report_dir = os.path.abspath(os.path.join(parent_dir, 'output', 'reports', output_subdir))
            image_logger.info(f"验证报告引用的图片文件是否存在，报告目录: {report_dir}")
            
            hybrid_comparison_path = os.path.join(report_dir, "hybrid_strategy_comparison.png")
            best_method_path = os.path.join(report_dir, f"{best_method_name}_equity_curve.png")
            
            image_logger.info(f"对比图路径: {hybrid_comparison_path}, 是否存在: {os.path.exists(hybrid_comparison_path)}")
            image_logger.info(f"最佳方法图路径: {best_method_path}, 是否存在: {os.path.exists(best_method_path)}")
            
            # 检查图片是否存在，如果不存在则添加警告
            if not os.path.exists(hybrid_comparison_path):
                image_logger.warning(f"对比图文件不存在: {hybrid_comparison_path}")
            if not os.path.exists(best_method_path):
                image_logger.warning(f"最佳方法图文件不存在: {best_method_path}")
                
            # 检查备份目录中是否有这些图片
            images_dir = os.path.abspath(os.path.join(parent_dir, 'output', 'images', datetime.now().strftime('%Y-%m-%d')))
            backup_comparison_path = os.path.join(images_dir, "hybrid_strategy_comparison.png")
            backup_best_method_path = os.path.join(images_dir, f"{best_method_name}_equity_curve.png")
            
            if os.path.exists(images_dir):
                image_logger.info(f"检查备份图像目录: {images_dir}")
                image_logger.info(f"备份对比图路径: {backup_comparison_path}, 是否存在: {os.path.exists(backup_comparison_path)}")
                image_logger.info(f"备份最佳方法图路径: {backup_best_method_path}, 是否存在: {os.path.exists(backup_best_method_path)}")
                
                # 如果报告目录中没有图片但备份目录有，尝试复制
                if not os.path.exists(hybrid_comparison_path) and os.path.exists(backup_comparison_path):
                    try:
                        import shutil
                        shutil.copy(backup_comparison_path, hybrid_comparison_path)
                        image_logger.info(f"已从备份目录复制对比图到报告目录")
                    except Exception as e:
                        image_logger.error(f"复制对比图失败: {str(e)}")
                
                if not os.path.exists(best_method_path) and os.path.exists(backup_best_method_path):
                    try:
                        import shutil
                        shutil.copy(backup_best_method_path, best_method_path)
                        image_logger.info(f"已从备份目录复制最佳方法图到报告目录")
                    except Exception as e:
                        image_logger.error(f"复制最佳方法图失败: {str(e)}")

            # 保存报告
            report_path = get_report_path("hybrid_strategy_report.md", subdirectory=output_subdir)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info(f"混合策略回测报告已保存至: {report_path}")
            
            return True
        except Exception as e:
            logger.error(f"生成报告时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
        logger.info(f"\n混合策略回测完成! 所有输出已保存至 {output_subdir} 目录")
        return True
    except Exception as e:
        logger.error(f"生成报告时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    args = parse_args()
    
    # 初始化图像调试日志
    image_debug_logger = setup_image_debug_logger()
    image_debug_logger.info(f"图像调试日志已初始化，开始运行混合策略回测: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    image_debug_logger.debug(f"命令行参数: {args}")
    
    success = run_hybrid_strategy(args)
    sys.exit(0 if success else 1) 