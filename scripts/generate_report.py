#!/usr/bin/env python
"""
从已有回测结果生成报告的工具脚本
"""
import os
import sys
import argparse
import pandas as pd
import pickle
import json
import traceback
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_quant.utils.logger import logger, set_log_level
from crypto_quant.backtesting.visualization.performance_visualizer import PerformanceVisualizer
from crypto_quant.utils.output_helper import get_report_path, get_data_path

# 设置更详细的日志级别
set_log_level('DEBUG')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="从已有回测结果生成报告")
    
    parser.add_argument("--data-dir", type=str, required=True,
                        help="包含回测结果的目录路径")
    parser.add_argument("--method", type=str, default=None,
                        help="要生成报告的方法(例如vote, weight, layered, expert)，如果不指定则生成所有方法的报告")
    parser.add_argument("--output-name", type=str, default=None,
                        help="输出报告的文件名，默认使用方法名")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录，默认使用与数据目录相同的子目录")
    
    return parser.parse_args()

def load_backtest_results_csv(filepath):
    """从CSV文件加载回测结果"""
    try:
        if not os.path.exists(filepath):
            logger.error(f"数据文件不存在: {filepath}")
            return None
            
        # 加载CSV文件
        df = pd.read_csv(filepath)
        
        # 检查必要的列
        required_cols = ['cumulative_strategy_returns', 'drawdown', 'equity_curve']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"回测结果缺少必要的列: {missing_cols}")
            return None
        
        # 尝试将第一列转换为日期索引
        try:
            df.index = pd.to_datetime(df.iloc[:, 0])
            # 删除原日期列
            df = df.drop(df.columns[0], axis=1)
        except:
            logger.warning("无法将第一列转换为日期索引，将使用原始索引")
        
        logger.info(f"成功加载回测结果: {len(df)} 行")
        return df
    except Exception as e:
        logger.error(f"加载CSV数据时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def load_strategy_parameters(data_dir):
    """加载策略参数"""
    try:
        params_path = os.path.join(data_dir, "strategy_parameters.json")
        if not os.path.exists(params_path):
            logger.warning(f"策略参数文件不存在: {params_path}")
            return None
            
        with open(params_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
            
        logger.info(f"成功加载策略参数")
        return params
    except Exception as e:
        logger.error(f"加载策略参数时出错: {str(e)}")
        return None

def load_performance_metrics(data_dir, method):
    """从回测结果计算或加载性能指标"""
    try:
        # 尝试从比较文件中加载性能指标
        comparison_path = os.path.join(data_dir, "hybrid_strategy_comparison.csv")
        if os.path.exists(comparison_path):
            comparison = pd.read_csv(comparison_path)
            if '组合方法' in comparison.columns and method in comparison['组合方法'].values:
                # 从比较表中提取性能指标
                metrics_row = comparison[comparison['组合方法'] == method].iloc[0]
                performance = {
                    'final_capital': metrics_row.get('最终资本', 0),
                    'total_return': metrics_row.get('总收益率', 0),
                    'annual_return': metrics_row.get('年化收益率', 0),
                    'max_drawdown': metrics_row.get('最大回撤', 0),
                    'sharpe_ratio': metrics_row.get('夏普比率', 0),
                    'calmar_ratio': metrics_row.get('卡尔马比率', 0),
                    'trade_count': metrics_row.get('交易次数', 0),
                    'win_rate': metrics_row.get('胜率', 0),
                    # 估计其他指标
                    'profit_loss_ratio': 1.5,  # 默认值
                    'winning_trades': int(metrics_row.get('交易次数', 0) * metrics_row.get('胜率', 0.5)),
                    'losing_trades': int(metrics_row.get('交易次数', 0) * (1 - metrics_row.get('胜率', 0.5))),
                    'risk_assessment': 'N/A'
                }
                logger.info(f"从比较文件中加载了{method}方法的性能指标")
                return performance
                
        # 如果没有比较文件，则从回测结果中估计性能指标
        results_path = os.path.join(data_dir, f"{method}_backtest_results.csv")
        if not os.path.exists(results_path):
            logger.error(f"回测结果文件不存在: {results_path}")
            return None
            
        results = load_backtest_results_csv(results_path)
        if results is None:
            return None
            
        # 基于结果估计性能指标
        performance = {
            'final_capital': results['equity_curve'].iloc[-1] if 'equity_curve' in results.columns else 0,
            'total_return': results['cumulative_strategy_returns'].iloc[-1] - 1 if 'cumulative_strategy_returns' in results.columns else 0,
            'annual_return': (results['cumulative_strategy_returns'].iloc[-1] ** (252 / len(results)) - 1) if 'cumulative_strategy_returns' in results.columns else 0,
            'max_drawdown': results['drawdown'].min() if 'drawdown' in results.columns else 0,
            'sharpe_ratio': 1.0,  # 默认值
            'calmar_ratio': 1.0,  # 默认值
            'win_rate': 0.5,  # 默认值
            'profit_loss_ratio': 1.5,  # 默认值
            'winning_trades': 10,  # 默认值
            'losing_trades': 10,  # 默认值
            'risk_assessment': 'N/A'
        }
        
        logger.info(f"已从回测结果估计{method}方法的性能指标")
        return performance
        
    except Exception as e:
        logger.error(f"加载或计算性能指标时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_report_for_method(data_dir, method, output_name=None, output_dir=None):
    """为特定方法生成报告"""
    try:
        # 加载回测结果
        results_path = os.path.join(data_dir, f"{method}_backtest_results.csv")
        if not os.path.exists(results_path):
            logger.error(f"回测结果文件不存在: {results_path}")
            return False
            
        results = load_backtest_results_csv(results_path)
        if results is None:
            return False
            
        # 加载性能指标
        performance = load_performance_metrics(data_dir, method)
        if performance is None:
            return False
            
        # 设置输出目录
        if output_dir is None:
            # 使用数据目录的最后一个部分作为输出子目录
            output_dir = os.path.basename(os.path.normpath(data_dir))
        
        # 设置输出文件名
        if output_name is None:
            output_name = f"{method}_strategy_report.html"
            
        # 创建可视化器
        visualizer = PerformanceVisualizer(results, performance)
        
        # 获取报告路径
        report_path = get_report_path(output_name, subdirectory=output_dir)
        
        # 尝试创建交互式仪表板
        logger.info(f"创建{method}方法的交互式仪表板...")
        dashboard = visualizer.create_interactive_dashboard()
        
        if dashboard is None:
            logger.error(f"{method}方法的交互式仪表板创建失败")
            return False
        
        # 保存报告
        logger.info(f"保存{method}方法的报告到: {report_path}")
        success = visualizer.save_report(report_path)
        
        if success:
            logger.info(f"{method}方法的报告已成功保存至: {report_path}")
        else:
            logger.error(f"{method}方法的报告保存失败")
        
        return success
    except Exception as e:
        logger.error(f"为{method}方法生成报告时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        logger.error(f"数据目录不存在: {args.data_dir}")
        return False
        
    # 加载策略参数（用于显示在报告中）
    params = load_strategy_parameters(args.data_dir)
    
    # 如果指定了方法，只生成该方法的报告
    if args.method:
        logger.info(f"生成{args.method}方法的报告...")
        return generate_report_for_method(
            args.data_dir, 
            args.method,
            args.output_name,
            args.output_dir
        )
    else:
        # 尝试发现可用的方法
        methods = []
        for filename in os.listdir(args.data_dir):
            if filename.endswith("_backtest_results.csv"):
                method = filename.replace("_backtest_results.csv", "")
                methods.append(method)
        
        if not methods:
            logger.error(f"在{args.data_dir}中没有发现回测结果文件")
            return False
            
        # 为每个方法生成报告
        success = True
        for method in methods:
            logger.info(f"生成{method}方法的报告...")
            result = generate_report_for_method(
                args.data_dir,
                method,
                args.output_name,
                args.output_dir
            )
            success = success and result
            
        return success

if __name__ == "__main__":
    logger.info("开始从已有回测结果生成报告...")
    success = main()
    if success:
        logger.info("报告生成成功!")
    else:
        logger.error("报告生成失败!") 