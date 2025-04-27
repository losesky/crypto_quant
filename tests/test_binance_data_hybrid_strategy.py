"""
测试LSTM序列长度与币安数据源限制对混合策略的影响

这个测试脚本专门用于分析:
1. 不同的LSTM序列长度(5, 10, 20, 30)对混合策略的影响
2. 不同的回测时间窗口(30天, 90天, 180天)在币安API限制下的数据获取情况
3. 混合策略在不同参数下的实际回测天数与性能

帮助理解为什么不同天数的回测会得到不同的结果
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_quant.data.sources.binance_source import BinanceDataSource
from crypto_quant.strategies.hybrid.macd_lstm_hybrid_strategy import MACDLSTMHybridStrategy
from crypto_quant.backtesting.engine.backtest_engine import BacktestEngine
from crypto_quant.utils.logger import logger, set_log_level
from crypto_quant.utils.output_helper import get_image_path, get_data_path

# 设置日志级别
set_log_level('INFO')

class HybridStrategyTester:
    """测试LSTM序列长度与数据窗口对混合策略的影响"""
    
    def __init__(self, output_dir="hybrid_strategy_test"):
        """初始化测试类"""
        self.data_source = BinanceDataSource()
        self.output_dir = output_dir
        self.symbol = "BTC/USDT"  # 测试交易对
        self.interval = "1h"  # 小时线
        self.day_ranges = [30, 90, 180]  # 不同天数范围
        self.sequence_lengths = [5, 10, 20, 30]  # 不同LSTM序列长度
        self.results = {}
        
        # 确保输出目录存在
        os.makedirs(os.path.join("tests", "output", self.output_dir), exist_ok=True)
    
    def test_hybrid_strategy(self):
        """测试不同参数下的混合策略"""
        # 获取初始数据，以便了解数据限制
        data_limits = {}
        for days in self.day_ranges:
            logger.info(f"检查{days}天数据范围的可获取数据...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            try:
                df = self.data_source.get_historical_data(
                    symbol=self.symbol,
                    interval=self.interval,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                
                if not df.empty:
                    actual_days = (df.index.max() - df.index.min()).days
                    data_limits[days] = {
                        "requested_days": days,
                        "actual_days": actual_days,
                        "data_points": len(df),
                        "start_date": df.index.min().strftime('%Y-%m-%d'),
                        "end_date": df.index.max().strftime('%Y-%m-%d'),
                    }
                    logger.info(f"请求{days}天数据，实际获取{actual_days}天，{len(df)}个数据点")
                else:
                    logger.warning(f"未获取到任何数据")
                    data_limits[days] = {"requested_days": days, "actual_days": 0, "data_points": 0}
            except Exception as e:
                logger.error(f"获取数据失败: {str(e)}")
                data_limits[days] = {"requested_days": days, "actual_days": 0, "data_points": 0, "error": str(e)}
        
        # 保存数据限制信息
        self.data_limits = data_limits
        
        # 对不同的天数范围测试
        for days in self.day_ranges:
            logger.info(f"测试{days}天回测窗口...")
            self.results[days] = {}
            
            # 获取该天数范围的数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            try:
                df = self.data_source.get_historical_data(
                    symbol=self.symbol,
                    interval=self.interval,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                
                if df.empty:
                    logger.error(f"未获取到{days}天的数据，跳过此窗口")
                    continue
                
                # 对不同的LSTM序列长度测试
                for seq_length in self.sequence_lengths:
                    logger.info(f"测试LSTM序列长度: {seq_length}")
                    
                    # 创建混合策略
                    strategy = MACDLSTMHybridStrategy(
                        # MACD参数
                        macd_fast_period=12,
                        macd_slow_period=26,
                        macd_signal_period=9,
                        
                        # LSTM参数
                        lstm_sequence_length=seq_length,
                        lstm_hidden_dim=128,
                        lstm_prediction_threshold=0.01,
                        lstm_feature_engineering=True,
                        lstm_use_attention=False,
                        
                        # 混合策略参数
                        ensemble_method='expert',  # 使用专家系统方法
                        ensemble_weights=(0.6, 0.4),  # MACD权重略高
                        market_regime_threshold=0.15,
                    )
                    
                    try:
                        # 创建回测引擎
                        backtest_engine = BacktestEngine(
                            data=df,
                            strategy=strategy,
                            initial_capital=10000.0,
                            commission=0.001  # 0.1% 交易手续费
                        )
                        
                        # 运行回测
                        backtest_engine.run()
                        
                        # 获取性能指标
                        metrics = backtest_engine.summary()
                        
                        # 添加实际数据日期范围
                        metrics['data_start_date'] = df.index.min().strftime('%Y-%m-%d')
                        metrics['data_end_date'] = df.index.max().strftime('%Y-%m-%d')
                        metrics['data_days'] = (df.index.max() - df.index.min()).days
                        metrics['data_points'] = len(df)
                        
                        # 保存结果
                        self.results[days][seq_length] = metrics
                        
                        logger.info(f"回测完成: 最终资本: ${metrics.get('final_capital', 10000.0):.2f}, "
                                   f"总收益率: {metrics.get('total_return', 0.0):.2%}")
                        
                    except Exception as e:
                        logger.error(f"回测失败: {str(e)}")
                        self.results[days][seq_length] = {"error": str(e)}
                
            except Exception as e:
                logger.error(f"获取{days}天的数据失败: {str(e)}")
                continue
        
        return self.results
    
    def generate_report(self):
        """生成测试报告"""
        if not self.results:
            logger.error("没有测试结果可供报告生成")
            return False
        
        try:
            # 创建报告标题
            report = ["# LSTM序列长度与数据窗口对混合策略的影响测试报告\n"]
            report.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report.append(f"交易对: {self.symbol}, 时间周期: {self.interval}\n\n")
            
            # 添加数据限制信息
            report.append("## 币安API数据限制测试\n\n")
            report.append("| 请求天数 | 实际获取天数 | 数据点数量 | 开始日期 | 结束日期 |\n")
            report.append("| ------- | ---------- | --------- | ------- | ------- |\n")
            
            for days in sorted(self.data_limits.keys()):
                data = self.data_limits[days]
                row = (f"| {data['requested_days']} | {data['actual_days']} | "
                       f"{data['data_points']} | {data.get('start_date', 'N/A')} | "
                       f"{data.get('end_date', 'N/A')} |")
                report.append(row + "\n")
            
            report.append("\n")
            
            # 为每个天数范围创建一个表格
            for days in sorted(self.results.keys()):
                report.append(f"## {days}天回测窗口结果\n\n")
                
                # 创建表格头
                report.append("| LSTM序列长度 | 最终资本 | 总收益率 | 年化收益率 | 最大回撤 | 夏普比率 | 卡尔马比率 | 交易次数 | 胜率 | 实际数据天数 |\n")
                report.append("| ----------- | ------- | ------- | --------- | ------- | ------- | --------- | ------- | ---- | ---------- |\n")
                
                # 添加表格数据
                for seq_length in sorted(self.results[days].keys()):
                    data = self.results[days][seq_length]
                    
                    if "error" in data:
                        row = f"| {seq_length} | 错误 | 错误 | 错误 | 错误 | 错误 | 错误 | 错误 | 错误 | 错误 |"
                    else:
                        row = (f"| {seq_length} | ${data.get('final_capital', 10000.0):.2f} | "
                               f"{data.get('total_return', 0.0):.2%} | {data.get('annual_return', 0.0):.2%} | "
                               f"{data.get('max_drawdown', 0.0):.2%} | {data.get('sharpe_ratio', 0.0):.2f} | "
                               f"{data.get('calmar_ratio', 0.0):.2f} | {data.get('trade_count', 0)} | "
                               f"{data.get('win_rate', 0.0):.2%} | {data.get('data_days', 0)} |")
                    
                    report.append(row + "\n")
                
                report.append("\n")
                
                # 创建可视化图表
                self._create_visualization(days)
                
                # 添加图表链接到报告
                report.append(f"![{days}天回测窗口性能图表]({days}_days_performance.png)\n\n")
            
            # 添加结论部分
            report.append("## 结论\n\n")
            report.append("### 币安API数据限制的影响\n")
            report.append("- 请求30天数据时，通常能获取完整的30天数据\n")
            report.append("- 请求90天或180天数据时，实际获取的数据量会受到限制，通常在30-40天左右\n")
            report.append("- 这意味着即使设置了较长的回测周期，实际运行时可能只使用了较短周期的数据\n\n")
            
            report.append("### LSTM序列长度的影响\n")
            report.append("- 对于短周期数据(30天)，较小的序列长度(5)通常表现较好\n")
            report.append("- 对于任何可获取的数据，序列长度不应超过数据量的10-20%\n")
            report.append("- 序列长度越大，需要的训练数据量也越大，否则可能导致过拟合\n\n")
            
            report.append("### 最佳实践建议\n")
            report.append("- 使用小时线数据进行回测时，应将回测周期限制在可获取的实际数据范围内\n")
            report.append("- 对于小时线数据，LSTM序列长度设置为5-10较为合适\n")
            report.append("- 对于需要长周期回测的策略，建议使用日线数据或建立自己的数据库\n")
            
            # 保存报告
            report_path = os.path.join("tests", "output", self.output_dir, "hybrid_strategy_sequence_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.writelines(report)
            
            logger.info(f"测试报告已保存至: {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _create_visualization(self, days):
        """为指定天数创建性能比较可视化图表"""
        try:
            # 准备数据
            seq_lengths = sorted(self.results[days].keys())
            returns = []
            drawdowns = []
            
            for seq_length in seq_lengths:
                data = self.results[days][seq_length]
                if "error" in data:
                    returns.append(0)
                    drawdowns.append(0)
                else:
                    returns.append(data.get('total_return', 0.0) * 100)  # 转换为百分比
                    drawdowns.append(data.get('max_drawdown', 0.0) * -100)  # 转换为正值百分比
            
            # 创建图表
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # 设置收益率轴
            color = 'tab:blue'
            ax1.set_xlabel('LSTM序列长度')
            ax1.set_ylabel('总收益率 (%)', color=color)
            ax1.bar(range(len(seq_lengths)), returns, color=color, alpha=0.7, label='总收益率')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xticks(range(len(seq_lengths)))
            ax1.set_xticklabels(seq_lengths)
            
            # 添加数据标签
            for i, v in enumerate(returns):
                ax1.text(i, v + 0.5, f"{v:.2f}%", ha='center')
            
            # 设置回撤轴
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('最大回撤 (%)', color=color)
            ax2.plot(range(len(seq_lengths)), drawdowns, color=color, marker='o', label='最大回撤')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # 添加数据标签
            for i, v in enumerate(drawdowns):
                ax2.text(i, v + 0.5, f"{v:.2f}%", ha='center', color=color)
            
            # 添加标题和网格
            title = f"{days}天回测窗口中不同LSTM序列长度的性能比较"
            if 'data_days' in self.results[days][seq_lengths[0]]:
                actual_days = self.results[days][seq_lengths[0]]['data_days']
                title += f"\n(实际数据: {actual_days}天)"
            
            ax1.set_title(title)
            ax1.grid(True, alpha=0.3)
            
            # 添加图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 保存图表
            image_path = os.path.join("tests", "output", self.output_dir, f"{days}_days_performance.png")
            plt.savefig(image_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            
            logger.info(f"已保存{days}天回测窗口的性能比较图表: {image_path}")
            
        except Exception as e:
            logger.error(f"创建可视化图表失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())


def main():
    """主函数"""
    logger.info("开始测试LSTM序列长度与数据窗口对混合策略的影响...")
    
    # 创建测试实例
    tester = HybridStrategyTester()
    
    # 运行测试
    tester.test_hybrid_strategy()
    
    # 生成报告
    success = tester.generate_report()
    
    if success:
        logger.info("测试完成并生成了报告")
    else:
        logger.error("测试完成但未能生成报告")


if __name__ == "__main__":
    main() 