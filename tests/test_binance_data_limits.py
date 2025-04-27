"""
测试币安数据源对不同时间周期(1h小时线和1d日线)数据的获取限制

这个测试脚本将系统地测试:
1. 不同时间区间(30天、90天、180天、365天)
2. 不同时间周期(小时线和日线)
3. 记录实际获取到的数据量和时间范围

帮助理解为什么某些回测会受到数据获取限制
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
from crypto_quant.utils.logger import logger, set_log_level
from crypto_quant.utils.output_helper import get_image_path, get_data_path

# 设置日志级别
set_log_level('INFO')

class BinanceDataLimitTester:
    """测试币安数据源获取限制的类"""
    
    def __init__(self, output_dir="data_limit_test"):
        """初始化测试类"""
        self.data_source = BinanceDataSource()
        self.output_dir = output_dir
        self.symbols = ["BTC/USDT", "ETH/USDT"]  # 测试多个交易对
        self.intervals = ["1h", "1d"]  # 小时线和日线
        self.day_ranges = [30, 90, 180, 365]  # 不同天数范围
        self.results = {}
        
        # 确保输出目录存在
        os.makedirs(os.path.join("tests", "output", self.output_dir), exist_ok=True)
    
    def test_data_limits(self):
        """测试数据获取限制"""
        for symbol in self.symbols:
            logger.info(f"测试交易对: {symbol}")
            self.results[symbol] = {}
            
            for interval in self.intervals:
                logger.info(f"测试时间周期: {interval}")
                self.results[symbol][interval] = {}
                
                for days in self.day_ranges:
                    logger.info(f"测试时间范围: {days}天")
                    
                    # 设置时间范围
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    # 记录开始时间以计算API请求时间
                    start_time = time.time()
                    
                    try:
                        # 获取数据
                        df = self.data_source.get_historical_data(
                            symbol=symbol,
                            interval=interval,
                            start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d')
                        )
                        
                        # 计算API请求时间
                        request_time = time.time() - start_time
                        
                        # 如果数据非空，记录结果
                        if not df.empty:
                            # 计算实际获取到的数据天数
                            actual_days = (df.index.max() - df.index.min()).days
                            
                            # 保存结果
                            self.results[symbol][interval][days] = {
                                "requested_days": days,
                                "actual_days": actual_days,
                                "data_points": len(df),
                                "start_date": df.index.min().strftime('%Y-%m-%d'),
                                "end_date": df.index.max().strftime('%Y-%m-%d'),
                                "request_time": request_time
                            }
                            
                            logger.info(f"请求{days}天数据，实际获取{actual_days}天，{len(df)}个数据点")
                            logger.info(f"数据范围: {df.index.min().strftime('%Y-%m-%d')} 至 {df.index.max().strftime('%Y-%m-%d')}")
                        else:
                            logger.warning(f"未获取到任何数据")
                            self.results[symbol][interval][days] = {
                                "requested_days": days,
                                "actual_days": 0,
                                "data_points": 0,
                                "start_date": None,
                                "end_date": None,
                                "request_time": request_time
                            }
                            
                    except Exception as e:
                        logger.error(f"获取数据失败: {str(e)}")
                        self.results[symbol][interval][days] = {
                            "requested_days": days,
                            "actual_days": 0,
                            "data_points": 0,
                            "start_date": None,
                            "end_date": None,
                            "error": str(e)
                        }
                    
                    # 等待一段时间避免API请求过快
                    time.sleep(1)
        
        return self.results
    
    def generate_report(self):
        """生成测试报告"""
        if not self.results:
            logger.error("没有测试结果可供报告生成")
            return False
        
        try:
            # 创建报告标题
            report = ["# 币安数据源API限制测试报告\n"]
            report.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 为每个交易对创建一个部分
            for symbol in self.results:
                report.append(f"## 交易对: {symbol}\n")
                
                # 为每个时间周期创建一个表格
                for interval in self.results[symbol]:
                    report.append(f"### 时间周期: {interval}\n")
                    
                    # 创建表格头
                    report.append("| 请求天数 | 实际天数 | 数据点数量 | 开始日期 | 结束日期 | 请求时间(秒) |\n")
                    report.append("| ------- | ------- | --------- | ------- | ------- | ----------- |\n")
                    
                    # 添加表格数据
                    for days in sorted(self.results[symbol][interval].keys()):
                        data = self.results[symbol][interval][days]
                        row = (f"| {data['requested_days']} | {data['actual_days']} | "
                               f"{data['data_points']} | {data['start_date'] or 'N/A'} | "
                               f"{data['end_date'] or 'N/A'} | {data.get('request_time', 'N/A'):.2f} |")
                        report.append(row + "\n")
                    
                    report.append("\n")
                
                # 创建可视化图表
                self._create_visualization(symbol)
                
                # 添加图表链接到报告
                report.append(f"![{symbol} 数据限制图表]({symbol.replace('/', '_')}_data_limits.png)\n\n")
            
            # 添加结论部分
            report.append("## 结论\n\n")
            report.append("### 小时线(1h)数据限制\n")
            report.append("- 币安API对小时线数据的历史范围通常限制在约XX天\n")
            report.append("- 请求更长时间范围的小时线数据时，实际返回的数据会受到限制\n\n")
            
            report.append("### 日线(1d)数据限制\n")
            report.append("- 币安API对日线数据的历史范围通常可以达到约XX天\n")
            report.append("- 日线数据相比小时线数据更容易获取更长的历史记录\n\n")
            
            report.append("### 建议\n")
            report.append("- 对于需要大量历史数据的回测，建议使用日线数据\n")
            report.append("- 如果必须使用小时线数据进行长期回测，建议建立自己的数据库并持续收集数据\n")
            report.append("- 对于超过可获取范围的回测，需要调整期望或使用其他数据源\n")
            
            # 保存报告
            report_path = os.path.join("tests", "output", self.output_dir, "binance_data_limits_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.writelines(report)
            
            logger.info(f"测试报告已保存至: {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _create_visualization(self, symbol):
        """为指定交易对创建可视化图表"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 为每个时间周期绘制一条线
            for interval in self.results[symbol]:
                x = []  # 请求天数
                y = []  # 实际获取天数
                
                for days in sorted(self.results[symbol][interval].keys()):
                    x.append(days)
                    y.append(self.results[symbol][interval][days]["actual_days"])
                
                ax.plot(x, y, marker='o', label=f"{interval}")
                
                # 添加数据标签
                for i, days in enumerate(x):
                    ax.annotate(f"{y[i]}", 
                               (x[i], y[i]),
                               textcoords="offset points",
                               xytext=(0, 10),
                               ha='center')
            
            # 添加对角线表示理想情况(请求多少天就获取多少天)
            max_days = max(self.day_ranges)
            ax.plot([0, max_days], [0, max_days], 'k--', alpha=0.3, label="理想情况")
            
            # 设置图表属性
            ax.set_xlabel("请求天数")
            ax.set_ylabel("实际获取天数")
            ax.set_title(f"{symbol} 不同时间周期的数据获取限制")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 保存图表
            image_path = os.path.join("tests", "output", self.output_dir, f"{symbol.replace('/', '_')}_data_limits.png")
            plt.savefig(image_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            
            logger.info(f"已保存{symbol}的可视化图表: {image_path}")
            
        except Exception as e:
            logger.error(f"创建可视化图表失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())


def main():
    """主函数"""
    logger.info("开始测试币安数据源对不同时间周期的数据获取限制...")
    
    # 创建测试实例
    tester = BinanceDataLimitTester()
    
    # 运行测试
    tester.test_data_limits()
    
    # 生成报告
    success = tester.generate_report()
    
    if success:
        logger.info("测试完成并生成了报告")
    else:
        logger.error("测试完成但未能生成报告")


if __name__ == "__main__":
    main() 