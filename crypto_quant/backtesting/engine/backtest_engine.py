"""
回测引擎模块，提供策略回测和评估功能
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from ...config.settings import BACKTEST_CONFIG, RISK_CONFIG
from ...utils.logger import logger


# 检查是否有中文字体可用
def check_chinese_font():
    """检查系统中是否有可用的中文字体"""
    try:
        # 首先尝试导入FontHelper
        from ...utils.font_helper import get_font_helper
        font_helper = get_font_helper()
        return font_helper.has_chinese_font, font_helper.chinese_font
    except ImportError:
        # 如果无法导入，使用简化的检测方法
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans CN', 'WenQuanYi Micro Hei']
        for font_name in chinese_fonts:
            try:
                font_prop = FontProperties(fname=font_name)
                if font_prop is not None:
                    return True, font_name
            except:
                continue
        return False, None

# 全局检查中文字体可用性
HAS_CHINESE_FONT, CHINESE_FONT_NAME = check_chinese_font()

# 如果找到中文字体，设置为默认字体
if HAS_CHINESE_FONT:
    try:
        plt.rcParams['font.family'] = ['sans-serif']
        if isinstance(CHINESE_FONT_NAME, str):
            plt.rcParams['font.sans-serif'] = [CHINESE_FONT_NAME, 'SimHei', 'Arial Unicode MS'] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        logger.info(f"已设置默认中文字体: {CHINESE_FONT_NAME}")
    except Exception as e:
        logger.warning(f"设置默认中文字体失败: {str(e)}")
        HAS_CHINESE_FONT = False


class BacktestEngine:
    """
    回测引擎类，用于执行策略回测和性能评估
    """

    def __init__(self, data, strategy, initial_capital=None, commission=None):
        """
        初始化回测引擎

        Args:
            data (pandas.DataFrame): 回测数据
            strategy: 交易策略对象
            initial_capital (float, optional): 初始资金
            commission (float, optional): 交易手续费率
        """
        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital or BACKTEST_CONFIG.get("initial_capital", 10000)
        self.commission = commission or BACKTEST_CONFIG.get("default_commission", 0.001)
        
        # 回测结果
        self.results = None
        self.performance = None
        
        logger.info(f"回测引擎初始化完成: 策略={strategy.name}, 初始资金={self.initial_capital}, 手续费率={self.commission}")

    def run(self):
        """
        执行回测

        Returns:
            self: 回测引擎实例
        """
        logger.info(f"开始回测策略: {self.strategy.name}")
        
        try:
            # 根据策略类型调用适当的run方法
            if 'MACDLSTMHybridStrategy' in self.strategy.__class__.__name__:
                # 混合策略的run方法不接受initial_capital和commission参数
                logger.info("检测到混合策略类型，使用特定调用方式")
                # 设置数据到策略对象
                if not hasattr(self.strategy, 'data') or self.strategy.data is None:
                    setattr(self.strategy, 'data', self.data.copy())
                    logger.info("已将数据设置到混合策略对象中")
                # 设置手续费到策略对象
                if hasattr(self.strategy, 'commission') and self.strategy.commission is None:
                    setattr(self.strategy, 'commission', self.commission)
                    logger.info(f"已设置手续费率: {self.commission}")
                self.results, self.performance = self.strategy.run(visualize=True)
            else:
                # 其他策略的run方法
                self.results, self.performance = self.strategy.run(
                    self.data, 
                    initial_capital=self.initial_capital,
                    commission=self.commission
                )
            
            # 验证回测结果
            if self.results is None or len(self.results) == 0:
                logger.error("策略没有生成有效的回测结果")
                return self
                
            if self.performance is None:
                logger.error("策略没有生成性能指标")
                return self
                
            # 检查回测结果必要的列
            required_columns = ['close', 'position', 'equity_curve', 'drawdown']
            missing_columns = [col for col in required_columns if col not in self.results.columns]
            if missing_columns:
                logger.error(f"回测结果缺少必要的列: {missing_columns}")
                
                # 尝试修复混合策略的结果
                if 'MACDLSTMHybridStrategy' in self.strategy.__class__.__name__:
                    logger.info("尝试修复混合策略的回测结果...")
                    
                    # 如果缺少equity_curve但有cumulative_strategy_returns
                    if 'equity_curve' not in self.results.columns and 'cumulative_strategy_returns' in self.results.columns:
                        self.results['equity_curve'] = self.initial_capital * self.results['cumulative_strategy_returns']
                        logger.info("已添加equity_curve列")
                    
                    # 如果缺少drawdown但有equity_curve
                    if 'drawdown' not in self.results.columns and 'equity_curve' in self.results.columns:
                        self.results['previous_peaks'] = self.results['equity_curve'].cummax()
                        self.results['drawdown'] = (self.results['equity_curve'] / self.results['previous_peaks']) - 1
                        logger.info("已添加drawdown列")
                    
                    # 再次检查是否还有缺失的列
                    missing_columns = [col for col in required_columns if col not in self.results.columns]
                    if missing_columns:
                        logger.error(f"尝试修复后仍缺少必要的列: {missing_columns}")
                        return self
                    else:
                        logger.info("修复成功，所有必要的列都已存在")
                else:
                    return self
            
            # 验证风险指标
            self._validate_risk_metrics()
            
            logger.info(f"策略回测成功完成: {self.strategy.name}")
        except Exception as e:
            logger.error(f"策略回测过程中出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 创建空回测结果
            self.results = pd.DataFrame()
            self.performance = {}
            
        return self

    def _validate_risk_metrics(self):
        """
        验证策略风险指标
        """
        if self.performance is None:
            logger.warning("无法验证风险指标：性能指标为空")
            return
            
        # 检查必要的指标是否存在
        required_metrics = ['max_drawdown', 'calmar_ratio']
        for metric in required_metrics:
            if metric not in self.performance:
                logger.warning(f"无法验证风险指标：缺少 {metric}")
                return
        
        try:
            # 获取风险配置
            max_drawdown_limit = RISK_CONFIG.get("max_drawdown_limit", 0.15)
            min_calmar_ratio = RISK_CONFIG.get("min_calmar_ratio", 2.5)
            
            if abs(self.performance['max_drawdown']) > max_drawdown_limit:
                logger.warning(
                    f"Strategy max drawdown ({self.performance['max_drawdown']:.2%}) exceeds limit ({max_drawdown_limit:.2%})"
                )
            
            if self.performance['calmar_ratio'] < min_calmar_ratio:
                logger.warning(
                    f"Strategy Calmar ratio ({self.performance['calmar_ratio']:.2f}) below required level ({min_calmar_ratio:.2f})"
                )
        except Exception as e:
            logger.error(f"验证风险指标时出错: {str(e)}")

    def plot_results(self, figsize=(12, 10)):
        """
        绘制回测结果图表
        
        Args:
            figsize (tuple, optional): 图表大小
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        if self.results is None:
            logger.error("Cannot plot: No backtest results available.")
            return None
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        
        # 检查全局字体助手是否可用
        try:
            from ...utils.font_helper import get_font_helper
            font_helper = get_font_helper()
            has_chinese_font = font_helper.has_chinese_font
        except ImportError:
            has_chinese_font = HAS_CHINESE_FONT
            font_helper = None
        
        # 设置标题文字（中英文）
        zh_titles = {
            'price': '价格与交易信号',
            'equity': '权益曲线',
            'drawdown': '回撤',
            'position': '仓位变化'
        }
        
        en_titles = {
            'price': 'Price and Trading Signals',
            'equity': 'Equity Curve',
            'drawdown': 'Drawdown',
            'position': 'Position Changes'
        }
        
        # 使用中文或英文标题
        titles = zh_titles if has_chinese_font else en_titles
        
        # 绘制价格和交易点
        ax1.plot(self.results.index, self.results['close'], label='价格' if has_chinese_font else 'Price')
        
        # 标记买入点
        buy_signals = self.results[self.results['signal'] > 0].index
        sell_signals = self.results[self.results['signal'] < 0].index
        
        if len(buy_signals) > 0:
            ax1.scatter(buy_signals, self.results.loc[buy_signals, 'close'], 
                       marker='^', color='green', s=100, label='买入' if has_chinese_font else 'Buy')
        
        if len(sell_signals) > 0:
            ax1.scatter(sell_signals, self.results.loc[sell_signals, 'close'], 
                       marker='v', color='red', s=100, label='卖出' if has_chinese_font else 'Sell')
        
        # 设置标题和图例
        ax1.set_title(titles['price'])
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # 绘制权益曲线
        if 'equity_curve' in self.results.columns:
            ax2.plot(self.results.index, self.results['equity_curve'], label='策略' if has_chinese_font else 'Strategy', color='blue')
            
            if 'benchmark_equity' in self.results.columns:
                ax2.plot(self.results.index, self.results['benchmark_equity'], 
                        label='基准' if has_chinese_font else 'Benchmark', color='gray', linestyle='--')
        elif 'cumulative_strategy_returns' in self.results.columns:
            ax2.plot(self.results.index, self.results['cumulative_strategy_returns'], 
                    label='策略' if has_chinese_font else 'Strategy', color='blue')
            
            if 'cumulative_returns' in self.results.columns:
                ax2.plot(self.results.index, self.results['cumulative_returns'], 
                        label='基准' if has_chinese_font else 'Benchmark', color='gray', linestyle='--')
        
        ax2.set_title(titles['equity'])
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # 绘制回撤
        if 'drawdown' in self.results.columns:
            ax3.fill_between(self.results.index, self.results['drawdown'] * 100, 0, 
                            color='red', alpha=0.3, label='回撤 %' if has_chinese_font else 'Drawdown %')
            ax3.set_title(titles['drawdown'])
            ax3.legend(loc='upper left')
            ax3.grid(True)
        
        # 绘制仓位变化
        if 'position' in self.results.columns:
            ax4.plot(self.results.index, self.results['position'], label='仓位' if has_chinese_font else 'Position')
            ax4.set_title(titles['position'])
        
        # 如果是MACD策略，添加MACD指标
        if 'macd' in self.results.columns and 'signal_line' in self.results.columns:
            # 创建次坐标轴
            ax5 = ax4.twinx()
            ax5.plot(self.results.index, self.results['macd'], label='MACD', color='blue')
            ax5.plot(self.results.index, self.results['signal_line'], label='Signal', color='red')
            ax5.bar(self.results.index, self.results['histogram'], label='Hist', color='gray', alpha=0.3)
            ax5.legend(loc='upper right')
        
        ax4.legend(loc='upper left')
        ax4.grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 应用字体到整个图表
        if font_helper:
            try:
                font_helper.apply_font_to_figure(fig)
                logger.info("已应用字体助手到图表")
            except Exception as e:
                logger.warning(f"应用字体助手到图表失败: {str(e)}")
        elif has_chinese_font and CHINESE_FONT_NAME:
            try:
                # 手动应用中文字体
                for ax in fig.axes:
                    title = ax.get_title()
                    if title:
                        ax.set_title(title, fontproperties=FontProperties(fname=CHINESE_FONT_NAME))
                    
                    # 处理坐标轴标签
                    xlabel = ax.get_xlabel()
                    if xlabel:
                        ax.set_xlabel(xlabel, fontproperties=FontProperties(fname=CHINESE_FONT_NAME))
                    
                    ylabel = ax.get_ylabel()
                    if ylabel:
                        ax.set_ylabel(ylabel, fontproperties=FontProperties(fname=CHINESE_FONT_NAME))
                    
                    # 处理图例
                    legend = ax.get_legend()
                    if legend:
                        for text in legend.get_texts():
                            text.set_fontproperties(FontProperties(fname=CHINESE_FONT_NAME))
            
                logger.info("已手动应用中文字体到图表")
            except Exception as e:
                logger.warning(f"手动应用中文字体到图表失败: {str(e)}")
        
        return fig

    def print_performance_report(self):
        """
        打印策略表现报告
        """
        if self.performance is None:
            logger.error("Cannot print report: Please run backtest first")
            return
        
        performance = self.performance
        
        print("\n" + "="*50)
        print(f"Strategy Performance Report: {self.strategy.name}")
        print("="*50)
        print(f"Backtest Period: {self.results.index[0].strftime('%Y-%m-%d')} to {self.results.index[-1].strftime('%Y-%m-%d')}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.initial_capital * (1 + performance['total_return']):,.2f}")
        print(f"Total Return: {performance['total_return']:.2%}")
        print(f"Annual Return: {performance['annual_return']:.2%}")
        print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Calmar Ratio: {performance['calmar_ratio']:.2f}")
        print(f"Win Rate: {performance['win_rate']:.2%}")
        print(f"Profit/Loss Ratio: {performance['profit_loss_ratio']:.2f}")
        print(f"Winning Trades: {performance['winning_trades']}")
        print(f"Losing Trades: {performance['losing_trades']}")
        print("="*50)

    def summary(self):
        """
        返回策略性能摘要

        Returns:
            dict: 性能摘要字典
        """
        # 检查回测结果
        if self.performance is None or not isinstance(self.performance, dict):
            logger.warning("无法生成性能概要：性能指标为空或无效")
            # 返回默认的性能指标字典
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'calmar_ratio': 0.0,
                'win_rate': 0.0,
                'profit_loss_ratio': 0.0,
                'winning_trades': 0,
                'losing_trades': 0,
                'risk_assessment': 'Failed',
                'trade_count': 0,
                'backtest_start': 'N/A',
                'backtest_end': 'N/A',
                'final_capital': self.initial_capital
            }
            
        # 结果为空
        if self.results is None or len(self.results) == 0:
            logger.warning("无法生成性能概要：回测结果为空")
            # 使用性能指标
            performance = self.performance.copy()
            # 添加默认回测期间
            performance['backtest_start'] = 'N/A'
            performance['backtest_end'] = 'N/A'
            # 添加其他默认值
            if 'risk_assessment' not in performance:
                performance['risk_assessment'] = 'Unknown'
            if 'trade_count' not in performance:
                if 'winning_trades' in performance and 'losing_trades' in performance:
                    performance['trade_count'] = performance['winning_trades'] + performance['losing_trades']
                else:
                    performance['trade_count'] = 0
            if 'final_capital' not in performance:
                if 'total_return' in performance:
                    performance['final_capital'] = self.initial_capital * (1 + performance['total_return'])
                else:
                    performance['final_capital'] = self.initial_capital
            return performance
        
        # 获取性能指标
        performance = self.performance.copy()
        
        # 添加风控评估
        performance['risk_assessment'] = self._get_risk_assessment()
        
        # 添加交易统计
        performance['trade_count'] = performance['winning_trades'] + performance['losing_trades']
        
        # 添加回测期间
        try:
            performance['backtest_start'] = self.results.index[0].strftime('%Y-%m-%d')
            performance['backtest_end'] = self.results.index[-1].strftime('%Y-%m-%d')
        except Exception as e:
            logger.warning(f"获取回测期间出错: {str(e)}")
            performance['backtest_start'] = 'N/A'
            performance['backtest_end'] = 'N/A'
        
        # 添加最终资金
        performance['final_capital'] = self.initial_capital * (1 + performance['total_return'])
        
        return performance

    def _get_risk_assessment(self):
        """
        获取风险评估

        Returns:
            str: 风险评估结果
        """
        # 检查性能指标是否存在
        if self.performance is None:
            return "Unknown"
            
        # 检查必要的风险指标是否存在
        if 'max_drawdown' not in self.performance or 'calmar_ratio' not in self.performance:
            return "Incomplete Data"
        
        try:
            max_drawdown_limit = RISK_CONFIG.get("max_drawdown_limit", 0.15)
            min_calmar_ratio = RISK_CONFIG.get("min_calmar_ratio", 2.5)
            
            max_drawdown = abs(self.performance['max_drawdown'])
            calmar_ratio = self.performance['calmar_ratio']
            
            if max_drawdown <= max_drawdown_limit / 2 and calmar_ratio >= min_calmar_ratio * 1.5:
                return "Excellent"
            elif max_drawdown <= max_drawdown_limit and calmar_ratio >= min_calmar_ratio:
                return "Good"
            elif max_drawdown <= max_drawdown_limit * 1.2 or calmar_ratio >= min_calmar_ratio * 0.8:
                return "Average"
            else:
                return "Below Standard"
        except Exception as e:
            logger.error(f"获取风险评估时出错: {str(e)}")
            return "Error" 