"""
策略性能可视化模块，提供各种图表来分析策略表现
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ...utils.logger import logger


class PerformanceVisualizer:
    """
    策略性能可视化类，使用matplotlib和plotly提供丰富的交互式可视化
    """

    def __init__(self, results=None, performance=None):
        """
        初始化性能可视化器

        Args:
            results (pandas.DataFrame, optional): 回测结果
            performance (dict, optional): 性能指标
        """
        self.results = results
        self.performance = performance

    def set_data(self, results, performance):
        """
        设置回测结果和性能指标

        Args:
            results (pandas.DataFrame): 回测结果
            performance (dict): 性能指标
        """
        self.results = results
        self.performance = performance

    def plot_equity_curve(self, figsize=(12, 6)):
        """
        绘制策略资金曲线

        Args:
            figsize (tuple): 图表尺寸

        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        if self.results is None:
            logger.error("无法绘制图表: 缺少回测结果数据")
            return None

        # 检查并处理必要的列
        if 'cumulative_returns' not in self.results.columns:
            logger.warning("结果中缺少 cumulative_returns 列，尝试计算...")
            
            # 尝试使用 daily_returns 列计算
            if 'daily_returns' in self.results.columns:
                self.results['cumulative_returns'] = (1 + self.results['daily_returns']).cumprod()
                logger.info("成功计算了 cumulative_returns 列")
            # 尝试使用 close 列计算
            elif 'close' in self.results.columns:
                self.results['cumulative_returns'] = self.results['close'] / self.results['close'].iloc[0]
                logger.info("使用 close 列创建了 cumulative_returns 列")
            else:
                logger.error("无法创建 cumulative_returns 列，没有可用的源数据")
                # 创建一个伪造的列以避免错误
                self.results['cumulative_returns'] = 1.0
        
        if 'cumulative_strategy_returns' not in self.results.columns:
            logger.warning("结果中缺少 cumulative_strategy_returns 列，尝试计算...")
            
            # 尝试使用 strategy_returns_after_commission 列计算
            if 'strategy_returns_after_commission' in self.results.columns:
                self.results['cumulative_strategy_returns'] = (1 + self.results['strategy_returns_after_commission']).cumprod()
                logger.info("成功计算了 cumulative_strategy_returns 列")
            # 尝试使用 strategy_returns 列计算
            elif 'strategy_returns' in self.results.columns:
                self.results['cumulative_strategy_returns'] = (1 + self.results['strategy_returns']).cumprod()
                logger.info("使用 strategy_returns 列创建了 cumulative_strategy_returns 列")
            # 使用 equity_curve 列（如果存在）
            elif 'equity_curve' in self.results.columns and self.results['equity_curve'].iloc[0] > 0:
                self.results['cumulative_strategy_returns'] = self.results['equity_curve'] / self.results['equity_curve'].iloc[0]
                logger.info("使用 equity_curve 列创建了 cumulative_strategy_returns 列")
            else:
                logger.error("无法创建 cumulative_strategy_returns 列，没有可用的源数据")
                # 创建一个伪造的列以避免错误
                self.results['cumulative_strategy_returns'] = 1.0

        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制基准收益曲线
        ax.plot(
            self.results.index, 
            self.results['cumulative_returns'], 
            color='blue', 
            alpha=0.5, 
            label='Buy & Hold'
        )
        
        # 绘制策略收益曲线
        ax.plot(
            self.results.index, 
            self.results['cumulative_strategy_returns'], 
            color='green', 
            label='Strategy'
        )
        
        # 检查是否有drawdown列
        if 'drawdown' in self.results.columns:
            # 标记回撤部分
            for i in range(len(self.results)):
                if self.results['drawdown'].iloc[i] <= -0.05:  # 回撤超过5%
                    ax.axvspan(
                        self.results.index[i],
                        self.results.index[i],
                        color='red',
                        alpha=0.3
                    )
        
        # 添加标题和标签
        ax.set_title('Equity Curve Comparison')
        ax.set_ylabel('Cumulative Returns')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True)
        
        return fig

    def plot_underwater_chart(self, figsize=(12, 6)):
        """
        绘制水下图表（回撤可视化）

        Args:
            figsize (tuple): 图表尺寸

        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        if self.results is None:
            logger.error("无法绘制图表: 缺少回测结果数据")
            return None
            
        # 检查并处理必要的列
        if 'drawdown' not in self.results.columns:
            logger.warning("结果中缺少 drawdown 列，尝试计算...")
            
            # 尝试使用累积策略收益计算回撤
            if 'cumulative_strategy_returns' in self.results.columns:
                peaks = self.results['cumulative_strategy_returns'].cummax()
                self.results['drawdown'] = self.results['cumulative_strategy_returns'] / peaks - 1
                logger.info("成功计算了 drawdown 列")
            # 尝试使用资金曲线计算回撤
            elif 'equity_curve' in self.results.columns:
                peaks = self.results['equity_curve'].cummax()
                self.results['drawdown'] = self.results['equity_curve'] / peaks - 1
                logger.info("使用 equity_curve 列创建了 drawdown 列")
            else:
                logger.error("无法创建 drawdown 列，没有可用的源数据")
                # 创建一个伪造的列以避免错误
                self.results['drawdown'] = 0.0

        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制回撤
        ax.fill_between(
            self.results.index, 
            self.results['drawdown'], 
            0, 
            color='red', 
            alpha=0.3,
            label='Drawdown'
        )
        
        # 标记最大回撤
        max_drawdown_idx = self.results['drawdown'].idxmin()
        ax.axvline(
            x=max_drawdown_idx, 
            color='black', 
            linestyle='--',
            label=f'Max Drawdown: {self.results["drawdown"].min():.2%}'
        )
        
        # 添加标题和标签
        ax.set_title('Underwater Chart (Drawdown)')
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True)
        
        return fig

    def plot_monthly_returns_heatmap(self, figsize=(12, 6)):
        """
        绘制月度收益热力图

        Args:
            figsize (tuple): 图表尺寸

        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        if self.results is None:
            logger.error("无法绘制图表: 缺少回测结果数据")
            return None

        # 准备数据
        monthly_returns = self.results['strategy_returns_after_commission'].groupby([
            lambda x: x.year, lambda x: x.month
        ]).apply(lambda x: (1 + x).prod() - 1)
        
        # 将数据转换为年份为行，月份为列的表格
        returns_table = monthly_returns.unstack()
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=figsize)
        heatmap = ax.pcolor(returns_table, cmap='RdYlGn', vmin=-0.1, vmax=0.1)
        
        # 设置坐标轴标签
        ax.set_xticks(np.arange(returns_table.shape[1]) + 0.5)
        ax.set_yticks(np.arange(returns_table.shape[0]) + 0.5)
        ax.set_xticklabels(returns_table.columns)
        ax.set_yticklabels(returns_table.index)
        
        # 添加颜色条
        plt.colorbar(heatmap, ax=ax, label='Monthly Return')
        
        # 添加标题
        ax.set_title('Monthly Returns Heatmap')
        
        # 将具体数值添加到每个单元格
        for i in range(returns_table.shape[0]):
            for j in range(returns_table.shape[1]):
                value = returns_table.iloc[i, j]
                if not np.isnan(value):
                    ax.text(j + 0.5, i + 0.5, f'{value:.2%}',
                           ha='center', va='center',
                           color='white' if abs(value) > 0.05 else 'black')
        
        return fig

    def create_interactive_dashboard(self):
        """
        创建交互式仪表板

        Returns:
            plotly.graph_objects.Figure: Plotly图表对象
        """
        if self.results is None or self.performance is None:
            logger.error("无法创建仪表板: 缺少回测结果或性能指标数据")
            return None
            
        # 检查并处理必要的列
        # 确保 cumulative_returns 列存在
        if 'cumulative_returns' not in self.results.columns:
            logger.warning("结果中缺少 cumulative_returns 列，尝试计算...")
            if 'daily_returns' in self.results.columns:
                self.results['cumulative_returns'] = (1 + self.results['daily_returns']).cumprod()
            elif 'close' in self.results.columns:
                self.results['cumulative_returns'] = self.results['close'] / self.results['close'].iloc[0]
            else:
                self.results['cumulative_returns'] = 1.0
                
        # 确保 cumulative_strategy_returns 列存在
        if 'cumulative_strategy_returns' not in self.results.columns:
            logger.warning("结果中缺少 cumulative_strategy_returns 列，尝试计算...")
            if 'strategy_returns_after_commission' in self.results.columns:
                self.results['cumulative_strategy_returns'] = (1 + self.results['strategy_returns_after_commission']).cumprod()
            elif 'strategy_returns' in self.results.columns:
                self.results['cumulative_strategy_returns'] = (1 + self.results['strategy_returns']).cumprod()
            elif 'equity_curve' in self.results.columns and self.results['equity_curve'].iloc[0] > 0:
                self.results['cumulative_strategy_returns'] = self.results['equity_curve'] / self.results['equity_curve'].iloc[0]
            else:
                self.results['cumulative_strategy_returns'] = 1.0
                
        # 确保 drawdown 列存在
        if 'drawdown' not in self.results.columns:
            logger.warning("结果中缺少 drawdown 列，尝试计算...")
            if 'cumulative_strategy_returns' in self.results.columns:
                peaks = self.results['cumulative_strategy_returns'].cummax()
                self.results['drawdown'] = self.results['cumulative_strategy_returns'] / peaks - 1
            elif 'equity_curve' in self.results.columns:
                peaks = self.results['equity_curve'].cummax()
                self.results['drawdown'] = self.results['equity_curve'] / peaks - 1
            else:
                self.results['drawdown'] = 0.0

        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{"type": "domain"}, {"type": "domain"}]
            ],
            subplot_titles=(
                "Equity Curve", 
                "Drawdown", 
                "Performance Metrics", 
                "Trade Analysis"
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # 添加资金曲线
        fig.add_trace(
            go.Scatter(
                x=self.results.index,
                y=self.results['cumulative_returns'],
                name="Buy & Hold",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.results.index,
                y=self.results['cumulative_strategy_returns'],
                name="Strategy",
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # 添加回撤图
        fig.add_trace(
            go.Scatter(
                x=self.results.index,
                y=self.results['drawdown'],
                name="Drawdown",
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # 添加性能指标仪表盘
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=self.performance['annual_return'],
                title={"text": "Annual Return"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, max(0.5, self.performance['annual_return'] * 1.5)]},
                    'steps': [
                        {'range': [0, 0.1], 'color': "lightgray"},
                        {'range': [0.1, 0.2], 'color': "gray"},
                        {'range': [0.2, 0.3], 'color': "yellow"},
                        {'range': [0.3, 0.5], 'color': "orange"},
                        {'range': [0.5, 1.0], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': self.performance['annual_return']
                    }
                },
                number_valueformat='.2%'
            ),
            row=3, col=1
        )
        
        # 添加交易分析饼图
        fig.add_trace(
            go.Pie(
                labels=["Winning Trades", "Losing Trades"],
                values=[self.performance['winning_trades'], self.performance['losing_trades']],
                hole=0.3,
                marker=dict(colors=['green', 'red'])
            ),
            row=3, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title_text="Strategy Performance Dashboard",
            height=800,
            width=1200,
            showlegend=True,
            template="plotly_white"
        )
        
        # 添加性能指标表格
        performance_text = f"""
        <b>Performance Summary:</b><br>
        Total Return: {self.performance['total_return']:.2%}<br>
        Annual Return: {self.performance['annual_return']:.2%}<br>
        Max Drawdown: {self.performance['max_drawdown']:.2%}<br>
        Sharpe Ratio: {self.performance['sharpe_ratio']:.2f}<br>
        Calmar Ratio: {self.performance['calmar_ratio']:.2f}<br>
        Win Rate: {self.performance['win_rate']:.2%}<br>
        Profit/Loss Ratio: {self.performance['profit_loss_ratio']:.2f}<br>
        """
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0,
            text=performance_text,
            showarrow=False,
            font=dict(size=14),
            align="center",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        
        return fig

    def plot_rolling_statistics(self, window=30, figsize=(12, 8)):
        """
        绘制滚动统计数据

        Args:
            window (int): 滚动窗口大小
            figsize (tuple): 图表尺寸

        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        if self.results is None:
            logger.error("无法绘制图表: 缺少回测结果数据")
            return None

        # 计算滚动统计
        rolling_return = self.results['strategy_returns_after_commission'].rolling(window).mean() * 252  # 年化
        rolling_vol = self.results['strategy_returns_after_commission'].rolling(window).std() * np.sqrt(252)  # 年化
        rolling_sharpe = rolling_return / rolling_vol
        
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # 绘制滚动收益率
        axes[0].plot(rolling_return, color='green')
        axes[0].axhline(y=0, color='black', linestyle='--')
        axes[0].set_title(f'{window}-Day Rolling Annual Return')
        axes[0].set_ylabel('Return')
        axes[0].grid(True)
        
        # 绘制滚动波动率
        axes[1].plot(rolling_vol, color='red')
        axes[1].set_title(f'{window}-Day Rolling Annual Volatility')
        axes[1].set_ylabel('Volatility')
        axes[1].grid(True)
        
        # 绘制滚动夏普比率
        axes[2].plot(rolling_sharpe, color='blue')
        axes[2].axhline(y=0, color='black', linestyle='--')
        axes[2].set_title(f'{window}-Day Rolling Sharpe Ratio')
        axes[2].set_ylabel('Sharpe Ratio')
        axes[2].set_xlabel('Date')
        axes[2].grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        return fig

    def save_report(self, output_path):
        """
        保存完整的HTML性能报告

        Args:
            output_path (str): 输出路径

        Returns:
            bool: 是否成功保存
        """
        try:
            if self.results is None or self.performance is None:
                logger.error("无法创建报告: 缺少回测结果或性能指标数据")
                return False
                
            # 创建交互式仪表板
            dashboard_fig = self.create_interactive_dashboard()
            
            # 构建HTML内容
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Strategy Performance Report</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ padding: 20px; }}
                    .container {{ max-width: 1200px; }}
                    .card {{ margin-bottom: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="text-center mb-4">Strategy Performance Report</h1>
                    
                    <div class="card">
                        <div class="card-header">
                            <h3>Strategy Dashboard</h3>
                        </div>
                        <div class="card-body">
                            <div id="dashboard"></div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <h3>Performance Metrics</h3>
                        </div>
                        <div class="card-body">
                            <table class="table table-striped">
                                <tr><th>Total Return</th><td>{self.performance['total_return']:.2%}</td></tr>
                                <tr><th>Annual Return</th><td>{self.performance['annual_return']:.2%}</td></tr>
                                <tr><th>Max Drawdown</th><td>{self.performance['max_drawdown']:.2%}</td></tr>
                                <tr><th>Sharpe Ratio</th><td>{self.performance['sharpe_ratio']:.2f}</td></tr>
                                <tr><th>Calmar Ratio</th><td>{self.performance['calmar_ratio']:.2f}</td></tr>
                                <tr><th>Win Rate</th><td>{self.performance['win_rate']:.2%}</td></tr>
                                <tr><th>Profit/Loss Ratio</th><td>{self.performance['profit_loss_ratio']:.2f}</td></tr>
                                <tr><th>Winning Trades</th><td>{self.performance['winning_trades']}</td></tr>
                                <tr><th>Losing Trades</th><td>{self.performance['losing_trades']}</td></tr>
                                <tr><th>Risk Assessment</th><td>{self.performance.get('risk_assessment', 'N/A')}</td></tr>
                            </table>
                        </div>
                    </div>
                </div>
                
                <script>
                    var dashboardDiv = document.getElementById('dashboard');
                    var dashboardData = {dashboard_fig.to_json()};
                    Plotly.newPlot(dashboardDiv, dashboardData.data, dashboardData.layout);
                </script>
            </body>
            </html>
            """
            
            # 保存HTML文件
            with open(output_path, 'w') as f:
                f.write(html_content)
                
            logger.info(f"性能报告已保存至: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存报告失败: {str(e)}")
            return False 