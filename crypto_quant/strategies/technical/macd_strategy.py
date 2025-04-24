"""
MACD技术分析策略模块
"""
import pandas as pd
import numpy as np
from ...utils.logger import logger


class MACDStrategy:
    """
    MACD (Moving Average Convergence Divergence) 策略
    
    使用MACD指标进行交易信号生成，经典的技术分析策略之一
    """

    def __init__(self, fast=12, slow=26, signal=9, stop_loss_pct=0, fast_period=None, slow_period=None, signal_period=None):
        """
        初始化MACD策略

        Args:
            fast (int): 快速EMA周期
            slow (int): 慢速EMA周期
            signal (int): 信号线周期
            stop_loss_pct (float): 止损百分比, 默认为0表示不使用止损
            fast_period (int, optional): 快速EMA周期 (与fast参数相同，用于参数优化)
            slow_period (int, optional): 慢速EMA周期 (与slow参数相同，用于参数优化)
            signal_period (int, optional): 信号线周期 (与signal参数相同，用于参数优化)
        """
        # 支持优化器中使用的参数名称
        self.fast = fast_period if fast_period is not None else fast
        self.slow = slow_period if slow_period is not None else slow
        self.signal = signal_period if signal_period is not None else signal
        self.stop_loss_pct = stop_loss_pct
        
        self.name = f"MACD({self.fast},{self.slow},{self.signal})"
        if self.stop_loss_pct is not None and self.stop_loss_pct > 0:
            self.name += f"_SL{self.stop_loss_pct:.1f}%"
            
        logger.info(f"MACD策略初始化完成: {self.name}")

    def calculate_indicators(self, df):
        """
        计算MACD指标

        Args:
            df (pandas.DataFrame): 包含OHLCV数据的DataFrame

        Returns:
            pandas.DataFrame: 添加了MACD指标的DataFrame
        """
        # 确保df是DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入数据必须是pandas DataFrame")

        # 确保df包含必要的列
        required_columns = ['close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"输入DataFrame必须包含以下列: {required_columns}")

        # 计算EMA
        df['ema_fast'] = df['close'].ewm(span=self.fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow, adjust=False).mean()

        # 计算MACD线和信号线
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['signal_line'] = df['macd'].ewm(span=self.signal, adjust=False).mean()

        # 计算MACD柱状图
        df['histogram'] = df['macd'] - df['signal_line']

        return df

    def generate_signals(self, df):
        """
        生成交易信号

        Args:
            df (pandas.DataFrame): 包含MACD指标的DataFrame

        Returns:
            pandas.DataFrame: 添加了交易信号的DataFrame
        """
        # 计算指标
        df = self.calculate_indicators(df)

        # 信号规则：
        # 1. 当MACD线上穿信号线时买入
        # 2. 当MACD线下穿信号线时卖出
        df['macd_crossover'] = np.where(
            (df['macd'] > df['signal_line']) & 
            (df['macd'].shift(1) <= df['signal_line'].shift(1)),
            1, 0
        )
        
        df['macd_crossunder'] = np.where(
            (df['macd'] < df['signal_line']) & 
            (df['macd'].shift(1) >= df['signal_line'].shift(1)),
            -1, 0
        )
        
        # 合并买入卖出信号
        df['signal'] = df['macd_crossover'] + df['macd_crossunder']
        
        # 生成仓位
        df['position'] = df['signal'].replace(to_replace=0, method='ffill')
        df['position'] = df['position'].fillna(0)
        
        # 应用止损
        if self.stop_loss_pct is not None and self.stop_loss_pct > 0:
            # 计算每个交易开始的价格
            df['entry_price'] = np.nan
            df.loc[df['signal'] == 1, 'entry_price'] = df.loc[df['signal'] == 1, 'close']
            df['entry_price'] = df['entry_price'].ffill()
            
            # 计算止损价格 (仅在持有多头仓位时)
            df['stop_loss_price'] = np.where(
                df['position'] > 0, 
                df['entry_price'] * (1 - self.stop_loss_pct / 100), 
                np.nan
            )
            
            # 根据止损规则调整仓位
            for i in range(1, len(df)):
                # 如果有多头仓位且价格低于止损价
                if df['position'].iloc[i-1] > 0 and df['close'].iloc[i] < df['stop_loss_price'].iloc[i-1]:
                    df.loc[df.index[i], 'position'] = 0
                    df.loc[df.index[i], 'signal'] = -1  # 记录止损信号
        
        return df

    def backtest(self, df, initial_capital=10000.0, commission=0.001):
        """
        回测策略

        Args:
            df (pandas.DataFrame): 包含OHLCV数据的DataFrame
            initial_capital (float): 初始资金
            commission (float): 手续费率，如0.001表示0.1%

        Returns:
            pandas.DataFrame: 回测结果
        """
        # 生成信号
        df = self.generate_signals(df)
        
        # 计算每日收益
        df['returns'] = df['close'].pct_change()
        
        # 计算策略收益 (今天的仓位 * 明天的收益率)
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        
        # 计算考虑手续费的策略收益
        # 当仓位发生变化时，收取手续费
        df['position_change'] = df['position'].diff().fillna(0) 
        df['commission_cost'] = abs(df['position_change']) * commission * df['close']
        
        # 调整策略收益
        df['strategy_returns_after_commission'] = df['strategy_returns'] * df['close'] - df['commission_cost']
        df['strategy_returns_after_commission'] = df['strategy_returns_after_commission'] / df['close'].shift(1)
        
        # 计算累积收益
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['cumulative_strategy_returns'] = (1 + df['strategy_returns_after_commission']).cumprod()
        
        # 计算资金曲线
        df['equity_curve'] = initial_capital * df['cumulative_strategy_returns']
        
        # 计算回撤
        df['previous_peaks'] = df['equity_curve'].cummax()
        df['drawdown'] = (df['equity_curve'] - df['previous_peaks']) / df['previous_peaks']
        
        return df
        
    def get_performance_metrics(self, df):
        """
        计算策略表现指标

        Args:
            df (pandas.DataFrame): 回测结果DataFrame

        Returns:
            dict: 包含策略表现指标的字典
        """
        # 年化收益率
        total_days = (df.index[-1] - df.index[0]).days
        annual_return = (df['cumulative_strategy_returns'].iloc[-1] ** (365 / total_days)) - 1
        
        # 最大回撤
        max_drawdown = df['drawdown'].min()
        
        # 夏普比率 (假设无风险利率为0.02)
        risk_free_rate = 0.02
        sharpe_ratio = ((df['strategy_returns_after_commission'].mean() * 252) - risk_free_rate) / \
                      (df['strategy_returns_after_commission'].std() * np.sqrt(252))
        
        # 卡尔玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
        
        # 胜率
        winning_trades = (df['strategy_returns_after_commission'] > 0).sum()
        losing_trades = (df['strategy_returns_after_commission'] < 0).sum()
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        # 盈亏比
        average_win = df.loc[df['strategy_returns_after_commission'] > 0, 'strategy_returns_after_commission'].mean()
        average_loss = abs(df.loc[df['strategy_returns_after_commission'] < 0, 'strategy_returns_after_commission'].mean())
        profit_loss_ratio = average_win / average_loss if average_loss != 0 else float('inf')
        
        # 总收益
        total_return = df['cumulative_strategy_returns'].iloc[-1] - 1
        
        return {
            "annual_return": annual_return,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades
        }

    def run(self, df, initial_capital=10000.0, commission=0.001):
        """
        运行策略并返回回测结果与性能指标

        Args:
            df (pandas.DataFrame): 包含OHLCV数据的DataFrame
            initial_capital (float): 初始资金
            commission (float): 手续费率

        Returns:
            tuple: (回测结果DataFrame, 性能指标字典)
        """
        # 运行回测
        backtest_results = self.backtest(df, initial_capital, commission)
        
        # 计算性能指标
        performance = self.get_performance_metrics(backtest_results)
        
        # 记录性能信息
        logger.info(f"策略 {self.name} 回测完成:")
        logger.info(f"年化收益率: {performance['annual_return']:.2%}")
        logger.info(f"最大回撤: {performance['max_drawdown']:.2%}")
        logger.info(f"卡尔玛比率: {performance['calmar_ratio']:.2f}")
        logger.info(f"夏普比率: {performance['sharpe_ratio']:.2f}")
        
        return backtest_results, performance 