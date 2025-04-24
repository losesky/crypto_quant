"""
高级风险管理模块
根据策略改进计划实现更强大的风险控制机制
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
import logging
from ..utils.logger import logger

class RiskManager:
    """
    高级风险管理类，提供多种风险控制机制：
    1. 动态头寸管理 - 根据波动率和回撤调整仓位大小
    2. 多层次止损机制 - 固定止损、追踪止损、时间止损
    3. 回撤控制 - 基于账户回撤自动降低风险敞口
    4. 风险绩效监控 - 持续评估策略风险指标
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.15,              # 最大允许回撤比例
        max_position_size: float = 0.2,          # 最大头寸比例（占总资金）
        base_position_size: float = 0.1,         # 基础头寸比例
        fixed_stop_loss: Optional[float] = 0.05, # 固定止损比例
        trailing_stop: Optional[float] = 0.03,   # 追踪止损比例
        take_profit: Optional[float] = 0.10,     # 止盈比例
        max_trades_per_day: Optional[int] = None,# 每日最大交易次数
        time_stop_bars: Optional[int] = None,    # 时间止损(K线数)
        consecutive_losses: int = 3,             # 允许连续亏损次数
        volatility_lookback: int = 20,           # 波动率计算回看周期
        volatility_scale_factor: float = 3.0,    # 波动率调整系数
        recovery_factor: float = 0.5,            # 回撤恢复系数
        use_atr_for_stops: bool = True,          # 是否使用ATR动态调整止损
        atr_stop_multiplier: float = 2.0,        # ATR止损乘数
        atr_period: int = 14,                    # ATR计算周期
        drawdown_position_reduce: bool = True,   # 回撤时是否降低仓位
        initial_capital: float = 10000.0         # 初始资金
    ):
        """
        初始化风险管理器
        
        Args:
            max_drawdown: 策略允许的最大回撤比例，超过此值将降低仓位或暂停交易
            max_position_size: 单个交易允许的最大仓位比例（占总资金的百分比）
            base_position_size: 默认基础仓位比例
            fixed_stop_loss: 固定止损比例，None表示不使用
            trailing_stop: 追踪止损比例，None表示不使用
            take_profit: 止盈比例，None表示不使用
            max_trades_per_day: 每日最大交易次数限制，None表示不限制
            time_stop_bars: 时间止损K线数，持仓超过此K线数自动平仓，None表示不使用
            consecutive_losses: 允许的最大连续亏损次数，超过此值降低仓位
            volatility_lookback: 计算波动率的回看周期
            volatility_scale_factor: 波动率调整系数，波动率越高，持仓规模越小
            recovery_factor: 从大回撤中恢复时的仓位调整系数
            use_atr_for_stops: 是否使用ATR动态调整止损
            atr_stop_multiplier: ATR止损乘数
            atr_period: ATR计算周期
            drawdown_position_reduce: 在回撤中是否自动降低仓位
            initial_capital: 初始资金，用于计算回撤和风险控制
        """
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.base_position_size = base_position_size
        self.fixed_stop_loss = fixed_stop_loss
        self.trailing_stop = trailing_stop
        self.take_profit = take_profit
        self.max_trades_per_day = max_trades_per_day
        self.time_stop_bars = time_stop_bars
        self.consecutive_losses = consecutive_losses
        self.volatility_lookback = volatility_lookback
        self.volatility_scale_factor = volatility_scale_factor
        self.recovery_factor = recovery_factor
        self.use_atr_for_stops = use_atr_for_stops
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_period = atr_period
        self.drawdown_position_reduce = drawdown_position_reduce
        self.initial_capital = initial_capital
        
        # 初始化状态变量
        self._entry_prices = {}  # 记录入场价格，格式: {date: (price, direction, stop_price)}
        self._trailing_stops = {}  # 追踪止损价格
        self._current_drawdown = 0.0  # 当前回撤
        self._max_equity = initial_capital  # 最大权益
        self._consecutive_loss_count = 0  # 连续亏损计数
        self._trade_count_today = 0  # 今日交易次数
        self._current_day = None  # 当前交易日
        self._trade_history = []  # 交易历史
        self._current_position_scale = 1.0  # 当前仓位缩放因子
        
        # 日志初始化
        self._init_logger()
        
        logger.info(f"风险管理器初始化完成，最大回撤限制: {self.max_drawdown:.2%}, "
                   f"最大仓位比例: {self.max_position_size:.2%}")
    
    def _init_logger(self):
        """初始化日志配置"""
        self.logger = logger
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        计算真实波动幅度均值(ATR)
        
        Args:
            df: 带有OHLC数据的DataFrame
            
        Returns:
            pd.Series: ATR值序列
        """
        if df.empty:
            return pd.Series()
            
        # 确保DataFrame有必要的列
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"计算ATR失败：缺少必要的列 {col}")
                return pd.Series(index=df.index)
        
        # 计算真实波动幅度(TR)
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # 计算ATR
        atr = tr.rolling(window=self.atr_period).mean()
        return atr
    
    def calculate_position_size(self, account_value: float, df: pd.DataFrame, current_index: int) -> float:
        """
        根据市场波动率和账户状态计算头寸大小
        
        Args:
            account_value: 当前账户价值
            df: 市场数据DataFrame
            current_index: 当前时间索引在df中的位置
            
        Returns:
            float: 计算的头寸大小比例(0.0-1.0)
        """
        # 默认使用基础仓位
        position_size = self.base_position_size
        
        # 如果数据不足，返回基础仓位
        if current_index < self.volatility_lookback or df.empty:
            self.logger.debug(f"数据点不足，使用基础仓位: {position_size:.2%}")
            return min(position_size, self.max_position_size)
        
        try:
            # 计算最近的波动率(使用收盘价的标准差)
            recent_volatility = df['close'].iloc[current_index-self.volatility_lookback:current_index].pct_change().std()
            
            # 波动率正常化(避免极端值)
            normalized_volatility = min(max(recent_volatility, 0.005), 0.05)
            
            # 波动率越高，头寸越小
            volatility_factor = 1.0 / (1.0 + self.volatility_scale_factor * normalized_volatility)
            
            # 考虑当前回撤状态
            if self.drawdown_position_reduce and self._current_drawdown > 0:
                # 回撤越大，头寸越小
                drawdown_factor = 1.0 - (self._current_drawdown / self.max_drawdown) ** 0.5
                drawdown_factor = max(0.1, drawdown_factor)  # 确保至少有10%的基础仓位
            else:
                drawdown_factor = 1.0
            
            # 考虑连续亏损
            if self._consecutive_loss_count >= self.consecutive_losses:
                # 连续亏损后降低仓位
                loss_factor = 1.0 - min(0.5, 0.1 * (self._consecutive_loss_count - self.consecutive_losses + 1))
            else:
                loss_factor = 1.0
            
            # 综合各因素计算最终头寸
            position_size = self.base_position_size * volatility_factor * drawdown_factor * loss_factor * self._current_position_scale
            
            # 确保头寸在允许范围内
            position_size = min(position_size, self.max_position_size)
            position_size = max(position_size, 0.01)  # 至少保留1%的仓位
            
            self.logger.debug(f"计算头寸大小: 基础={self.base_position_size:.2%}, "
                             f"波动率因子={volatility_factor:.2f}, 回撤因子={drawdown_factor:.2f}, "
                             f"亏损因子={loss_factor:.2f}, 最终={position_size:.2%}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"计算头寸大小时出错: {str(e)}")
            return min(self.base_position_size, self.max_position_size)
    
    def update_account_state(self, current_equity: float, trade_result: Optional[float] = None) -> None:
        """
        更新账户状态，包括回撤跟踪和连续亏损计数
        
        Args:
            current_equity: 当前账户权益
            trade_result: 最近交易的盈亏(可选)
        """
        # 更新最大权益和回撤
        if current_equity > self._max_equity:
            self._max_equity = current_equity
            # 如果创新高，可以开始恢复仓位
            if self._current_position_scale < 1.0:
                self._current_position_scale = min(1.0, self._current_position_scale + self.recovery_factor * 0.1)
                self.logger.info(f"账户创新高，提高仓位系数至 {self._current_position_scale:.2f}")
        
        # 计算当前回撤
        if self._max_equity > 0:
            self._current_drawdown = 1.0 - (current_equity / self._max_equity)
            
            # 如果回撤超过阈值，降低仓位
            if self._current_drawdown > self.max_drawdown and self.drawdown_position_reduce:
                old_scale = self._current_position_scale
                self._current_position_scale *= 0.75  # 降低25%的仓位
                self.logger.warning(f"回撤超过阈值({self._current_drawdown:.2%} > {self.max_drawdown:.2%})，"
                                   f"降低仓位系数: {old_scale:.2f} → {self._current_position_scale:.2f}")
        
        # 更新连续亏损计数
        if trade_result is not None:
            if trade_result < 0:
                self._consecutive_loss_count += 1
                if self._consecutive_loss_count >= self.consecutive_losses:
                    self.logger.warning(f"连续亏损 {self._consecutive_loss_count} 次，风险管理将降低头寸大小")
            else:
                if self._consecutive_loss_count > 0:
                    self.logger.info(f"盈利交易，重置连续亏损计数(之前: {self._consecutive_loss_count})")
                self._consecutive_loss_count = 0
    
    def apply_risk_management(self, df: pd.DataFrame, positions_column: str = 'position') -> pd.DataFrame:
        """
        应用风险管理规则到策略结果
        
        Args:
            df: 策略生成的带有仓位的DataFrame
            positions_column: 仓位列名
            
        Returns:
            pd.DataFrame: 应用了风险管理的DataFrame
        """
        # 重置入场价格记录和追踪止损
        self._entry_prices = {}
        self._trailing_stops = {}
        
        # 确保数据框有必要的列
        if df.empty or positions_column not in df.columns:
            self.logger.warning(f"应用风险管理: DataFrame为空或缺少仓位列 '{positions_column}'")
            return df
        
        # 添加ATR列用于动态止损
        if self.use_atr_for_stops:
            df['atr'] = self._calculate_atr(df)
        
        # 添加账户权益列
        if 'equity_curve' not in df.columns:
            # 初始化权益曲线
            df['equity_curve'] = self.initial_capital
        
        # 添加交易计数列
        df['trades_today'] = 0
        
        # 添加仓位大小列
        df['position_size'] = 0.0
        
        # 防止连续交易
        df['last_trade_bar'] = 0
        
        result_df = df.copy()
        
        # 当前的持仓状态(0=无持仓, 1=多头, -1=空头)
        current_position = 0
        
        # 最后交易的K线索引
        last_trade_idx = -999
        
        try:
            # 逐行处理数据
            for i in range(1, len(result_df)):
                current_idx = result_df.index[i]
                previous_idx = result_df.index[i-1]
                
                # 检查日期变更，重置每日交易计数
                current_day = pd.Timestamp(current_idx).date()
                if self._current_day is None or current_day != self._current_day:
                    self._trade_count_today = 0
                    self._current_day = current_day
                
                # 更新账户状态
                current_equity = result_df['equity_curve'].iloc[i-1]
                self.update_account_state(current_equity)
                
                # 记录回撤状态
                result_df.loc[current_idx, 'drawdown'] = self._current_drawdown
                
                # 计算当前头寸大小
                position_size = self.calculate_position_size(
                    current_equity, 
                    result_df.iloc[:i+1], 
                    i
                )
                result_df.loc[current_idx, 'position_size'] = position_size
                
                # 获取当前和之前的仓位
                current_signal = result_df[positions_column].iloc[i]
                previous_position = current_position
                
                # 获取当前价格
                current_price = result_df['close'].iloc[i]
                
                # 初始化本行的仓位为上一行的仓位
                result_df.loc[current_idx, positions_column] = previous_position
                
                # 记录距上次交易的K线数
                bar_since_last_trade = i - last_trade_idx
                result_df.loc[current_idx, 'last_trade_bar'] = bar_since_last_trade
                
                # 如果有持仓，应用止损止盈逻辑
                if previous_position != 0:
                    # 获取入场信息
                    entry_info = None
                    for date in reversed(list(self._entry_prices.keys())):
                        if date <= current_idx:
                            entry_info = self._entry_prices[date]
                            break
                    
                    if entry_info:
                        entry_price, entry_direction, stop_price = entry_info
                        
                        # 1. 时间止损
                        if self.time_stop_bars is not None and bar_since_last_trade > self.time_stop_bars:
                            result_df.loc[current_idx, positions_column] = 0
                            self.logger.info(f"时间止损触发: K线数={bar_since_last_trade} > {self.time_stop_bars}")
                            current_position = 0
                            # 记录本次交易的结果
                            if previous_position > 0:
                                trade_result = (current_price - entry_price) / entry_price
                            else:
                                trade_result = (entry_price - current_price) / entry_price
                            self.update_account_state(current_equity, trade_result)
                            continue
                        
                        # 2. 固定止损和追踪止损
                        if previous_position > 0:  # 多头
                            # 计算当前价格相对入场价格的变化
                            price_change = (current_price - entry_price) / entry_price
                            
                            # 更新追踪止损价格
                            if self.trailing_stop is not None:
                                if current_idx in self._trailing_stops:
                                    current_trail = self._trailing_stops[current_idx]
                                else:
                                    # 初始追踪止损价
                                    current_trail = entry_price * (1 - self.trailing_stop)
                                
                                # 如果价格上涨，提高追踪止损价格
                                new_trail_price = current_price * (1 - self.trailing_stop)
                                if new_trail_price > current_trail:
                                    self._trailing_stops[current_idx] = new_trail_price
                                    current_trail = new_trail_price
                                
                                # 检查是否触发追踪止损
                                if current_price <= current_trail:
                                    result_df.loc[current_idx, positions_column] = 0
                                    self.logger.info(f"追踪止损触发: 价格={current_price:.2f}, 止损价={current_trail:.2f}")
                                    current_position = 0
                                    self.update_account_state(current_equity, price_change)
                                    continue
                            
                            # 检查是否触发固定止损
                            if self.fixed_stop_loss is not None:
                                # 使用ATR动态调整止损幅度
                                if self.use_atr_for_stops and 'atr' in result_df.columns:
                                    current_atr = result_df['atr'].iloc[i]
                                    if not pd.isna(current_atr) and current_atr > 0:
                                        dynamic_stop = self.atr_stop_multiplier * current_atr / entry_price
                                        actual_stop = max(dynamic_stop, self.fixed_stop_loss)
                                    else:
                                        actual_stop = self.fixed_stop_loss
                                else:
                                    actual_stop = self.fixed_stop_loss
                                
                                if price_change < -actual_stop:
                                    result_df.loc[current_idx, positions_column] = 0
                                    self.logger.info(f"固定止损触发: 价格={current_price:.2f}, 入场价={entry_price:.2f}, "
                                                   f"跌幅={price_change:.2%}, 止损阈值={-actual_stop:.2%}")
                                    current_position = 0
                                    self.update_account_state(current_equity, price_change)
                                    continue
                            
                            # 检查是否触发止盈
                            if self.take_profit is not None and price_change > self.take_profit:
                                result_df.loc[current_idx, positions_column] = 0
                                self.logger.info(f"止盈触发: 价格={current_price:.2f}, 入场价={entry_price:.2f}, "
                                               f"涨幅={price_change:.2%}, 止盈阈值={self.take_profit:.2%}")
                                current_position = 0
                                self.update_account_state(current_equity, price_change)
                                continue
                                
                        elif previous_position < 0:  # 空头
                            # 计算当前价格相对入场价格的变化(对于空头，价格下跌是盈利)
                            price_change = (entry_price - current_price) / entry_price
                            
                            # 更新追踪止损价格
                            if self.trailing_stop is not None:
                                if current_idx in self._trailing_stops:
                                    current_trail = self._trailing_stops[current_idx]
                                else:
                                    # 初始追踪止损价
                                    current_trail = entry_price * (1 + self.trailing_stop)
                                
                                # 如果价格下跌，降低追踪止损价格
                                new_trail_price = current_price * (1 + self.trailing_stop)
                                if new_trail_price < current_trail:
                                    self._trailing_stops[current_idx] = new_trail_price
                                    current_trail = new_trail_price
                                
                                # 检查是否触发追踪止损
                                if current_price >= current_trail:
                                    result_df.loc[current_idx, positions_column] = 0
                                    self.logger.info(f"空头追踪止损触发: 价格={current_price:.2f}, 止损价={current_trail:.2f}")
                                    current_position = 0
                                    self.update_account_state(current_equity, price_change)
                                    continue
                            
                            # 检查是否触发固定止损
                            if self.fixed_stop_loss is not None:
                                # 使用ATR动态调整止损幅度
                                if self.use_atr_for_stops and 'atr' in result_df.columns:
                                    current_atr = result_df['atr'].iloc[i]
                                    if not pd.isna(current_atr) and current_atr > 0:
                                        dynamic_stop = self.atr_stop_multiplier * current_atr / entry_price
                                        actual_stop = max(dynamic_stop, self.fixed_stop_loss)
                                    else:
                                        actual_stop = self.fixed_stop_loss
                                else:
                                    actual_stop = self.fixed_stop_loss
                                
                                if price_change < -actual_stop:  # 对于空头，价格上涨导致亏损
                                    result_df.loc[current_idx, positions_column] = 0
                                    self.logger.info(f"空头固定止损触发: 价格={current_price:.2f}, 入场价={entry_price:.2f}, "
                                                   f"上涨={-price_change:.2%}, 止损阈值={actual_stop:.2%}")
                                    current_position = 0
                                    self.update_account_state(current_equity, price_change)
                                    continue
                            
                            # 检查是否触发止盈
                            if self.take_profit is not None and price_change > self.take_profit:
                                result_df.loc[current_idx, positions_column] = 0
                                self.logger.info(f"空头止盈触发: 价格={current_price:.2f}, 入场价={entry_price:.2f}, "
                                               f"下跌={price_change:.2%}, 止盈阈值={self.take_profit:.2%}")
                                current_position = 0
                                self.update_account_state(current_equity, price_change)
                                continue
                
                # 处理新的交易信号
                if current_signal != 0 and current_signal != previous_position:
                    # 检查每日交易次数限制
                    if self.max_trades_per_day is not None and self._trade_count_today >= self.max_trades_per_day:
                        self.logger.warning(f"达到每日最大交易次数({self.max_trades_per_day})，忽略新信号")
                        continue
                    
                    # 执行开仓或换仓操作
                    result_df.loc[current_idx, positions_column] = current_signal
                    current_position = current_signal
                    
                    # 记录入场价格和方向
                    self._entry_prices[current_idx] = (current_price, current_signal, 0.0)
                    
                    # 更新最后交易K线索引
                    last_trade_idx = i
                    
                    # 增加每日交易计数
                    self._trade_count_today += 1
                    result_df.loc[current_idx, 'trades_today'] = self._trade_count_today
                    
                    self.logger.info(f"新仓位: 日期={current_idx}, 价格={current_price:.2f}, "
                                   f"方向={current_signal:+.0f}, 头寸比例={position_size:.2%}")
        
        except Exception as e:
            self.logger.error(f"应用风险管理时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return result_df
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """
        获取当前风险指标
        
        Returns:
            Dict[str, float]: 风险指标字典
        """
        return {
            'current_drawdown': self._current_drawdown,
            'position_scale': self._current_position_scale,
            'consecutive_losses': self._consecutive_loss_count,
            'max_equity': self._max_equity,
            'trades_today': self._trade_count_today
        }