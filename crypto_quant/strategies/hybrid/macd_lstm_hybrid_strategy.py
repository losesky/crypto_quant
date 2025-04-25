"""
MACD-LSTM混合策略
结合技术分析和机器学习的优势，创建更稳健的交易策略
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
from matplotlib.font_manager import FontProperties

from ...strategies.technical.macd_strategy import MACDStrategy
from ...strategies.ml_based.enhanced_lstm_strategy import EnhancedLSTMStrategy
from ...utils.logger import logger
from ...utils.font_helper import get_font_helper
from ...utils.output_helper import get_image_path

# 获取字体助手
font_helper = get_font_helper()


class MACDLSTMHybridStrategy:
    """
    MACD-LSTM混合策略类，结合传统技术分析和机器学习的优势
    
    支持多种策略组合方式：
    1. 一致性投票：只有当两个策略信号一致时才产生交易信号
    2. 加权组合：根据各策略权重组合信号
    3. 分层决策：一个策略决定方向，另一个策略决定入场时机
    4. 专家系统：根据市场状态动态选择最适合的策略
    """
    
    def __init__(
        self,
        # MACD策略参数
        macd_fast_period: int = 12,
        macd_slow_period: int = 26,
        macd_signal_period: int = 9,
        macd_stop_loss_pct: Optional[float] = 0.03,
        
        # LSTM策略参数
        lstm_sequence_length: int = 20,
        lstm_hidden_dim: int = 128,
        lstm_prediction_threshold: float = 0.01,
        lstm_feature_engineering: bool = True,
        lstm_use_attention: bool = True,
        lstm_model_path: Optional[str] = None,
        
        # 混合策略参数
        ensemble_method: str = 'vote',  # 'vote', 'weight', 'layered', 'expert'
        ensemble_weights: Tuple[float, float] = (0.5, 0.5),  # MACD权重, LSTM权重
        market_regime_threshold: float = 0.15,  # 市场制度识别阈值
        stop_loss_pct: Optional[float] = 0.05,  # 全局止损比例
        take_profit_pct: Optional[float] = 0.10,  # 全局止盈比例
        moving_average_period: int = 50,  # 用于市场趋势识别的均线周期
        output_dir: str = "",  # 输出目录
        commission: float = 0.001  # 交易手续费率
    ):
        """
        初始化MACD-LSTM混合策略
        
        Args:
            macd_fast_period: MACD快线周期
            macd_slow_period: MACD慢线周期
            macd_signal_period: MACD信号线周期
            macd_stop_loss_pct: MACD策略止损百分比
            
            lstm_sequence_length: LSTM序列长度
            lstm_hidden_dim: LSTM隐藏层维度
            lstm_prediction_threshold: LSTM预测变化阈值
            lstm_feature_engineering: 是否启用特征工程
            lstm_use_attention: 是否使用注意力机制
            lstm_model_path: LSTM预训练模型路径
            
            ensemble_method: 组合方法
            ensemble_weights: 策略权重
            market_regime_threshold: 市场制度识别阈值
            stop_loss_pct: 全局止损比例
            take_profit_pct: 全局止盈比例
            moving_average_period: 均线周期
            output_dir: 输出目录
            commission: 交易手续费率
        """
        # 存储参数
        self.ensemble_method = ensemble_method
        self.ensemble_weights = ensemble_weights
        self.market_regime_threshold = market_regime_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.moving_average_period = moving_average_period
        self.output_dir = output_dir
        self.commission = commission
        self.data = None  # 初始化数据属性
        
        # 创建MACD策略
        self.macd_strategy = MACDStrategy(
            fast_period=macd_fast_period,
            slow_period=macd_slow_period,
            signal_period=macd_signal_period,
            stop_loss_pct=None  # 在混合策略中统一管理止损
        )
        
        # 创建增强型LSTM策略
        self.lstm_strategy = EnhancedLSTMStrategy(
            sequence_length=lstm_sequence_length,
            hidden_dim=lstm_hidden_dim,
            prediction_threshold=lstm_prediction_threshold,
            feature_engineering=lstm_feature_engineering,
            use_attention=lstm_use_attention,
            model_path=lstm_model_path,
            stop_loss_pct=None  # 在混合策略中统一管理止损
        )
        
        # 初始化状态变量
        self._market_state = None  # 市场状态：'uptrend', 'downtrend', 'sideways'
        self._entry_prices = {}  # 记录入场价格，用于止损止盈
        
        # 生成策略名称
        macd_name = f"MACD({macd_fast_period},{macd_slow_period},{macd_signal_period})"
        lstm_name = f"LSTM({lstm_sequence_length},{lstm_prediction_threshold:.2%})"
        
        if ensemble_method == 'vote':
            method_str = "Vote"
        elif ensemble_method == 'weight':
            method_str = f"W({ensemble_weights[0]:.1f},{ensemble_weights[1]:.1f})"
        elif ensemble_method == 'layered':
            method_str = "Layer"
        elif ensemble_method == 'expert':
            method_str = "Expert"
        
        self.name = f"Hybrid_{macd_name}_{lstm_name}_{method_str}"
        
        if stop_loss_pct is not None:
            self.name += f"_SL{stop_loss_pct:.1%}"
        if take_profit_pct is not None:
            self.name += f"_TP{take_profit_pct:.1%}"
        
        logger.info(f"混合策略初始化完成: {self.name}")
        logger.info(f"组合方法: {ensemble_method}, "
                   f"MACD权重: {ensemble_weights[0]}, "
                   f"LSTM权重: {ensemble_weights[1]}")
    
    def prepare(self, df: pd.DataFrame) -> None:
        """
        准备策略，确保LSTM模型已训练
        
        Args:
            df: 历史数据DataFrame
        """
        # 训练LSTM模型（如果需要）
        if not hasattr(self.lstm_strategy, '_is_trained') or not self.lstm_strategy._is_trained:
            logger.info("训练LSTM模型...")
            # 使用70%的数据进行训练
            train_size = int(len(df) * 0.7)
            train_df = df.iloc[:train_size]
            self.lstm_strategy.train(train_df)
    
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        检测市场状态：使用多种技术指标进行更精确的市场状态判断
        
        Args:
            df: 历史数据DataFrame
            
        Returns:
            str: 市场状态 - 'trending_volatile', 'ranging_volatile', 'trending_stable', 'ranging_stable'
        """
        try:
            # 导入技术指标计算类
            from ...indicators.technical_indicators import TechnicalIndicators
            
            # 使用技术指标模块进行市场状态判断
            market_regime = TechnicalIndicators.identify_market_regime(
                df, 
                volatility_threshold=self.market_regime_threshold,
                trend_threshold=25,  # ADX阈值
                fast_ma_period=10,
                slow_ma_period=self.moving_average_period
            )
            
            logger.info(f"识别到的市场状态: {market_regime}")
            return market_regime
            
        except ImportError:
            logger.warning("技术指标模块导入失败，使用基础市场状态判断")
            # 回退到基础判断逻辑
            return self._basic_market_regime_detection(df)
        except Exception as e:
            logger.error(f"市场状态判断出错: {str(e)}")
            # 回退到基础判断逻辑
            return self._basic_market_regime_detection(df)
    
    def _basic_market_regime_detection(self, df: pd.DataFrame) -> str:
        """
        基础的市场状态检测逻辑
        
        Args:
            df: 历史数据DataFrame
            
        Returns:
            str: 市场状态 - 'uptrend', 'downtrend', 'sideways'
        """
        # 计算移动平均线
        if f'ma_{self.moving_average_period}' not in df.columns:
            df[f'ma_{self.moving_average_period}'] = df['close'].rolling(window=self.moving_average_period).mean()
        
        # 计算最近价格与移动平均线的偏差
        last_price = df['close'].iloc[-1]
        last_ma = df[f'ma_{self.moving_average_period}'].iloc[-1]
        deviation = (last_price - last_ma) / last_ma
        
        # 计算最近的价格波动率
        volatility = df['close'].pct_change().rolling(window=20).std().iloc[-1]
        
        # 根据偏差和波动率判断市场状态
        if deviation > self.market_regime_threshold:
            return 'uptrend'
        elif deviation < -self.market_regime_threshold:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_market_features(self, df: pd.DataFrame, row_index: int) -> dict:
        """
        计算当前时间点的市场特征
        
        Args:
            df: 历史数据DataFrame
            row_index: 当前行索引
            
        Returns:
            dict: 市场特征字典
        """
        # 如果数据不足，返回基本特征
        if row_index < 50:
            return self._calculate_simple_features(df, row_index)
        
        # 尝试计算更复杂的市场特征
        try:
            from ...indicators.technical_indicators import TechnicalIndicators
            
            # 获取历史数据窗口
            hist_window = df.iloc[:row_index+1].copy()
            
            # 识别市场状态
            market_regime = TechnicalIndicators.identify_market_regime(hist_window)
            
            # 将市场状态转换为数值表示
            regime_map = {
                'trending_volatile': 3, 
                'ranging_volatile': 2, 
                'trending_stable': 1, 
                'ranging_stable': 0,
                'unknown': -1
            }
            market_regime_num = regime_map.get(market_regime, -1)
            
            # 计算其他市场特征
            features = TechnicalIndicators.get_market_features(hist_window)
            
            # 确保所有特征值都是数值类型
            for key, value in list(features.items()):
                if not isinstance(value, (int, float, np.number)):
                    # 如果是非数值类型，尝试转换或移除
                    if key == 'market_regime':
                        # 将市场状态转换为数值
                        features['market_regime_num'] = market_regime_num
                        del features[key]
                    else:
                        # 对于其他非数值特征，直接移除
                        del features[key]
            
            # 添加最重要的市场状态数值表示
            features['market_regime_num'] = market_regime_num
            
            return features
            
        except ImportError:
            logger.warning("无法导入TechnicalIndicators，将使用简单特征")
            return self._calculate_simple_features(df, row_index)
        except Exception as e:
            logger.error(f"计算市场特征时出错: {str(e)}")
            return self._calculate_simple_features(df, row_index)
    
    def _calculate_simple_features(self, df: pd.DataFrame, row_index: int) -> dict:
        """
        计算简单的市场特征
        
        Args:
            df: 历史数据DataFrame
            row_index: 当前行索引
            
        Returns:
            dict: 市场特征字典
        """
        features = {}
        
        # 确保数据点足够
        if row_index < 5:
            features['volatility'] = 0.01  # 默认低波动率
            return features
            
        # 计算波动率
        features['volatility'] = df['close'].iloc[max(0, row_index-20):row_index+1].pct_change().std()
        
        # 计算短期和长期移动平均
        if row_index >= 10:
            short_ma = df['close'].iloc[row_index-9:row_index+1].mean()
            if row_index >= 50:
                long_ma = df['close'].iloc[row_index-49:row_index+1].mean()
                features['ma_trend'] = 1 if short_ma > long_ma else -1
        
        return features
    
    def _vote_ensemble(self, macd_signal: float, lstm_signal: float) -> float:
        """
        投票组合法：只有当两个策略信号一致时才产生交易信号
        
        Args:
            macd_signal: MACD信号
            lstm_signal: LSTM信号
            
        Returns:
            float: 组合信号
        """
        if macd_signal == lstm_signal and macd_signal != 0:
            return macd_signal
        return 0.0
    
    def _weighted_ensemble(self, macd_signal: float, lstm_signal: float) -> float:
        """
        加权组合法：根据权重组合信号
        
        Args:
            macd_signal: MACD信号
            lstm_signal: LSTM信号
            
        Returns:
            float: 组合信号
        """
        # 计算加权信号
        weighted_signal = (
            self.ensemble_weights[0] * macd_signal +
            self.ensemble_weights[1] * lstm_signal
        )
        
        # 信号确定性级别
        if abs(weighted_signal) >= 0.5:
            return 1.0 if weighted_signal > 0 else -1.0
        else:
            return 0.0
    
    def _layered_ensemble(self, macd_signal: float, lstm_signal: float, row_index: int, df: pd.DataFrame) -> float:
        """
        分层决策法：MACD决定方向，LSTM决定入场时机
        
        Args:
            macd_signal: MACD信号
            lstm_signal: LSTM信号
            row_index: 当前行索引
            df: 历史数据DataFrame
            
        Returns:
            float: 组合信号
        """
        if row_index < 1:
            return 0.0
        
        # 获取之前的MACD趋势
        prev_macd_hist = df['macd_hist'].iloc[row_index-1]
        curr_macd_hist = df['macd_hist'].iloc[row_index]
        
        # MACD直方图上穿零轴，确认上升趋势
        if prev_macd_hist <= 0 and curr_macd_hist > 0:
            # 当LSTM也看涨时确认买入
            if lstm_signal > 0:
                return 1.0
        
        # MACD直方图下穿零轴，确认下降趋势
        elif prev_macd_hist >= 0 and curr_macd_hist < 0:
            # 当LSTM也看跌时确认卖出
            if lstm_signal < 0:
                return -1.0
        
        # 保持当前持仓
        return 0.0
    
    def _expert_ensemble(self, macd_signal: float, lstm_signal: float, market_state: str) -> float:
        """
        改进的专家系统法：根据更精确的市场状态动态选择最适合的策略
        
        Args:
            macd_signal: MACD信号
            lstm_signal: LSTM信号
            market_state: 市场状态
            
        Returns:
            float: 组合信号
        """
        # 获取适合当前市场状态的权重
        macd_weight, lstm_weight = self._get_adaptive_weights(market_state)
        
        logger.debug(f"市场状态: {market_state}, MACD权重: {macd_weight}, LSTM权重: {lstm_weight}")
        
        # 应用加权组合
        weighted_signal = macd_weight * macd_signal + lstm_weight * lstm_signal
        
        # 信号确定性级别 - 权重和信号强度都需要足够
        if abs(weighted_signal) >= 0.3:
            return 1.0 if weighted_signal > 0 else -1.0
        
        # 两个信号一致且非零时也生成交易信号
        if macd_signal == lstm_signal and macd_signal != 0:
            return macd_signal
            
        # 信号不确定时保持观望
        return 0.0
    
    def _get_adaptive_weights(self, market_state: str) -> Tuple[float, float]:
        """
        根据市场状态自适应调整策略权重
        
        Args:
            market_state: 市场状态
            
        Returns:
            Tuple[float, float]: (MACD权重, LSTM权重)
        """
        # 针对不同市场状态优化权重配置
        if market_state in ('trending_volatile', 'ranging_volatile', 'trending_stable', 'ranging_stable'):
            # 新的四分类市场状态
            weights = {
                'trending_volatile': (0.3, 0.7),  # 波动趋势市场优先LSTM
                'ranging_volatile': (0.2, 0.8),   # 波动震荡市场强依赖LSTM
                'trending_stable': (0.7, 0.3),    # 稳定趋势市场优先MACD
                'ranging_stable': (0.5, 0.5)      # 稳定震荡市场平衡配置
            }
            return weights.get(market_state, self.ensemble_weights)
            
        else:
            # 兼容旧版市场状态
            weights = {
                'uptrend': (0.4, 0.6),      # 上升趋势市场，LSTM稍占优
                'downtrend': (0.6, 0.4),    # 下降趋势市场，MACD稍占优
                'sideways': (0.5, 0.5)      # 横盘市场，平衡配置
            }
            return weights.get(market_state, self.ensemble_weights)
    
    def generate_signals(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        生成混合策略信号
        
        Args:
            df: 历史数据DataFrame，如果为None则使用self.data
            
        Returns:
            pd.DataFrame: 带有信号的DataFrame
        """
        # 如果没有提供数据，使用实例的data属性
        if df is None:
            df = self.data
            
        if df is None:
            logger.error("没有提供数据，无法生成信号")
            return None
        
        try:
            # 复制数据以避免修改原始数据
            df = df.copy()
            
            # 检测市场状态
            self._market_state = self._detect_market_regime(df)
            logger.info(f"当前市场状态: {self._market_state}")
            
            # 确保基础列存在
            if 'close' not in df.columns:
                logger.error("DataFrame中缺少'close'列")
                return None
            
            # 生成MACD策略信号
            logger.info("生成MACD策略信号...")
            macd_df = self.macd_strategy.generate_signals(df)
            
            # 生成LSTM策略信号
            logger.info("生成LSTM策略信号...")
            lstm_df = self.lstm_strategy.generate_signals(df)
            
            # 合并信号到原始DataFrame
            if 'macd' in macd_df.columns:
                df['macd'] = macd_df['macd']
            if 'macd_signal' in macd_df.columns:
                df['macd_signal'] = macd_df['macd_signal']
            if 'macd_hist' in macd_df.columns:
                df['macd_hist'] = macd_df['macd_hist']
            if 'position' in macd_df.columns:
                df['macd_position'] = macd_df['position']
            if 'position' in lstm_df.columns:
                df['lstm_position'] = lstm_df['position']
            if 'predicted_close' in lstm_df.columns:
                df['predicted_close'] = lstm_df['predicted_close']
            
            # 检查是否成功合并了信号
            if 'macd_position' not in df.columns:
                logger.warning("无法合并MACD仓位信号")
                df['macd_position'] = 0
            if 'lstm_position' not in df.columns:
                logger.warning("无法合并LSTM仓位信号")
                df['lstm_position'] = 0
            
            # 根据组合方法生成最终信号
            logger.info(f"使用{self.ensemble_method}方法组合信号...")
            signals = []
            
            for i in range(len(df)):
                macd_signal = df['macd_position'].iloc[i]
                lstm_signal = df['lstm_position'].iloc[i]
                
                # 获取当前市场特征
                market_features = self._calculate_market_features(df, i)
                
                # 存储市场特征
                for key, value in market_features.items():
                    if f'market_{key}' not in df.columns:
                        df[f'market_{key}'] = None
                    df.at[df.index[i], f'market_{key}'] = value
                
                # 获取当前市场状态
                current_market_state = market_features.get('market_regime', self._market_state)
                
                # 基于选择的组合方法组合信号
                if self.ensemble_method == 'vote':
                    signal = self._vote_ensemble(macd_signal, lstm_signal)
                elif self.ensemble_method == 'weight':
                    signal = self._weighted_ensemble(macd_signal, lstm_signal)
                elif self.ensemble_method == 'layered':
                    signal = self._layered_ensemble(macd_signal, lstm_signal, i, df)
                elif self.ensemble_method == 'expert':
                    signal = self._expert_ensemble(macd_signal, lstm_signal, current_market_state)
                else:
                    logger.warning(f"未知的组合方法: {self.ensemble_method}，使用投票法")
                    signal = self._vote_ensemble(macd_signal, lstm_signal)
                
                signals.append(signal)
            
            # 添加信号到DataFrame
            df['signal'] = signals
            
            # 生成仓位
            df['position'] = df['signal']
            
            # 应用风险管理
            logger.info("应用风险管理规则...")
            if hasattr(self, 'risk_manager') and self.risk_manager is not None:
                # 使用高级风险管理器
                logger.info("使用高级风险管理器...")
                try:
                    df = self.risk_manager.apply_risk_management(df, 'position')
                    
                    # 检查是否添加了position_size列
                    if 'position_size' not in df.columns and hasattr(self.risk_manager, 'calculate_position_size'):
                        # 计算头寸大小
                        position_sizes = []
                        for i in range(len(df)):
                            # 获取当前权益值，如果不存在则使用初始资金
                            current_equity = df['equity_curve'].iloc[i] if 'equity_curve' in df.columns and i > 0 else self.risk_manager.initial_capital
                            pos_size = self.risk_manager.calculate_position_size(current_equity, df, i)
                            position_sizes.append(pos_size)
                        
                        df['position_size'] = position_sizes
                        logger.info("已添加头寸大小列")
                    
                    logger.info("高级风险管理应用完成")
                except Exception as e:
                    logger.error(f"应用高级风险管理时出错: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # 回退到基本风险管理
                    logger.info("回退到基本风险管理...")
                    df = self._apply_risk_management(df)
            else:
                # 使用基本风险管理
                logger.info("使用基本风险管理...")
                df = self._apply_risk_management(df)
            
            return df
            
        except Exception as e:
            logger.error(f"生成信号时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _apply_risk_management(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用风险管理：止损和止盈
        
        Args:
            df: 带有仓位的DataFrame
            
        Returns:
            pd.DataFrame: 应用了风险管理的DataFrame
        """
        # 重置入场价格记录
        self._entry_prices = {}
        
        # 如果没有足够的数据，直接返回
        if len(df) <= 1:
            logger.warning("数据不足，无法应用风险管理")
            return df
        
        try:
            # 记录入场价格并应用止损止盈
            for i in range(1, len(df)):
                if i >= len(df):
                    break
                    
            current_idx = df.index[i]
            previous_idx = df.index[i-1]
            
            # 检测仓位变化
            if df['position'].iloc[i] != df['position'].iloc[i-1] and df['position'].iloc[i] != 0:
                # 新开仓，记录入场价格
                    # 根据仓位方向存储入场价格，对于空头仓位，将价格存为负数以便于后续识别
                    if df['position'].iloc[i] > 0:  # 多头仓位
                        self._entry_prices[current_idx] = df['close'].iloc[i]
                    else:  # 空头仓位
                        self._entry_prices[current_idx] = -df['close'].iloc[i]  # 用负值标记空头入场价格
                    
                    logger.info(f"新开仓: 日期={current_idx}, 价格={df['close'].iloc[i]:.2f}, 仓位={df['position'].iloc[i]}")
            
            # 如果持有多头仓位
            if df['position'].iloc[i] > 0:
                # 寻找最近的入场价格
                entry_price = None
                for date in reversed(list(self._entry_prices.keys())):
                    if date <= current_idx and self._entry_prices[date] > 0:
                        entry_price = self._entry_prices[date]
                        break
                
                if entry_price is not None:
                    current_price = df['close'].iloc[i]
                    price_change = (current_price - entry_price) / entry_price
                    
                    # 应用止损
                    if self.stop_loss_pct is not None and price_change < -self.stop_loss_pct:
                        df.loc[current_idx, 'position'] = 0
                        logger.info(f"触发止损: 日期={current_idx}, 入场价={entry_price:.2f}, 当前价={current_price:.2f}, 跌幅={price_change:.2%}")
                    
                    # 应用止盈
                    elif self.take_profit_pct is not None and price_change > self.take_profit_pct:
                        df.loc[current_idx, 'position'] = 0
                        logger.info(f"触发止盈: 日期={current_idx}, 入场价={entry_price:.2f}, 当前价={current_price:.2f}, 涨幅={price_change:.2%}")
            
            # 如果持有空头仓位
            elif df['position'].iloc[i] < 0:
                # 寻找最近的入场价格
                entry_price = None
                for date in reversed(list(self._entry_prices.keys())):
                    if date <= current_idx and self._entry_prices[date] < 0:
                            entry_price = abs(self._entry_prices[date])  # 转回正值用于计算
                            break
                
                if entry_price is not None:
                    current_price = df['close'].iloc[i]
                    price_change = (entry_price - current_price) / entry_price
                    
                    # 应用止损
                    if self.stop_loss_pct is not None and price_change < -self.stop_loss_pct:
                        df.loc[current_idx, 'position'] = 0
                        logger.info(f"触发止损: 日期={current_idx}, 入场价={entry_price:.2f}, 当前价={current_price:.2f}, 上涨={-price_change:.2%}")
                    
                    # 应用止盈
                    elif self.take_profit_pct is not None and price_change > self.take_profit_pct:
                        df.loc[current_idx, 'position'] = 0
                        logger.info(f"触发止盈: 日期={current_idx}, 入场价={entry_price:.2f}, 当前价={current_price:.2f}, 下跌={price_change:.2%}")
        except Exception as e:
            logger.error(f"应用风险管理时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return df
    
    def visualize_signals(self, df: pd.DataFrame, title: str = None) -> plt.Figure:
        """
        可视化混合策略信号
        
        Args:
            df: 包含信号的DataFrame
            title: 图表标题
            
        Returns:
            plt.Figure: 图表对象
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # 1. 价格图
        axes[0].plot(df.index, df['close'], label='价格')
        axes[0].set_title('价格走势')
        
        # 标记买入和卖出信号
        buy_signals = df[df['signal'] > 0].index
        sell_signals = df[df['signal'] < 0].index
        
        axes[0].plot(buy_signals, df.loc[buy_signals, 'close'], '^', markersize=10, color='g', alpha=0.7, label='买入信号')
        axes[0].plot(sell_signals, df.loc[sell_signals, 'close'], 'v', markersize=10, color='r', alpha=0.7, label='卖出信号')
        
        # 2. MACD图
        axes[1].plot(df.index, df['macd'], label='MACD')
        axes[1].plot(df.index, df['macd_signal'], label='MACD信号')
        axes[1].bar(df.index, df['macd_hist'], label='MACD柱状图', alpha=0.5)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1].set_title('MACD指标')
        
        # 3. LSTM预测图
        if 'predicted_close' in df.columns:
            axes[2].plot(df.index, df['close'], label='实际价格')
            axes[2].plot(df.index, df['predicted_close'], label='预测价格', linestyle='--')
            axes[2].set_title('LSTM价格预测')
        else:
            axes[2].plot(df.index, df['lstm_position'], label='LSTM仓位')
            axes[2].set_title('LSTM策略仓位')
        
        # 4. 混合策略仓位图
        axes[3].plot(df.index, df['position'], label='混合策略仓位')
        
        # 检查列是否存在，然后再绘制
        if 'macd_position' in df.columns:
            axes[3].plot(df.index, df['macd_position'], label='MACD仓位', linestyle='--', alpha=0.7)
        elif 'position' in df.columns and hasattr(self, 'macd_strategy'):
            # 尝试使用普通的position列
            logger.warning("找不到macd_position列，尝试使用替代方法显示MACD仓位")
        
        if 'lstm_position' in df.columns:
            axes[3].plot(df.index, df['lstm_position'], label='LSTM仓位', linestyle=':', alpha=0.7)
        
        axes[3].set_title('策略仓位对比')
        axes[3].set_ylim(-1.2, 1.2)
        
        # 设置图表属性
        for ax in axes:
            ax.legend(loc='best')
            ax.grid(True)
        
        # 设置总标题
        if title is None:
            title = f'混合策略信号: {self.name}'
        
        # 使用fig.suptitle替代错误的set_chinese_title(fig, title)调用
        if font_helper.has_chinese_font:
            fig.suptitle(title, fontproperties=FontProperties(fname=font_helper.chinese_font), fontsize=14)
        else:
            fig.suptitle(title, fontsize=14)
            
        # 应用字体到图形
        font_helper.apply_font_to_figure(fig)
        
        plt.tight_layout()
        return fig
    
    def run(self, visualize: bool = True) -> tuple:
        """
        运行混合策略
        
        Args:
            visualize: 是否可视化结果
            
        Returns:
            tuple: (results_df, performance_metrics)
        """
        try:
            logger.info("开始运行MACD-LSTM混合策略...")
        # 生成信号
            try:
                signals_df = self.generate_signals()
                if signals_df is None:
                    logger.error("生成的信号DataFrame为None")
                    return pd.DataFrame(), {}
                
                if len(signals_df) == 0:
                    logger.error("生成的信号DataFrame为空")
                    return pd.DataFrame(), {}
                    
                logger.info(f"成功生成信号，数据长度: {len(signals_df)}")
                logger.debug(f"信号DataFrame列: {signals_df.columns.tolist()}")
                
                # 检查必需的列是否存在
                required_columns = ['close', 'position']
                missing_columns = [col for col in required_columns if col not in signals_df.columns]
                if missing_columns:
                    logger.error(f"信号DataFrame缺少必需的列: {missing_columns}")
                    return pd.DataFrame(), {}
            except Exception as e:
                logger.error(f"生成信号时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return pd.DataFrame(), {}
        
        # 可视化信号
            try:
                if visualize:
                    self.visualize_signals(signals_df, self.output_dir)
                    logger.info(f"信号可视化已保存到 {self.output_dir}")
            except Exception as e:
                logger.warning(f"可视化信号时出错: {str(e)}")
                # 继续执行，可视化问题不应该中断策略运行
            
            # 复制数据以避免修改原始数据
            results_df = signals_df.copy()
            
            # 检查并记录状态
            if 'position' not in results_df.columns:
                logger.error("缺少'position'列，无法计算回报")
                results_df['position'] = 0
            
            if 'close' not in results_df.columns:
                logger.error("缺少'close'列，无法计算回报")
                return pd.DataFrame(), {}
            
            # 计算每日收益率
            try:
                results_df['daily_returns'] = results_df['close'].pct_change()
                
                # 检查是否有无效的日收益率
                invalid_returns = results_df['daily_returns'].isna().sum()
                if invalid_returns > 1:  # 第一个值通常为NaN，这是正常的
                    logger.warning(f"日收益率中有 {invalid_returns} 个无效值")
                
                # 填充NaN值以避免后续计算问题
                results_df['daily_returns'] = results_df['daily_returns'].fillna(0)
            except Exception as e:
                logger.error(f"计算每日收益率时出错: {str(e)}")
                results_df['daily_returns'] = 0
            
            # 计算策略收益率(未扣除手续费)
            try:
                # 确保position列只包含[-1, 0, 1]
                valid_positions = results_df['position'].isin([-1, 0, 1]).all()
                if not valid_positions:
                    logger.warning("position列包含[-1, 0, 1]以外的值")
                    results_df['position'] = results_df['position'].clip(-1, 1)
                
                # 适当移位position，使得今天的position影响明天的收益
                results_df['strategy_returns'] = results_df['position'].shift(1) * results_df['daily_returns']
                results_df['strategy_returns'] = results_df['strategy_returns'].fillna(0)
                
                # 记录一些统计数据以帮助调试
                num_trades = (results_df['position'].diff() != 0).sum()
                logger.debug(f"总交易次数: {num_trades}")
                long_trades = ((results_df['position'].shift(1) == 0) & (results_df['position'] == 1)).sum()
                short_trades = ((results_df['position'].shift(1) == 0) & (results_df['position'] == -1)).sum()
                logger.debug(f"做多交易: {long_trades}, 做空交易: {short_trades}")
            except Exception as e:
                logger.error(f"计算策略收益率时出错: {str(e)}")
                results_df['strategy_returns'] = 0
            
            # 计算手续费
            try:
                # 检测交易信号变化来判定是否发生交易
                results_df['trade'] = results_df['position'].diff() != 0
                # 对于每次交易，手续费为交易金额的一个百分比
                results_df['commission'] = np.where(results_df['trade'], abs(results_df['close'] * self.commission), 0)
                # 将手续费转换为收益率形式
                results_df['commission_rate'] = results_df['commission'] / results_df['close']
                # 扣除手续费后的策略收益率
                results_df['strategy_returns_after_commission'] = results_df['strategy_returns'] - results_df['commission_rate']
                
                # 检查手续费计算是否合理
                commission_sum = results_df['commission_rate'].sum()
                if commission_sum > 0.1:  # 总手续费超过10%警告
                    logger.warning(f"总手续费率看起来异常高: {commission_sum:.4f}")
            except Exception as e:
                logger.error(f"计算手续费时出错: {str(e)}")
                results_df['commission'] = 0
                results_df['commission_rate'] = 0
                results_df['strategy_returns_after_commission'] = results_df['strategy_returns']
            
            # 计算累积收益率
            try:
                # 初始资金为1
                initial_capital = 1.0
                results_df['cumulative_strategy_returns'] = (1 + results_df['strategy_returns_after_commission']).cumprod()
                
                # 计算资金曲线，这是回测引擎必需的
                results_df['equity_curve'] = initial_capital * results_df['cumulative_strategy_returns']
                
                # 添加 cumulative_returns 列以满足可视化器的需求
                results_df['cumulative_returns'] = (1 + results_df['daily_returns']).cumprod()
                
                # 前一个峰值，用于计算回撤
                results_df['previous_peaks'] = results_df['equity_curve'].cummax()
                
                # 检查最终资金是否合理
                final_capital = results_df['cumulative_strategy_returns'].iloc[-1]
                if pd.isna(final_capital):
                    logger.error("最终资金值为NaN")
                    results_df['cumulative_strategy_returns'] = results_df['cumulative_strategy_returns'].fillna(initial_capital)
                    results_df['equity_curve'] = results_df['equity_curve'].fillna(initial_capital)
                    results_df['previous_peaks'] = results_df['previous_peaks'].fillna(initial_capital)
                    results_df['cumulative_returns'] = results_df['cumulative_returns'].fillna(1.0)
                    final_capital = initial_capital
                
                logger.info(f"起始资金: {initial_capital:.2f}, 最终资金: {final_capital:.2f}")
                logger.info(f"总收益率: {((final_capital/initial_capital)-1)*100:.2f}%")
            except Exception as e:
                logger.error(f"计算累积收益率时出错: {str(e)}")
                results_df['cumulative_strategy_returns'] = 1.0
                results_df['equity_curve'] = initial_capital
                results_df['previous_peaks'] = initial_capital
                results_df['cumulative_returns'] = 1.0
            
            # 计算回撤
            try:
                results_df['equity_peak'] = results_df['cumulative_strategy_returns'].cummax()
                results_df['drawdown'] = results_df['cumulative_strategy_returns'] / results_df['equity_peak'] - 1
                
                # 检查最大回撤
                max_drawdown = results_df['drawdown'].min()
                max_drawdown_time = results_df.loc[results_df['drawdown'] == max_drawdown].index[0] if max_drawdown < 0 else None
                if max_drawdown < -0.2:  # 回撤超过20%
                    logger.warning(f"最大回撤过大: {max_drawdown:.2%} at {max_drawdown_time}")
                else:
                    logger.info(f"最大回撤: {max_drawdown:.2%}")
            except Exception as e:
                logger.error(f"计算回撤时出错: {str(e)}")
                results_df['equity_peak'] = 1.0
                results_df['drawdown'] = 0.0
            
            # 添加买入和卖出点的标记
            try:
                results_df['buy_signal'] = ((results_df['position'].shift(1) <= 0) & (results_df['position'] > 0)).astype(int)
                results_df['sell_signal'] = ((results_df['position'].shift(1) >= 0) & (results_df['position'] < 0)).astype(int)
                
                buy_count = results_df['buy_signal'].sum()
                sell_count = results_df['sell_signal'].sum()
                logger.info(f"买入信号数: {buy_count}, 卖出信号数: {sell_count}")
            except Exception as e:
                logger.error(f"添加买入卖出点标记时出错: {str(e)}")
                results_df['buy_signal'] = 0
                results_df['sell_signal'] = 0
            
            # 计算性能指标
            try:
                performance_metrics = self._calculate_performance_metrics(results_df)
                # 记录关键性能指标
                logger.info(f"策略性能指标:")
                logger.info(f"  总收益率: {performance_metrics['total_return']:.2%}")
                logger.info(f"  年化收益率: {performance_metrics['annual_return']:.2%}")
                logger.info(f"  最大回撤: {performance_metrics['max_drawdown']:.2%}")
                logger.info(f"  夏普比率: {performance_metrics['sharpe_ratio']:.2f}")
                logger.info(f"  胜率: {performance_metrics['win_rate']:.2f}")
            except Exception as e:
                logger.error(f"记录性能指标时出错: {str(e)}")
                performance_metrics = self._get_default_performance_metrics()
            
            # 最终检查
            if len(results_df) == 0:
                logger.error("结果DataFrame为空，无法完成策略回测")
                return pd.DataFrame(), {}
            
            logger.info("MACD-LSTM混合策略运行完成")
            return results_df, performance_metrics
            
        except Exception as e:
            logger.error(f"运行策略时发生未捕获的异常: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame(), self._get_default_performance_metrics()
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> dict:
        """
        计算策略表现指标
        
        Args:
            df: 回测结果DataFrame
            
        Returns:
            dict: 包含性能指标的字典
        """
        try:
            # 检查数据框是否有效
            if df is None or len(df) == 0:
                logger.error("无法计算性能指标：数据框为空")
                return self._get_default_performance_metrics()
                
            # 检查必需的列是否存在
            required_columns = ['cumulative_strategy_returns', 'drawdown', 'strategy_returns_after_commission']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"无法计算性能指标：缺少必需的列 {missing_columns}")
                return self._get_default_performance_metrics()
            
            # 年化收益率
            try:
                total_days = (df.index[-1] - df.index[0]).days
                if total_days <= 0:
                    logger.error("计算年化收益率失败：数据周期小于等于0天")
                    annual_return = 0
                else:
                    last_cum_return = df['cumulative_strategy_returns'].iloc[-1]
                    if pd.isna(last_cum_return) or last_cum_return <= 0:
                        logger.warning("累积收益率非正值，无法计算年化收益率")
                        annual_return = 0
                    else:
                        annual_return = (last_cum_return ** (365 / total_days)) - 1
            except Exception as e:
                logger.error(f"计算年化收益率时出错: {str(e)}")
                annual_return = 0
            
            # 最大回撤
            try:
                max_drawdown = df['drawdown'].min()
                if pd.isna(max_drawdown):
                    logger.warning("计算最大回撤失败：值为NaN")
                    max_drawdown = 0
            except Exception as e:
                logger.error(f"计算最大回撤时出错: {str(e)}")
                max_drawdown = 0
            
            # 夏普比率 (假设无风险利率为0.02)
            try:
                risk_free_rate = 0.02
                returns_std = df['strategy_returns_after_commission'].std()
                returns_mean = df['strategy_returns_after_commission'].mean()
                
                if pd.isna(returns_std) or returns_std <= 0 or pd.isna(returns_mean):
                    logger.warning("计算夏普比率失败：标准差为0或NaN")
                    sharpe_ratio = 0
                else:
                    sharpe_ratio = ((returns_mean * 252) - risk_free_rate) / (returns_std * np.sqrt(252))
            except Exception as e:
                logger.error(f"计算夏普比率时出错: {str(e)}")
                sharpe_ratio = 0
            
            # 卡尔玛比率
            try:
                if max_drawdown >= 0 or abs(max_drawdown) < 0.0001:
                    logger.warning("计算卡尔玛比率失败：最大回撤接近0或为正值")
                    calmar_ratio = 0
                else:
                    calmar_ratio = annual_return / abs(max_drawdown)
            except Exception as e:
                logger.error(f"计算卡尔玛比率时出错: {str(e)}")
                calmar_ratio = 0
            
            # 胜率
            try:
                winning_trades = (df['strategy_returns_after_commission'] > 0).sum()
                losing_trades = (df['strategy_returns_after_commission'] < 0).sum()
                total_trades = winning_trades + losing_trades
                
                if total_trades <= 0:
                    logger.warning("计算胜率失败：无交易记录")
                    win_rate = 0
                else:
                    win_rate = winning_trades / total_trades
            except Exception as e:
                logger.error(f"计算胜率时出错: {str(e)}")
                winning_trades = 0
                losing_trades = 0
                win_rate = 0
            
            # 盈亏比
            try:
                if winning_trades > 0:
                    average_win = df.loc[df['strategy_returns_after_commission'] > 0, 'strategy_returns_after_commission'].mean()
                    if pd.isna(average_win):
                        average_win = 0
                else:
                    average_win = 0
                    
                if losing_trades > 0:
                    average_loss = abs(df.loc[df['strategy_returns_after_commission'] < 0, 'strategy_returns_after_commission'].mean())
                    if pd.isna(average_loss):
                        average_loss = 1
                else:
                    average_loss = 1
                    
                if average_loss > 0:
                    profit_loss_ratio = average_win / average_loss
                else:
                    logger.warning("计算盈亏比失败：平均亏损为0")
                    profit_loss_ratio = 0
            except Exception as e:
                logger.error(f"计算盈亏比时出错: {str(e)}")
                profit_loss_ratio = 0
            
            # 总收益
            try:
                last_cum_return = df['cumulative_strategy_returns'].iloc[-1]
                if pd.isna(last_cum_return):
                    logger.warning("计算总收益率失败：累积收益率为NaN")
                    total_return = 0
                else:
                    total_return = last_cum_return - 1
            except Exception as e:
                logger.error(f"计算总收益率时出错: {str(e)}")
                total_return = 0
            
            # 返回经过处理的指标
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
        except Exception as e:
            logger.error(f"计算性能指标时发生未处理的异常: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return self._get_default_performance_metrics()
    
    def _get_default_performance_metrics(self) -> dict:
        """
        返回默认的性能指标字典
        
        Returns:
            dict: 包含默认值的性能指标字典
        """
        return {
            "annual_return": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0,
            "winning_trades": 0,
            "losing_trades": 0
        } 