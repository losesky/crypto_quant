"""
特征工程模块，生成高级特征，如交易信号和市场情绪指标
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ...utils.logger import logger


class FeatureEngineering:
    """
    特征工程类，用于从基础数据生成高级特征
    """

    def __init__(self):
        """
        初始化特征工程实例
        """
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        logger.info("特征工程模块初始化完成")

    def generate_trading_signals(self, df):
        """
        生成交易信号特征

        Args:
            df (pandas.DataFrame): 带有技术指标的DataFrame

        Returns:
            pandas.DataFrame: 添加交易信号特征的DataFrame
        """
        # 复制数据避免修改原始数据
        df = df.copy()
        
        # 检查必要的列是否存在
        required_cols = ['close', 'ma_20', 'ma_50', 'rsi_14', 'macd', 'macd_signal']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"缺少生成交易信号所需的列: {missing_cols}")
            return df
        
        # 1. 移动平均线信号
        df['ma_crossover'] = np.where(
            (df['ma_20'] > df['ma_50']) & (df['ma_20'].shift(1) <= df['ma_50'].shift(1)),
            1, 0  # 1表示金叉信号（买入）
        )
        
        df['ma_crossunder'] = np.where(
            (df['ma_20'] < df['ma_50']) & (df['ma_20'].shift(1) >= df['ma_50'].shift(1)),
            1, 0  # 1表示死叉信号（卖出）
        )
        
        # 2. RSI超买超卖信号
        df['rsi_oversold'] = np.where(df['rsi_14'] < 30, 1, 0)  # 1表示超卖（买入机会）
        df['rsi_overbought'] = np.where(df['rsi_14'] > 70, 1, 0)  # 1表示超买（卖出机会）
        
        # 3. MACD信号
        df['macd_crossover'] = np.where(
            (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)),
            1, 0  # 1表示MACD金叉信号（买入）
        )
        
        df['macd_crossunder'] = np.where(
            (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)),
            1, 0  # 1表示MACD死叉信号（卖出）
        )
        
        # 4. 布林带信号
        if all(col in df.columns for col in ['bb_lower', 'bb_upper']):
            df['bb_breakout_up'] = np.where(
                (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1)),
                1, 0  # 1表示向上突破上轨（可能继续上涨或超买）
            )
            
            df['bb_breakout_down'] = np.where(
                (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1)),
                1, 0  # 1表示向下突破下轨（可能继续下跌或超卖）
            )
            
            df['bb_squeeze'] = np.where(
                (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] < 0.1,
                1, 0  # 1表示布林带收窄（可能即将剧烈波动）
            )
        
        # 5. 价格动量信号
        df['price_momentum'] = df['close'].pct_change(5)  # 5日价格动量
        df['volume_momentum'] = df['volume'].pct_change(5)  # 5日成交量动量
        
        # 6. 综合买入信号
        df['buy_signal'] = np.where(
            (df['ma_crossover'] == 1) |  # 移动平均线金叉
            (df['macd_crossover'] == 1) |  # MACD金叉
            (df['rsi_oversold'] == 1),  # RSI超卖
            1, 0
        )
        
        # 7. 综合卖出信号
        df['sell_signal'] = np.where(
            (df['ma_crossunder'] == 1) |  # 移动平均线死叉
            (df['macd_crossunder'] == 1) |  # MACD死叉
            (df['rsi_overbought'] == 1),  # RSI超买
            1, 0
        )
        
        # 8. 信号强度（多个指标同时发出信号）
        # 买入信号强度（0-3，数字越大信号越强）
        df['buy_signal_strength'] = df['ma_crossover'] + df['macd_crossover'] + df['rsi_oversold']
        
        # 卖出信号强度（0-3，数字越大信号越强）
        df['sell_signal_strength'] = df['ma_crossunder'] + df['macd_crossunder'] + df['rsi_overbought']
        
        logger.info("交易信号特征生成完成")
        return df

    def calculate_market_sentiment(self, df):
        """
        计算市场情绪指标

        Args:
            df (pandas.DataFrame): 价格和交易数据DataFrame

        Returns:
            pandas.DataFrame: 添加市场情绪指标的DataFrame
        """
        # 复制数据避免修改原始数据
        df = df.copy()
        
        # 1. 恐慌贪婪指数 (简化版)
        # 使用RSI、波动率、价格与移动平均线的关系计算
        if 'rsi_14' in df.columns and 'ma_50' in df.columns and 'volatility_14' in df.columns:
            # RSI贡献 (0-100)
            rsi_contribution = df['rsi_14']
            
            # 价格与MA50偏离度贡献，大的偏离表示可能过度贪婪或恐慌
            # 将偏离度标准化到0-100
            if 'close' in df.columns:
                price_ma_deviation = abs((df['close'] - df['ma_50']) / df['ma_50']) * 100
                price_ma_contribution = 100 - np.minimum(price_ma_deviation, 100)
                
                # 波动率贡献 - 高波动率表示恐慌
                # 将波动率标准化到0-100
                volatility_normalized = df['volatility_14'] / df['volatility_14'].rolling(window=30).max() * 100
                volatility_contribution = 100 - volatility_normalized
                
                # 计算恐慌贪婪指数 (0-100)：0表示极度恐慌，100表示极度贪婪
                df['fear_greed_index'] = (
                    0.5 * rsi_contribution + 
                    0.3 * price_ma_contribution + 
                    0.2 * volatility_contribution
                )
        
        # 2. 市场动量
        if 'price_change' in df.columns:
            # 计算10日平均价格变动方向
            df['price_direction_10d'] = df['price_change'].rolling(window=10).mean()
            
            # 计算价格与短期均线的距离，正值表示价格高于均线
            if all(col in df.columns for col in ['close', 'ma_20']):
                df['price_ma20_gap'] = (df['close'] - df['ma_20']) / df['ma_20']
        
        # 3. 震荡指标 (Choppiness Index)
        # 用于识别市场是处于趋势还是震荡状态
        if all(col in df.columns for col in ['high', 'low', 'close', 'atr_14']):
            # 计算N日真实波动范围总和
            n = 14
            df['true_range_sum'] = df['atr_14'] * n
            
            # 计算N日高点与低点的范围
            df['high_low_range'] = df['high'].rolling(window=n).max() - df['low'].rolling(window=n).min()
            
            # 计算震荡指标
            df['choppiness_index'] = 100 * np.log10(df['true_range_sum'] / df['high_low_range']) / np.log10(n)
            
            # 删除临时列
            df.drop(['true_range_sum', 'high_low_range'], axis=1, inplace=True)
        
        # 4. 价格与200日均线关系 (牛熊市指标)
        if all(col in df.columns for col in ['close', 'ma_200']):
            df['bull_market'] = np.where(df['close'] > df['ma_200'], 1, 0)
            
            # 连续X日收盘价高于/低于200日均线的天数
            df['bull_market_days'] = df['bull_market'].rolling(window=200, min_periods=1).sum()
            df['bear_market_days'] = 200 - df['bull_market_days']
        
        logger.info("市场情绪指标计算完成")
        return df
    
    def normalize_features(self, df, columns=None, method='standard'):
        """
        标准化/归一化特征

        Args:
            df (pandas.DataFrame): 数据
            columns (list, optional): 要标准化的列，默认为所有数值列
            method (str, optional): 标准化方法，'standard'或'minmax'

        Returns:
            pandas.DataFrame: 标准化后的DataFrame
        """
        # 复制数据避免修改原始数据
        df = df.copy()
        
        # 如果未指定列，则选择所有数值列
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # 排除日期相关的数值列
            date_columns = ['year', 'month', 'day', 'dayofweek', 'quarter', 'dayofyear', 'weekofyear']
            columns = [col for col in columns if col not in date_columns]
        
        # 标准化处理
        if method == 'standard':
            # 使用StandardScaler (均值为0，标准差为1)
            df[columns] = self.standard_scaler.fit_transform(df[columns])
        elif method == 'minmax':
            # 使用MinMaxScaler (范围0-1)
            df[columns] = self.minmax_scaler.fit_transform(df[columns])
        else:
            logger.warning(f"不支持的标准化方法: {method}")
        
        logger.info(f"已使用{method}方法标准化{len(columns)}个特征")
        return df
    
    def generate_lag_features(self, df, columns=None, lag_periods=[1, 3, 5, 7]):
        """
        生成时间滞后特征

        Args:
            df (pandas.DataFrame): 数据
            columns (list, optional): 要生成滞后特征的列，默认为所有技术指标
            lag_periods (list, optional): 滞后周期列表

        Returns:
            pandas.DataFrame: 添加滞后特征的DataFrame
        """
        # 复制数据避免修改原始数据
        df = df.copy()
        
        # 默认使用主要技术指标和价格列
        if columns is None:
            columns = ['close', 'volume', 'rsi_14', 'macd', 'macd_signal', 'volatility_14']
            columns = [col for col in columns if col in df.columns]
        
        # 为每个指定的列创建滞后特征
        for col in columns:
            for lag in lag_periods:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        logger.info(f"为{len(columns)}个列生成了{len(lag_periods)}个滞后周期特征")
        return df
    
    def generate_return_features(self, df, periods=[1, 3, 5, 7, 14, 30]):
        """
        生成不同周期的收益率特征

        Args:
            df (pandas.DataFrame): 数据
            periods (list, optional): 周期列表

        Returns:
            pandas.DataFrame: 添加收益率特征的DataFrame
        """
        # 复制数据避免修改原始数据
        df = df.copy()
        
        # 检查close列是否存在
        if 'close' not in df.columns:
            logger.warning("缺少'close'列，无法计算收益率特征")
            return df
        
        # 计算不同周期的收益率
        for period in periods:
            df[f'return_{period}d'] = df['close'].pct_change(periods=period)
        
        logger.info(f"生成了{len(periods)}个周期的收益率特征")
        return df
    
    def generate_volatility_features(self, df, windows=[5, 10, 20, 30, 60]):
        """
        生成不同窗口的波动率特征

        Args:
            df (pandas.DataFrame): 数据
            windows (list, optional): 窗口列表

        Returns:
            pandas.DataFrame: 添加波动率特征的DataFrame
        """
        # 复制数据避免修改原始数据
        df = df.copy()
        
        # 检查close列是否存在
        if 'close' not in df.columns:
            logger.warning("缺少'close'列，无法计算波动率特征")
            return df
        
        # 计算日收益率
        returns = df['close'].pct_change()
        
        # 计算不同窗口的波动率 (年化)
        for window in windows:
            df[f'volatility_{window}d'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        logger.info(f"生成了{len(windows)}个窗口的波动率特征")
        return df
    
    def generate_all_features(self, df):
        """
        生成所有高级特征

        Args:
            df (pandas.DataFrame): 原始数据DataFrame

        Returns:
            pandas.DataFrame: 添加了所有高级特征的DataFrame
        """
        # 复制数据避免修改原始数据
        df = df.copy()
        
        # 1. 生成交易信号
        df = self.generate_trading_signals(df)
        
        # 2. 计算市场情绪指标
        df = self.calculate_market_sentiment(df)
        
        # 3. 生成收益率特征
        df = self.generate_return_features(df)
        
        # 4. 生成波动率特征
        df = self.generate_volatility_features(df)
        
        # 5. 生成时间滞后特征
        df = self.generate_lag_features(df)
        
        # 注意：这里不应该标准化特征，因为可能会影响可解释性
        # 标准化应该在模型训练前单独进行
        
        logger.info("所有高级特征生成完成")
        return df 