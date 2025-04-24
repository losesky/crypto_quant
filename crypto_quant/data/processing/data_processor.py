"""
数据处理模块，用于清洗和转换原始数据，生成特征
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ...utils.logger import logger
from ...config.settings import RISK_CONFIG


class DataProcessor:
    """
    数据处理类，提供数据清洗、特征工程等功能
    """

    def __init__(self):
        """
        初始化数据处理器
        """
        self.zscore_threshold = RISK_CONFIG.get("zscore_threshold", 5.0)
        logger.info(f"数据处理器初始化完成: Z-Score阈值={self.zscore_threshold}")

    def clean_data(self, df):
        """
        数据清洗，去除缺失值和异常值

        Args:
            df (pandas.DataFrame): 原始数据

        Returns:
            pandas.DataFrame: 清洗后的数据
        """
        # 复制数据避免修改原始数据
        df = df.copy()
        
        # 检查并处理缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"检测到缺失值: {missing_values[missing_values > 0]}")
            
            # 用前一个有效值填充缺失值
            df.fillna(method='ffill', inplace=True)
            
            # 如果仍有缺失值，用后一个有效值填充
            df.fillna(method='bfill', inplace=True)
            
            logger.info("缺失值处理完成")
        
        # 过滤异常值
        df = self.filter_outliers(df)
        
        return df
    
    def filter_outliers(self, df, columns=None):
        """
        过滤异常值，对±5σ外的数据自动触发Z-Score过滤

        Args:
            df (pandas.DataFrame): 原始数据
            columns (list, optional): 需要过滤的列名列表，默认处理OHLCV列

        Returns:
            pandas.DataFrame: 过滤后的数据
        """
        # 复制数据避免修改原始数据
        df = df.copy()
        
        # 默认处理OHLCV列
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
            columns = [col for col in columns if col in df.columns]
        
        total_outliers = 0
        
        # 对每列进行异常值处理
        for column in columns:
            # 计算涨跌幅
            df[f'{column}_change'] = df[column].pct_change()
            
            # 计算Z-Score
            df[f'{column}_zscore'] = (df[f'{column}_change'] - df[f'{column}_change'].mean()) / df[f'{column}_change'].std()
            
            # 标记异常值
            df[f'{column}_is_outlier'] = np.abs(df[f'{column}_zscore']) > self.zscore_threshold
            
            # 统计异常值
            outliers_count = df[f'{column}_is_outlier'].sum()
            total_outliers += outliers_count
            
            if outliers_count > 0:
                logger.warning(f"列 {column} 检测到 {outliers_count} 个异常值，已过滤")
                
                # 将异常值替换为前一个有效值
                outlier_indices = df[df[f'{column}_is_outlier']].index
                df.loc[outlier_indices, column] = df[column].shift(1).loc[outlier_indices]
            
            # 删除临时列
            df.drop([f'{column}_change', f'{column}_zscore', f'{column}_is_outlier'], axis=1, inplace=True)
        
        if total_outliers > 0:
            logger.info(f"共处理 {total_outliers} 个异常值")
        
        return df

    def add_technical_indicators(self, df):
        """
        添加常用技术指标

        Args:
            df (pandas.DataFrame): 原始价格数据

        Returns:
            pandas.DataFrame: 添加技术指标后的数据
        """
        # 复制数据避免修改原始数据
        df = df.copy()
        
        # 确保存在必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"缺少必要的列: {[col for col in required_columns if col not in df.columns]}")
            return df
        
        # 1. 移动平均线
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        df['ma_200'] = df['close'].rolling(window=200).mean()
        
        # 2. 指数移动平均线
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # 3. MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 4. RSI (相对强弱指标)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain_14 = gain.rolling(window=14).mean()
        avg_loss_14 = loss.rolling(window=14).mean()
        
        rs_14 = avg_gain_14 / avg_loss_14
        df['rsi_14'] = 100 - (100 / (1 + rs_14))
        
        # 5. Bollinger Bands (布林带)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        
        # 6. ATR (平均真实范围)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()
        
        # 7. 交易量变化率
        df['volume_change'] = df['volume'].pct_change()
        
        # 8. 价格波动率
        df['volatility_14'] = df['close'].pct_change().rolling(window=14).std() * np.sqrt(14)
        
        return df

    def add_market_features(self, df):
        """
        添加市场相关特征，如日期特征、价格变化率等

        Args:
            df (pandas.DataFrame): 原始数据

        Returns:
            pandas.DataFrame: 添加市场特征后的数据
        """
        # 复制数据避免修改原始数据
        df = df.copy()
        
        # 确保索引是日期时间类型
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("数据索引不是DatetimeIndex类型，无法添加日期特征")
            return df
        
        # 1. 日期特征
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek  # 0=周一, 6=周日
        df['quarter'] = df.index.quarter
        df['dayofyear'] = df.index.dayofyear
        df['weekofyear'] = df.index.isocalendar().week
        
        # 2. 价格变化率
        df['price_change'] = df['close'].pct_change()
        df['price_change_1d'] = df['close'].pct_change(periods=1)
        df['price_change_3d'] = df['close'].pct_change(periods=3)
        df['price_change_7d'] = df['close'].pct_change(periods=7)
        df['price_change_14d'] = df['close'].pct_change(periods=14)
        df['price_change_30d'] = df['close'].pct_change(periods=30)
        
        # 3. 价格范围特征
        df['day_range'] = (df['high'] - df['low']) / df['low']  # 当天价格范围比率
        df['open_to_close'] = (df['close'] - df['open']) / df['open']  # 开盘到收盘价格变化率
        
        # 4. 振幅
        df['amplitude'] = df['high'] / df['low'] - 1
        
        # 5. 波动率
        df['volatility_7d'] = df['close'].pct_change().rolling(window=7).std() * np.sqrt(7)
        df['volatility_30d'] = df['close'].pct_change().rolling(window=30).std() * np.sqrt(30)
        
        # 6. 自相关特征
        df['lag_1'] = df['close'].shift(1)
        df['lag_3'] = df['close'].shift(3)
        df['lag_7'] = df['close'].shift(7)
        
        return df
    
    def process_data(self, df, add_features=True):
        """
        数据处理主函数，包括数据清洗和特征工程

        Args:
            df (pandas.DataFrame): 原始数据
            add_features (bool): 是否添加额外特征

        Returns:
            pandas.DataFrame: 处理后的数据
        """
        # 1. 数据清洗
        df = self.clean_data(df)
        
        # 2. 添加特征
        if add_features:
            # 添加技术指标
            df = self.add_technical_indicators(df)
            
            # 添加市场特征
            df = self.add_market_features(df)
            
            logger.info(f"数据处理完成: 形状={df.shape}, 列={list(df.columns)}")
        
        return df 