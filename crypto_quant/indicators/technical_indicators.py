"""
技术指标计算模块
提供各种常用技术指标的计算功能
"""
import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
from ..utils.logger import logger


class TechnicalIndicators:
    """
    技术指标计算类
    提供各种常用技术指标的静态方法
    """
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14, price_col: str = 'close') -> pd.Series:
        """
        计算相对强弱指标(RSI)
        
        Args:
            data: 包含价格数据的DataFrame
            period: RSI计算周期
            price_col: 价格列名
            
        Returns:
            pd.Series: RSI值序列
        """
        if price_col not in data.columns:
            logger.warning(f"RSI计算失败：找不到{price_col}列")
            return pd.Series(index=data.index)
            
        # 计算价格变动
        delta = data[price_col].diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 首次计算平均涨跌幅
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 后续计算使用平滑方法
        for i in range(period, len(delta)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
        
        # 计算RS和RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算平均趋向指标(ADX)
        
        Args:
            data: 包含OHLC数据的DataFrame
            period: ADX计算周期
            
        Returns:
            pd.Series: ADX值序列
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                logger.warning(f"ADX计算失败：缺少{col}列")
                return pd.Series(index=data.index)
        
        # 计算方向性移动(+DM和-DM)
        high_diff = data['high'].diff()
        low_diff = data['low'].diff().multiply(-1)
        
        plus_dm = ((high_diff > low_diff) & (high_diff > 0)).astype(int) * high_diff
        minus_dm = ((low_diff > high_diff) & (low_diff > 0)).astype(int) * low_diff
        
        # 计算真实波动幅度(TR)
        tr1 = data['high'] - data['low']
        tr2 = (data['high'] - data['close'].shift(1)).abs()
        tr3 = (data['low'] - data['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算平滑值
        atr = tr.rolling(window=period).mean()
        plus_di = (plus_dm.rolling(window=period).mean() / atr) * 100
        minus_di = (minus_dm.rolling(window=period).mean() / atr) * 100
        
        # 计算方向性指标差(DX)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di) * 100).fillna(0)
        
        # 计算ADX
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_volatility(data: pd.DataFrame, period: int = 20, price_col: str = 'close', annualize: bool = False) -> float:
        """
        计算价格波动率
        
        Args:
            data: 包含价格数据的DataFrame
            period: 波动率计算周期
            price_col: 价格列名
            annualize: 是否年化波动率
            
        Returns:
            float: 波动率值
        """
        if price_col not in data.columns:
            logger.warning(f"波动率计算失败：找不到{price_col}列")
            return 0.0
            
        if len(data) < period:
            logger.warning(f"波动率计算失败：数据长度({len(data)})小于计算周期({period})")
            # 使用可用的数据计算
            if len(data) > 1:
                price_changes = data[price_col].pct_change().dropna()
                volatility = price_changes.std()
            else:
                return 0.01  # 默认低波动率
        else:
            # 计算价格变化百分比
            price_changes = data[price_col].iloc[-period:].pct_change().dropna()
            # 计算标准差作为波动率
            volatility = price_changes.std()
        
        # 年化处理
        if annualize:
            # 假设数据频率是日线，年化系数为sqrt(365)
            volatility = volatility * np.sqrt(365)
            
        return volatility
    
    @staticmethod
    def calculate_bb_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0, price_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算布林带(Bollinger Bands)
        
        Args:
            data: 包含价格数据的DataFrame
            period: 移动平均周期
            std_dev: 标准差倍数
            price_col: 价格列名
            
        Returns:
            Tuple: (上轨, 中轨, 下轨)
        """
        if price_col not in data.columns:
            logger.warning(f"布林带计算失败：找不到{price_col}列")
            return pd.Series(index=data.index), pd.Series(index=data.index), pd.Series(index=data.index)
        
        # 计算中轨(SMA)
        middle_band = data[price_col].rolling(window=period).mean()
        
        # 计算标准差
        std = data[price_col].rolling(window=period).std()
        
        # 计算上下轨
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, price_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算MACD指标
        
        Args:
            data: 包含价格数据的DataFrame
            fast_period: 快线EMA周期
            slow_period: 慢线EMA周期
            signal_period: 信号线EMA周期
            price_col: 价格列名
            
        Returns:
            Tuple: (MACD线, 信号线, 柱状图)
        """
        if price_col not in data.columns:
            logger.warning(f"MACD计算失败：找不到{price_col}列")
            return pd.Series(index=data.index), pd.Series(index=data.index), pd.Series(index=data.index)
        
        # 计算快线和慢线EMA
        ema_fast = data[price_col].ewm(span=fast_period, adjust=False).mean()
        ema_slow = data[price_col].ewm(span=slow_period, adjust=False).mean()
        
        # 计算MACD线
        macd_line = ema_fast - ema_slow
        
        # 计算信号线
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # 计算柱状图
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_obv(data: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume') -> pd.Series:
        """
        计算能量潮指标(OBV)
        
        Args:
            data: 包含价格和成交量数据的DataFrame
            price_col: 价格列名
            volume_col: 成交量列名
            
        Returns:
            pd.Series: OBV值序列
        """
        required_cols = [price_col, volume_col]
        for col in required_cols:
            if col not in data.columns:
                logger.warning(f"OBV计算失败：缺少{col}列")
                return pd.Series(index=data.index)
        
        # 计算价格变动方向
        price_change = data[price_col].diff()
        
        # 根据价格变动方向确定OBV变动
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = 0
        
        for i in range(1, len(data)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + data[volume_col].iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - data[volume_col].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def calculate_divergence(price: pd.Series, indicator: pd.Series, window: int = 5) -> pd.Series:
        """
        检测价格与指标之间的背离
        
        Args:
            price: 价格序列
            indicator: 指标序列
            window: 极值点检测窗口
            
        Returns:
            pd.Series: 背离信号序列(1=看涨背离, -1=看跌背离, 0=无背离)
        """
        if len(price) != len(indicator):
            logger.warning("背离计算失败：价格序列与指标序列长度不一致")
            return pd.Series(0, index=price.index)
            
        # 初始化结果序列
        divergence = pd.Series(0, index=price.index)
        
        # 寻找局部极值点
        def is_local_min(series, i, window):
            if i < window or i >= len(series) - window:
                return False
            return all(series.iloc[i] <= series.iloc[j] for j in range(i-window, i+window+1) if j != i)
            
        def is_local_max(series, i, window):
            if i < window or i >= len(series) - window:
                return False
            return all(series.iloc[i] >= series.iloc[j] for j in range(i-window, i+window+1) if j != i)
        
        # 查找潜在背离
        for i in range(window, len(price) - window):
            # 看涨背离：价格创新低但指标不创新低
            if is_local_min(price, i, window):
                prev_mins = [j for j in range(max(0, i-3*window), i-window) if is_local_min(price, j, window)]
                if prev_mins and price.iloc[i] < price.iloc[min(prev_mins, key=lambda j: price.iloc[j])]:
                    # 价格创新低
                    min_idx = min(prev_mins, key=lambda j: price.iloc[j])
                    if indicator.iloc[i] > indicator.iloc[min_idx]:
                        # 指标不创新低 -> 看涨背离
                        divergence.iloc[i] = 1
            
            # 看跌背离：价格创新高但指标不创新高
            if is_local_max(price, i, window):
                prev_maxs = [j for j in range(max(0, i-3*window), i-window) if is_local_max(price, j, window)]
                if prev_maxs and price.iloc[i] > price.iloc[max(prev_maxs, key=lambda j: price.iloc[j])]:
                    # 价格创新高
                    max_idx = max(prev_maxs, key=lambda j: price.iloc[j])
                    if indicator.iloc[i] < indicator.iloc[max_idx]:
                        # 指标不创新高 -> 看跌背离
                        divergence.iloc[i] = -1
        
        return divergence
    
    @staticmethod
    def identify_market_regime(data: pd.DataFrame, volatility_threshold: float = 0.03, 
                              trend_threshold: float = 25, fast_ma_period: int = 10, 
                              slow_ma_period: int = 50) -> str:
        """
        识别市场状态
        
        Args:
            data: 包含价格数据的DataFrame
            volatility_threshold: 波动率阈值，高于此值视为高波动市场
            trend_threshold: ADX阈值，高于此值视为趋势市场
            fast_ma_period: 快速移动平均周期
            slow_ma_period: 慢速移动平均周期
            
        Returns:
            str: 市场状态 - 
                'trending_volatile': 波动趋势市场
                'ranging_volatile': 波动震荡市场
                'trending_stable': 稳定趋势市场
                'ranging_stable': 稳定震荡市场
        """
        # 计算波动率
        volatility = TechnicalIndicators.calculate_volatility(data, period=20)
        
        # 计算ADX (趋势强度)
        adx = TechnicalIndicators.calculate_adx(data, period=14).iloc[-1]
        
        # 计算移动平均线
        if 'fast_ma' not in data.columns:
            fast_ma = data['close'].rolling(window=fast_ma_period).mean()
        else:
            fast_ma = data['fast_ma']
            
        if 'slow_ma' not in data.columns:
            slow_ma = data['close'].rolling(window=slow_ma_period).mean()
        else:
            slow_ma = data['slow_ma']
        
        # 判断趋势方向
        trend_direction = 1 if fast_ma.iloc[-1] > slow_ma.iloc[-1] else -1
        
        # 市场状态分类
        if volatility > volatility_threshold:
            if adx > trend_threshold:
                return 'trending_volatile'  # 波动趋势市场
            else:
                return 'ranging_volatile'   # 波动震荡市场
        else:
            if adx > trend_threshold:
                return 'trending_stable'    # 稳定趋势市场
            else:
                return 'ranging_stable'     # 稳定震荡市场
    
    @staticmethod
    def get_market_features(data: pd.DataFrame) -> dict:
        """
        获取市场特征集合
        
        Args:
            data: 包含价格数据的DataFrame
            
        Returns:
            dict: 市场特征字典
        """
        # 确保数据足够
        if len(data) < 50:
            logger.warning(f"市场特征计算失败：数据长度不足({len(data)}<50)")
            return {}
        
        try:
            features = {}
            
            # 计算波动率
            features['volatility'] = TechnicalIndicators.calculate_volatility(data, period=20)
            
            # 计算ADX (趋势强度)
            features['adx'] = TechnicalIndicators.calculate_adx(data, period=14).iloc[-1]
            
            # 计算RSI
            features['rsi'] = TechnicalIndicators.calculate_rsi(data, period=14).iloc[-1]
            
            # 计算MACD
            macd, signal, hist = TechnicalIndicators.calculate_macd(data)
            features['macd'] = macd.iloc[-1]
            features['macd_signal'] = signal.iloc[-1]
            features['macd_hist'] = hist.iloc[-1]
            
            # 计算移动平均趋势
            ma_10 = data['close'].rolling(window=10).mean().iloc[-1]
            ma_50 = data['close'].rolling(window=50).mean().iloc[-1]
            features['ma_trend'] = 1 if ma_10 > ma_50 else -1
            
            # 计算价格相对MA的位置
            features['price_vs_ma'] = (data['close'].iloc[-1] / ma_50) - 1
            
            # 布林带宽度
            upper, middle, lower = TechnicalIndicators.calculate_bb_bands(data)
            features['bb_width'] = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
            
            # 价格动量
            features['momentum'] = (data['close'].iloc[-1] / data['close'].iloc[-5]) - 1
            
            # 市场状态识别
            features['market_regime'] = TechnicalIndicators.identify_market_regime(data)
            
            return features
            
        except Exception as e:
            logger.error(f"计算市场特征时出错: {str(e)}")
            return {} 