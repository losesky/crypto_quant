"""
Binance数据源适配器，提供与Binance交易所的数据交互接口
"""
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from binance.client import Client
import aiohttp
import asyncio
from ...config.settings import EXCHANGE_APIS
from ...utils.logger import logger

# 自定义异步客户端类
class AsyncClient:
    def __init__(self, api_key=None, api_secret=None):
        self.client = Client(api_key, api_secret)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

class BinanceDataSource:
    """
    Binance交易所数据源适配器
    提供历史K线数据、交易深度、最新价格等数据获取功能
    """

    def __init__(self, use_async=False):
        """
        初始化Binance数据源

        Args:
            use_async (bool, optional): 是否使用异步客户端
        """
        api_config = EXCHANGE_APIS.get("binance", {})
        self.api_key = api_config.get("api_key", "")
        self.api_secret = api_config.get("api_secret", "")
        self.use_testnet = api_config.get("use_testnet", True)
        
        # 设置同步客户端
        self.client = Client(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.use_testnet,
        )
        
        # 异步客户端初始化状态
        self.async_client = None
        self.use_async = use_async
        
        # 缓存数据
        self._exchange_info = None
        
        logger.info(f"Binance数据源初始化完成: {'测试网' if self.use_testnet else '主网'}")

    async def _ensure_async_client(self):
        """
        确保异步客户端已初始化
        """
        if self.async_client is None:
            self.async_client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.use_testnet,
            )
            logger.info("Binance异步客户端初始化完成")

    def _interval_to_milliseconds(self, interval):
        """
        将时间间隔字符串转换为毫秒数

        Args:
            interval (str): 时间间隔字符串，如 "1m", "1h", "1d"

        Returns:
            int: 对应的毫秒数
        """
        intervals_map = {
            "1m": 60 * 1000,
            "3m": 3 * 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "2h": 2 * 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "6h": 6 * 60 * 60 * 1000,
            "8h": 8 * 60 * 60 * 1000,
            "12h": 12 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
            "3d": 3 * 24 * 60 * 60 * 1000,
            "1w": 7 * 24 * 60 * 60 * 1000,
            "1M": 30 * 24 * 60 * 60 * 1000,
        }
        return intervals_map.get(interval, 0)

    def get_historical_klines(self, symbol, interval, start_str=None, end_str=None, limit=1000):
        """
        获取历史K线数据

        Args:
            symbol (str): 交易对，如 "BTCUSDT"
            interval (str): K线间隔，如 "1m", "1h", "1d"
            start_str (str, optional): 开始时间，如 "1 Jan, 2020"
            end_str (str, optional): 结束时间，如 "1 Jan, 2021"
            limit (int, optional): 限制获取的K线数量，最大1000

        Returns:
            pandas.DataFrame: 包含K线数据的DataFrame
        """
        try:
            # 获取历史K线数据
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=limit
            )
            
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=[
                'datetime', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 数据类型转换
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                              'quote_asset_volume', 'taker_buy_base_asset_volume', 
                              'taker_buy_quote_asset_volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
            
            # 设置索引
            df.set_index('datetime', inplace=True)
            
            # 过滤异常值
            df = self._filter_outliers(df)
            
            logger.info(f"已获取{symbol} {interval}历史K线数据: {len(df)}条")
            return df
            
        except Exception as e:
            logger.error(f"获取{symbol} {interval}历史K线数据失败: {str(e)}")
            raise

    def _filter_outliers(self, df, z_threshold=5.0):
        """
        过滤异常值，对±5σ外的价格波动自动触发Z-Score过滤
        
        Args:
            df (pandas.DataFrame): K线数据DataFrame
            z_threshold (float): Z-Score阈值，默认为5.0
            
        Returns:
            pandas.DataFrame: 过滤后的DataFrame
        """
        # 计算价格变化率
        df['price_change'] = df['close'].pct_change()
        
        # 计算Z-Score
        df['z_score'] = (df['price_change'] - df['price_change'].mean()) / df['price_change'].std()
        
        # 标记异常值
        df['is_outlier'] = np.abs(df['z_score']) > z_threshold
        
        # 记录异常值
        outliers = df[df['is_outlier']]
        if not outliers.empty:
            logger.warning(f"检测到{len(outliers)}个异常价格波动，已过滤")
            
        # 将异常值替换为前一个有效值
        df.loc[df['is_outlier'], 'open'] = df['open'].shift(1)
        df.loc[df['is_outlier'], 'high'] = df['high'].shift(1)
        df.loc[df['is_outlier'], 'low'] = df['low'].shift(1)
        df.loc[df['is_outlier'], 'close'] = df['close'].shift(1)
        
        # 删除辅助列
        df.drop(['price_change', 'z_score', 'is_outlier'], axis=1, inplace=True)
        
        return df

    def get_latest_price(self, symbol):
        """
        获取最新价格

        Args:
            symbol (str): 交易对，如 "BTCUSDT"

        Returns:
            float: 最新价格
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"获取{symbol}最新价格失败: {str(e)}")
            raise

    def get_order_book(self, symbol, limit=100):
        """
        获取订单簿(深度数据)

        Args:
            symbol (str): 交易对，如 "BTCUSDT"
            limit (int): 深度级别，可选值: 5, 10, 20, 50, 100, 500, 1000, 5000

        Returns:
            dict: 包含订单簿数据的字典
        """
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            return depth
        except Exception as e:
            logger.error(f"获取{symbol}订单簿数据失败: {str(e)}")
            raise

    def get_exchange_info(self, reload=False):
        """
        获取交易所信息

        Args:
            reload (bool): 是否重新加载，不使用缓存

        Returns:
            dict: 交易所信息
        """
        if self._exchange_info is None or reload:
            try:
                self._exchange_info = self.client.get_exchange_info()
            except Exception as e:
                logger.error(f"获取交易所信息失败: {str(e)}")
                raise
        return self._exchange_info

    def get_all_tickers(self):
        """
        获取所有交易对的最新价格

        Returns:
            pandas.DataFrame: 包含所有交易对最新价格的DataFrame
        """
        try:
            tickers = self.client.get_all_tickers()
            df = pd.DataFrame(tickers)
            df['price'] = pd.to_numeric(df['price'])
            return df
        except Exception as e:
            logger.error(f"获取所有交易对价格失败: {str(e)}")
            raise

    def get_futures_funding_rate(self, symbol=None):
        """
        获取永续合约资金费率

        Args:
            symbol (str, optional): 交易对，如 "BTCUSDT"，默认为None获取所有

        Returns:
            pandas.DataFrame: 包含资金费率的DataFrame
        """
        try:
            if symbol:
                rates = self.client.futures_funding_rate(symbol=symbol)
            else:
                rates = self.client.futures_funding_rate()
                
            df = pd.DataFrame(rates)
            
            # 转换数据类型
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = pd.to_numeric(df['fundingRate'])
            
            return df
        except Exception as e:
            logger.error(f"获取资金费率失败: {str(e)}")
            raise

    def get_historical_data_segmented(self, symbol, interval, start, end=None, max_segment_days=365):
        """
        分段获取历史数据，解决长时间范围数据获取的限制问题
        
        Args:
            symbol (str): 交易对，如 "BTC/USDT"
            interval (str): K线间隔，如 "1d", "1h"
            start (str): 开始时间
            end (str): 结束时间，默认为当前时间
            max_segment_days (int): 每段最大天数，默认365天
            
        Returns:
            pandas.DataFrame: 合并后的历史数据
        """
        # 转换符号格式 (BTC/USDT -> BTCUSDT)
        formatted_symbol = symbol.replace('/', '')
        
        # 转换日期
        start_date = pd.to_datetime(start)
        if end:
            end_date = pd.to_datetime(end)
        else:
            end_date = datetime.now()
        
        # 创建分段日期列表
        segment_dates = []
        current_date = start_date
        
        while current_date < end_date:
            segment_dates.append(current_date)
            current_date = current_date + timedelta(days=max_segment_days)
        
        # 确保最后一个日期是结束日期
        if current_date != end_date:
            segment_dates.append(end_date)
        
        # 分段获取数据
        all_dfs = []
        for i in range(len(segment_dates) - 1):
            start_segment = segment_dates[i].strftime('%Y-%m-%d')
            end_segment = segment_dates[i+1].strftime('%Y-%m-%d')
            
            try:
                logger.info(f"获取分段数据 ({i+1}/{len(segment_dates)-1}): {start_segment} 至 {end_segment}")
                
                segment_df = self.get_historical_klines(
                    symbol=formatted_symbol,
                    interval=interval,
                    start_str=start_segment,
                    end_str=end_segment
                )
                
                if not segment_df.empty:
                    all_dfs.append(segment_df)
                    
            except Exception as e:
                logger.error(f"获取分段数据失败 ({start_segment} 至 {end_segment}): {str(e)}")
        
        # 合并数据
        if not all_dfs:
            logger.warning(f"未获取到任何数据")
            return pd.DataFrame()
        
        df_combined = pd.concat(all_dfs)
        
        # 删除重复的日期索引
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
        
        # 按日期排序
        df_combined = df_combined.sort_index()
        
        logger.info(f"共获取到 {len(df_combined)} 条历史数据，日期范围: {df_combined.index.min()} 至 {df_combined.index.max()}")
        
        return df_combined

    def get_historical_data(self, symbol, interval, start='1 day ago UTC', end=None):
        """
        获取历史数据，并格式化为策略友好的DataFrame
        使用分段获取方法解决长时间范围数据获取的限制问题
        
        Args:
            symbol (str): 交易对，如 "BTC/USDT"
            interval (str): K线间隔，如 "1m", "1h", "1d"
            start (str): 开始时间
            end (str): 结束时间，默认为当前时间

        Returns:
            pandas.DataFrame: 格式化后的历史数据
        """
        # 检查时间跨度，决定是否使用分段获取
        try:
            start_date = pd.to_datetime(start)
            if end:
                end_date = pd.to_datetime(end)
            else:
                end_date = datetime.now()
                
            # 计算天数差
            days_diff = (end_date - start_date).days
            
            # 如果时间跨度大于500天（一般安全范围），使用分段获取
            if days_diff > 500:
                logger.info(f"检测到大时间跨度请求 ({days_diff} 天)，使用分段获取方法")
                return self.get_historical_data_segmented(symbol, interval, start, end)
        except Exception as e:
            logger.warning(f"日期解析失败，使用原始方法: {str(e)}")
        
        # 对于小时间跨度或日期解析失败的情况，使用原方法
        # 转换符号格式 (BTC/USDT -> BTCUSDT)
        formatted_symbol = symbol.replace('/', '')
        
        # 获取K线数据
        df = self.get_historical_klines(
            symbol=formatted_symbol,
            interval=interval,
            start_str=start,
            end_str=end
        )
        
        return df 