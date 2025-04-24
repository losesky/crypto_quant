"""
数据适配器模块，连接数据源、数据处理和数据存储
"""
import pandas as pd
from datetime import datetime, timedelta
from ..sources.binance_source import BinanceDataSource
from ..storage.db_manager import ClickHouseManager
from .data_processor import DataProcessor
from ...utils.logger import logger


class DataAdapter:
    """
    数据适配器类，负责获取、处理和存储数据
    """

    def __init__(self, source_type="binance"):
        """
        初始化数据适配器

        Args:
            source_type (str): 数据源类型，目前支持"binance"
        """
        # 初始化数据源
        if source_type.lower() == "binance":
            self.data_source = BinanceDataSource()
        else:
            raise ValueError(f"不支持的数据源类型: {source_type}")
        
        # 初始化数据处理器和数据库连接
        self.processor = DataProcessor()
        self.db_manager = ClickHouseManager()
        
        logger.info(f"数据适配器初始化完成: 数据源={source_type}")

    def fetch_and_store_klines(self, symbol, interval, start_date, end_date=None, 
                              process_data=True, store_data=True, table_name=None):
        """
        获取、处理并存储K线数据

        Args:
            symbol (str): 交易对，如"BTC/USDT"
            interval (str): K线间隔，如"1d", "1h"
            start_date (str): 开始日期，如"2022-01-01"
            end_date (str, optional): 结束日期，默认为当前时间
            process_data (bool): 是否处理数据
            store_data (bool): 是否存储数据
            table_name (str, optional): 表名，默认为"klines_{symbol}_{interval}"

        Returns:
            pandas.DataFrame: 获取并处理后的数据
        """
        try:
            # 1. 获取数据
            logger.info(f"从{symbol} {interval}获取K线数据: {start_date}至{end_date or '现在'}")
            df = self.data_source.get_historical_data(
                symbol=symbol,
                interval=interval,
                start=start_date,
                end=end_date
            )
            
            # 2. 处理数据
            if process_data:
                logger.info(f"处理{symbol} {interval}数据")
                df = self.processor.process_data(df)
            
            # 3. 存储数据
            if store_data:
                # 如果没有指定表名，生成默认表名
                if table_name is None:
                    # 清理符号中的特殊字符
                    clean_symbol = symbol.replace('/', '_').replace('-', '_')
                    table_name = f"klines_{clean_symbol}_{interval}"
                
                # 创建表（如果不存在）
                self._create_klines_table(table_name)
                
                # 存储数据
                self._store_df_to_clickhouse(df, table_name)
            
            return df
            
        except Exception as e:
            logger.error(f"获取、处理并存储{symbol} {interval}数据失败: {str(e)}")
            raise

    def _create_klines_table(self, table_name):
        """
        创建K线数据表（如果不存在）

        Args:
            table_name (str): 表名
        """
        # 定义K线表结构，包括矿工持仓变化和期货基差字段
        columns_definition = """
            datetime DateTime,
            
            open Float64,
            high Float64,
            low Float64,
            close Float64,
            volume Float64,
            
            quote_asset_volume Float64,
            number_of_trades UInt32,
            taker_buy_base_asset_volume Float64,
            taker_buy_quote_asset_volume Float64,
            
            miner_position_change Float64 DEFAULT 0,
            futures_basis Float64 DEFAULT 0,
            
            ma_5 Float64 DEFAULT 0,
            ma_10 Float64 DEFAULT 0,
            ma_20 Float64 DEFAULT 0,
            ma_50 Float64 DEFAULT 0,
            ma_200 Float64 DEFAULT 0,
            
            ema_5 Float64 DEFAULT 0,
            ema_10 Float64 DEFAULT 0,
            ema_20 Float64 DEFAULT 0,
            ema_50 Float64 DEFAULT 0,
            ema_200 Float64 DEFAULT 0,
            
            macd Float64 DEFAULT 0,
            macd_signal Float64 DEFAULT 0,
            macd_hist Float64 DEFAULT 0,
            
            rsi_14 Float64 DEFAULT 0,
            
            bb_middle Float64 DEFAULT 0,
            bb_upper Float64 DEFAULT 0,
            bb_lower Float64 DEFAULT 0,
            
            atr_14 Float64 DEFAULT 0,
            volume_change Float64 DEFAULT 0,
            volatility_14 Float64 DEFAULT 0,
            
            price_change Float64 DEFAULT 0,
            price_change_1d Float64 DEFAULT 0,
            price_change_3d Float64 DEFAULT 0,
            price_change_7d Float64 DEFAULT 0,
            price_change_30d Float64 DEFAULT 0,
            
            day_range Float64 DEFAULT 0,
            amplitude Float64 DEFAULT 0,
            
            year UInt16 DEFAULT 0,
            month UInt8 DEFAULT 0,
            day UInt8 DEFAULT 0,
            dayofweek UInt8 DEFAULT 0
        """
        
        # 创建表
        self.db_manager.create_table_if_not_exists(table_name, columns_definition)
        logger.info(f"表已准备: {table_name}")

    def _store_df_to_clickhouse(self, df, table_name):
        """
        将DataFrame存储到ClickHouse表

        Args:
            df (pandas.DataFrame): 要存储的数据
            table_name (str): 表名

        Returns:
            int: 插入的行数
        """
        try:
            # 确保datetime列作为第一列
            if 'datetime' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            
            # 过滤不存在的列
            existing_columns = self.db_manager.execute(
                f"SELECT name FROM system.columns WHERE table = '{table_name}'"
            )
            existing_columns = [col[0] for col in existing_columns]
            
            # 仅保留表中存在的列
            columns_to_keep = [col for col in df.columns if col in existing_columns]
            df_to_insert = df[columns_to_keep].copy()
            
            # 插入数据
            inserted_rows = self.db_manager.insert_df(table_name, df_to_insert)
            logger.info(f"已插入{inserted_rows}行数据到表{table_name}")
            
            return inserted_rows
            
        except Exception as e:
            logger.error(f"存储数据到表{table_name}失败: {str(e)}")
            raise

    def load_data_from_db(self, symbol, interval, start_date, end_date=None, table_name=None):
        """
        从数据库加载K线数据

        Args:
            symbol (str): 交易对，如"BTC/USDT"
            interval (str): K线间隔，如"1d", "1h"
            start_date (str): 开始日期，如"2022-01-01"
            end_date (str, optional): 结束日期，默认为当前时间
            table_name (str, optional): 表名，默认为"klines_{symbol}_{interval}"

        Returns:
            pandas.DataFrame: 从数据库加载的数据
        """
        try:
            # 如果没有指定表名，生成默认表名
            if table_name is None:
                # 清理符号中的特殊字符
                clean_symbol = symbol.replace('/', '_').replace('-', '_')
                table_name = f"klines_{clean_symbol}_{interval}"
            
            # 转换日期格式
            start_date_dt = pd.to_datetime(start_date)
            start_date_str = start_date_dt.strftime('%Y-%m-%d %H:%M:%S')
            
            if end_date:
                end_date_dt = pd.to_datetime(end_date)
                end_date_str = end_date_dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                end_date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 构建SQL查询
            query = f"""
            SELECT * 
            FROM {table_name} 
            WHERE datetime >= '{start_date_str}' AND datetime <= '{end_date_str}'
            ORDER BY datetime
            """
            
            # 执行查询
            result = self.db_manager.execute(query, with_column_types=True)
            rows, columns = result
            
            # 转换为DataFrame
            column_names = [col[0] for col in columns]
            df = pd.DataFrame(rows, columns=column_names)
            
            # 设置日期索引
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            
            logger.info(f"从表{table_name}加载了{len(df)}行数据: {start_date}至{end_date or '现在'}")
            
            return df
            
        except Exception as e:
            logger.error(f"从表{table_name}加载数据失败: {str(e)}")
            
            # 如果表不存在或加载失败，尝试直接从数据源获取
            logger.info(f"尝试从数据源直接获取数据")
            return self.fetch_and_store_klines(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                process_data=True,
                store_data=True,
                table_name=table_name
            )

    def update_data(self, symbol, interval, days=7, table_name=None):
        """
        更新最近几天的数据

        Args:
            symbol (str): 交易对，如"BTC/USDT"
            interval (str): K线间隔，如"1d", "1h"
            days (int): 要更新的天数
            table_name (str, optional): 表名

        Returns:
            pandas.DataFrame: 更新后的数据
        """
        # 计算开始日期
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # 获取并存储数据
        df = self.fetch_and_store_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            process_data=True,
            store_data=True,
            table_name=table_name
        )
        
        logger.info(f"已更新{symbol} {interval}最近{days}天的数据")
        return df

    def get_symbols_list(self):
        """
        获取交易所支持的交易对列表

        Returns:
            list: 交易对列表
        """
        try:
            # 从数据源获取所有交易对
            tickers = self.data_source.get_all_tickers()
            symbols = tickers['symbol'].tolist()
            
            # 过滤USDT交易对
            usdt_symbols = [s for s in symbols if s.endswith('USDT')]
            
            # 转换格式为BTC/USDT样式
            formatted_symbols = [f"{s[:-4]}/{s[-4:]}" for s in usdt_symbols]
            
            return formatted_symbols
            
        except Exception as e:
            logger.error(f"获取交易对列表失败: {str(e)}")
            raise 