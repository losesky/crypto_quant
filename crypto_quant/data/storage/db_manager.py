"""
数据库连接管理器模块，提供数据库连接池和操作接口
"""
from clickhouse_driver import Client
from clickhouse_driver.errors import Error as ClickHouseError
from ...config.settings import DB_CONFIG
from ...utils.logger import logger
import pandas as pd


class ClickHouseManager:
    """
    ClickHouse数据库管理器，使用连接池处理数据库操作
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """
        实现单例模式
        """
        if cls._instance is None:
            cls._instance = super(ClickHouseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        初始化ClickHouse连接管理器
        """
        if not self._initialized:
            self.config = DB_CONFIG.get("clickhouse", {})
            self.client = Client(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 9000),
                user=self.config.get("user", "default"),
                password=self.config.get("password", "hello123"),
                database=self.config.get("database", "crypto_quant"),
                settings={
                    "use_numpy": False,  # 禁用NumPy支持
                    "max_execution_time": 60,  # 60秒执行超时
                    "connect_timeout": 10,     # 10秒连接超时
                    "send_timeout": 10,        # 10秒发送超时
                    "receive_timeout": 10,     # 10秒接收超时
                    "insert_block_size": 100   # 每次插入100行
                },
                compression=False,
            )
            self._initialized = True
            self._ensure_database_exists()
            logger.info(f"已连接到ClickHouse数据库: {self.config.get('host')}:{self.config.get('port')}")

    def _ensure_database_exists(self):
        """
        确保数据库存在，不存在则创建
        """
        db_name = self.config.get("database", "crypto_quant")
        try:
            # 切换到系统数据库
            self.client.execute("SELECT 1")
            
            # 检查数据库是否存在
            result = self.client.execute(
                f"SELECT name FROM system.databases WHERE name = '{db_name}'"
            )
            
            if not result:
                self.client.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
                logger.info(f"已创建ClickHouse数据库: {db_name}")
                
            # 切换到我们的数据库
            self.client.execute(f"USE {db_name}")
            
        except ClickHouseError as e:
            logger.error(f"数据库初始化错误: {str(e)}")
            raise

    def execute(self, query, params=None, with_column_types=False):
        """
        执行SQL查询

        Args:
            query (str): SQL查询语句
            params (dict, optional): 查询参数
            with_column_types (bool, optional): 是否返回列类型信息

        Returns:
            result: 查询结果
        """
        try:
            return self.client.execute(query, params, with_column_types=with_column_types)
        except ClickHouseError as e:
            logger.error(f"SQL执行错误: {str(e)}\nQuery: {query}\nParams: {params}")
            raise

    def create_table_if_not_exists(self, table_name, columns_definition):
        """
        如果表不存在则创建

        Args:
            table_name (str): 表名
            columns_definition (str): 列定义字符串
        """
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {columns_definition}
        ) ENGINE = MergeTree()
        ORDER BY (datetime)
        """
        self.execute(query)
        logger.info(f"已确保ClickHouse表存在: {table_name}")

    def insert_df(self, table_name, df):
        """
        将DataFrame插入ClickHouse表

        Args:
            table_name (str): 表名
            df (pandas.DataFrame): 要插入的数据

        Returns:
            int: 插入的行数
        """
        try:
            # 获取表结构
            table_schema = self.execute(f"DESCRIBE TABLE {table_name}")
            table_columns = [col[0] for col in table_schema]
            
            # 检查是否需要添加datetime列
            if 'datetime' in table_columns and 'datetime' not in df.columns:
                # 尝试从索引或其他列创建datetime列
                if df.index.name == 'datetime' or isinstance(df.index, pd.DatetimeIndex):
                    logger.info("从索引创建datetime列")
                    df = df.reset_index()
                elif 'close_time' in df.columns:
                    logger.info("从close_time创建datetime列")
                    df['datetime'] = pd.to_datetime(df['close_time'], unit='ms')
                else:
                    logger.error("DataFrame中缺少必要的datetime列，无法插入数据")
                    return 0
            
            # 只保留表中存在的列
            valid_columns = [col for col in df.columns if col in table_columns]
            if not valid_columns:
                logger.error(f"DataFrame中没有表 {table_name} 中的有效列")
                return 0
                
            # 创建一个新的DataFrame，只包含有效列并处理数据类型
            insert_df = df[valid_columns].copy()
            
            # 确保datetime列的格式正确
            if 'datetime' in valid_columns:
                insert_df['datetime'] = pd.to_datetime(insert_df['datetime'])
            
            # 将DataFrame转换为字典列表
            records = insert_df.to_dict('records')
            logger.info(f"准备插入{len(records)}行数据到{table_name}表")
            
            # 构建查询
            columns_str = ", ".join(valid_columns)
            query = f"INSERT INTO {table_name} ({columns_str}) VALUES"
            
            # 执行插入
            self.client.execute(query, records)
            logger.info(f"已向{table_name}表插入{len(records)}行数据")
            return len(records)
            
        except Exception as e:
            logger.error(f"数据插入错误: {str(e)}")
            raise 