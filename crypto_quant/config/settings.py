"""
全局配置设置模块，包含框架的主要配置项
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 基础路径
BASE_DIR = Path(__file__).parents[2]
DATA_DIR = os.path.join(BASE_DIR, "data_storage")

# 数据库配置
DB_CONFIG = {
    "clickhouse": {
        "host": os.getenv("CLICKHOUSE_HOST", "localhost"),
        "port": int(os.getenv("CLICKHOUSE_PORT", 9000)),
        "user": os.getenv("CLICKHOUSE_USER", "default"),
        "password": os.getenv("CLICKHOUSE_PASSWORD", "hello123"),
        "database": os.getenv("CLICKHOUSE_DB", "crypto_quant"),
        "pool_size": 5,
    }
}

# API配置 - 交易所
EXCHANGE_APIS = {
    "binance": {
        "api_key": os.getenv("BINANCE_API_KEY", ""),
        "api_secret": os.getenv("BINANCE_API_SECRET", ""),
        "use_testnet": os.getenv("USE_BINANCE_TESTNET", "True").lower() == "true",
    },
    "okex": {
        "api_key": os.getenv("OKEX_API_KEY", ""),
        "api_secret": os.getenv("OKEX_API_SECRET", ""),
        "passphrase": os.getenv("OKEX_PASSPHRASE", ""),
        "use_testnet": os.getenv("USE_OKEX_TESTNET", "True").lower() == "true",
    }
}

# 回测配置
BACKTEST_CONFIG = {
    "default_commission": 0.001,  # 0.1% 交易手续费
    "default_slippage": 0.0005,   # 0.05% 滑点
    "default_leverage": 1,        # 默认杠杆率
    "risk_free_rate": 0.02,       # 无风险利率 (年化)
}

# 风险管理配置
RISK_CONFIG = {
    "max_drawdown_limit": 0.15,   # 最大回撤限制 (15%)
    "min_calmar_ratio": 2.5,      # 最小Calmar比率
    "max_position_size": 0.2,     # 单个资产最大仓位占比 (20%)
    "volatility_lookback": 20,    # 波动率计算回溯期 (日)
    "zscore_threshold": 5.0,      # Z-Score异常数据过滤阈值
}

# 机器学习配置
ML_CONFIG = {
    "default_split_ratio": 0.8,    # 训练集/测试集分割比例
    "cv_folds": 5,                 # 交叉验证折数
    "optuna_trials": 100,          # Optuna超参数优化尝试次数
    "model_save_path": os.path.join(BASE_DIR, "models", "saved"),
}

# 日志配置
LOG_CONFIG = {
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
    "log_dir": os.path.join(BASE_DIR, "logs"),
    "log_to_console": True,
    "log_to_file": True,
}

# Web服务配置
WEB_CONFIG = {
    "host": os.getenv("WEB_HOST", "0.0.0.0"),
    "port": int(os.getenv("WEB_PORT", 8000)),
    "debug": os.getenv("WEB_DEBUG", "False").lower() == "true",
    "workers": int(os.getenv("WEB_WORKERS", 4)),
}

# Telegram机器人配置
TELEGRAM_CONFIG = {
    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", "5978160206:AAFEJJFsfJQpuesB_rxsFulHmbjmzsmvgxY"),
    "chat_id": os.getenv("TELEGRAM_CHAT_ID", "1850814972"),
    "enabled": os.getenv("TELEGRAM_ENABLED", "False").lower() == "true",
} 