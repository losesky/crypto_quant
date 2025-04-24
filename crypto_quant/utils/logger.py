"""
日志工具模块，提供统一的日志记录功能
"""
import os
import sys
import time
from pathlib import Path
from loguru import logger
from ..config.settings import LOG_CONFIG


def setup_logger():
    """
    设置全局日志记录器

    Returns:
        logger: 配置好的loguru记录器实例
    """
    # 移除默认处理程序
    logger.remove()

    # 确保日志目录存在
    log_dir = LOG_CONFIG.get("log_dir")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 获取日志级别
    log_level = LOG_CONFIG.get("log_level", "INFO")

    # 添加控制台处理程序
    if LOG_CONFIG.get("log_to_console", True):
        logger.add(
            sys.stderr,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level=log_level,
            backtrace=True,
            diagnose=True,
        )

    # 添加文件处理程序
    if LOG_CONFIG.get("log_to_file", True):
        log_file = os.path.join(
            log_dir, f"crypto_quant_{time.strftime('%Y%m%d')}.log"
        )
        logger.add(
            log_file,
            rotation="00:00",  # 每天午夜轮换
            retention="30 days",  # 保留30天
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
            level=log_level,
            backtrace=True,
            diagnose=True,
        )

    return logger


def set_log_level(level):
    """
    设置日志级别
    
    Args:
        level (str): 日志级别，可选 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    """
    # 移除所有现有处理程序
    logger.remove()
    
    # 重新添加具有新日志级别的处理程序
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level=level,
        backtrace=True,
        diagnose=True,
    )
    
    # 如果配置了文件日志
    if LOG_CONFIG.get("log_to_file", True):
        log_dir = LOG_CONFIG.get("log_dir")
        log_file = os.path.join(
            log_dir, f"crypto_quant_{time.strftime('%Y%m%d')}.log"
        )
        logger.add(
            log_file,
            rotation="00:00",  # 每天午夜轮换
            retention="30 days",  # 保留30天
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
            level=level,
            backtrace=True,
            diagnose=True,
        )
    
    logger.info(f"日志级别已设置为: {level}")


# 导出配置好的记录器
logger = setup_logger() 