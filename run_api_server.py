#!/usr/bin/env python
"""
启动API服务器脚本
"""
import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from crypto_quant.api.rest.api_service import start_api_server
from crypto_quant.utils.logger import logger


if __name__ == "__main__":
    logger.info("启动API服务器...")
    try:
        start_api_server()
    except KeyboardInterrupt:
        logger.info("API服务器已停止")
    except Exception as e:
        logger.error(f"API服务器发生错误: {str(e)}")
        sys.exit(1) 