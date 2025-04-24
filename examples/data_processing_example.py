#!/usr/bin/env python
"""
数据处理示例脚本，展示如何使用数据适配器获取、处理和存储数据
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_quant.data.processing import DataAdapter
from crypto_quant.utils.logger import logger
from crypto_quant.utils.font_helper import get_font_helper
from crypto_quant.utils.output_helper import get_image_path

# 获取字体助手实例
font_helper = get_font_helper()

def fetch_and_process_data():
    """
    获取并处理比特币数据
    """
    # 初始化数据适配器
    adapter = DataAdapter(source_type="binance")
    
    # 设置参数
    symbol = "BTC/USDT"
    interval = "1d"
    start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')  # 三年前
    
    logger.info(f"获取并处理{symbol} {interval}数据: {start_date}至今")
    
    # 获取、处理并存储数据
    df = adapter.fetch_and_store_klines(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        process_data=True,
        store_data=True
    )
    
    # 打印数据信息
    logger.info(f"数据形状: {df.shape}")
    logger.info(f"数据列: {list(df.columns)}")
    logger.info(f"数据日期范围: {df.index.min()} 至 {df.index.max()}")
    
    return df


def plot_data(df):
    """
    绘制数据图表
    
    Args:
        df (pandas.DataFrame): 数据
    """
    # 根据中文字体支持情况选择标题和标签
    if font_helper.has_chinese_font:
        price_title = 'BTC/USDT 价格和移动平均线'
        volume_title = '成交量'
        rsi_title = 'RSI(14) 指标'
        macd_title = 'MACD 指标'
        price_label = '价格'
        volume_label = '成交量'
        signal_label = '信号线'
        hist_label = 'MACD柱状图'
        date_label = '日期'
        price_y_label = '价格 (USDT)'
        main_title = 'BTC/USDT 技术指标分析'
    else:
        price_title = 'BTC/USDT Price and Moving Averages'
        volume_title = 'Volume'
        rsi_title = 'RSI(14) Indicator'
        macd_title = 'MACD Indicator'
        price_label = 'Price'
        volume_label = 'Volume'
        signal_label = 'Signal Line'
        hist_label = 'MACD Histogram'
        date_label = 'Date'
        price_y_label = 'Price (USDT)'
        main_title = 'BTC/USDT Technical Indicators Analysis'

    # 创建图表
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # 设置全局标题
    if font_helper.has_chinese_font:
        fig.suptitle(main_title, fontsize=16, fontweight='bold', fontproperties=font_helper.chinese_font)
    else:
        fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    # 1. 价格和移动平均线
    font_helper.set_chinese_title(axes[0], price_title)
    axes[0].plot(df.index, df['close'], label=price_label)
    axes[0].plot(df.index, df['ma_20'], label='MA20')
    axes[0].plot(df.index, df['ma_50'], label='MA50')
    axes[0].plot(df.index, df['ma_200'], label='MA200')
    font_helper.set_chinese_label(axes[0], ylabel=price_y_label)
    font_helper.set_chinese_legend(axes[0])
    axes[0].grid(True)
    
    # 2. 成交量
    font_helper.set_chinese_title(axes[1], volume_title)
    axes[1].bar(df.index, df['volume'], label=volume_label, alpha=0.5)
    font_helper.set_chinese_label(axes[1], ylabel=volume_title)
    font_helper.set_chinese_legend(axes[1])
    axes[1].grid(True)
    
    # 3. RSI
    font_helper.set_chinese_title(axes[2], rsi_title)
    axes[2].plot(df.index, df['rsi_14'], label='RSI', color='purple')
    axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
    axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
    font_helper.set_chinese_label(axes[2], ylabel='RSI')
    axes[2].grid(True)
    
    # 4. MACD
    font_helper.set_chinese_title(axes[3], macd_title)
    axes[3].plot(df.index, df['macd'], label='MACD')
    axes[3].plot(df.index, df['macd_signal'], label=signal_label)
    axes[3].bar(df.index, df['macd_hist'], label=hist_label, alpha=0.5)
    font_helper.set_chinese_label(axes[3], xlabel=date_label, ylabel='MACD')
    font_helper.set_chinese_legend(axes[3])
    axes[3].grid(True)
    
    # 应用中文字体到整个图表
    font_helper.apply_font_to_figure(fig)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为上方的标题留出空间
    
    # 使用输出助手保存图像
    output_path = get_image_path("btc_data_analysis.png")
    plt.savefig(output_path)
    logger.info(f"图表已保存至: {output_path}")


def load_data_from_db():
    """
    从数据库加载数据
    """
    # 初始化数据适配器
    adapter = DataAdapter()
    
    # 设置参数
    symbol = "BTC/USDT"
    interval = "1d"
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 一年前，从180天扩展到365天
    
    # 从数据库加载数据
    logger.info(f"从数据库加载{symbol} {interval}数据: {start_date}至今")
    df = adapter.load_data_from_db(
        symbol=symbol,
        interval=interval,
        start_date=start_date
    )
    
    logger.info(f"从数据库加载了{len(df)}行数据")
    
    return df


def get_available_symbols():
    """
    获取可用的交易对
    """
    # 初始化数据适配器
    adapter = DataAdapter()
    
    # 获取交易对列表
    symbols = adapter.get_symbols_list()
    
    # 打印交易对信息
    logger.info(f"发现{len(symbols)}个交易对")
    logger.info(f"前10个交易对: {symbols[:10]}")
    
    return symbols


if __name__ == "__main__":
    # 1. 获取并处理数据
    df = fetch_and_process_data()
    
    # 2. 绘制数据图表
    plot_data(df)
    
    # 3. 从数据库加载数据
    db_df = load_data_from_db()
    
    # 4. 获取可用的交易对
    symbols = get_available_symbols()
    
    logger.info("数据处理示例运行完成") 