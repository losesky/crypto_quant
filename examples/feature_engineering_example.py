#!/usr/bin/env python
"""
特征工程示例脚本，展示如何使用特征工程生成高级特征
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_quant.data.processing import DataAdapter, FeatureEngineering
from crypto_quant.utils.logger import logger
from crypto_quant.utils.font_helper import get_font_helper
from crypto_quant.utils.output_helper import get_image_path, get_report_path

# 获取字体助手实例
font_helper = get_font_helper()

# 标签对照表 - 当中文字体不可用时，使用英文标签
labels = {
    '价格': 'Price',
    '价格与移动平均线': 'Price and Moving Averages',
    'BTC/USDT价格与移动平均线': 'BTC/USDT Price and Moving Averages',
    'BTC/USDT价格': 'BTC/USDT Price',
    'BTC/USDT 价格走势图': 'BTC/USDT Price Chart',
    '买入信号': 'Buy Signal',
    '卖出信号': 'Sell Signal',
    '移动平均线': 'Moving Average',
    'MA买入信号': 'MA Buy Signal',
    'MA卖出信号': 'MA Sell Signal',
    'RSI和超买超卖信号': 'RSI and Overbought/Oversold Signals',
    '超买信号': 'Overbought Signal',
    '超卖信号': 'Oversold Signal',
    '综合买入信号': 'Combined Buy Signal',
    '综合卖出信号': 'Combined Sell Signal',
    '综合买卖信号': 'Combined Trading Signals',
    '恐慌贪婪指数': 'Fear & Greed Index',
    '极度贪婪': 'Extreme Greed',
    '极度恐慌': 'Extreme Fear',
    '特征相关性热力图': 'Feature Correlation Heatmap',
    '累计收益率': 'Cumulative Return',
    '信号策略': 'Signal Strategy',
    '信号策略 vs Buy & Hold': 'Signal Strategy vs Buy & Hold',
    '日期': 'Date',
    'RSI': 'RSI',
    '成交量': 'Volume',
    '警告': 'Warning',
    '恐慌贪婪指数不存在，跳过可视化': 'Fear & Greed Index not found, skipping visualization',
    '缺少买卖信号列，无法回测': 'Missing buy/sell signal columns, cannot backtest',
    '特征工程示例运行完成': 'Feature engineering example completed',
    '交易信号图表已保存为': 'Trading signal chart saved as',
    '市场情绪图表已保存为': 'Market sentiment chart saved as',
    '特征相关性热力图已保存为': 'Feature correlation heatmap saved as',
    '回测结果图表已保存为': 'Backtest result chart saved as',
    '策略累计收益率': 'Strategy cumulative return',
    'Buy & Hold累计收益率': 'Buy & Hold cumulative return',
    '获取了': 'Retrieved',
    '行数据': 'rows of data',
    '生成特征后数据形状': 'Data shape after generating features',
    '特征列': 'Feature columns',
    '数据源初始化完成': 'Data source initialized',
    '数据处理器初始化完成': 'Data processor initialized',
    '已连接到数据库': 'Connected to database',
    '数据适配器初始化完成': 'Data adapter initialized',
    '获取K线数据': 'Fetching K-line data',
    '已获取历史K线数据': 'Retrieved historical K-line data',
    '处理数据': 'Processing data',
    '数据处理完成': 'Data processing completed',
    '特征工程模块初始化完成': 'Feature engineering module initialized',
    '交易信号特征生成完成': 'Trading signal features generated',
    '市场情绪指标计算完成': 'Market sentiment indicators calculated',
    '生成了': 'Generated',
    '周期的收益率特征': 'periods of return features',
    '窗口的波动率特征': 'windows of volatility features',
    '条': 'records'
}

# 获取标签的辅助函数
def get_label(zh_label):
    """根据中文字体可用性返回适当的标签"""
    if font_helper.has_chinese_font:
        return zh_label
    else:
        return labels.get(zh_label, zh_label)  # 如果没有对应的英文标签，使用原始标签


def get_data():
    """
    获取比特币数据
    """
    # 初始化数据适配器
    adapter = DataAdapter(source_type="binance")
    
    # 设置参数
    symbol = "BTC/USDT"
    interval = "1d"
    start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')  # 三年前
    
    # 获取并处理数据
    df = adapter.fetch_and_store_klines(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        process_data=True,
        store_data=False  # 不存储到数据库
    )
    
    logger.info(f"{get_label('获取了')} {len(df)} {get_label('行数据')}")
    return df


def generate_features(df):
    """
    生成高级特征
    
    Args:
        df (pandas.DataFrame): 基础数据
        
    Returns:
        pandas.DataFrame: 添加了高级特征的数据
    """
    # 初始化特征工程工具
    fe = FeatureEngineering()
    
    # 生成交易信号
    df = fe.generate_trading_signals(df)
    
    # 计算市场情绪指标
    df = fe.calculate_market_sentiment(df)
    
    # 生成收益率特征
    df = fe.generate_return_features(df)
    
    # 生成波动率特征
    df = fe.generate_volatility_features(df)
    
    logger.info(f"{get_label('生成特征后数据形状')}: {df.shape}")
    logger.info(f"{get_label('特征列')}: {list(df.columns)}")
    
    return df


def visualize_trading_signals(df):
    """
    可视化交易信号
    
    Args:
        df (pandas.DataFrame): 带有交易信号的数据
    """
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    # 设置全局标题
    title = get_label('BTC/USDT 技术分析')
    if font_helper.has_chinese_font:
        fig.suptitle(title, fontsize=16, fontweight='bold', fontproperties=font_helper.chinese_font)
    else:
        fig.suptitle('BTC/USDT Technical Analysis', fontsize=16, fontweight='bold')
    
    # 1. 价格和MA线
    font_helper.set_chinese_title(axes[0], get_label('价格与移动平均线'))
    axes[0].plot(df.index, df['close'], label=get_label('价格'), alpha=0.7)
    axes[0].plot(df.index, df['ma_20'], label='MA20', alpha=0.7)
    axes[0].plot(df.index, df['ma_50'], label='MA50', alpha=0.7)
    
    # 标记MA交叉信号
    buy_signals = df[df['ma_crossover'] == 1].index
    sell_signals = df[df['ma_crossunder'] == 1].index
    
    axes[0].scatter(buy_signals, df.loc[buy_signals, 'close'], marker='^', color='green', s=100, label=get_label('MA买入信号'))
    axes[0].scatter(sell_signals, df.loc[sell_signals, 'close'], marker='v', color='red', s=100, label=get_label('MA卖出信号'))
    
    font_helper.set_chinese_label(axes[0], ylabel=get_label('价格 (USDT)'))
    font_helper.set_chinese_legend(axes[0])
    axes[0].grid(True)
    
    # 2. RSI和超买超卖信号
    font_helper.set_chinese_title(axes[1], 'RSI (14) ' + get_label('指标'))
    axes[1].plot(df.index, df['rsi_14'], label='RSI', color='purple')
    axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5)
    axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5)
    
    # 标记RSI超买超卖信号
    oversold_signals = df[df['rsi_oversold'] == 1].index
    overbought_signals = df[df['rsi_overbought'] == 1].index
    
    axes[1].scatter(oversold_signals, df.loc[oversold_signals, 'rsi_14'], marker='^', color='green', s=100, label=get_label('超卖信号'))
    axes[1].scatter(overbought_signals, df.loc[overbought_signals, 'rsi_14'], marker='v', color='red', s=100, label=get_label('超买信号'))
    
    axes[1].set_ylabel('RSI')
    font_helper.set_chinese_legend(axes[1])
    axes[1].grid(True)
    
    # 3. 综合买卖信号
    font_helper.set_chinese_title(axes[2], get_label('综合买卖信号'))
    axes[2].plot(df.index, df['close'], label=get_label('价格'), alpha=0.7)
    
    # 标记综合买卖信号
    combined_buy = df[df['buy_signal'] == 1].index
    combined_sell = df[df['sell_signal'] == 1].index
    
    # 按信号强度调整标记大小
    buy_sizes = df.loc[combined_buy, 'buy_signal_strength'] * 50 + 50
    sell_sizes = df.loc[combined_sell, 'sell_signal_strength'] * 50 + 50
    
    axes[2].scatter(combined_buy, df.loc[combined_buy, 'close'], marker='^', color='green', s=buy_sizes, label=get_label('综合买入信号'))
    axes[2].scatter(combined_sell, df.loc[combined_sell, 'close'], marker='v', color='red', s=sell_sizes, label=get_label('综合卖出信号'))
    
    font_helper.set_chinese_label(axes[2], xlabel=get_label('日期'), ylabel=get_label('价格 (USDT)'))
    font_helper.set_chinese_legend(axes[2])
    axes[2].grid(True)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为上方的全局标题留出空间
    
    # 使用输出助手保存图像
    output_path = get_image_path("trading_signals.png")
    plt.savefig(output_path)
    logger.info(f"{get_label('交易信号图表已保存为')} {output_path}")


def visualize_market_sentiment(df):
    """
    可视化市场情绪指标
    
    Args:
        df (pandas.DataFrame): 带有市场情绪指标的数据
    """
    # 检查是否有恐慌贪婪指数
    if 'fear_greed_index' not in df.columns:
        logger.warning(get_label("恐慌贪婪指数不存在，跳过可视化"))
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # 设置全局标题
    title = get_label('BTC/USDT 市场情绪分析')
    if font_helper.has_chinese_font:
        fig.suptitle(title, fontsize=16, fontweight='bold', fontproperties=font_helper.chinese_font)
    else:
        fig.suptitle('BTC/USDT Market Sentiment Analysis', fontsize=16, fontweight='bold')
    
    # 1. 价格和成交量
    font_helper.set_chinese_title(axes[0], get_label('价格和成交量'))
    axes[0].plot(df.index, df['close'], label=get_label('价格'), color='blue')
    
    # 在第二个Y轴显示成交量
    ax2 = axes[0].twinx()
    ax2.bar(df.index, df['volume'], label=get_label('成交量'), alpha=0.3, color='gray')
    font_helper.set_chinese_label(ax2, ylabel=get_label('成交量'))
    
    font_helper.set_chinese_label(axes[0], ylabel=get_label('价格 (USDT)'))
    font_helper.set_chinese_legend(axes[0])
    axes[0].grid(True)
    
    # 2. 恐慌贪婪指数
    font_helper.set_chinese_title(axes[1], get_label('恐慌贪婪指数'))
    
    # 绘制恐慌贪婪指数
    axes[1].plot(df.index, df['fear_greed_index'], label=get_label('恐慌贪婪指数'), color='purple', linewidth=2)
    
    # 添加填充颜色以表示不同区域
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    levels = [0, 20, 40, 60, 80, 100]
    labels = [get_label('极度恐慌'), get_label('恐慌'), get_label('中性'), get_label('贪婪'), get_label('极度贪婪')]
    
    for i in range(len(colors)):
        axes[1].fill_between(df.index, levels[i], levels[i+1], color=colors[i], alpha=0.2)
    
    # 添加水平线表示区域边界
    for level in levels[1:-1]:
        axes[1].axhline(y=level, linestyle='--', alpha=0.5, color='gray')
    
    # 设置Y轴范围和标签
    axes[1].set_ylim(0, 100)
    font_helper.set_chinese_label(axes[1], xlabel=get_label('日期'), ylabel=get_label('恐慌贪婪指数'))
    
    # 添加最后一个值的注释
    last_date = df.index[-1]
    last_value = df['fear_greed_index'].iloc[-1]
    axes[1].annotate(f'{last_value:.0f}', 
                     xy=(last_date, last_value),
                     xytext=(10, 0),
                     textcoords="offset points",
                     ha='left', va='center',
                     fontsize=12, fontweight='bold')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为上方的全局标题留出空间
    
    # 使用输出助手保存图像
    output_path = get_image_path("market_sentiment.png")
    plt.savefig(output_path)
    logger.info(f"{get_label('市场情绪图表已保存为')} {output_path}")


def analyze_feature_correlation(df):
    """
    分析特征相关性
    
    Args:
        df (pandas.DataFrame): 带有特征的数据
    """
    # 选择数值型列
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # 从相关性分析中排除一些列
    exclude_columns = ['timestamp', 'volume', 'quote_volume', 'trades_count', 'taker_buy_volume', 
                       'taker_sell_volume', 'open', 'high', 'low']
    
    correlation_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    # 计算相关系数
    correlation_matrix = df[correlation_columns].corr()
    
    # 创建相关性热力图
    plt.figure(figsize=(16, 14))
    mask = np.triu(correlation_matrix)
    
    # 生成热力图
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, vmin=-1, vmax=1)
    
    # 设置标题
    font_helper.set_chinese_title(plt.gca(), get_label('特征相关性热力图'))
    
    # 保存图表
    plt.tight_layout()
    
    # 使用输出助手保存图像
    output_path = get_image_path("feature_correlation.png")
    plt.savefig(output_path, dpi=150)
    logger.info(f"{get_label('特征相关性热力图已保存为')} {output_path}")


def backtest_simple_signal_strategy(df):
    """
    使用生成的信号回测简单策略
    
    Args:
        df (pandas.DataFrame): 带有信号的数据
    """
    # 检查是否包含买卖信号列
    if 'buy_signal' not in df.columns or 'sell_signal' not in df.columns:
        logger.warning(get_label('缺少买卖信号列，无法回测'))
        return
    
    # 计算策略收益
    df['position'] = 0
    
    # 买入信号后持仓为1，卖出信号后持仓为0
    for i in range(1, len(df)):
        if df['buy_signal'].iloc[i-1] == 1:
            df.loc[df.index[i], 'position'] = 1
        elif df['sell_signal'].iloc[i-1] == 1:
            df.loc[df.index[i], 'position'] = 0
        else:
            df.loc[df.index[i], 'position'] = df['position'].iloc[i-1]
    
    # 计算每日收益率
    df['daily_return'] = df['close'].pct_change()
    
    # 计算策略收益率
    df['strategy_return'] = df['position'].shift(1) * df['daily_return']
    
    # 计算累计收益率
    df['cum_return'] = (1 + df['daily_return']).cumprod() - 1
    df['cum_strategy_return'] = (1 + df['strategy_return']).cumprod() - 1
    
    # 创建回测结果图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制Buy & Hold收益曲线
    ax.plot(df.index, df['cum_return'], label=get_label('Buy & Hold累计收益率'), linestyle='--')
    
    # 绘制策略收益曲线
    ax.plot(df.index, df['cum_strategy_return'], label=get_label('策略累计收益率'))
    
    font_helper.set_chinese_title(ax, get_label('信号策略 vs Buy & Hold'))
    font_helper.set_chinese_label(ax, xlabel=get_label('日期'), ylabel=get_label('累计收益率'))
    font_helper.set_chinese_legend(ax)
    
    ax.grid(True)
    plt.tight_layout()
    
    # 保存图表
    output_path = get_image_path("signal_strategy_backtest.png")
    plt.savefig(output_path)
    logger.info(f"{get_label('回测结果图表已保存为')} {output_path}")
    
    # 输出策略的收益率
    final_return = df['cum_strategy_return'].iloc[-1]
    bh_return = df['cum_return'].iloc[-1]
    logger.info(f"{get_label('策略累计收益率')}: {final_return:.2%}")
    logger.info(f"{get_label('Buy & Hold累计收益率')}: {bh_return:.2%}")
    
    return df[['cum_return', 'cum_strategy_return']]


def main():
    """
    主函数
    """
    # 1. 获取数据
    df = get_data()
    
    # 2. 生成特征
    df = generate_features(df)
    
    # 3. 可视化交易信号
    visualize_trading_signals(df)
    
    # 4. 可视化市场情绪
    visualize_market_sentiment(df)
    
    # 5. 分析特征相关性
    analyze_feature_correlation(df)
    
    # 6. 回测简单信号策略
    backtest_simple_signal_strategy(df)
    
    logger.info(get_label("特征工程示例运行完成"))


if __name__ == "__main__":
    # 检查是否已导入numpy
    try:
        import numpy as np
    except ImportError:
        import numpy as np
    
    main() 