# 特征工程模块文档

本文档详细说明了比特币量化交易框架中特征工程模块的功能和使用方法。

## 概述

特征工程模块(`FeatureEngineering`)设计用于从基础价格和交易数据中创建高级特征，这些特征可以：

1. 生成交易信号
2. 计算市场情绪指标
3. 提供时间序列模型所需的滞后特征
4. 计算和标准化各类市场特征

这些功能对于构建量化交易策略或训练机器学习模型至关重要。

## 核心功能

### 1. 交易信号生成

`generate_trading_signals()` 方法从技术指标中生成以下交易信号：

- **移动平均线信号**
  - `ma_crossover`: MA20上穿MA50的金叉买入信号
  - `ma_crossunder`: MA20下穿MA50的死叉卖出信号

- **RSI信号**
  - `rsi_oversold`: RSI低于30的超卖买入信号
  - `rsi_overbought`: RSI高于70的超买卖出信号

- **MACD信号**
  - `macd_crossover`: MACD上穿信号线的金叉买入信号
  - `macd_crossunder`: MACD下穿信号线的死叉卖出信号

- **布林带信号**
  - `bb_breakout_up`: 价格突破上轨信号
  - `bb_breakout_down`: 价格突破下轨信号
  - `bb_squeeze`: 布林带收窄信号

- **综合信号**
  - `buy_signal`: 综合买入信号
  - `sell_signal`: 综合卖出信号
  - `buy_signal_strength`: 买入信号强度(0-3)
  - `sell_signal_strength`: 卖出信号强度(0-3)

### 2. 市场情绪指标

`calculate_market_sentiment()` 方法计算以下市场情绪指标：

- **恐慌贪婪指数**
  - `fear_greed_index`: 0-100的指数，0表示极度恐慌，100表示极度贪婪

- **市场动量指标**
  - `price_direction_10d`: 10日平均价格变动方向
  - `price_ma20_gap`: 价格与20日均线的距离

- **震荡指标**
  - `choppiness_index`: 市场震荡指标，分辨趋势和震荡市场

- **牛熊市指标**
  - `bull_market`: 价格高于200日均线为牛市(值为1)，反之为熊市(值为0)
  - `bull_market_days`: 连续牛市的天数
  - `bear_market_days`: 连续熊市的天数

### 3. 收益率特征

`generate_return_features()` 方法计算不同周期的收益率特征：

- `return_1d`: 1日收益率
- `return_3d`: 3日收益率
- `return_5d`: 5日收益率
- `return_7d`: 7日收益率
- `return_14d`: 14日收益率
- `return_30d`: 30日收益率

### 4. 波动率特征

`generate_volatility_features()` 方法计算不同时间窗口的波动率特征：

- `volatility_5d`: 5日波动率(年化)
- `volatility_10d`: 10日波动率(年化)
- `volatility_20d`: 20日波动率(年化)
- `volatility_30d`: 30日波动率(年化)
- `volatility_60d`: 60日波动率(年化)

### 5. 时间滞后特征

`generate_lag_features()` 方法生成时间滞后特征，默认为以下字段生成滞后值：

- 'close', 'volume', 'rsi_14', 'macd', 'macd_signal', 'volatility_14'

默认生成的滞后周期为 [1, 3, 5, 7]，例如:
- `close_lag_1`: 收盘价滞后1天
- `rsi_14_lag_3`: RSI指标滞后3天

### 6. 特征标准化

`normalize_features()` 方法提供两种标准化方式：

- **标准化(standard)**: 将特征转换为均值为0，标准差为1的分布
- **归一化(minmax)**: 将特征缩放到[0,1]区间

## 使用示例

### 基本使用

```python
from crypto_quant.data.processing import DataAdapter, FeatureEngineering

# 获取数据
adapter = DataAdapter()
df = adapter.fetch_and_store_klines(
    symbol="BTC/USDT",
    interval="1d",
    start_date="2022-01-01",
    store_data=False
)

# 创建特征工程实例
fe = FeatureEngineering()

# 生成交易信号
df = fe.generate_trading_signals(df)

# 计算市场情绪指标
df = fe.calculate_market_sentiment(df)

# 生成所有高级特征
df_all_features = fe.generate_all_features(df)
```

### 自定义特征生成

```python
# 生成自定义滞后特征
df = fe.generate_lag_features(
    df, 
    columns=['close', 'volume', 'rsi_14'], 
    lag_periods=[1, 5, 10, 20]
)

# 生成自定义收益率特征
df = fe.generate_return_features(
    df,
    periods=[1, 7, 14, 30, 60, 90]
)

# 标准化特征
df_scaled = fe.normalize_features(
    df,
    columns=['close', 'volume', 'rsi_14', 'macd'],
    method='minmax'
)
```

### 特征选择

由于生成的特征数量可能很多，建议根据特征相关性或重要性进行选择：

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 分析特征相关性
corr = df[['close', 'rsi_14', 'macd', 'volume', 'return_7d']].corr()

# 可视化相关性矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("特征相关性矩阵")
plt.show()
```

## 注意事项

1. **数据预处理**: 在生成高级特征前，确保数据已经过清洗和基础处理
2. **缺失值**: 许多特征依赖于滚动窗口计算，会在开始部分产生NaN值
3. **特征爆炸**: 使用所有特征可能导致维度灾难，建议进行特征选择
4. **前瞻偏差**: 确保在回测中不使用未来数据生成的特征
5. **标准化时机**: 对于模型训练，标准化应该在训练集上拟合，并应用到测试集

## 参考文献

- Murphy, J.J. (1999). Technical Analysis of the Financial Markets
- Prado, M.L. (2018). Advances in Financial Machine Learning
- Kakushadze, Z., & Serur, J.A. (2018). 151 Trading Strategies 