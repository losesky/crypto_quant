# Bitcoin Quant Trading Framework (BTC-QTF)

全面的比特币及加密资产量化交易解决方案，集成传统量化策略与机器学习技术。

## 核心特性

- **统一数据接口**：处理市场数据、链上数据和交易所数据
- **智能策略优化**：自动因子生成和策略迭代优化
- **机器学习整合**：支持PyTorch，时间序列预测和强化学习
- **可视化回测系统**：实时策略绩效监控
- **风险管理**：自动异常数据过滤和风险控制

## 项目结构

```
crypto_quant/
├── data/                  # 数据获取、处理和存储
│   ├── sources/           # 数据源适配器
│   ├── processing/        # 数据预处理和特征工程
│   └── storage/           # 数据存储和管理
├── models/                # 机器学习模型
│   ├── classic/           # 传统机器学习模型
│   ├── deep_learning/     # 深度学习模型
│   └── reinforcement/     # 强化学习模型
├── strategies/            # 交易策略
│   ├── technical/         # 技术分析策略
│   ├── fundamental/       # 基本面策略
│   ├── ml_based/          # 机器学习策略
│   └── hybrid/            # 混合策略
├── backtesting/           # 回测系统
│   ├── engine/            # 回测引擎
│   ├── evaluation/        # 策略评估
│   └── visualization/     # 结果可视化
├── trading/               # 实盘交易
│   ├── execution/         # 订单执行
│   ├── risk/              # 风险管理
│   └── portfolio/         # 投资组合管理
├── api/                   # API接口
│   ├── rest/              # REST API
│   └── websocket/         # WebSocket API
├── ui/                    # 用户界面
│   ├── dashboard/         # 控制面板
│   └── reports/           # 报告生成
├── utils/                 # 通用工具
├── config/                # 配置文件
├── optimization/          # 参数优化模块
├── outputs/               # 输出文件目录
│   ├── images/            # 图表输出
│   └── reports/           # 报告输出
└── tests/                 # 测试代码
```

## 安装指南

1. 克隆仓库
```bash
git clone https://github.com/losesky/crypto_quant.git
cd crypto_quant
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 设置环境变量
```bash
cp .env.example .env
# 编辑.env文件，填入必要的API密钥和配置
```

4. 安装中文字体（解决matplotlib显示问题）
```bash
# 运行字体安装脚本
python install_fonts.py

# 或者手动安装字体
# Ubuntu/Debian:
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei

# Windows:
# 下载并安装中文字体，如微软雅黑或宋体

# macOS:
# 使用Homebrew安装
brew install --cask font-wqy-microhei
```

## 快速开始

```python
from crypto_quant.backtesting import Backtest
from crypto_quant.strategies.technical import MACDStrategy
from crypto_quant.data.sources import BinanceDataSource

# 获取数据
data_source = BinanceDataSource()
btc_data = data_source.get_historical_data('BTC/USDT', '1d', start='2022-01-01')

# 初始化策略
strategy = MACDStrategy(fast=12, slow=26, signal=9)

# 回测
backtest = Backtest(data=btc_data, strategy=strategy, initial_capital=10000)
results = backtest.run()

# 查看结果
print(results.summary())
results.plot()
```

## 项目特点

1. **自动化投研平台**：全流程智能化策略开发
2. **智能迭代闭环**：「数据-建模-验证」的持续优化
3. **异常数据处理**：对±5σ外的价格波动自动触发Z-Score过滤
4. **风险控制**：验证Calmar比率≥2.5、最大回撤≤15%的策略筛选标准

## 常见问题解决

### 中文字体显示问题

如果在运行可视化相关代码时遇到中文字体显示问题，会出现类似以下警告：

```
UserWarning: Glyph xxxxx (\N{CJK UNIFIED IDEOGRAPH-xxxx}) missing from current font.
```

解决方法：

1. 运行提供的字体安装脚本：
```bash
python install_fonts.py
```

2. 手动配置matplotlib字体：
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Microsoft YaHei']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
```

3. 确认已正确安装中文字体：
```python
# 检查可用字体
import matplotlib.font_manager
[f.name for f in matplotlib.font_manager.fontManager.ttflist if 'Micro' in f.name]
```

## 贡献指南

欢迎提交Issues和Pull Requests！请遵循以下代码规范：

- 变量命名：`snake_case`（策略类用`CamelCase`）
- 函数注释覆盖率≥90%，复杂逻辑需添加详细说明
- 数据库操作强制使用连接池，禁止明文存储API密钥

## 许可证

MIT 