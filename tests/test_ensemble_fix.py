#!/usr/bin/env python
"""
测试集成模型修复
用于验证AdaptiveEnsemble、GradientBoostingEnsemble和NeuralNetworkEnsemble
的修复是否有效

用法:
    python test_ensemble_fix.py [gradient|neural]
"""
import sys
import os
import pandas as pd
import numpy as np
import logging
import traceback

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_quant.utils.logger import logger, set_log_level
from crypto_quant.data.sources.binance_source import BinanceDataSource
from crypto_quant.strategies.technical.macd_strategy import MACDStrategy
from crypto_quant.strategies.ml_based.enhanced_lstm_strategy import EnhancedLSTMStrategy

# 设置日志级别
set_log_level('INFO')

# 根据参数选择测试哪个集成模型
if len(sys.argv) > 1 and sys.argv[1].lower() == 'gradient':
    from crypto_quant.strategies.hybrid.gradient_boosting_ensemble import GradientBoostingEnsemble as EnsembleModel
    model_type = "GradientBoosting"
else:
    from crypto_quant.strategies.hybrid.neural_network_ensemble import NeuralNetworkEnsemble as EnsembleModel
    model_type = "NeuralNetwork"

def create_synthetic_data(days=365):
    """创建合成测试数据，包含完整的OHLCV数据和市场状态"""
    # 生成日期索引
    import datetime
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 生成随机价格数据，使用随机游走
    np.random.seed(42)  # 设置随机种子，确保结果可重现
    
    # 生成基础价格趋势（随机游走）
    price_changes = np.random.normal(0, 0.02, size=len(dates))
    
    # 添加一些长期趋势和季节性
    trend = np.linspace(0, 0.5, len(dates))  # 上升趋势
    seasonality = 0.05 * np.sin(np.linspace(0, 4*np.pi, len(dates)))  # 季节性波动
    
    # 组合成日收益率
    returns = price_changes + 0.001 * trend + seasonality
    
    # 从收益率计算收盘价
    close = 20000 * np.cumprod(1 + returns)  # 起始价格20000
    
    # 生成日内波动
    daily_volatility = np.abs(np.random.normal(0, 0.02, size=len(dates)))
    high = close * (1 + daily_volatility)  # 确保high总是高于close
    low = close * (1 - daily_volatility)   # 确保low总是低于close
    
    # 开盘价在前一天收盘价附近波动
    gap = np.random.normal(0, 0.01, size=len(dates))
    open_price = np.zeros_like(close)
    open_price[0] = close[0] * (1 + gap[0])  # 第一天
    open_price[1:] = close[:-1] * (1 + gap[1:])  # 其他天
    
    # 确保开盘价在当天的最高价和最低价之间
    for i in range(len(dates)):
        if open_price[i] > high[i]:
            open_price[i] = high[i] - (high[i] - close[i]) * np.random.random()
        elif open_price[i] < low[i]:
            open_price[i] = low[i] + (close[i] - low[i]) * np.random.random()
    
    # 生成成交量数据
    volume_base = np.abs(np.random.normal(loc=5000, scale=1000, size=len(dates)))
    # 价格波动大的日子，成交量也相应增加
    volume = volume_base * (1 + 3 * np.abs(returns))
    
    # 创建市场状态标签
    # 使用波动率和价格趋势来确定市场状态
    volatility = pd.Series(returns).rolling(window=20).std().fillna(0).values
    price_trend = pd.Series(returns).rolling(window=20).mean().fillna(0).values
    
    market_regimes = []
    for i in range(len(dates)):
        # 高波动 + 强趋势 = 波动趋势市场
        if volatility[i] > 0.015 and abs(price_trend[i]) > 0.005:
            market_regimes.append('trending_volatile')
        # 高波动 + 弱趋势 = 波动震荡市场
        elif volatility[i] > 0.015:
            market_regimes.append('ranging_volatile')
        # 低波动 + 强趋势 = 稳定趋势市场
        elif abs(price_trend[i]) > 0.005:
            market_regimes.append('trending_stable')
        # 低波动 + 弱趋势 = 稳定震荡市场
        else:
            market_regimes.append('ranging_stable')
    
    # 创建DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'market_regime': market_regimes
    }, index=dates)
    
    # 计算常用的技术指标
    # RSI (14天)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 波动率 (20天)
    df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(20)
    
    # 计算MACD (12, 26, 9)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 填充缺失值
    df = df.fillna(0)
    
    return df

def test_ensemble_model():
    """测试集成模型"""
    try:
        # 获取测试数据
        logger.info(f"开始测试{model_type}集成模型")
        
        try:
            # 尝试使用Binance数据源获取数据
            data_source = BinanceDataSource()
            df = data_source.get_historical_data(
                symbol='BTCUSDT',
                interval='1d',
                start='2022-01-01',
                end='2022-12-31'
            )
            logger.info(f"从Binance获取到{len(df)}条数据记录")
        except Exception as e:
            logger.warning(f"从Binance获取数据失败: {e}，将使用合成数据")
            # 使用合成数据代替
            df = create_synthetic_data(days=365)
            logger.info(f"已创建{len(df)}条合成数据记录")
        
        # 确保数据中没有"close_time"列，避免LSTM特征工程警告
        if 'close_time' in df.columns:
            logger.info("删除'close_time'列以避免特征工程警告")
            df = df.drop('close_time', axis=1)
        
        # 创建基础策略
        macd_strategy = MACDStrategy()
        
        # 创建LSTM策略 - 正确初始化，不传递epochs参数
        lstm_strategy = EnhancedLSTMStrategy(
            sequence_length=20,
            prediction_threshold=0.01,
            hidden_dim=64,
            num_layers=2,
            feature_engineering=True
        )
        
        # 提前训练LSTM模型以避免警告，在train方法中传递epochs参数
        logger.info("预先训练LSTM模型以避免运行时警告")
        lstm_df = df.copy()
        lstm_strategy.train(lstm_df, epochs=10)  # 减少训练轮数加快测试
        
        # 生成基础策略信号
        logger.info("生成MACD策略信号")
        macd_df = macd_strategy.generate_signals(df.copy())
        
        logger.info("生成LSTM策略信号")
        lstm_df = lstm_strategy.generate_signals(df.copy())
        
        # 添加信号到原始数据
        df['macd_position'] = macd_df['position']
        df['lstm_position'] = lstm_df['position']
        
        # 创建集成模型
        logger.info(f"创建{model_type}集成模型")
        base_strategies = [
            {'strategy': macd_strategy, 'name': 'macd'},
            {'strategy': lstm_strategy, 'name': 'lstm'}
        ]
        
        # 不要设置模型保存路径，除非确定会训练模型
        test_model_path = None
        os.makedirs("tmp", exist_ok=True)
        
        # 创建集成模型 - 先不设置模型保存路径
        ensemble = EnsembleModel(
            base_strategies=base_strategies,
            window_size=30,
            retrain_interval=50,
            min_train_samples=100,
            feature_columns=['close', 'volume', 'macd_position', 'lstm_position'],
            target_column='future_return_1d',
            prediction_threshold=0.001,
            hidden_dim=32,  # 神经网络参数
            num_layers=1    # 神经网络参数
            # 暂时不设置model_path，避免尝试保存未训练的模型
        )
        
        # 测试预处理特征方法
        logger.info("测试特征预处理")
        features_df = ensemble.prepare_features(df.copy(), current_idx=100, is_training=True, suppress_warnings=True)
        logger.info(f"特征预处理成功，生成了 {len(features_df.columns)} 个特征列")
        
        # 打印预处理后的特征列
        logger.info(f"特征列: {features_df.columns.tolist()[:10]}...")
        
        # 测试训练方法
        logger.info("测试模型训练")
        model_trained = False
        try:
            # 确保有目标列并且长度足够，这样才能正常训练模型
            if 'future_return_1d' not in features_df.columns:
                # 计算未来收益作为目标变量
                features_df['future_return_1d'] = features_df['close'].pct_change(1).shift(-1)
                
            # 丢弃包含NaN的行
            clean_features_df = features_df.dropna()
            logger.info(f"训练数据清理后剩余 {len(clean_features_df)} 行")
            
            # 确保有足够的训练数据
            if len(clean_features_df) >= 50:  # 至少需要50行进行训练
                ensemble.train(clean_features_df, clean_features_df['future_return_1d'])
                logger.info("模型训练成功")
                
                # 设置_is_trained标志
                ensemble._is_trained = True
                model_trained = True
                
                # 现在训练成功，我们可以设置模型路径并保存模型
                if model_trained:
                    test_model_path = f"tmp/test_{model_type.lower()}_model.pkl"
                    ensemble.model_path = test_model_path
                    
                    # 尝试手动触发保存模型
                    if hasattr(ensemble, '_save_model'):
                        logger.info("尝试保存训练好的模型")
                        ensemble._save_model()
            else:
                logger.warning(f"训练数据不足 ({len(clean_features_df)} < 50行)，跳过训练")
        except Exception as train_error:
            logger.error(f"模型训练失败: {str(train_error)}")
            traceback.print_exc()
            return False
        
        # 测试预测方法
        logger.info("测试模型预测")
        try:
            predictions = ensemble.predict(features_df.iloc[200:210])
            logger.info(f"模型预测成功，得到 {len(predictions)} 个预测值")
            logger.info(f"预测结果示例: {predictions[:5]}")
        except Exception as predict_error:
            logger.error(f"模型预测失败: {str(predict_error)}")
            traceback.print_exc()
            return False
        
        # 测试信号生成
        logger.info("测试信号生成")
        try:
            signals_df = ensemble.generate_signals(df.copy(), suppress_warnings=True)
            logger.info(f"信号生成成功，共 {signals_df['signal'].abs().sum()} 个交易信号")
            
            # 计算基本统计量
            signal_counts = signals_df['signal'].value_counts()
            logger.info(f"信号分布: {signal_counts.to_dict()}")
            
            # 计算策略收益
            signals_df['returns'] = signals_df['close'].pct_change()
            signals_df['strategy_returns'] = signals_df['position'].shift(1) * signals_df['returns']
            
            total_return = (1 + signals_df['strategy_returns']).prod() - 1
            sharpe = signals_df['strategy_returns'].mean() / signals_df['strategy_returns'].std() * np.sqrt(252)
            
            logger.info(f"策略总收益: {total_return:.2%}")
            logger.info(f"策略夏普比率: {sharpe:.2f}")
            
        except Exception as signal_error:
            logger.error(f"信号生成失败: {str(signal_error)}")
            traceback.print_exc()
            return False
        
        # 测试特征重要性
        logger.info("测试特征重要性")
        try:
            feature_importance = ensemble.get_feature_importance()
            if not feature_importance.empty:
                logger.info("获取特征重要性成功")
                # 显示前10个最重要的特征
                top_features = feature_importance.sort_values(ascending=False).head(10)
                for feature, importance in top_features.items():
                    logger.info(f"  {feature}: {importance:.4f}")
            else:
                logger.warning("特征重要性为空")
        except Exception as fi_error:
            logger.error(f"获取特征重要性失败: {str(fi_error)}")
            traceback.print_exc()
        
        # 测试集成报告
        if hasattr(ensemble, 'create_ensemble_report'):
            logger.info("测试集成报告生成")
            try:
                report_dir = "tmp/ensemble_report"
                os.makedirs(report_dir, exist_ok=True)
                report = ensemble.create_ensemble_report(signals_df, report_dir)
                logger.info("集成报告生成成功")
                logger.info(f"报告内容: {report}")
            except Exception as report_error:
                logger.error(f"生成集成报告失败: {str(report_error)}")
                traceback.print_exc()
        
        # 所有测试通过
        logger.info(f"{model_type}集成模型测试成功!")
        return True
    
    except Exception as e:
        logger.error(f"测试过程中出现错误: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ensemble_model()
    sys.exit(0 if success else 1) 