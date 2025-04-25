#!/usr/bin/env python
"""
测试logger修复
用于验证AdaptiveEnsemble及其子类中的logger修复是否有效
"""
import sys
import os
import pandas as pd
import numpy as np
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_quant.utils.logger import logger, set_log_level
from crypto_quant.strategies.technical.macd_strategy import MACDStrategy
from crypto_quant.strategies.ml_based.enhanced_lstm_strategy import EnhancedLSTMStrategy
from crypto_quant.strategies.hybrid.gradient_boosting_ensemble import GradientBoostingEnsemble
from crypto_quant.strategies.hybrid.neural_network_ensemble import NeuralNetworkEnsemble

# 设置日志级别
set_log_level('INFO')

def create_test_data(rows=200):
    """创建测试数据"""
    # 生成日期索引
    import datetime
    start_date = datetime.datetime.now() - datetime.timedelta(days=rows)
    dates = [start_date + datetime.timedelta(days=i) for i in range(rows)]
    
    # 生成随机价格数据
    np.random.seed(42)  # 设置随机种子
    close = np.random.randn(rows).cumsum() + 100  # 初始价格100
    
    # 添加一些波动性，但确保high > close > low (合理的价格关系)
    daily_volatility = np.abs(np.random.normal(0, 0.02, size=rows))
    high = close * (1 + daily_volatility)  # 确保high总是高于close
    low = close * (1 - daily_volatility)   # 确保low总是低于close
    open_price = low + (high - low) * np.random.random(size=rows)  # open在high和low之间
    volume = np.abs(np.random.normal(loc=1000, scale=200, size=len(dates)))
    
    # 创建直接的数值市场状态，避免分类问题
    # 0=ranging_stable, 1=trending_stable, 2=ranging_volatile, 3=trending_volatile
    market_regime_num = np.array([i % 4 for i in range(rows)])
    
    # 创建DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'market_regime_num': market_regime_num  # 直接使用数值编码，而不是分类类型
    }, index=dates)
    
    return df

def test_gradient_boosting_ensemble():
    """测试梯度提升集成策略的logger修复"""
    logger.info("测试梯度提升集成策略的logger修复...")
    
    # 创建测试数据
    df = create_test_data()
    
    # 创建基础策略
    macd_strategy = MACDStrategy()
    lstm_strategy = EnhancedLSTMStrategy(sequence_length=10, hidden_dim=32)
    
    # 生成基础策略信号
    df['macd_position'] = macd_strategy.generate_signals(df.copy())['position']
    
    # 不训练LSTM模型，直接生成随机信号作为测试
    np.random.seed(42)
    df['lstm_position'] = np.random.choice([-1, 0, 1], size=len(df))
    
    # 创建集成策略
    base_strategies = [
        {'strategy': macd_strategy, 'name': 'macd'},
        {'strategy': lstm_strategy, 'name': 'lstm'}
    ]
    
    # 临时模型路径
    model_path = "tmp/test_gb_model.joblib"
    os.makedirs("tmp", exist_ok=True)
    
    gb_ensemble = GradientBoostingEnsemble(
        base_strategies=base_strategies,
        window_size=20,
        retrain_interval=50,
        min_train_samples=50,
        feature_columns=['close', 'volume', 'macd_position', 'lstm_position', 'market_regime_num'],
        target_column='future_return_1d',
        prediction_threshold=0.001,
        model_path=model_path
    )
    
    # 测试prepare_features方法（应该使用self.logger）
    try:
        logger.info("测试prepare_features方法...")
        # 设置suppress_warnings=True来抑制非关键警告
        features_df = gb_ensemble.prepare_features(df.copy(), current_idx=100, is_training=True, suppress_warnings=True)
        logger.info("prepare_features方法测试通过")
    except Exception as e:
        logger.error(f"prepare_features方法测试失败: {str(e)}")
        return False
    
    # 测试generate_signals方法（应该使用self.logger）
    try:
        logger.info("测试generate_signals方法...")
        # 只测试前150条数据，减少训练时间
        # 设置suppress_warnings=True来抑制非关键警告
        signals_df = gb_ensemble.generate_signals(df.copy().iloc[:150], suppress_warnings=True)
        logger.info("generate_signals方法测试通过")
    except Exception as e:
        logger.error(f"generate_signals方法测试失败: {str(e)}")
        return False
    
    # 所有测试通过
    logger.info("梯度提升集成策略logger修复测试通过")
    return True

def test_neural_network_ensemble():
    """测试神经网络集成策略的logger修复"""
    logger.info("测试神经网络集成策略的logger修复...")
    
    # 创建测试数据
    df = create_test_data()
    
    # 创建基础策略
    macd_strategy = MACDStrategy()
    lstm_strategy = EnhancedLSTMStrategy(sequence_length=10, hidden_dim=32)
    
    # 生成基础策略信号
    df['macd_position'] = macd_strategy.generate_signals(df.copy())['position']
    
    # 不训练LSTM模型，直接生成随机信号作为测试
    np.random.seed(42)
    df['lstm_position'] = np.random.choice([-1, 0, 1], size=len(df))
    
    # 创建集成策略
    base_strategies = [
        {'strategy': macd_strategy, 'name': 'macd'},
        {'strategy': lstm_strategy, 'name': 'lstm'}
    ]
    
    # 临时模型路径
    model_path = "tmp/test_nn_model.pt"
    os.makedirs("tmp", exist_ok=True)
    
    nn_ensemble = NeuralNetworkEnsemble(
        base_strategies=base_strategies,
        window_size=20,
        retrain_interval=50,
        min_train_samples=50,
        feature_columns=['close', 'volume', 'macd_position', 'lstm_position', 'market_regime_num'],
        target_column='future_return_1d',
        prediction_threshold=0.001,
        hidden_dim=16,
        num_layers=1,
        epochs=5,  # 使用较少的训练轮数加快测试
        model_path=model_path
    )
    
    # 测试prepare_features方法（应该使用self.logger）
    try:
        logger.info("测试prepare_features方法...")
        # 设置suppress_warnings=True来抑制非关键警告
        features_df = nn_ensemble.prepare_features(df.copy(), current_idx=100, is_training=True, suppress_warnings=True)
        logger.info("prepare_features方法测试通过")
    except Exception as e:
        logger.error(f"prepare_features方法测试失败: {str(e)}")
        return False
    
    # 测试generate_signals方法（应该使用self.logger）
    try:
        logger.info("测试generate_signals方法...")
        # 只测试前150条数据，减少训练时间
        # 设置suppress_warnings=True来抑制非关键警告
        signals_df = nn_ensemble.generate_signals(df.copy().iloc[:150], suppress_warnings=True)
        logger.info("generate_signals方法测试通过")
    except Exception as e:
        logger.error(f"generate_signals方法测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # 所有测试通过
    logger.info("神经网络集成策略logger修复测试通过")
    return True

def main():
    """主函数"""
    logger.info("开始测试logger修复...")
    
    # 测试梯度提升集成策略
    gb_success = test_gradient_boosting_ensemble()
    
    # 测试神经网络集成策略
    nn_success = test_neural_network_ensemble()
    
    # 输出结果
    if gb_success and nn_success:
        logger.info("所有测试通过，logger修复有效!")
        return 0
    else:
        logger.error("测试失败，logger修复可能有问题!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 