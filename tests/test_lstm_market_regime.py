"""
测试LSTM策略和混合策略对市场状态的处理

专门测试修复后的代码处理市场状态字符串的能力
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_quant.strategies.ml_based.enhanced_lstm_strategy import EnhancedLSTMStrategy
from crypto_quant.strategies.hybrid.macd_lstm_hybrid_strategy import MACDLSTMHybridStrategy
from crypto_quant.utils.logger import logger, set_log_level

# 设置日志级别
set_log_level('INFO')


def create_test_data_with_regime(rows=200):
    """创建带有市场状态的测试数据"""
    # 生成日期索引
    start_date = datetime.now() - timedelta(days=rows)
    dates = [start_date + timedelta(days=i) for i in range(rows)]
    
    # 生成随机价格数据
    np.random.seed(42)  # 设置随机种子，使结果可复现
    close = np.random.randn(rows).cumsum() + 100  # 初始价格100，随机游走
    
    # 添加一些波动性
    volatility = np.abs(np.random.randn(rows) * 2)
    high = close + volatility
    low = close - volatility
    open_price = close - volatility / 2 + np.random.randn(rows)
    volume = np.abs(np.random.randn(rows) * 1000) + 1000
    
    # 创建DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # 添加市场状态列
    market_regimes = ['trending_volatile', 'ranging_volatile', 'trending_stable', 'ranging_stable', 'unknown']
    df['market_regime'] = [market_regimes[i % len(market_regimes)] for i in range(rows)]
    
    return df


def test_enhanced_lstm_strategy_with_market_regime():
    """测试增强型LSTM策略处理市场状态"""
    logger.info("测试增强型LSTM策略处理市场状态...")
    
    # 创建测试数据
    df = create_test_data_with_regime()
    
    # 创建增强型LSTM策略
    lstm_strategy = EnhancedLSTMStrategy(
        sequence_length=10,  # 使用较小的序列长度加快测试
        hidden_dim=64,  # 使用较小的隐藏层维度加快测试
        feature_engineering=True
    )
    
    try:
        # 训练策略
        logger.info("训练LSTM策略...")
        lstm_strategy.train(df, epochs=3)  # 仅训练3个epoch加快测试
        
        # 生成信号
        logger.info("生成LSTM策略信号...")
        lstm_df = lstm_strategy.generate_signals(df)
        
        if lstm_df is not None and isinstance(lstm_df, pd.DataFrame) and 'position' in lstm_df.columns:
            logger.info("LSTM策略成功生成信号")
            return True
        else:
            logger.error("LSTM策略未生成有效信号")
            return False
            
    except Exception as e:
        logger.error(f"LSTM策略测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_macd_lstm_hybrid_strategy_with_market_regime():
    """测试MACD-LSTM混合策略处理市场状态"""
    logger.info("测试MACD-LSTM混合策略处理市场状态...")
    
    # 创建测试数据
    df = create_test_data_with_regime()
    
    # 创建MACD-LSTM混合策略
    hybrid_strategy = MACDLSTMHybridStrategy(
        # LSTM参数
        lstm_sequence_length=10,
        lstm_hidden_dim=64,
        lstm_feature_engineering=True,
        
        # 混合参数
        ensemble_method='expert'
    )
    
    try:
        # 生成信号
        logger.info("生成混合策略信号...")
        hybrid_df = hybrid_strategy.generate_signals(df)
        
        if hybrid_df is not None and isinstance(hybrid_df, pd.DataFrame) and 'position' in hybrid_df.columns:
            logger.info("混合策略成功生成信号")
            return True
        else:
            logger.error("混合策略未生成有效信号")
            return False
            
    except Exception as e:
        logger.error(f"混合策略测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """主函数"""
    logger.info("开始测试市场状态处理修复...")
    
    # 测试增强型LSTM策略
    lstm_success = test_enhanced_lstm_strategy_with_market_regime()
    
    # 测试MACD-LSTM混合策略
    hybrid_success = test_macd_lstm_hybrid_strategy_with_market_regime()
    
    # 输出结果
    if lstm_success and hybrid_success:
        logger.info("所有测试通过，修复有效!")
    else:
        logger.error("测试失败，修复可能有问题!")


if __name__ == "__main__":
    main() 