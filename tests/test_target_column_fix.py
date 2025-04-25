#!/usr/bin/env python
"""
测试目标列一致性修复
用于验证AdaptiveEnsemble及其子类中针对目标列一致性的修复是否有效
"""
import sys
import os
import pandas as pd
import numpy as np
import datetime
import unittest

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_quant.utils.logger import logger, set_log_level
from crypto_quant.strategies.technical.macd_strategy import MACDStrategy
from crypto_quant.strategies.ml_based.enhanced_lstm_strategy import EnhancedLSTMStrategy
from crypto_quant.strategies.hybrid.gradient_boosting_ensemble import GradientBoostingEnsemble

# 设置日志级别
set_log_level('INFO')

def create_test_data(rows=200):
    """创建测试数据"""
    # 生成日期索引
    start_date = datetime.datetime.now() - datetime.timedelta(days=rows)
    dates = [start_date + datetime.timedelta(days=i) for i in range(rows)]
    
    # 生成随机价格数据
    np.random.seed(42)  # 设置随机种子
    close = np.random.randn(rows).cumsum() + 100  # 初始价格100
    
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
    
    return df

class TestTargetColumnFix(unittest.TestCase):
    """测试目标列一致性修复"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试数据
        self.df = create_test_data()
        
        # 创建基础策略
        self.macd_strategy = MACDStrategy()
        self.lstm_strategy = EnhancedLSTMStrategy(sequence_length=10, hidden_dim=32)
        
        # 生成基础策略信号
        self.df['macd_position'] = self.macd_strategy.generate_signals(self.df.copy())['position']
        
        # 不训练LSTM模型，直接生成随机信号作为测试
        np.random.seed(42)
        self.df['lstm_position'] = np.random.choice([-1, 0, 1], size=len(self.df))
        
        # 创建基础策略列表
        self.base_strategies = [
            {'strategy': self.macd_strategy, 'name': 'macd'},
            {'strategy': self.lstm_strategy, 'name': 'lstm'}
        ]
        
        # 临时模型路径
        self.model_path = "tmp/test_target_fix_model.joblib"
        os.makedirs("tmp", exist_ok=True)
    
    def test_existing_target_column(self):
        """测试目标列已存在的情况"""
        logger.info("测试目标列已存在的情况")
        
        # 添加目标列
        self.df['future_return_1d'] = self.df['close'].pct_change(-1)
        
        # 创建集成策略
        ensemble = GradientBoostingEnsemble(
            base_strategies=self.base_strategies,
            window_size=20,
            retrain_interval=50,
            min_train_samples=50,
            feature_columns=['close', 'volume', 'macd_position', 'lstm_position'],
            target_column='future_return_1d',
            prediction_threshold=0.001,
            model_path=self.model_path
        )
        
        # 获取一个特征子集
        features_df = ensemble.prepare_features(self.df.copy(), current_idx=100, is_training=True)
        
        # 验证目标列是否存在
        self.assertIn('future_return_1d', features_df.columns, "目标列应该存在于特征DataFrame中")
        
        # 获取目标列
        target = ensemble._ensure_target_column(features_df)
        
        # 验证目标列是否正确
        self.assertIsInstance(target, pd.Series, "目标应该是Series类型")
        self.assertEqual(len(target), len(features_df), "目标长度应该与特征相同")
        logger.info("目标列已存在的测试通过")
    
    def test_nonexistent_target_column(self):
        """测试目标列不存在的情况"""
        logger.info("测试目标列不存在的情况")
        
        # 创建集成策略，使用不存在的目标列
        ensemble = GradientBoostingEnsemble(
            base_strategies=self.base_strategies,
            window_size=20,
            retrain_interval=50,
            min_train_samples=50,
            feature_columns=['close', 'volume', 'macd_position', 'lstm_position'],
            target_column='nonexistent_target',
            prediction_threshold=0.001,
            model_path=self.model_path
        )
        
        # 获取一个特征子集
        features_df = ensemble.prepare_features(self.df.copy(), current_idx=100, is_training=True)
        
        # 验证目标列是否被生成
        self.assertIn('future_return_1d', features_df.columns, "应该生成默认的未来收益列")
        
        # 获取目标列
        target = ensemble._ensure_target_column(features_df)
        
        # 验证目标列是否有效
        self.assertIsInstance(target, pd.Series, "目标应该是Series类型")
        self.assertEqual(len(target), len(features_df), "目标长度应该与特征相同")
        logger.info("目标列不存在的测试通过")
    
    def test_future_return_parsing(self):
        """测试未来收益列名解析"""
        logger.info("测试未来收益列名解析")
        
        # 创建集成策略，使用未来5天收益作为目标
        ensemble = GradientBoostingEnsemble(
            base_strategies=self.base_strategies,
            window_size=20,
            retrain_interval=50,
            min_train_samples=50,
            feature_columns=['close', 'volume', 'macd_position', 'lstm_position'],
            target_column='future_return_5d',
            prediction_threshold=0.001,
            model_path=self.model_path
        )
        
        # 获取一个特征子集
        features_df = ensemble.prepare_features(self.df.copy(), current_idx=100, is_training=True)
        
        # 验证未来收益列是否被正确生成
        self.assertIn('future_return_5d', features_df.columns, "应该正确生成5天未来收益列")
        self.assertIn('future_return_1d', features_df.columns, "应该同时生成1天未来收益列")
        self.assertIn('future_return_3d', features_df.columns, "应该同时生成3天未来收益列")
        
        # 获取目标列
        target = ensemble._ensure_target_column(features_df)
        
        # 验证目标列是否有效
        self.assertIsInstance(target, pd.Series, "目标应该是Series类型")
        self.assertEqual(len(target), len(features_df), "目标长度应该与特征相同")
        logger.info("未来收益列名解析测试通过")
    
    def test_returns_target(self):
        """测试使用returns作为目标列"""
        logger.info("测试使用returns作为目标列")
        
        # 创建集成策略，使用returns作为目标
        ensemble = GradientBoostingEnsemble(
            base_strategies=self.base_strategies,
            window_size=20,
            retrain_interval=50,
            min_train_samples=50,
            feature_columns=['close', 'volume', 'macd_position', 'lstm_position'],
            target_column='returns',
            prediction_threshold=0.001,
            model_path=self.model_path
        )
        
        # 获取一个特征子集
        features_df = ensemble.prepare_features(self.df.copy(), current_idx=100, is_training=True)
        
        # 验证returns列是否被生成
        self.assertIn('returns', features_df.columns, "应该生成returns列")
        
        # 获取目标列
        target = ensemble._ensure_target_column(features_df)
        
        # 验证目标列是否有效
        self.assertIsInstance(target, pd.Series, "目标应该是Series类型")
        self.assertEqual(len(target), len(features_df), "目标长度应该与特征相同")
        logger.info("使用returns作为目标列的测试通过")
    
    def test_fallback_mechanisms(self):
        """测试回退机制"""
        logger.info("测试回退机制")
        
        # 创建一个不包含close列的数据框
        df_no_close = self.df.copy()
        df_no_close.drop(columns=['close'], inplace=True)
        
        # 创建集成策略
        ensemble = GradientBoostingEnsemble(
            base_strategies=self.base_strategies,
            window_size=20,
            retrain_interval=50,
            min_train_samples=50,
            feature_columns=['volume', 'macd_position', 'lstm_position'],
            target_column='nonexistent_target',
            prediction_threshold=0.001,
            model_path=self.model_path
        )
        
        # 测试position列回退
        features_df = pd.DataFrame({
            'macd_position': [1, 0, -1, 1, 0],
            'lstm_position': [-1, 1, 0, 0, 1]
        })
        
        target = ensemble._ensure_target_column(features_df)
        
        # 验证目标列是否有效
        self.assertIsInstance(target, pd.Series, "目标应该是Series类型")
        self.assertEqual(len(target), len(features_df), "目标长度应该与特征相同")
        
        # 测试零序列回退
        features_df = pd.DataFrame({
            'some_feature': [1, 2, 3, 4, 5]
        })
        
        target = ensemble._ensure_target_column(features_df)
        
        # 验证目标列是否有效
        self.assertIsInstance(target, pd.Series, "目标应该是Series类型")
        self.assertEqual(len(target), len(features_df), "目标长度应该与特征相同")
        self.assertTrue((target == 0).all(), "所有值应该为0")
        
        logger.info("回退机制测试通过")
    
    def test_generate_signals(self):
        """测试generate_signals方法中的目标列处理"""
        logger.info("测试generate_signals方法中的目标列处理")
        
        # 创建集成策略
        ensemble = GradientBoostingEnsemble(
            base_strategies=self.base_strategies,
            window_size=20,
            retrain_interval=20,  # 小间隔，确保会触发多次训练
            min_train_samples=50,
            feature_columns=['close', 'volume', 'macd_position', 'lstm_position'],
            target_column='future_return_1d',
            prediction_threshold=0.001,
            model_path=self.model_path
        )
        
        # 只使用部分数据进行测试，以加快速度
        test_df = self.df.iloc[:100].copy()
        
        # 生成信号
        try:
            result_df = ensemble.generate_signals(test_df)
            self.assertIn('signal', result_df.columns, "结果应包含signal列")
            self.assertIn('position', result_df.columns, "结果应包含position列")
            self.assertIn('ensemble_confidence', result_df.columns, "结果应包含ensemble_confidence列")
            logger.info("generate_signals测试通过")
        except Exception as e:
            self.fail(f"generate_signals抛出异常: {e}")
    
    def tearDown(self):
        """清理测试环境"""
        # 清理临时文件
        if os.path.exists(self.model_path):
            try:
                os.remove(self.model_path)
            except:
                pass

def main():
    """主函数"""
    logger.info("开始测试目标列一致性修复...")
    
    # 运行测试
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    logger.info("目标列一致性测试完成")

if __name__ == "__main__":
    main() 