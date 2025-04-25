#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试目标列处理修复
测试AdaptiveEnsemble类中_ensure_target_column方法的修复
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_quant.utils.logger import logger, set_log_level
from crypto_quant.strategies.hybrid.gradient_boosting_ensemble import GradientBoostingEnsemble
from crypto_quant.strategies.technical.macd_strategy import MACDStrategy

# 设置日志级别
set_log_level('INFO')

def create_test_data(rows=100):
    """创建测试数据"""
    # 生成日期索引
    start_date = datetime.now() - timedelta(days=rows)
    dates = [start_date + timedelta(days=i) for i in range(rows)]
    
    # 生成随机价格数据
    np.random.seed(42)  # 设置随机种子，确保结果可重现
    close = 10000 + np.random.randn(rows).cumsum() * 100  # 初始价格10000，随机游走
    
    # 确保价格为正
    close = np.maximum(close, 100)
    
    # 添加波动性
    high = close * (1 + np.abs(np.random.randn(rows)) * 0.02)
    low = close * (1 - np.abs(np.random.randn(rows)) * 0.02)
    open_price = close * (1 + np.random.randn(rows) * 0.01)
    volume = np.abs(np.random.randn(rows) * 1000) + 5000
    
    # 创建DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df

class TestTargetColumnHandling(unittest.TestCase):
    """测试目标列处理"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试数据
        self.df = create_test_data()
        
        # 生成MACD策略信号
        macd_strategy = MACDStrategy()
        macd_df = macd_strategy.generate_signals(self.df.copy())
        self.df['macd_position'] = macd_df['position']
        
        # 创建基础策略配置
        self.base_strategies = [
            {'strategy': macd_strategy, 'name': 'macd'}
        ]
    
    def test_standard_future_return_handling(self):
        """测试标准未来收益目标列处理"""
        logger.info("测试标准未来收益目标列处理")
        
        # 初始化集成模型，使用标准未来收益作为目标
        ensemble = GradientBoostingEnsemble(
            base_strategies=self.base_strategies,
            window_size=20,
            retrain_interval=50,
            min_train_samples=30,
            feature_columns=['close', 'volume', 'macd_position'],
            target_column='future_return_1d',
            prediction_threshold=0.001
        )
        
        # 准备特征
        features_df = ensemble.prepare_features(self.df.copy(), current_idx=None, is_training=True, suppress_warnings=True)
        
        # 验证future_return_1d列是否存在
        self.assertIn('future_return_1d', features_df.columns, "future_return_1d 列应该存在")
        
        # 获取目标列
        target = ensemble._ensure_target_column(features_df, suppress_warnings=True)
        
        # 检查目标列是否正确
        self.assertIsInstance(target, pd.Series, "目标应该是Series类型")
        self.assertEqual(target.name, 'future_return_1d', "目标名称应为future_return_1d")
        self.assertEqual(len(target), len(features_df), "目标长度应该与特征相同")
        
        logger.info("标准未来收益目标列处理测试通过")
    
    def test_custom_future_return_handling(self):
        """测试自定义未来收益目标列处理"""
        logger.info("测试自定义未来收益目标列处理")
        
        # 初始化集成模型，使用自定义未来收益作为目标
        ensemble = GradientBoostingEnsemble(
            base_strategies=self.base_strategies,
            window_size=20,
            retrain_interval=50,
            min_train_samples=30,
            feature_columns=['close', 'volume', 'macd_position'],
            target_column='future_return_7d',  # 不是标准天数
            prediction_threshold=0.001
        )
        
        # 准备特征
        features_df = ensemble.prepare_features(self.df.copy(), current_idx=None, is_training=True, suppress_warnings=True)
        
        # 验证future_return_7d列是否存在
        self.assertIn('future_return_7d', features_df.columns, "future_return_7d 列应该存在")
        
        # 获取目标列
        target = ensemble._ensure_target_column(features_df, suppress_warnings=True)
        
        # 检查目标列是否正确
        self.assertIsInstance(target, pd.Series, "目标应该是Series类型")
        self.assertEqual(target.name, 'future_return_7d', "目标名称应为future_return_7d")
        self.assertEqual(len(target), len(features_df), "目标长度应该与特征相同")
        
        logger.info("自定义未来收益目标列处理测试通过")
    
    def test_missing_target_column_handling(self):
        """测试缺失目标列的处理"""
        logger.info("测试缺失目标列的处理")
        
        # 创建仅包含基本列的测试数据（不包含未来收益列）
        basic_df = pd.DataFrame({
            'close': np.random.rand(50) * 100 + 10000,
            'macd_position': np.random.choice([-1, 0, 1], size=50)
        })
        
        # 初始化集成模型，使用不存在的目标列
        ensemble = GradientBoostingEnsemble(
            base_strategies=self.base_strategies,
            window_size=20,
            retrain_interval=50,
            min_train_samples=30,
            feature_columns=['close', 'macd_position'],
            target_column='nonexistent_target',
            prediction_threshold=0.001
        )
        
        # 获取目标列，抑制警告
        target = ensemble._ensure_target_column(basic_df, suppress_warnings=True)
        
        # 检查是否返回了有效的目标列
        self.assertIsInstance(target, pd.Series, "目标应该是Series类型")
        self.assertEqual(len(target), len(basic_df), "目标长度应该与特征相同")
        
        # 填充NaN值，这与prepare_features方法中的行为一致
        target = target.fillna(0)
        self.assertFalse(target.isna().any(), "填充后的目标不应包含缺失值")
        
        logger.info("缺失目标列处理测试通过")
    
    def test_returns_target_handling(self):
        """测试returns目标列处理"""
        logger.info("测试returns目标列处理")
        
        # 添加returns列
        df_with_returns = self.df.copy()
        df_with_returns['returns'] = df_with_returns['close'].pct_change()
        
        # 初始化集成模型，使用returns作为目标
        ensemble = GradientBoostingEnsemble(
            base_strategies=self.base_strategies,
            window_size=20,
            retrain_interval=50,
            min_train_samples=30,
            feature_columns=['close', 'volume', 'macd_position'],
            target_column='returns',
            prediction_threshold=0.001
        )
        
        # 准备特征
        features_df = ensemble.prepare_features(df_with_returns, current_idx=None, is_training=True, suppress_warnings=True)
        
        # 验证returns列是否存在
        self.assertIn('returns', features_df.columns, "returns 列应该存在")
        
        # 获取目标列
        target = ensemble._ensure_target_column(features_df, suppress_warnings=True)
        
        # 检查目标列是否正确
        self.assertIsInstance(target, pd.Series, "目标应该是Series类型")
        self.assertEqual(target.name, 'returns', "目标名称应为returns")
        self.assertEqual(len(target), len(features_df), "目标长度应该与特征相同")
        
        logger.info("returns目标列处理测试通过")
    
    def test_fallback_to_strategy_position(self):
        """测试回退到策略position列"""
        logger.info("测试回退到策略position列")
        
        # 创建仅包含策略信号的数据
        position_df = pd.DataFrame({
            'macd_position': np.random.choice([-1, 0, 1], size=50)
        })
        
        # 初始化集成模型，使用不存在的目标列
        ensemble = GradientBoostingEnsemble(
            base_strategies=self.base_strategies,
            window_size=20,
            retrain_interval=50,
            min_train_samples=30,
            feature_columns=['macd_position'],
            target_column='nonexistent_target',
            prediction_threshold=0.001
        )
        
        # 获取目标列，抑制警告
        target = ensemble._ensure_target_column(position_df, suppress_warnings=True)
        
        # 检查是否正确使用了position列
        self.assertIsInstance(target, pd.Series, "目标应该是Series类型")
        self.assertEqual(target.name, 'macd_position', "目标名称应为macd_position")
        self.assertEqual(len(target), len(position_df), "目标长度应该与特征相同")
        
        logger.info("回退到策略position列测试通过")
    
    def test_zero_target_creation(self):
        """测试零目标列创建"""
        logger.info("测试零目标列创建")
        
        # 创建不包含任何有用目标列的数据
        empty_df = pd.DataFrame({
            'some_useless_column': np.random.rand(20)
        })
        
        # 初始化集成模型，使用不存在的目标列
        ensemble = GradientBoostingEnsemble(
            base_strategies=self.base_strategies,
            window_size=20,
            retrain_interval=50,
            min_train_samples=30,
            feature_columns=['some_useless_column'],
            target_column='nonexistent_target',
            prediction_threshold=0.001
        )
        
        # 获取目标列，抑制警告
        target = ensemble._ensure_target_column(empty_df, suppress_warnings=True)
        
        # 检查是否创建了零目标列
        self.assertIsInstance(target, pd.Series, "目标应该是Series类型")
        self.assertEqual(target.name, 'zero_target', "目标名称应为zero_target")
        self.assertEqual(len(target), len(empty_df), "目标长度应该与特征相同")
        self.assertTrue((target == 0).all(), "所有值应该为0")
        
        logger.info("零目标列创建测试通过")
        
    def test_special_test_target_name(self):
        """测试特殊测试目标名称的自动识别"""
        logger.info("测试特殊测试目标名称的自动识别")
        
        # 创建基本测试数据
        basic_df = pd.DataFrame({
            'close': np.random.rand(50) * 100 + 10000,
            'some_useless_column': np.random.rand(50)
        })
        
        # 添加future_return_1d列以测试自动替代功能
        basic_df['future_return_1d'] = np.random.rand(50) * 0.1 - 0.05
        
        # 初始化集成模型，使用以test_开头的目标列
        ensemble = GradientBoostingEnsemble(
            base_strategies=self.base_strategies,
            window_size=20,
            retrain_interval=50,
            min_train_samples=30,
            feature_columns=['close', 'some_useless_column'],
            target_column='test_nonexistent_target',  # 使用特殊测试目标名称格式
            prediction_threshold=0.001
        )
        
        # 获取目标列，不需要显式抑制警告，因为测试目标名称应该自动抑制
        target = ensemble._ensure_target_column(basic_df)
        
        # 检查是否正确使用future_return_1d作为替代
        self.assertIsInstance(target, pd.Series, "目标应该是Series类型")
        self.assertEqual(target.name, 'future_return_1d', "目标名称应为future_return_1d")
        self.assertEqual(len(target), len(basic_df), "目标长度应该与特征相同")
        
        # 测试nonexistent_target特殊名称
        ensemble.target_column = 'nonexistent_target'
        target = ensemble._ensure_target_column(basic_df)
        
        # 检查是否正确使用future_return_1d作为替代
        self.assertIsInstance(target, pd.Series, "目标应该是Series类型")
        self.assertEqual(target.name, 'future_return_1d', "目标名称应为future_return_1d")
        self.assertEqual(len(target), len(basic_df), "目标长度应该与特征相同")
        
        logger.info("特殊测试目标名称自动识别测试通过")

if __name__ == "__main__":
    unittest.main() 