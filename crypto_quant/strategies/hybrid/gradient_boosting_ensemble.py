"""
基于梯度提升的自适应集成策略
使用XGBoost实现高级策略集成
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import os
import joblib
from datetime import datetime
import traceback

from .adaptive_ensemble import AdaptiveEnsemble
from ...utils.logger import logger


class GradientBoostingEnsemble(AdaptiveEnsemble):
    """
    基于梯度提升的集成策略
    
    使用XGBoost对多个基础策略信号进行集成，动态适应不同市场环境
    """
    
    def __init__(
        self,
        base_strategies: List[Dict],
        window_size: int = 50,
        retrain_interval: int = 100,
        min_train_samples: int = 200,
        feature_columns: Optional[List[str]] = None,
        target_column: str = 'returns',
        prediction_threshold: float = 0.0,
        model_params: Optional[Dict] = None,
        model_path: Optional[str] = None,
    ):
        """
        初始化梯度提升集成策略
        
        Args:
            base_strategies: 基础策略列表，每个策略是一个字典，包含策略对象和权重
            window_size: 用于特征计算的窗口大小
            retrain_interval: 模型重新训练的间隔
            min_train_samples: 训练所需的最小样本数
            feature_columns: 用于训练的特征列名列表
            target_column: 目标变量列名
            prediction_threshold: 预测阈值，预测值超过此阈值才产生交易信号
            model_params: XGBoost模型参数
            model_path: 模型保存路径，如果不为None则尝试加载已保存的模型
        """
        # 调用父类初始化
        super().__init__(
            base_strategies=base_strategies,
            window_size=window_size,
            retrain_interval=retrain_interval,
            min_train_samples=min_train_samples,
            feature_columns=feature_columns,
            target_column=target_column,
            prediction_threshold=prediction_threshold,
        )
        
        # 确保logger属性存在（增加健壮性）
        if not hasattr(self, 'logger'):
            self.logger = logger
        
        # 设置默认特征列
        if not self.feature_columns:
            # 基础特征列
            self.feature_columns = [
                'close', 'rsi', 'volatility', 'macd', 'macd_signal', 'macd_hist'
            ]
            # 添加基础策略信号列
            for i, strategy_info in enumerate(self.base_strategies):
                strategy_name = strategy_info.get('name', f"strategy_{i}")
                self.feature_columns.append(f"{strategy_name}_position")
        
        # XGBoost模型参数
        self.model_params = model_params or {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'n_estimators': 100,
            'random_state': 42
        }
        
        self.model_path = model_path
        
        # 尝试加载预训练模型
        if self.model_path and os.path.exists(self.model_path):
            try:
                # 使用新的_load_model方法
                load_success = self._load_model()
                if load_success:
                    logger.info(f"加载预训练模型: {self.model_path}")
                else:
                    logger.warning(f"加载模型失败，将在运行时重新训练")
            except Exception as e:
                logger.error(f"加载模型失败: {str(e)}")
                self._model = None
    
    def prepare_features(self, df: pd.DataFrame, current_idx=None, is_training=False, suppress_warnings=False) -> pd.DataFrame:
        """
        准备集成模型的特征数据
        
        Args:
            df: 包含基础策略信号和市场数据的DataFrame
            current_idx: 当前处理的索引位置，如果为None则使用最后一个索引
            is_training: 是否是训练模式
            suppress_warnings: 是否抑制警告
            
        Returns:
            pd.DataFrame: 带有特征的DataFrame
        """
        # 确保current_idx有值
        if current_idx is None and len(df) > 0:
            current_idx = df.index[-1]
            
        # 调用父类方法处理基本特征
        df_features = super().prepare_features(df, current_idx, is_training, suppress_warnings)
        
        # market_regime处理已移至基类AdaptiveEnsemble中的_ensure_numeric_dataframe方法
        
        return df_features

    def train(self, features_df: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        训练XGBoost集成模型
        
        Args:
            features_df: 特征DataFrame
            y: 目标变量，如果为None则使用features_df中的target_column
        """
        try:
            import xgboost as xgb
            
            # 准备目标变量
            if y is None:
                if self.target_column not in features_df.columns:
                    logger.error(f"目标列 {self.target_column} 不在特征DataFrame中")
                    return
                y = features_df[self.target_column]
            
            # 过滤特征列，只使用存在的列
            available_columns = [col for col in self.feature_columns if col in features_df.columns]
            if len(available_columns) == 0:
                logger.error("没有可用的特征列")
                return
                
            if len(available_columns) < len(self.feature_columns):
                logger.warning(f"部分特征列不可用: {set(self.feature_columns) - set(available_columns)}")
            
            # 筛选特征列
            X = features_df[available_columns].copy()
            X = X.fillna(0)  # 填充缺失值
            
            # 处理分类特征
            categorical_cols = []
            for col_name in X.columns:
                # 获取列的数据类型
                col_series = X[col_name]
                if not isinstance(col_series, pd.Series):
                    logger.warning(f"列 {col_name} 不是Series类型，跳过处理")
                    continue
                    
                col_dtype = col_series.dtype
                if col_dtype == 'object' or col_dtype == 'category':
                    # 如果是分类特征，转换为数值编码
                    try:
                        X[col_name] = pd.Categorical(col_series).codes
                        categorical_cols.append(col_name)
                    except Exception as e:
                        logger.warning(f"处理列 {col_name} 时出错: {str(e)}")
                        # 移除无法转换的列
                        X = X.drop(col_name, axis=1)
            
            if len(categorical_cols) > 0:
                logger.info(f"转换了 {len(categorical_cols)} 个分类特征")
            
            logger.info(f"训练XGBoost模型，特征数量: {len(X.columns)}, 样本数量: {len(X)}")
            
            # 创建XGBoost模型
            model = xgb.XGBRegressor(**self.model_params)
            
            # 训练模型
            model.fit(X, y)
            
            # 存储模型和特征重要性
            self._model = model
            self._is_trained = True
            
            # 更新使用的特征列
            self.feature_columns = list(X.columns)
            
            # 计算特征重要性
            importance = model.feature_importances_
            self._feature_importance = pd.Series(importance, index=X.columns).sort_values(ascending=False)
            
            # 输出特征重要性
            logger.info("特征重要性:")
            for feature, importance in self._feature_importance.iloc[:10].items():
                logger.info(f"  {feature}: {importance:.4f}")
            
            # 保存模型
            if self.model_path:
                model_dir = os.path.dirname(self.model_path)
                if model_dir and not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                
                # 创建一个包含模型和特征列的字典
                save_dict = {
                    'model': model,
                    'feature_columns': self.feature_columns,
                    'feature_importance': self._feature_importance
                }
                
                joblib.dump(save_dict, self.model_path)
                logger.info(f"模型已保存至: {self.model_path}, 特征数量: {len(self.feature_columns)}")
            
        except ImportError:
            logger.error("无法导入xgboost库，请确保已安装")
        except Exception as e:
            logger.error(f"训练模型时出错: {str(e)}")
            traceback.print_exc()
            
    def _load_model(self, model_path: str = None) -> bool:
        """
        从磁盘加载预训练模型
        
        Args:
            model_path: 模型文件路径，如果为None则使用self.model_path
            
        Returns:
            bool: 是否成功加载模型
        """
        path = model_path or self.model_path
        if not path or not os.path.exists(path):
            return False
            
        try:
            # 加载模型
            save_dict = joblib.load(path)
            
            # 如果保存的是字典（新格式），提取模型和特征列
            if isinstance(save_dict, dict) and 'model' in save_dict:
                self._model = save_dict['model']
                
                if 'feature_columns' in save_dict:
                    self.feature_columns = save_dict['feature_columns']
                    logger.info(f"加载特征列: {len(self.feature_columns)}个")
                    
                if 'feature_importance' in save_dict:
                    self._feature_importance = save_dict['feature_importance']
            else:
                # 兼容旧格式，直接加载模型
                self._model = save_dict
                logger.warning("模型文件使用旧格式，没有特征列信息")
                
            self._is_trained = True
            logger.info(f"成功加载模型: {path}")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return False
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        使用XGBoost模型进行预测
        
        Args:
            features: 特征DataFrame
            
        Returns:
            np.ndarray: 预测结果
        """
        if not self._is_trained or self._model is None:
            logger.warning("模型尚未训练，无法进行预测")
            return np.zeros(len(features))
        
        try:
            # 记录开始信息
            logger.info(f"开始预测，特征形状: {features.shape}，特征列: {len(self.feature_columns)}")
            
            # 获取数值型特征
            numeric_features = features.select_dtypes(include=['number'])
            logger.info(f"数值型特征: {numeric_features.shape}")
            
            # 检查自身特征列列表是否为空
            if not self.feature_columns:
                logger.warning("特征列列表为空，将使用所有数值特征")
                self.feature_columns = numeric_features.columns.tolist()
            
            # 检查特征列是否存在
            missing_features = set(self.feature_columns) - set(numeric_features.columns)
            if missing_features:
                logger.warning(f"缺少 {len(missing_features)} 个训练时使用的特征: {missing_features}")
                
                # 添加缺失的特征列，以0填充
                for col in missing_features:
                    numeric_features[col] = 0.0
                    logger.info(f"添加缺失特征 '{col}' 并以0值填充")
            
            # 确保使用与训练时完全相同的特征列和顺序
            X = pd.DataFrame(index=numeric_features.index)
            for col in self.feature_columns:
                if col in numeric_features.columns:
                    X[col] = numeric_features[col]
                else:
                    X[col] = 0.0
                    logger.warning(f"特征 '{col}' 不可用，使用0填充")
            
            # 填充缺失值
            X = X.fillna(0)
            logger.info(f"准备好的特征形状: {X.shape}")
            
            # 检查特征维度是否匹配
            if X.shape[1] != len(self.feature_columns):
                logger.error(f"特征维度不匹配: 预期 {len(self.feature_columns)}，实际 {X.shape[1]}")
                logger.error(f"预期特征: {self.feature_columns}")
                logger.error(f"实际特征: {X.columns.tolist()}")
                # 作为最后的保护措施，确保特征列顺序与训练时一致
                if set(X.columns) == set(self.feature_columns):
                    X = X[self.feature_columns]
                else:
                    # 有可能特征列不匹配，添加缺失列
                    for col in self.feature_columns:
                        if col not in X.columns:
                            X[col] = 0.0
                    X = X[self.feature_columns]
            
            # 输出一些调试信息
            self._debug_feature_info(X)
            
            # 处理分类特征
            for col_name in X.columns:
                # 获取列的数据类型
                col_series = X[col_name]
                if not isinstance(col_series, pd.Series):
                    logger.warning(f"列 {col_name} 不是Series类型，跳过处理")
                    continue
                    
                col_dtype = col_series.dtype
                if col_dtype == 'object' or col_dtype == 'category':
                    # 如果是分类特征，转换为数值编码
                    try:
                        X[col_name] = pd.Categorical(col_series).codes
                    except Exception as e:
                        logger.warning(f"处理列 {col_name} 时出错: {str(e)}")
                        # 对于无法转换的列，尝试使用0填充
                        X[col_name] = 0
            
            # 预测
            try:
                predictions = self._model.predict(X)
                logger.info(f"预测成功，结果形状: {predictions.shape}")
                return predictions
            except Exception as predict_error:
                logger.error(f"模型预测失败: {str(predict_error)}")
                # 特征不匹配的特殊处理
                if "Feature names must match" in str(predict_error):
                    logger.error("特征名称不匹配，尝试使用特征顺序匹配...")
                    import xgboost as xgb
                    # 创建DMatrix直接使用数值，忽略特征名称
                    dmatrix = xgb.DMatrix(X.values)
                    predictions = self._model.predict(dmatrix)
                    logger.info(f"使用DMatrix预测成功，结果形状: {predictions.shape}")
                    return predictions
                else:
                    raise predict_error
            
        except Exception as e:
            logger.error(f"预测时出错: {str(e)}")
            traceback.print_exc()
            return np.zeros(len(features))
    
    def _debug_feature_info(self, X: pd.DataFrame) -> None:
        """
        输出特征信息，用于调试
        
        Args:
            X: 特征DataFrame
        """
        try:
            # 输出基本信息
            logger.info(f"特征维度: {X.shape}")
            logger.info(f"特征列: {X.columns.tolist()}")
            
            # 检查特征统计信息
            logger.info("特征统计信息:")
            for col in X.columns:
                # 获取基本统计量
                mean = X[col].mean()
                std = X[col].std()
                min_val = X[col].min()
                max_val = X[col].max()
                null_count = X[col].isnull().sum()
                
                logger.info(f"  {col}: 均值={mean:.4f}, 标准差={std:.4f}, 最小值={min_val:.4f}, 最大值={max_val:.4f}, 缺失值={null_count}")
            
            # 检查模型类型和特征期望
            if hasattr(self._model, 'feature_names_in_'):
                model_feature_names = self._model.feature_names_in_
                logger.info(f"模型期望的特征名称: {model_feature_names}")
                
                # 检查特征匹配情况
                missing_in_input = set(model_feature_names) - set(X.columns)
                if missing_in_input:
                    logger.warning(f"输入中缺少模型期望的特征: {missing_in_input}")
                
                extra_in_input = set(X.columns) - set(model_feature_names)
                if extra_in_input:
                    logger.warning(f"输入中包含模型未使用的特征: {extra_in_input}")
        
        except Exception as e:
            logger.warning(f"输出调试信息时出错: {str(e)}")
            
    def debug_model_structure(self) -> None:
        """
        调试模型结构，输出模型信息
        """
        if not self._is_trained or self._model is None:
            logger.warning("模型尚未训练，无法调试结构")
            return
            
        try:
            # 输出模型的基本信息
            logger.info(f"模型类型: {type(self._model)}")
            
            # 检查是否是XGBoost模型
            if hasattr(self._model, 'get_booster'):
                booster = self._model.get_booster()
                logger.info(f"Booster类型: {type(booster)}")
                
                # 获取特征信息
                if hasattr(self._model, 'feature_names_in_'):
                    logger.info(f"特征数量: {len(self._model.feature_names_in_)}")
                    logger.info(f"特征名称: {self._model.feature_names_in_}")
                
                # 获取模型参数
                params = self._model.get_params()
                logger.info(f"模型参数: {params}")
                
                # 获取特征重要性
                if self._feature_importance is not None:
                    logger.info("特征重要性前10:")
                    for feature, importance in self._feature_importance.iloc[:10].items():
                        logger.info(f"  {feature}: {importance:.4f}")
            else:
                logger.info("模型不是标准XGBoost模型，无法获取详细信息")
                
        except Exception as e:
            logger.error(f"调试模型结构时出错: {str(e)}")
            traceback.print_exc()
    
    def create_ensemble_report(self, df: pd.DataFrame, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        创建集成策略报告
        
        Args:
            df: 带有信号和价格数据的DataFrame
            output_dir: 输出目录，如果为None则不保存图表
            
        Returns:
            Dict: 报告内容
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        report = {}
        
        # 计算策略性能
        if 'position' in df.columns and 'close' in df.columns:
            # 计算策略收益
            df['strategy_returns'] = df['position'].shift(1) * df['close'].pct_change()
            df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
            
            report['final_return'] = df['cumulative_returns'].iloc[-1] - 1
            report['sharpe_ratio'] = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
            report['max_drawdown'] = (df['cumulative_returns'] / df['cumulative_returns'].cummax() - 1).min()
            
            # 统计交易次数和胜率
            df['trade'] = df['position'].diff().abs()
            report['trade_count'] = df['trade'].sum() / 2  # 每次交易包含进出两次操作
            
            df['trade_profit'] = (df['strategy_returns'] > 0).astype(int)
            if df['trade'].sum() > 0:
                report['win_rate'] = df.loc[df['trade'] > 0, 'trade_profit'].sum() / df['trade'].sum()
            else:
                report['win_rate'] = 0
        
        # 创建特征重要性图表
        if self._feature_importance is not None and len(self._feature_importance) > 0:
            report['feature_importance'] = self._feature_importance.to_dict()
            
            if output_dir:
                plt.figure(figsize=(10, 6))
                self._feature_importance.iloc[:20].plot(kind='barh')
                plt.title('特征重要性')
                plt.tight_layout()
                
                # 保存图表
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
                plt.close()
        
        # 创建策略收益图表
        if 'cumulative_returns' in df.columns and output_dir:
            plt.figure(figsize=(12, 6))
            df['cumulative_returns'].plot()
            plt.title('策略累积收益')
            plt.tight_layout()
            
            # 保存图表
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(os.path.join(output_dir, 'cumulative_returns.png'))
            plt.close()
        
        # 创建市场状态分析
        if 'market_regime' in df.columns:
            regime_stats = df.groupby('market_regime')['strategy_returns'].agg(['mean', 'std', 'count'])
            regime_stats['sharpe'] = regime_stats['mean'] / regime_stats['std'] * np.sqrt(252)
            report['regime_performance'] = regime_stats.to_dict()
            
            if output_dir:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='market_regime', y='strategy_returns', data=df)
                plt.title('不同市场状态下的策略收益分布')
                plt.tight_layout()
                
                # 保存图表
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plt.savefig(os.path.join(output_dir, 'regime_performance.png'))
                plt.close()
        
        return report 