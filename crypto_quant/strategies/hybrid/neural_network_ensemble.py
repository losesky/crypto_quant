"""
基于神经网络的自适应集成策略
使用PyTorch实现高级策略集成
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
import traceback
import os
import joblib
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from .adaptive_ensemble import AdaptiveEnsemble
from ...utils.logger import logger


class NeuralNetworkEnsemble(AdaptiveEnsemble):
    """
    基于神经网络的集成策略
    
    使用PyTorch构建神经网络，实现针对不同市场环境的自适应集成
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
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        model_path: Optional[str] = None,
        use_attention: bool = True
    ):
        """
        初始化神经网络集成策略
        
        Args:
            base_strategies: 基础策略列表，每个策略是一个字典，包含策略对象和权重
            window_size: 用于特征计算的窗口大小
            retrain_interval: 模型重新训练的间隔
            min_train_samples: 训练所需的最小样本数
            feature_columns: 用于训练的特征列名列表
            target_column: 目标变量列名
            prediction_threshold: 预测阈值，预测值超过此阈值才产生交易信号
            hidden_dim: 隐藏层维度
            num_layers: 隐藏层数量
            dropout: Dropout比例
            learning_rate: 学习率
            batch_size: 批处理大小
            epochs: 训练轮数
            model_path: 模型保存路径
            use_attention: 是否使用注意力机制
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
        
        # 神经网络参数
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_attention = use_attention
        
        # 模型路径
        self.model_path = model_path
        
        # 模型和优化器
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._scaler = None
        
        # 初始化特征缩放器，避免临时创建
        self.feature_scaler = StandardScaler()
        self.has_fitted_scaler = False
        
        # 尝试加载预训练模型
        if self.model_path and os.path.exists(self.model_path):
            try:
                # 使用_load_model方法加载模型
                load_success = self._load_model()
                if load_success:
                    logger.info(f"加载预训练模型: {self.model_path}")
                else:
                    logger.warning(f"加载模型失败，将在运行时重新训练")
            except Exception as e:
                logger.error(f"加载模型失败: {str(e)}")
                traceback.print_exc()
    
    def _build_model(self, input_size: int):
        """
        构建神经网络模型
        
        Args:
            input_size: 输入特征维度
            
        Returns:
            bool: 模型构建是否成功
        """
        try:
            import torch
            import torch.nn as nn
            
            # 定义注意力层
            class AttentionLayer(nn.Module):
                """注意力机制层"""
                
                def __init__(self, input_dim):
                    super().__init__()
                    self.attention = nn.Sequential(
                        nn.Linear(input_dim, input_dim),
                        nn.Tanh(),
                        nn.Linear(input_dim, 1),
                        nn.Softmax(dim=1)
                    )
                    
                def forward(self, x):
                    # 计算注意力权重
                    attention_weights = self.attention(x)
                    
                    # 应用注意力权重
                    context = attention_weights * x
                    return context, attention_weights
            
            # 定义集成模型
            class EnsembleModel(nn.Module):
                """集成神经网络模型"""
                
                def __init__(self, input_dim, hidden_dim, num_layers, dropout, use_attention):
                    super().__init__()
                    
                    self.use_attention = use_attention
                    
                    # 输入层
                    self.input_layer = nn.Linear(input_dim, hidden_dim)
                    
                    # 隐藏层
                    self.hidden_layers = nn.ModuleList()
                    for _ in range(num_layers):
                        self.hidden_layers.append(nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout)
                        ))
                    
                    # 注意力层
                    if use_attention:
                        self.attention = AttentionLayer(hidden_dim)
                    
                    # 输出层
                    self.output_layer = nn.Linear(hidden_dim, 1)
                    
                def forward(self, x):
                    # 输入层
                    x = self.input_layer(x)
                    x = torch.relu(x)
                    
                    # 隐藏层
                    for layer in self.hidden_layers:
                        x = layer(x)
                    
                    # 注意力层
                    if self.use_attention:
                        x, _ = self.attention(x)
                    
                    # 输出层
                    x = self.output_layer(x)
                    return x
            
            # 构建模型
            self._model = EnsembleModel(
                input_dim=input_size,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                use_attention=self.use_attention
            )
            
            # 定义损失函数和优化器
            self._criterion = nn.MSELoss()
            self._optimizer = torch.optim.Adam(
                self._model.parameters(), 
                lr=self.learning_rate
            )
            
            logger.info(f"神经网络模型构建成功: 输入维度={input_size}, 隐藏层={self.num_layers}层x{self.hidden_dim}维")
            return True
            
        except ImportError:
            logger.error("无法导入PyTorch库，请确保已安装")
            return False
        except Exception as e:
            logger.error(f"构建模型时出错: {str(e)}")
            traceback.print_exc()
            return False
    
    def _save_model(self):
        """
        保存模型到磁盘
        """
        try:
            import torch
            import os
            
            if not self._is_trained or self._model is None:
                logger.warning("模型尚未训练，无法保存")
                return
                
            if self.model_path is None:
                logger.warning("未指定模型保存路径")
                return
                
            # 创建保存目录
            model_dir = os.path.dirname(self.model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)
                
            # 准备保存的内容
            save_dict = {
                'model_state_dict': self._model.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
                'feature_columns': self.feature_columns,  # 保存特征列列表
                'scaler': self._scaler  # 保存标准化器
            }
            
            # 保存模型
            torch.save(save_dict, self.model_path)
            logger.info(f"模型已保存至: {self.model_path}, 特征数量: {len(self.feature_columns)}")
            
        except Exception as e:
            logger.error(f"保存模型时出错: {str(e)}")
            
    def _load_model(self):
        """
        从磁盘加载模型
        """
        try:
            import torch
            import os
            from sklearn.preprocessing import StandardScaler
            
            if not os.path.exists(self.model_path):
                self.logger.warning(f"模型文件不存在: {self.model_path}")
                return False
                
            # 加载模型
            try:
                save_dict = torch.load(self.model_path)
            except Exception as load_error:
                self.logger.error(f"加载模型文件失败: {str(load_error)}")
                return False
            
            # 构建模型
            feature_count = len(save_dict.get('feature_columns', []))
            if feature_count == 0:
                self.logger.warning("模型文件中没有特征列信息，使用默认特征数量")
                feature_count = 10
            
            success = self._build_model(feature_count)
            if not success:
                return False
            
            # 检查是否是旧模型格式
            is_old_format = False
            has_attention = False
            if 'model_state_dict' in save_dict:
                model_state = save_dict['model_state_dict']
                # 检查是否包含旧格式的键
                if any('model.' in key for key in model_state.keys()):
                    is_old_format = True
                    self.logger.warning("检测到旧格式模型，将进行结构适配...")
                # 检查是否包含注意力机制
                if any('attention' in key for key in model_state.keys()):
                    has_attention = True
                    self.logger.info("检测到旧格式模型包含注意力机制")
            
            if is_old_format:
                # 直接使用新模型，不尝试迁移旧模型参数
                self.logger.warning("旧格式模型将不会应用参数，使用新初始化的模型")
                # 仅加载特征列信息
                if 'feature_columns' in save_dict:
                    self.feature_columns = save_dict['feature_columns']
                    self.logger.info(f"已加载特征列: {len(self.feature_columns)}个")
            else:
                # 新格式模型，直接加载
                try:
                    self._model.load_state_dict(save_dict['model_state_dict'])
                    self.logger.info("成功加载模型参数")
                except Exception as load_error:
                    self.logger.error(f"加载模型状态字典时出错: {str(load_error)}")
                    # 继续使用新初始化的模型
                    self.logger.warning("将使用新初始化的模型")
            
            # 加载优化器状态 - 改进处理逻辑
            if 'optimizer_state_dict' in save_dict:
                try:
                    # 优化器加载逻辑重写，避免警告
                    param_groups_count_saved = 0
                    if 'param_groups' in save_dict['optimizer_state_dict']:
                        param_groups_count_saved = len(save_dict['optimizer_state_dict']['param_groups'])
                    
                    # 初始化新的优化器而不是尝试加载可能不兼容的状态
                    if param_groups_count_saved > 0:
                        current_param_count = len(list(self._model.parameters()))
                        
                        # 只有在参数组数量匹配时才尝试加载
                        if current_param_count == param_groups_count_saved:
                            try:
                                self._optimizer.load_state_dict(save_dict['optimizer_state_dict'])
                                self.logger.info("成功加载优化器状态")
                            except Exception as opt_error:
                                # 忽略不兼容问题，使用新初始化的优化器
                                self.logger.debug(f"优化器状态加载失败，使用新初始化的优化器: {str(opt_error)}")
                        else:
                            # 明确记录但减少警告日志级别
                            self.logger.debug(f"优化器参数组数量不匹配，跳过加载优化器状态: 当前={current_param_count}, 保存={param_groups_count_saved}")
                    else:
                        self.logger.debug("优化器状态没有参数组信息，跳过加载")
                except Exception as opt_error:
                    self.logger.debug(f"加载优化器状态时出错，使用新初始化的优化器")
            
            # 加载特征列和标准化器
            if 'feature_columns' in save_dict:
                self.feature_columns = save_dict['feature_columns']
                self.logger.info(f"已加载特征列: {len(self.feature_columns)}个")
            
            if 'scaler' in save_dict:
                self._scaler = save_dict['scaler']
                self.logger.info("已加载特征标准化器")
            else:
                self._scaler = StandardScaler()
                
            self._is_trained = True
            self.logger.info(f"模型加载完成: {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载模型时出错: {str(e)}")
            traceback.print_exc()
            return False
    
    def prepare_features(self, df: pd.DataFrame, current_idx=None, is_training=False, suppress_warnings=False) -> pd.DataFrame:
        """
        准备集成模型的特征数据
        
        Args:
            df: 包含基础策略信号和市场数据的DataFrame
            current_idx: 当前处理的索引位置，如果为None则使用最后一个索引
            is_training: 是否是训练模式
            suppress_warnings: 是否抑制非关键警告（用于测试环境）
            
        Returns:
            pd.DataFrame: 带有特征的DataFrame
        """
        # 确保current_idx有值
        if current_idx is None and len(df) > 0:
            current_idx = df.index[-1]
            
        # 调用父类方法处理基本特征
        df_features = super().prepare_features(df, current_idx, is_training, suppress_warnings)
        
        # market_regime处理已移至基类AdaptiveEnsemble中的_ensure_numeric_dataframe方法
        
        # 检查并移除所有非数值列
        non_numeric_cols = df_features.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            if not suppress_warnings:
                self.logger.warning(f"移除非数值特征列: {non_numeric_cols}")
            df_features = df_features.drop(non_numeric_cols, axis=1)
        
        return df_features
    
    def train(self, df, current_idx=None):
        """
        训练神经网络模型。
        
        参数:
            df (pd.DataFrame): 历史数据
            current_idx: 当前索引
        
        返回:
            bool: 训练是否成功
        """
        try:
            # 准备特征
            logger.info("准备训练特征数据")
            features_df = self.prepare_features(df, current_idx, is_training=True, suppress_warnings=True)
            
            if len(features_df) < self.min_train_samples:
                logger.warning(f"样本数量 {len(features_df)} 小于最小训练样本数量 {self.min_train_samples}，跳过训练")
                return False
            
            # 提取特征列和目标列
            X = features_df[self.feature_columns].copy()
            
            # 使用新的辅助方法确保目标列存在
            y = self._ensure_target_column(features_df)
                
            # 确保数据是数值类型
            X = self._ensure_numeric_dataframe(X)
            
            # 过滤掉缺失值
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            if valid_idx.sum() < self.min_train_samples:
                logger.warning(f"有效样本数量 {valid_idx.sum()} 小于最小训练样本数量 {self.min_train_samples}，跳过训练")
                return False
                
            X = X[valid_idx]
            y = y[valid_idx]
            
            # 记录训练数据的形状
            logger.info(f"训练数据: X形状={X.shape}, y形状={y.shape}")
            logger.debug(f"特征列: {self.feature_columns}")
            
            # 数据预处理
            X_scaled = self.feature_scaler.fit_transform(X)
            self.has_fitted_scaler = True
            
            # 转换为PyTorch张量
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y.values.reshape(-1, 1))
            
            # 创建数据集和数据加载器
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # 构建和训练模型
            self._build_model(X.shape[1])
            
            # 使用Adam优化器
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
            
            # 使用均方误差损失函数
            loss_fn = nn.MSELoss()
            
            # 训练模型
            logger.info(f"开始训练神经网络模型，epochs={self.epochs}")
            self._model.train()
            losses = []
            
            for epoch in range(self.epochs):
                epoch_losses = []
                for batch_X, batch_y in dataloader:
                    # 梯度清零
                    optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = self._model(batch_X)
                    
                    # 计算损失
                    loss = loss_fn(outputs, batch_y)
                    
                    # 反向传播
                    loss.backward()
                    
                    # 参数更新
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                losses.append(avg_loss)
                
                if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.epochs - 1:
                    logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
            
            # 计算特征重要性
            try:
                self._calculate_feature_importance(X_scaled, y.values)
                logger.info("已计算特征重要性")
            except Exception as e:
                logger.error(f"计算特征重要性失败: {e}")
            
            # 保存模型
            if self.model_path:
                self._save_model()
                logger.info(f"模型已保存到 {self.model_path}")
            
            # 更新训练索引
            self.last_train_idx = current_idx
            
            logger.info("模型训练完成")
            return True
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            traceback.print_exc()
            return False
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """
        计算特征重要性（使用排列重要性方法）
        
        Args:
            X: 特征数组
            y: 目标变量数组
        """
        try:
            import torch
            import numpy as np
            from sklearn.metrics import mean_squared_error
            
            # 确保模型处于评估模式
            self._model.eval()
            
            # 转换为PyTorch张量
            X_tensor = torch.FloatTensor(X)
            
            # 计算原始误差
            with torch.no_grad():
                y_pred = self._model(X_tensor).numpy().flatten()
            baseline_error = mean_squared_error(y, y_pred)
            
            # 计算每个特征的重要性
            feature_importance = np.zeros(X.shape[1])
            
            for i in range(X.shape[1]):
                # 复制数据并打乱特征列
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                
                # 预测并计算误差
                X_permuted_tensor = torch.FloatTensor(X_permuted)
                with torch.no_grad():
                    y_pred_permuted = self._model(X_permuted_tensor).numpy().flatten()
                permuted_error = mean_squared_error(y, y_pred_permuted)
                
                # 重要性 = 打乱后误差 - 原始误差
                feature_importance[i] = permuted_error - baseline_error
            
            # 归一化重要性
            if feature_importance.sum() > 0:
                feature_importance = feature_importance / feature_importance.sum()
            
            # 存储为Series
            self._feature_importance = pd.Series(
                feature_importance, 
                index=self.feature_columns
            ).sort_values(ascending=False)
            
            # 输出特征重要性
            logger.info("特征重要性:")
            for feature, importance in self._feature_importance.iloc[:10].items():
                logger.info(f"  {feature}: {importance:.4f}")
                
        except Exception as e:
            logger.error(f"计算特征重要性时出错: {str(e)}")
    
    def predict(self, df, current_idx=None):
        """
        使用训练好的模型进行预测。
        
        参数:
            df (pd.DataFrame): 历史数据
            current_idx: 当前索引
        
        返回:
            np.ndarray: 预测结果
        """
        try:
            # 准备特征
            logger.info("准备预测特征数据")
            features_df = self.prepare_features(df, current_idx, is_training=False, suppress_warnings=True)
            
            # 提取特征列
            if self.feature_columns:
                # 获取存在于features_df中的特征列
                available_features = [col for col in self.feature_columns if col in features_df.columns]
                missing_features = [col for col in self.feature_columns if col not in features_df.columns]
                
                if missing_features:
                    logger.warning(f"以下特征列在预测数据中不存在: {missing_features}")
                
                if not available_features:
                    logger.error("没有可用的特征列用于预测")
                    return np.zeros(1)
                
                X = features_df[available_features].copy()
                
                # 记录特征统计数据
                logger.info(f"预测特征形状: {X.shape}")
                for col in X.columns[:5]:  # 只记录前5列
                    stats = X[col].describe()
                    logger.debug(f"特征 {col} 统计: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, 最小值={stats['min']:.4f}, 最大值={stats['max']:.4f}, 缺失值={X[col].isna().sum()}")
                
                # 确保数据是数值类型
                X = self._ensure_numeric_dataframe(X)
                
                # 如果存在缺失值，填充为0
                if X.isna().any().any():
                    logger.warning(f"预测数据中存在 {X.isna().sum().sum()} 个缺失值，将使用0填充")
                    X.fillna(0, inplace=True)
                
                # 使用一致的特征缩放方法
                if self.has_fitted_scaler:
                    # 处理在训练中存在但在预测中不存在的特征列
                    if X.shape[1] != len(self.feature_scaler.mean_):
                        logger.warning(f"特征列数量不匹配: 训练={len(self.feature_scaler.mean_)}, 预测={X.shape[1]}")
                        # 如果预测特征数量少于训练特征数量，创建一个零填充的临时DataFrame
                        temp_X = pd.DataFrame(0, index=X.index, columns=self.feature_columns)
                        for col in X.columns:
                            if col in temp_X.columns:
                                temp_X[col] = X[col]
                        X = temp_X
                    
                    X_scaled = self.feature_scaler.transform(X)
                else:
                    # 如果特征缩放器尚未训练，使用当前数据进行拟合和转换
                    X_scaled = self.feature_scaler.fit_transform(X)
                    self.has_fitted_scaler = True
                    logger.info("已使用当前数据拟合特征缩放器")
                
                # 转换为PyTorch张量
                X_tensor = torch.FloatTensor(X_scaled)
                
                # 预测
                self._model.eval()
                with torch.no_grad():
                    predictions = self._model(X_tensor).numpy()
                
                logger.info(f"预测结果形状: {predictions.shape}")
                return predictions.flatten()
            else:
                logger.error("没有定义特征列")
                return np.zeros(1)
                
        except Exception as e:
            logger.error(f"预测失败: {e}")
            traceback.print_exc()
            return np.zeros(1)
    
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