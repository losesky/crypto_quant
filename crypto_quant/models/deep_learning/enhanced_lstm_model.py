"""
增强版LSTM深度学习模型，用于时间序列预测
添加了注意力机制、高级特征工程和正则化
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ...utils.logger import logger
import ta
import math

class AttentionModule(nn.Module):
    """
    注意力机制模块，用于LSTM模型
    """
    def __init__(self, hidden_dim):
        """
        初始化注意力模块
        
        Args:
            hidden_dim (int): 隐藏层维度
        """
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, lstm_output):
        """
        前向传播
        
        Args:
            lstm_output (torch.Tensor): LSTM输出，形状为 (batch_size, seq_len, hidden_dim)
            
        Returns:
            tuple: (attention_output, attention_weights)
                attention_output: 形状为 (batch_size, hidden_dim)
                attention_weights: 形状为 (batch_size, seq_len)
        """
        # 计算注意力权重
        attention_weights = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # (batch_size, seq_len, 1)
        
        # 应用注意力权重
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_dim)
        
        return context_vector, attention_weights.squeeze(-1)  # 返回压缩后的权重


class EnhancedLSTMModel(nn.Module):
    """
    增强型LSTM神经网络模型
    使用多层LSTM和注意力机制进行时间序列预测
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2, bidirectional=True):
        """
        初始化增强型LSTM模型

        Args:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            num_layers (int): LSTM层数
            output_dim (int): 输出维度
            dropout (float): Dropout比率
            bidirectional (bool): 是否使用双向LSTM
        """
        super(EnhancedLSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        
        # 方向数
        self.directions = 2 if bidirectional else 1
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # 注意力机制
        self.attention = AttentionModule(hidden_dim * self.directions)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * self.directions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 批归一化
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        logger.info(f"增强型LSTM模型初始化成功: input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"num_layers={num_layers}, bidirectional={bidirectional}")

    def forward(self, x):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 应用注意力机制
        attn_output, attention_weights = self.attention(lstm_out)
        
        # 全连接层
        out = self.fc1(attn_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.batch_norm(out)
        out = self.fc2(out)
        
        return out, attention_weights 

class EnhancedLSTMPricePredictor:
    """
    增强型LSTM价格预测器
    提供端到端的数据处理、高级特征工程、训练、预测和评估功能
    """

    def __init__(self, sequence_length=20, hidden_dim=128, num_layers=3, dropout=0.3, 
                 learning_rate=0.001, bidirectional=True, feature_engineering=True,
                 use_attention=True, scheduler_step_size=20, scheduler_gamma=0.5):
        """
        初始化增强型LSTM价格预测器

        Args:
            sequence_length (int): 序列长度，使用多少天的数据进行预测
            hidden_dim (int): LSTM隐藏层维度
            num_layers (int): LSTM层数
            dropout (float): Dropout比率
            learning_rate (float): 学习率
            bidirectional (bool): 是否使用双向LSTM
            feature_engineering (bool): 是否进行特征工程
            use_attention (bool): 是否使用注意力机制
            scheduler_step_size (int): 学习率调度器步长
            scheduler_gamma (float): 学习率调度器衰减率
        """
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.feature_engineering = feature_engineering
        self.use_attention = use_attention
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        
        # 初始化状态
        self.model = None
        self.scaler_X = StandardScaler()  # 使用StandardScaler替代MinMaxScaler
        self.scaler_y = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_weights = None
        
        logger.info(f"增强型LSTM价格预测器初始化成功: sequence_length={sequence_length}, device={self.device}, "
                   f"bidirectional={bidirectional}, feature_engineering={feature_engineering}")

    def _engineer_features(self, df):
        """
        高级特征工程
        
        Args:
            df (pandas.DataFrame): 输入数据
            
        Returns:
            pandas.DataFrame: 包含工程特征的数据
        """
        if not self.feature_engineering:
            return df
        
        df_processed = df.copy()
        
        try:
            # 确保索引是日期时间类型
            if not isinstance(df_processed.index, pd.DatetimeIndex):
                df_processed.index = pd.to_datetime(df_processed.index)
            
            # 预处理分类特征 - 处理字符串类型的列（如market_regime）
            # 将常见的分类特征转换为数值
            categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns
            for col in categorical_columns:
                if col == 'market_regime':
                    # 专门处理market_regime列
                    regime_map = {
                        'trending_volatile': 3, 
                        'ranging_volatile': 2, 
                        'trending_stable': 1, 
                        'ranging_stable': 0,
                        'unknown': -1
                    }
                    if 'market_regime_num' not in df_processed.columns:
                        df_processed['market_regime_num'] = df_processed[col].map(regime_map).fillna(-1)
                    # 删除原始字符串列
                    df_processed = df_processed.drop(col, axis=1)
                    logger.info(f"已将 '{col}' 转换为数值特征 'market_regime_num'")
                elif df_processed[col].nunique() <= 30:  # 只处理值不太多的分类特征
                    try:
                        # 使用类别编码
                        df_processed[f'{col}_encoded'] = pd.Categorical(df_processed[col]).codes
                        df_processed = df_processed.drop(col, axis=1)
                        logger.info(f"已将分类特征 '{col}' 编码为数值")
                    except Exception as e:
                        logger.warning(f"无法编码分类特征 '{col}': {str(e)}")
                        df_processed = df_processed.drop(col, axis=1)
                else:
                    # 分类值太多，直接删除
                    logger.warning(f"分类特征 '{col}' 有太多唯一值 ({df_processed[col].nunique()})，将被移除")
                    df_processed = df_processed.drop(col, axis=1)
            
            # 价格特征
            df_processed['log_return'] = np.log(df_processed['close'] / df_processed['close'].shift(1))
            df_processed['price_range'] = (df_processed['high'] - df_processed['low']) / df_processed['close']
            df_processed['gap'] = (df_processed['open'] - df_processed['close'].shift(1)) / df_processed['close'].shift(1)
            
            # 移动平均特征
            for window in [5, 10, 20, 50]:
                df_processed[f'ma_{window}'] = df_processed['close'].rolling(window=window).mean()
                df_processed[f'ma_ratio_{window}'] = df_processed['close'] / df_processed[f'ma_{window}']
            
            # 波动率特征
            for window in [5, 10, 20]:
                df_processed[f'volatility_{window}'] = df_processed['log_return'].rolling(window=window).std()
            
            # 相对强弱指标 (RSI)
            df_processed['rsi_14'] = ta.momentum.rsi(df_processed['close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df_processed['close'])
            df_processed['macd'] = macd.macd()
            df_processed['macd_signal'] = macd.macd_signal()
            df_processed['macd_diff'] = macd.macd_diff()
            
            # 布林带
            bollinger = ta.volatility.BollingerBands(df_processed['close'])
            df_processed['bollinger_high'] = bollinger.bollinger_hband()
            df_processed['bollinger_low'] = bollinger.bollinger_lband()
            df_processed['bollinger_pct'] = bollinger.bollinger_pband()
            
            # 交易量特征
            df_processed['volume_ma_5'] = df_processed['volume'].rolling(window=5).mean()
            df_processed['volume_ma_10'] = df_processed['volume'].rolling(window=10).mean()
            df_processed['volume_ratio_5'] = df_processed['volume'] / df_processed['volume_ma_5']
            df_processed['volume_ratio_10'] = df_processed['volume'] / df_processed['volume_ma_10']
            
            # 日期特征 - 将其转换为数值特征，而不是直接使用时间戳
            df_processed['day_of_week'] = df_processed.index.dayofweek.astype(float)
            df_processed['month'] = df_processed.index.month.astype(float)
            df_processed['day_of_month'] = df_processed.index.day.astype(float)
            
            # 最后的检查 - 确保所有列都是数值型
            non_numeric_cols = df_processed.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                logger.warning(f"依然存在非数值特征，将被移除: {non_numeric_cols}")
                df_processed = df_processed.drop(non_numeric_cols, axis=1)
            
            # 去除NaN值
            df_processed = df_processed.fillna(method='bfill').fillna(method='ffill')
            
            # 如果存在预定义的特征列表，确保数据框只包含这些特征
            if hasattr(self, '_all_feature_names') and self._all_feature_names:
                # 检查是否有缺失的特征
                missing_features = [col for col in self._all_feature_names if col not in df_processed.columns]
                for col in missing_features:
                    logger.warning(f"特征工程中缺失特征: {col}，将使用0填充")
                    df_processed[col] = 0
                
                # 只保留预定义的特征列
                df_processed = df_processed[self._all_feature_names]
            else:
                # 存储特征名称列表以确保一致性
                self._all_feature_names = list(df_processed.columns)
            
            logger.info(f"特征工程完成，添加了{len(df_processed.columns) - len(df.columns)}个新特征")
        except Exception as e:
            logger.error(f"特征工程过程中出错: {str(e)}")
            logger.warning("将使用原始数据，不进行特征工程")
            return df
        
        return df_processed

    def _create_sequences(self, data, target_column='close'):
        """
        创建序列数据

        Args:
            data (pandas.DataFrame): 输入数据
            target_column (str): 目标列名

        Returns:
            tuple: (X, y) 特征序列和目标值
        """
        # 先进行特征工程
        data_processed = self._engineer_features(data)
        
        # 选择特征和目标
        if self.feature_engineering:
            # 使用当前实例的特征列表
            if hasattr(self, '_model_feature_columns') and self._model_feature_columns:
                logger.debug("使用已保存的模型特征列表")
                features_cols = self._model_feature_columns.copy()
                
                # 检查所有需要的列是否存在
                missing_cols = [col for col in features_cols if col not in data_processed.columns]
                if missing_cols:
                    logger.warning(f"数据中缺少以下特征列: {missing_cols}, 将使用0填充")
                    for col in missing_cols:
                        data_processed[col] = 0
                
                # 确保close列在特征列表中（即使是目标列）
                if 'close' not in features_cols:
                    logger.info("向特征列表中添加'close'列")
                    features_cols.append('close')
            else:
                logger.debug("创建新的特征列表")
                # 使用所有工程特征，排除非数值列
                features_cols = []
                for col in data_processed.columns:
                    # 排除非数值列，但保留close列
                    if (col == 'close' or col != target_column) and np.issubdtype(data_processed[col].dtype, np.number):
                        features_cols.append(col)
                
                # 确保close列在特征列表中
                if 'close' not in features_cols and 'close' in data_processed.columns:
                    logger.info("向特征列表中添加'close'列")
                    features_cols.append('close')
                
                # 存储特征列以确保一致性
                self._model_feature_columns = features_cols.copy()
                logger.debug(f"保存特征列表，共{len(features_cols)}个特征")
        else:
            # 只使用OHLCV
            features_cols = ['open', 'high', 'low', 'close', 'volume']
            # 检查这些基本列是否存在
            missing_cols = [col for col in features_cols if col not in data_processed.columns]
            if missing_cols:
                logger.warning(f"数据中缺少基本OHLCV列: {missing_cols}, 将使用0填充")
                for col in missing_cols:
                    data_processed[col] = 0
                    
            self._model_feature_columns = features_cols.copy()
        
        try:
            # 确保所有特征都是数值型
            numeric_features_cols = []
            for col in features_cols:
                if col in data_processed.columns and np.issubdtype(data_processed[col].dtype, np.number):
                    numeric_features_cols.append(col)
                else:
                    if col in data_processed.columns:
                        logger.warning(f"跳过非数值特征列: {col}, 类型: {data_processed[col].dtype}")
                    else:
                        logger.warning(f"特征列 {col} 不存在于数据中")
            
            # 再次确保close列在特征中
            if 'close' not in numeric_features_cols and 'close' in data_processed.columns:
                numeric_features_cols.append('close')
                logger.info("重新添加close列到特征列表中")
            
            # 记录特征数量变化
            if len(numeric_features_cols) != len(features_cols):
                logger.warning(f"过滤掉 {len(features_cols) - len(numeric_features_cols)} 个非数值特征列")
                # 更新存储的特征列
                self._model_feature_columns = numeric_features_cols.copy()
                features_cols = numeric_features_cols
            
            # 确保有特征可用
            if not features_cols:
                raise ValueError("没有可用的数值特征列")
            
            # 记录使用的特征列数量
            logger.debug(f"使用 {len(features_cols)} 个特征: {features_cols}")
            
            # 提取特征和目标数据
            features = data_processed[features_cols].values
            target = data_processed[[target_column]].values
            
            # 检查是否有NaN值
            if np.isnan(features).any():
                nan_count = np.isnan(features).sum()
                
                # 创建特征DataFrame以便使用更高级的填充方法
                features_df = pd.DataFrame(features, columns=self._model_feature_columns if hasattr(self, '_model_feature_columns') else None)
                
                # 1. 先尝试前向填充
                features_df = features_df.ffill()
                
                # 2. 对仍然有NaN的值使用列均值填充
                for col in features_df.columns:
                    if features_df[col].isna().any():
                        col_mean = features_df[col].mean()
                        # 如果均值也是NaN（整列都是NaN），则使用0
                        if pd.isna(col_mean):
                            col_mean = 0
                        features_df[col] = features_df[col].fillna(col_mean)
                
                # 将处理后的数据转回numpy数组
                features = features_df.values
                
                # 仍有NaN值（极少数情况）就使用0填充
                if np.isnan(features).any():
                    remaining_nan = np.isnan(features).sum()
                    features = np.nan_to_num(features, nan=0.0)
                    logger.warning(f"处理后仍有 {remaining_nan} 个NaN值，已使用0填充")
                else:
                    # 只在第一次预测或样本数据较少时输出提示信息
                    if len(data_processed) < 50 or not hasattr(self, '_nan_warning_shown'):
                        logger.info(f"已处理 {nan_count} 个NaN值，使用前向填充和均值填充")
                        self._nan_warning_shown = True
            
            if np.isnan(target).any():
                nan_count = np.isnan(target).sum()
                logger.warning(f"目标数据中存在 {nan_count} 个NaN值，将使用前值填充")
                # 使用前值填充NaN
                for i in range(1, len(target)):
                    if np.isnan(target[i]):
                        target[i] = target[i-1]
                # 如果第一个值是NaN，使用后值填充
                if np.isnan(target[0]) and len(target) > 1:
                    target[0] = target[1]
                # 如果仍然有NaN，使用0填充
                target = np.nan_to_num(target, nan=0.0)
            
            # 保存特征维度，以便后续检查
            self._feature_dim = features.shape[1]
            logger.debug(f"特征维度: {self._feature_dim}")
            
            # 标准化
            features_normalized = self.scaler_X.fit_transform(features)
            target_normalized = self.scaler_y.fit_transform(target)
            
            # 创建序列
            X, y = [], []
            for i in range(len(features_normalized) - self.sequence_length):
                X.append(features_normalized[i:i + self.sequence_length])
                y.append(target_normalized[i + self.sequence_length])
            
            return np.array(X), np.array(y), features_cols
        except Exception as e:
            logger.error(f"创建序列数据时出错: {str(e)}")
            logger.error(f"问题列: {[col for col in features_cols if col in data_processed.columns and not np.issubdtype(data_processed[col].dtype, np.number)]}")
            raise ValueError(f"创建序列数据失败: {str(e)}")

    def prepare_data(self, df, target_column='close', test_size=0.2, batch_size=32):
        """
        准备训练和测试数据

        Args:
            df (pandas.DataFrame): 输入数据
            target_column (str): 目标列名
            test_size (float): 测试集比例
            batch_size (int): 批次大小

        Returns:
            tuple: (train_loader, test_loader, X_test, y_test, features_cols) 
                   训练和测试数据加载器及测试数据
        """
        # 创建序列
        X, y, features_cols = self._create_sequences(df, target_column)
        
        # 检查样本数量是否足够
        min_samples = 10  # 最小样本数量
        actual_samples = len(X)
        
        if actual_samples < min_samples:
            logger.warning(f"样本数量({actual_samples})过少，建议增加数据量或减少序列长度")
            if actual_samples == 0:
                raise ValueError(f"没有足够的数据生成序列。确保数据长度({len(df)})大于序列长度({self.sequence_length})。")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), shuffle=False)
        
        logger.info(f"数据准备完成: 训练样本={len(X_train)}, 测试样本={len(X_test)}, 特征数量={len(features_cols)}")
        
        return train_loader, test_loader, X_test, y_test, features_cols 

    def build_model(self, input_dim=None):
        """
        构建增强型LSTM模型

        Args:
            input_dim (int, optional): 输入特征维度，如果为None则使用原始5维特征

        Returns:
            EnhancedLSTMModel: 构建的模型
        """
        if input_dim is None:
            input_dim = 5  # 默认OHLCV五个特征
            
        self.model = EnhancedLSTMModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=1,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)
        
        return self.model

    def train(self, train_loader, test_loader=None, num_epochs=100, patience=15, weight_decay=1e-5):
        """
        训练模型，使用学习率调度、早停和L2正则化

        Args:
            train_loader (DataLoader): 训练数据加载器
            test_loader (DataLoader, optional): 测试数据加载器，用于验证
            num_epochs (int): 训练轮数
            patience (int): 早停耐心值
            weight_decay (float): L2正则化系数

        Returns:
            dict: 训练历史记录
        """
        # 获取输入维度
        x_sample, _ = next(iter(train_loader))
        input_dim = x_sample.shape[2]
        
        if self.model is None:
            self.build_model(input_dim=input_dim)
            
        # 定义损失函数和优化器，使用L2正则化
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.scheduler_step_size, 
            gamma=self.scheduler_gamma
        )
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        # 早停机制
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_weights = None
        
        # 训练循环
        logger.info(f"开始训练增强型LSTM模型: epochs={num_epochs}, patience={patience}")
        
        for epoch in range(num_epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # 前向传播
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                
                # 计算损失
                loss = criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # 验证
            val_loss = 0.0
            if test_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        outputs, _ = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(test_loader)
                history['val_loss'].append(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    # 保存最佳模型权重
                    best_model_weights = self.model.state_dict().copy()
                else:
                    early_stop_counter += 1
                
                if early_stop_counter >= patience:
                    logger.info(f"早停触发于epoch {epoch+1}, 最佳验证损失: {best_val_loss:.6f}")
                    # 恢复最佳模型
                    self.model.load_state_dict(best_model_weights)
                    break
            
            # 学习率调度
            scheduler.step()
            
            # 打印进度
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                if test_loader is not None:
                    logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                else:
                    logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        logger.info(f"LSTM模型训练完成: 最终训练损失={history['train_loss'][-1]:.6f}")
        
        return history

    def evaluate(self, test_loader):
        """
        评估模型

        Args:
            test_loader (DataLoader): 测试数据加载器

        Returns:
            dict: 评估结果
        """
        if self.model is None:
            raise ValueError("模型未初始化，请先构建或加载模型")
        
        # 评估模式
        self.model.eval()
        
        # 准备评估指标
        criterion = nn.MSELoss()
        total_loss = 0.0
        predictions = []
        targets = []
        attention_weights_list = []
        
        # 预测
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs, attention_weights = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # 收集预测和目标
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())
                
                # 收集注意力权重
                if self.use_attention:
                    attention_weights_list.append(attention_weights.cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / len(test_loader)
        
        # 计算均方根误差 (RMSE)
        rmse = np.sqrt(avg_loss)
        
        # 计算平均注意力权重
        if self.use_attention and attention_weights_list:
            self.attention_weights = np.mean(np.vstack(attention_weights_list), axis=0)
        
        # 逆变换预测和目标
        predictions = self.scaler_y.inverse_transform(np.array(predictions))
        targets = self.scaler_y.inverse_transform(np.array(targets))
        
        # 计算平均绝对误差百分比 (MAPE)
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        
        # 计算方向准确率
        pred_direction = np.diff(predictions.flatten())
        true_direction = np.diff(targets.flatten())
        direction_accuracy = np.mean((pred_direction * true_direction) > 0) * 100
        
        logger.info(f"模型评估结果: MSE={avg_loss:.6f}, RMSE={rmse:.6f}, MAPE={mape:.2f}%, 方向准确率={direction_accuracy:.2f}%")
        
        return {
            'mse': avg_loss,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'predictions': predictions,
            'targets': targets,
            'attention_weights': self.attention_weights if self.use_attention else None
        }

    def predict(self, X):
        """
        预测

        Args:
            X (numpy.ndarray): 输入特征，形状为 (sequence_length, features)

        Returns:
            numpy.ndarray: 预测结果
        """
        if self.model is None:
            raise ValueError("模型未初始化，请先构建或加载模型")
        
        # 评估模式
        self.model.eval()
        
        # 准备输入
        X = X.reshape(1, X.shape[0], X.shape[1])
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # 预测
        with torch.no_grad():
            output, attention_weights = self.model(X_tensor)
            
            # 保存注意力权重
            if self.use_attention:
                self.attention_weights = attention_weights.cpu().numpy()
        
        # 逆变换预测结果
        prediction = self.scaler_y.inverse_transform(output.cpu().numpy())
        
        return prediction

    def predict_next_day(self, df, n_steps=1, return_attention=False):
        """
        预测未来n天的价格

        Args:
            df (pandas.DataFrame): 历史数据
            n_steps (int): 预测天数
            return_attention (bool): 是否返回注意力权重

        Returns:
            tuple or list: 预测结果列表，如果return_attention=True，则返回(predictions, attention_weights)
        """
        if self.model is None:
            logger.warning("模型未初始化，无法进行预测")
            return [] if not return_attention else ([], [])
            
        try:
            # 确保模型已经初始化并训练过
            if not hasattr(self, '_model_feature_columns') or not self._model_feature_columns:
                logger.error("模型特征列表未初始化，可能是模型还未训练")
                return [] if not return_attention else ([], [])
            
            if not hasattr(self, '_feature_dim'):
                logger.error("特征维度未保存，可能是模型还未训练")
                return [] if not return_attention else ([], [])
            
            # 复制数据以防修改原始数据
            df_copy = df.copy()
            
            # 确保具有'close'列
            if 'close' not in df_copy.columns:
                logger.error("输入数据缺少'close'列，无法进行预测")
                return [] if not return_attention else ([], [])
            
            # 进行特征工程获取标准化特征
            processed_df = self._engineer_features(df_copy)
            
            # 使用我们的模型特征列表，确保包含'close'列
            feature_names = self._model_feature_columns.copy()
            
            # 检查并添加'close'列到特征列表
            if 'close' not in feature_names:
                logger.info("向预测特征列表中添加'close'列")
                feature_names.append('close')
                # 更新模型特征列表
                self._model_feature_columns = feature_names.copy()
                # 更新特征维度
                self._feature_dim = len(feature_names)
            
            logger.debug(f"预测使用 {len(feature_names)} 个特征: {feature_names}")
            
            # 确保所有需要的特征都存在
            missing_features = [col for col in feature_names if col not in processed_df.columns]
            if missing_features:
                logger.warning(f"在预测数据中缺少以下特征: {missing_features}，将使用0填充")
                for feature in missing_features:
                    processed_df[feature] = 0
            
            # 从处理后的数据中提取特征
            features = processed_df[feature_names].values
            
            # 检查特征维度
            if features.shape[1] != self._feature_dim:
                logger.warning(f"特征维度不匹配: 预期 {self._feature_dim}, 实际 {features.shape[1]}")
                # 动态调整特征维度以匹配
                self._feature_dim = features.shape[1]
            
            # 检查NaN值
            if np.isnan(features).any():
                nan_count = np.isnan(features).sum()
                
                # 创建特征DataFrame以便使用更高级的填充方法
                features_df = pd.DataFrame(features, columns=self._model_feature_columns if hasattr(self, '_model_feature_columns') else None)
                
                # 1. 先尝试前向填充
                features_df = features_df.ffill()
                
                # 2. 对仍然有NaN的值使用列均值填充
                for col in features_df.columns:
                    if features_df[col].isna().any():
                        col_mean = features_df[col].mean()
                        # 如果均值也是NaN（整列都是NaN），则使用0
                        if pd.isna(col_mean):
                            col_mean = 0
                        features_df[col] = features_df[col].fillna(col_mean)
                
                # 将处理后的数据转回numpy数组
                features = features_df.values
                
                # 仍有NaN值（极少数情况）就使用0填充
                if np.isnan(features).any():
                    remaining_nan = np.isnan(features).sum()
                    features = np.nan_to_num(features, nan=0.0)
                    logger.warning(f"处理后仍有 {remaining_nan} 个NaN值，已使用0填充")
                else:
                    # 只在第一次预测或样本数据较少时输出提示信息
                    if len(df_copy) < 50 or not hasattr(self, '_nan_warning_shown'):
                        logger.info(f"已处理 {nan_count} 个NaN值，使用前向填充和均值填充")
                        self._nan_warning_shown = True
            
            # 标准化特征
            try:
                features_normalized = self.scaler_X.transform(features)
            except Exception as e:
                logger.error(f"标准化特征时出错: {str(e)}")
                # 如果标准化失败，尝试重新拟合
                logger.warning("尝试重新拟合标准化器...")
                features_normalized = self.scaler_X.fit_transform(features)
            
            # 使用最后sequence_length天的数据进行预测
            if len(features_normalized) < self.sequence_length:
                logger.error(f"数据量不足: 需要至少{self.sequence_length}天，实际只有{len(features_normalized)}天")
                return [] if not return_attention else ([], [])
                
            last_sequence = features_normalized[-self.sequence_length:]
            
            # 找到close列的索引
            close_idx = -1
            if 'close' in feature_names:
                close_idx = feature_names.index('close')
                logger.debug(f"找到'close'列，索引为{close_idx}")
            else:
                # 这种情况不应该发生，因为我们已经确保了'close'列在特征名称中
                logger.error("致命错误: 即使我们添加了'close'列，仍然找不到它!")
                return [] if not return_attention else ([], [])
            
            # 预测多步
            predictions = []
            attention_weights_list = []
            
            for step in range(n_steps):
                try:
                    # 预测
                    input_seq = last_sequence.reshape(1, self.sequence_length, -1)
                    input_tensor = torch.FloatTensor(input_seq).to(self.device)
                    
                    with torch.no_grad():
                        output, attention_weights = self.model(input_tensor)
                        
                        # 保存注意力权重
                        if self.use_attention:
                            attention_weights_list.append(attention_weights.cpu().numpy())
                    
                    # 转换回原始尺度
                    pred_value = self.scaler_y.inverse_transform(output.cpu().numpy())[0]
                    predictions.append(float(pred_value[0]))
                    
                    # 如果需要预测多步，更新序列
                    if n_steps > 1 and step < n_steps - 1:
                        # 获取最新的特征向量
                        last_features = last_sequence[-1].copy()
                        
                        # 更新收盘价
                        last_features[close_idx] = output.cpu().numpy()[0][0]
                        
                        # 添加到序列
                        last_sequence = np.vstack([last_sequence[1:], last_features])
                        logger.debug(f"已更新序列用于多步预测，步骤 {step+1}/{n_steps}")
                except Exception as e:
                    logger.error(f"预测过程中出错: {str(e)}")
                    # 如果单步预测失败，返回已预测的结果
                    break
            
            if predictions:
                logger.info(f"预测未来{len(predictions)}天价格: {predictions}")
            
            if return_attention and self.use_attention:
                return predictions, attention_weights_list
            return predictions
            
        except Exception as e:
            logger.error(f"预测过程中发生全局错误: {str(e)}")
            return [] if not return_attention else ([], [])

    def plot_attention(self, df=None, attention_weights=None, sequence_length=None, feature_names=None):
        """
        可视化注意力权重

        Args:
            df (pandas.DataFrame, optional): 用于显示日期的数据
            attention_weights (numpy.ndarray, optional): 注意力权重
            sequence_length (int, optional): 序列长度
            feature_names (list, optional): 特征名称
            
        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        if attention_weights is None:
            if self.attention_weights is None:
                logger.warning("没有可用的注意力权重，请先调用predict或evaluate方法")
                return None
            attention_weights = self.attention_weights
            
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        plt.figure(figsize=(10, 6))
        plt.bar(range(sequence_length), attention_weights, alpha=0.7)
        plt.xlabel('时间步')
        plt.ylabel('注意力权重')
        plt.title('LSTM注意力权重分布')
        
        # 如果提供了数据框，使用日期作为x轴标签
        if df is not None and len(df) >= sequence_length:
            recent_dates = df.index[-sequence_length:].strftime('%Y-%m-%d').tolist()
            plt.xticks(range(sequence_length), recent_dates, rotation=45)
            
        plt.tight_layout()
        
        return plt.gcf()

    def save_model(self, path=None):
        """
        保存模型

        Args:
            path (str, optional): 保存路径，如果为None则使用默认路径

        Returns:
            str: 保存路径
        """
        if self.model is None:
            raise ValueError("模型未初始化，无法保存")
            
        if path is None:
            # 创建模型目录
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs', 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            # 生成路径
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(model_dir, f'enhanced_lstm_model_{timestamp}.pth')
        
        # 保存模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'feature_engineering': self.feature_engineering,
            'use_attention': self.use_attention,
            '_all_feature_names': self._all_feature_names if hasattr(self, '_all_feature_names') else None,
            '_model_feature_columns': self._model_feature_columns if hasattr(self, '_model_feature_columns') else None,
            '_feature_dim': self._feature_dim if hasattr(self, '_feature_dim') else None
        }, path)
        
        logger.info(f"模型已保存至: {path}")
        
        return path

    def load_model(self, path):
        """
        加载模型

        Args:
            path (str): 模型路径

        Returns:
            bool: 加载是否成功
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(path):
                logger.error(f"模型文件不存在: {path}")
                return False
                
            # 加载模型
            checkpoint = torch.load(path, map_location=self.device)
            
            # 恢复模型配置
            self.sequence_length = checkpoint.get('sequence_length', self.sequence_length)
            self.hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
            self.num_layers = checkpoint.get('num_layers', self.num_layers)
            self.bidirectional = checkpoint.get('bidirectional', self.bidirectional)
            self.feature_engineering = checkpoint.get('feature_engineering', self.feature_engineering)
            self.use_attention = checkpoint.get('use_attention', self.use_attention)
            
            # 恢复缩放器
            self.scaler_X = checkpoint.get('scaler_X', self.scaler_X)
            self.scaler_y = checkpoint.get('scaler_y', self.scaler_y)
            
            # 恢复特征名称相关属性
            if '_all_feature_names' in checkpoint:
                self._all_feature_names = checkpoint.get('_all_feature_names')
                logger.debug(f"加载了 {len(self._all_feature_names)} 个原始特征名")
            
            if '_model_feature_columns' in checkpoint:
                self._model_feature_columns = checkpoint.get('_model_feature_columns')
                logger.debug(f"加载了 {len(self._model_feature_columns)} 个模型特征列")
            
            if '_feature_dim' in checkpoint:
                self._feature_dim = checkpoint.get('_feature_dim')
                logger.debug(f"加载了特征维度: {self._feature_dim}")
            
            # 创建模型
            input_dim = self._feature_dim if hasattr(self, '_feature_dim') else 5
            if hasattr(self.scaler_X, 'n_features_in_'):
                # 检查是否与保存的特征维度一致
                scaler_dims = self.scaler_X.n_features_in_
                if hasattr(self, '_feature_dim') and self._feature_dim != scaler_dims:
                    logger.warning(f"特征维度不一致：保存的为 {self._feature_dim}，Scaler的为 {scaler_dims}")
                # 使用Scaler的维度，因为它是用于执行变换的
                input_dim = scaler_dims
                
            self.build_model(input_dim=input_dim)
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # 设置为评估模式
            
            logger.info(f"模型已从 {path} 加载成功")
            
            return True
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            return False

    def feature_importance(self, df, target_column='close'):
        """
        计算特征重要性

        Args:
            df (pandas.DataFrame): 输入数据
            target_column (str): 目标列名

        Returns:
            pandas.DataFrame: 特征重要性
        """
        if not self.feature_engineering:
            logger.warning("未启用特征工程，无法计算特征重要性")
            return None
            
        # 进行特征工程
        processed_df = self._engineer_features(df.copy())
        
        # 选择特征和目标
        exclude_cols = ['day_of_week', 'month']
        features_cols = [col for col in processed_df.columns if col != target_column and col not in exclude_cols]
        
        # 移除所有非数值型列，避免计算相关系数时出错
        numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
        numeric_features = [col for col in features_cols if col in numeric_cols]
        
        if target_column not in numeric_cols:
            logger.error(f"目标列 {target_column} 不是数值类型，无法计算相关性")
            return None
            
        # 检查是否有被排除的特征
        excluded_features = [col for col in features_cols if col not in numeric_features]
        if excluded_features:
            logger.warning(f"以下非数值特征将被排除在相关性计算之外: {excluded_features}")
        
        if len(numeric_features) == 0:
            logger.error("没有可用的数值型特征，无法计算相关性")
            return None
        
        # 计算特征与目标的相关性 - 仅使用数值列
        correlations = processed_df[numeric_features + [target_column]].corr()[target_column].drop(target_column)
        
        # 计算绝对相关性
        abs_correlations = correlations.abs().sort_values(ascending=False)
        
        # 创建特征重要性表格
        importance_df = pd.DataFrame({
            'feature': abs_correlations.index,
            'importance': abs_correlations.values,
            'correlation': correlations[abs_correlations.index].values
        })
        
        return importance_df 