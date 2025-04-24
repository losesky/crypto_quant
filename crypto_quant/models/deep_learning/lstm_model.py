"""
LSTM深度学习模型，用于时间序列预测
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ...utils.logger import logger
from ...config.settings import ML_CONFIG


class LSTMModel(nn.Module):
    """
    LSTM神经网络模型
    使用多层LSTM进行时间序列预测
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        初始化LSTM模型

        Args:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            num_layers (int): LSTM层数
            output_dim (int): 输出维度
            dropout (float): Dropout比率
        """
        super(LSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 激活函数
        self.relu = nn.ReLU()
        
        logger.info(f"LSTM模型初始化成功: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")

    def forward(self, x):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, output_dim)
        """
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # 获取最后一个时间步的输出
        out = self.relu(out[:, -1, :])
        
        # 全连接层
        out = self.fc(out)
        
        return out


class LSTMPricePredictor:
    """
    基于LSTM的价格预测器
    提供端到端的数据处理、训练、预测和评估功能
    """

    def __init__(self, sequence_length=10, hidden_dim=64, num_layers=2, dropout=0.2, learning_rate=0.001):
        """
        初始化LSTM价格预测器

        Args:
            sequence_length (int): 序列长度，使用多少天的数据进行预测
            hidden_dim (int): LSTM隐藏层维度
            num_layers (int): LSTM层数
            dropout (float): Dropout比率
            learning_rate (float): 学习率
        """
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # 初始化状态
        self.model = None
        self.scaler_X = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"LSTM价格预测器初始化成功: sequence_length={sequence_length}, device={self.device}")

    def _create_sequences(self, data, target_column='close'):
        """
        创建序列数据

        Args:
            data (pandas.DataFrame): 输入数据
            target_column (str): 目标列名

        Returns:
            tuple: (X, y) 特征序列和目标值
        """
        # 选择特征
        features = data[['open', 'high', 'low', 'close', 'volume']].values
        target = data[[target_column]].values
        
        # 标准化
        features_normalized = self.scaler_X.fit_transform(features)
        target_normalized = self.scaler_y.fit_transform(target)
        
        # 创建序列
        X, y = [], []
        for i in range(len(features_normalized) - self.sequence_length):
            X.append(features_normalized[i:i + self.sequence_length])
            y.append(target_normalized[i + self.sequence_length])
            
        return np.array(X), np.array(y)

    def prepare_data(self, df, target_column='close', test_size=0.2, batch_size=64):
        """
        准备训练和测试数据

        Args:
            df (pandas.DataFrame): 输入数据
            target_column (str): 目标列名
            test_size (float): 测试集比例
            batch_size (int): 批次大小

        Returns:
            tuple: (train_loader, test_loader, X_test, y_test) 训练和测试数据加载器及测试数据
        """
        # 创建序列
        X, y = self._create_sequences(df, target_column)
        
        # 检查样本数量是否足够
        min_samples = 5  # 最小样本数
        actual_samples = len(X)
        
        if actual_samples == 0:
            raise ValueError(f"没有足够的数据生成序列。确保数据长度({len(df)})大于序列长度({self.sequence_length})。")
        
        # 对于小样本调整测试集比例
        if actual_samples < min_samples:
            logger.warning(f"样本数量({actual_samples})过少，将使用所有样本进行训练，不进行测试集划分。")
            test_size = 0.0
        elif actual_samples < 2 * min_samples:
            adjusted_test_size = 1 / actual_samples
            logger.warning(f"样本数量({actual_samples})较少，将测试集比例从{test_size}调整为{adjusted_test_size}。")
            test_size = adjusted_test_size
        
        # 分割训练集和测试集
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
        else:
            # 全部用于训练
            X_train, y_train = X, y
            X_test, y_test = X[:1], y[:1]  # 用一个样本作为测试，避免空测试集
            logger.warning("所有样本用于训练，测试结果将不可靠。")
        
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
        
        logger.info(f"数据准备完成: 训练样本={len(X_train)}, 测试样本={len(X_test)}")
        
        return train_loader, test_loader, X_test, y_test

    def build_model(self, input_dim=5, output_dim=1):
        """
        构建LSTM模型

        Args:
            input_dim (int): 输入特征维度
            output_dim (int): 输出维度

        Returns:
            LSTMModel: 构建的模型
        """
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=output_dim,
            dropout=self.dropout
        ).to(self.device)
        
        return self.model

    def train(self, train_loader, num_epochs=100, patience=10):
        """
        训练模型

        Args:
            train_loader (DataLoader): 训练数据加载器
            num_epochs (int): 训练轮数
            patience (int): 早停耐心值

        Returns:
            dict: 训练历史记录
        """
        if self.model is None:
            self.build_model()
            
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 训练历史
        history = {'loss': []}
        
        # 早停
        best_loss = float('inf')
        no_improvement = 0
        
        # 训练循环
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # 前向传播
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            # 计算平均损失
            avg_loss = epoch_loss / len(train_loader)
            history['loss'].append(avg_loss)
            
            # 日志
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
            
            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improvement = 0
                # 保存最佳模型
                self._save_model()
            else:
                no_improvement += 1
                
            if no_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        logger.info(f"训练完成: 最终损失={history['loss'][-1]:.6f}")
        
        return history

    def evaluate(self, test_loader):
        """
        评估模型

        Args:
            test_loader (DataLoader): 测试数据加载器

        Returns:
            dict: 评估指标
        """
        if self.model is None:
            logger.error("模型未训练，无法评估")
            return {}
            
        self.model.eval()
        criterion = nn.MSELoss()
        
        test_loss = 0.0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                
                # 收集预测和实际值
                predictions.append(outputs.cpu().numpy())
                actuals.append(batch_y.cpu().numpy())
                
        # 计算平均损失
        avg_loss = test_loss / len(test_loader)
        
        # 合并批次结果
        predictions = np.vstack(predictions)
        actuals = np.vstack(actuals)
        
        # 转换回原始值
        predictions = self.scaler_y.inverse_transform(predictions)
        actuals = self.scaler_y.inverse_transform(actuals)
        
        # 计算评估指标
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        
        logger.info(f"模型评估: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'test_loss': avg_loss,
            'predictions': predictions,
            'actuals': actuals
        }

    def predict(self, X):
        """
        进行预测

        Args:
            X (numpy.ndarray): 输入序列

        Returns:
            numpy.ndarray: 预测结果
        """
        if self.model is None:
            logger.error("模型未训练，无法预测")
            return None
            
        self.model.eval()
        
        # 确保输入是正确形状
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=0)
            
        # 标准化
        X_normalized = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(X_tensor)
            
        # 转换回原始值
        prediction = self.scaler_y.inverse_transform(output.cpu().numpy())
        
        return prediction

    def predict_next_day(self, df, n_steps=1):
        """
        预测未来n天的价格

        Args:
            df (pandas.DataFrame): 历史数据
            n_steps (int): 预测天数

        Returns:
            list: 预测结果列表
        """
        if self.model is None:
            logger.error("模型未训练，无法预测")
            return []
            
        # 获取最近的序列数据
        recent_data = df.iloc[-self.sequence_length:][['open', 'high', 'low', 'close', 'volume']].values
        
        # 预测未来n天
        predictions = []
        current_data = recent_data.copy()
        
        for _ in range(n_steps):
            # 预测下一天
            next_day = self.predict(current_data)
            predictions.append(next_day[0][0])
            
            # 更新序列（移除最早的一天，添加预测的一天）
            current_data = np.vstack([
                current_data[1:],
                [current_data[-1, 0], current_data[-1, 1], current_data[-1, 2], next_day[0][0], current_data[-1, 4]]
            ])
            
        return predictions

    def _save_model(self, path=None):
        """
        保存模型

        Args:
            path (str, optional): 保存路径
        """
        if self.model is None:
            logger.error("模型未训练，无法保存")
            return
            
        if path is None:
            # 使用默认路径
            model_dir = ML_CONFIG.get("model_save_path", "models/saved")
            os.makedirs(model_dir, exist_ok=True)
            path = os.path.join(model_dir, "lstm_model.pth")
            
        # 保存模型状态
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
        }, path)
        
        logger.info(f"模型已保存至: {path}")

    def load_model(self, path):
        """
        加载模型

        Args:
            path (str): 模型路径

        Returns:
            bool: 是否成功加载
        """
        try:
            # 加载模型参数
            checkpoint = torch.load(path, map_location=self.device)
            
            # 恢复模型超参数
            self.sequence_length = checkpoint['sequence_length']
            self.hidden_dim = checkpoint['hidden_dim']
            self.num_layers = checkpoint['num_layers']
            self.dropout = checkpoint['dropout']
            
            # 恢复缩放器
            self.scaler_X = checkpoint['scaler_X']
            self.scaler_y = checkpoint['scaler_y']
            
            # 构建模型
            self.build_model()
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info(f"模型已从{path}加载")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return False

    def plot_predictions(self, predictions, actuals, title='LSTM Predictions vs Actuals'):
        """
        绘制预测结果与实际值对比图

        Args:
            predictions (numpy.ndarray): 预测值
            actuals (numpy.ndarray): 实际值
            title (str): 图表标题

        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        plt.figure(figsize=(12, 6))
        plt.plot(actuals, label='Actual', color='blue')
        plt.plot(predictions, label='Predicted', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf() 