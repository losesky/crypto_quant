"""
基于LSTM机器学习的交易策略
"""
import numpy as np
import pandas as pd
from ...models.deep_learning.lstm_model import LSTMPricePredictor
from ...utils.logger import logger


class LSTMStrategy:
    """
    LSTM策略类，使用LSTM模型进行价格预测并生成交易信号
    """

    def __init__(self, sequence_length=10, prediction_threshold=0.01, hidden_dim=64, num_layers=2, model_path=None):
        """
        初始化LSTM策略

        Args:
            sequence_length (int): 用于预测的历史序列长度
            prediction_threshold (float): 预测变化阈值，超过此值才产生信号
            hidden_dim (int): LSTM隐藏层维度
            num_layers (int): LSTM层数
            model_path (str, optional): 预训练模型路径，如果提供则加载模型
        """
        self.sequence_length = sequence_length
        self.prediction_threshold = prediction_threshold
        self.name = f"LSTM({sequence_length},{prediction_threshold:.2%})"
        
        # 创建LSTM预测器
        self.predictor = LSTMPricePredictor(
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # 加载预训练模型（如果提供）
        if model_path:
            success = self.predictor.load_model(model_path)
            if success:
                logger.info(f"LSTM策略已加载预训练模型: {model_path}")
            else:
                logger.warning(f"LSTM策略加载预训练模型失败，将使用未训练模型")
        
        # 初始化缓存
        self._is_trained = (model_path is not None)
        
        logger.info(f"LSTM策略初始化完成: {self.name}")

    def train(self, df, target_column='close', test_size=0.2, epochs=100, patience=10):
        """
        训练LSTM模型

        Args:
            df (pandas.DataFrame): 训练数据
            target_column (str): 目标列名
            test_size (float): 测试集比例
            epochs (int): 训练轮数
            patience (int): 早停耐心值

        Returns:
            dict: 训练历史和评估结果
        """
        logger.info(f"开始训练LSTM模型: 数据量={len(df)}, epochs={epochs}")
        
        # 准备数据
        train_loader, test_loader, X_test, y_test = self.predictor.prepare_data(
            df, target_column=target_column, test_size=test_size
        )
        
        # 构建模型
        self.predictor.build_model()
        
        # 训练模型
        history = self.predictor.train(train_loader, num_epochs=epochs, patience=patience)
        
        # 评估模型
        evaluation = self.predictor.evaluate(test_loader)
        
        # 标记为已训练
        self._is_trained = True
        
        logger.info(f"LSTM模型训练完成: MSE={evaluation['mse']:.4f}, RMSE={evaluation['rmse']:.4f}")
        
        return {
            'history': history,
            'evaluation': evaluation
        }

    def predict_next_day(self, df, n_steps=1):
        """
        预测未来n天的价格

        Args:
            df (pandas.DataFrame): 历史数据
            n_steps (int): 预测天数

        Returns:
            list: 预测结果列表
        """
        if not self._is_trained:
            logger.warning("LSTM模型未训练，无法进行可靠预测")
            return []
            
        predictions = self.predictor.predict_next_day(df, n_steps=n_steps)
        
        return predictions

    def calculate_indicators(self, df):
        """
        计算技术指标

        Args:
            df (pandas.DataFrame): 包含OHLCV数据的DataFrame

        Returns:
            pandas.DataFrame: 添加了指标的DataFrame
        """
        # 复制数据避免修改原始数据
        df = df.copy()
        
        # 确保LSTM模型已训练
        if not self._is_trained:
            logger.warning("LSTM模型未训练，将进行实时训练")
            self.train(df)
        
        # 创建预测列
        df['predicted_close'] = np.nan
        df['predicted_change'] = np.nan
        
        # 对每个预测窗口进行预测
        for i in range(self.sequence_length, len(df)):
            # 获取窗口数据
            window_data = df.iloc[i-self.sequence_length:i][['open', 'high', 'low', 'close', 'volume']].values
            
            # 预测
            try:
                predicted_price = self.predictor.predict(window_data)[0][0]
                
                # 填充预测值
                df.loc[df.index[i], 'predicted_close'] = predicted_price
                
                # 计算预测变化率
                current_price = df.iloc[i]['close']
                previous_price = df.iloc[i-1]['close']
                
                df.loc[df.index[i], 'predicted_change'] = (predicted_price - current_price) / current_price
            except Exception as e:
                logger.error(f"预测失败: {str(e)}")
        
        return df

    def generate_signals(self, df):
        """
        生成交易信号

        Args:
            df (pandas.DataFrame): 包含价格数据的DataFrame

        Returns:
            pandas.DataFrame: 添加了交易信号的DataFrame
        """
        # 计算指标
        df = self.calculate_indicators(df)
        
        # 生成信号
        # 1. 当预测价格变化超过阈值时买入
        # 2. 当预测价格变化低于负阈值时卖出
        # 3. 其他情况保持不变
        
        # 初始化信号列
        df['signal'] = 0
        
        # 当预测变化率超过阈值时买入
        df.loc[df['predicted_change'] > self.prediction_threshold, 'signal'] = 1
        
        # 当预测变化率低于负阈值时卖出
        df.loc[df['predicted_change'] < -self.prediction_threshold, 'signal'] = -1
        
        # 生成仓位
        df['position'] = df['signal'].replace(to_replace=0, method='ffill')
        df['position'] = df['position'].fillna(0)
        
        return df

    def backtest(self, df, initial_capital=10000.0, commission=0.001):
        """
        回测策略

        Args:
            df (pandas.DataFrame): 包含OHLCV数据的DataFrame
            initial_capital (float): 初始资金
            commission (float): 手续费率，如0.001表示0.1%

        Returns:
            pandas.DataFrame: 回测结果
        """
        # 生成信号
        df = self.generate_signals(df)
        
        # 计算每日收益
        df['returns'] = df['close'].pct_change()
        
        # 计算策略收益 (今天的仓位 * 明天的收益率)
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        
        # 计算考虑手续费的策略收益
        # 当仓位发生变化时，收取手续费
        df['position_change'] = df['position'].diff().fillna(0) 
        df['commission_cost'] = abs(df['position_change']) * commission * df['close']
        
        # 调整策略收益
        df['strategy_returns_after_commission'] = df['strategy_returns'] * df['close'] - df['commission_cost']
        df['strategy_returns_after_commission'] = df['strategy_returns_after_commission'] / df['close'].shift(1)
        
        # 计算累积收益
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['cumulative_strategy_returns'] = (1 + df['strategy_returns_after_commission']).cumprod()
        
        # 计算资金曲线
        df['equity_curve'] = initial_capital * df['cumulative_strategy_returns']
        
        # 计算回撤
        df['previous_peaks'] = df['equity_curve'].cummax()
        df['drawdown'] = (df['equity_curve'] - df['previous_peaks']) / df['previous_peaks']
        
        return df

    def get_performance_metrics(self, df):
        """
        计算策略表现指标

        Args:
            df (pandas.DataFrame): 回测结果DataFrame

        Returns:
            dict: 包含策略表现指标的字典
        """
        # 年化收益率
        total_days = (df.index[-1] - df.index[0]).days
        annual_return = (df['cumulative_strategy_returns'].iloc[-1] ** (365 / total_days)) - 1
        
        # 最大回撤
        max_drawdown = df['drawdown'].min()
        
        # 夏普比率 (假设无风险利率为0.02)
        risk_free_rate = 0.02
        sharpe_ratio = ((df['strategy_returns_after_commission'].mean() * 252) - risk_free_rate) / \
                      (df['strategy_returns_after_commission'].std() * np.sqrt(252))
        
        # 卡尔玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
        
        # 胜率
        winning_trades = (df['strategy_returns_after_commission'] > 0).sum()
        losing_trades = (df['strategy_returns_after_commission'] < 0).sum()
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        # 盈亏比
        average_win = df.loc[df['strategy_returns_after_commission'] > 0, 'strategy_returns_after_commission'].mean()
        average_loss = abs(df.loc[df['strategy_returns_after_commission'] < 0, 'strategy_returns_after_commission'].mean())
        profit_loss_ratio = average_win / average_loss if average_loss != 0 else float('inf')
        
        # 总收益
        total_return = df['cumulative_strategy_returns'].iloc[-1] - 1
        
        return {
            "annual_return": annual_return,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades
        }

    def run(self, df, initial_capital=10000.0, commission=0.001, train_test_split=0.7):
        """
        策略运行方法（回测）

        Args:
            df (pandas.DataFrame): 包含OHLCV数据的DataFrame
            initial_capital (float): 初始资金
            commission (float): 手续费率，如0.001表示0.1%
            train_test_split (float): 训练集比例，仅在需要训练模型时使用

        Returns:
            tuple: (results, performance_metrics) 结果和性能指标
        """
        # 确保LSTM模型已训练
        if not self._is_trained:
            logger.info("模型未训练，将使用部分数据进行训练")
            
            # 使用前70%数据进行训练
            train_size = int(len(df) * train_test_split)
            if train_size < self.sequence_length + 5:  # 确保训练数据足够
                logger.warning(f"数据量({len(df)})过少，无法进行有效的训练/测试分割。使用所有数据训练。")
                train_df = df
            else:
                train_df = df.iloc[:train_size]
                logger.info(f"使用前{train_size}条数据训练模型")
            
            # 减小epochs数量，避免过拟合小样本
            epochs = min(100, max(20, len(train_df) // 2))
            
            # 训练模型，对小数据集使用小的测试集比例
            test_size = 0.1 if len(train_df) < 50 else 0.2
            self.train(train_df, test_size=test_size, epochs=epochs, patience=5)
        
        # 执行回测
        results = self.backtest(df, initial_capital, commission)
        
        # 计算性能指标
        performance_metrics = self.get_performance_metrics(results)
        
        # 检查性能指标是否符合风控要求
        max_drawdown = abs(performance_metrics.get('max_drawdown', 0))
        calmar_ratio = performance_metrics.get('calmar_ratio', 0)
        
        # 根据crypto-quant-rules中的风控要求检查
        if max_drawdown > 0.15:  # 15%
            logger.warning(f"Strategy max drawdown ({max_drawdown:.2%}) exceeds limit (15.00%)")
        
        if calmar_ratio < 2.5:
            logger.warning(f"Strategy Calmar ratio ({calmar_ratio:.2f}) below required level (2.50)")
        
        logger.info(f"策略回测完成: {self.name}")
        
        return results, performance_metrics 