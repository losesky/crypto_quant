"""
基于增强型LSTM机器学习的交易策略
包含注意力机制和高级特征工程
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from ...models.deep_learning.enhanced_lstm_model import EnhancedLSTMPricePredictor
from ...utils.logger import logger
from ...utils.output_helper import get_image_path


class EnhancedLSTMStrategy:
    """
    增强型LSTM策略类，使用带注意力机制的LSTM模型进行价格预测并生成交易信号
    """

    def __init__(self, sequence_length=20, prediction_threshold=0.01, hidden_dim=128, 
                 num_layers=3, feature_engineering=True, use_attention=True, 
                 model_path=None, bidirectional=True, stop_loss_pct=None, take_profit_pct=None):
        """
        初始化增强型LSTM策略

        Args:
            sequence_length (int): 用于预测的历史序列长度
            prediction_threshold (float): 预测变化阈值，超过此值才产生信号
            hidden_dim (int): LSTM隐藏层维度
            num_layers (int): LSTM层数
            feature_engineering (bool): 是否使用特征工程
            use_attention (bool): 是否使用注意力机制
            model_path (str, optional): 预训练模型路径，如果提供则加载模型
            bidirectional (bool): 是否使用双向LSTM
            stop_loss_pct (float, optional): 止损百分比，如果提供则启用止损
            take_profit_pct (float, optional): 止盈百分比，如果提供则启用止盈
        """
        self.sequence_length = sequence_length
        self.prediction_threshold = prediction_threshold
        self.feature_engineering = feature_engineering
        self.use_attention = use_attention
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.name = f"EnhancedLSTM({sequence_length},{prediction_threshold:.2%},{hidden_dim},{num_layers})"
        
        # 如果启用止损，添加到策略名称
        if self.stop_loss_pct is not None:
            self.name += f",SL={self.stop_loss_pct:.1%}"
            
        # 如果启用止盈，添加到策略名称
        if self.take_profit_pct is not None:
            self.name += f",TP={self.take_profit_pct:.1%}"
        
        # 创建LSTM预测器
        self.predictor = EnhancedLSTMPricePredictor(
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            feature_engineering=feature_engineering,
            use_attention=use_attention,
            bidirectional=bidirectional
        )
        
        # 加载预训练模型（如果提供）
        if model_path:
            success = self.predictor.load_model(model_path)
            if success:
                logger.info(f"增强型LSTM策略已加载预训练模型: {model_path}")
            else:
                logger.warning(f"增强型LSTM策略加载预训练模型失败，将使用未训练模型")
        
        # 初始化缓存
        self._is_trained = (model_path is not None)
        self.entry_price = None  # 入场价格，用于止损和止盈
        
        # 添加使用特征工程和注意力机制的信息
        feature_info = "特征工程" if feature_engineering else "仅OHLCV"
        attention_info = "注意力机制" if use_attention else "标准LSTM"
        direction_info = "双向" if bidirectional else "单向"
        
        logger.info(f"增强型LSTM策略初始化完成: {self.name}")
        logger.info(f"使用{feature_info}, {attention_info}, {direction_info}LSTM, {num_layers}层, {hidden_dim}维隐藏层")

    def train(self, df, target_column='close', test_size=0.2, epochs=150, patience=15, batch_size=32):
        """
        训练LSTM模型

        Args:
            df (pandas.DataFrame): 训练数据
            target_column (str): 目标列名
            test_size (float): 测试集比例
            epochs (int): 训练轮数
            patience (int): 早停耐心值
            batch_size (int): 批次大小

        Returns:
            dict: 训练历史和评估结果
        """
        logger.info(f"开始训练增强型LSTM模型: 数据量={len(df)}, epochs={epochs}, patience={patience}")
        
        # 准备数据
        train_loader, test_loader, X_test, y_test, features_cols = self.predictor.prepare_data(
            df, target_column=target_column, test_size=test_size, batch_size=batch_size
        )
        
        # 构建模型 (prepare_data中会确定input_dim)
        self.predictor.build_model(input_dim=X_test.shape[2])
        
        # 训练模型
        history = self.predictor.train(train_loader, test_loader, num_epochs=epochs, patience=patience)
        
        # 评估模型
        evaluation = self.predictor.evaluate(test_loader)
        
        # 标记为已训练
        self._is_trained = True
        
        # 绘制训练历史
        self._plot_training_history(history)
        
        # 绘制预测结果
        self._plot_predictions(evaluation['predictions'], evaluation['targets'], df.index[-len(evaluation['predictions']):])
        
        # 如果使用注意力机制，绘制注意力权重
        if self.use_attention and 'attention_weights' in evaluation and evaluation['attention_weights'] is not None:
            self.predictor.plot_attention(df)
            attention_path = get_image_path("enhanced_lstm_attention.png")
            plt.savefig(attention_path)
            logger.info(f"注意力权重可视化已保存至: {attention_path}")
        
        # 如果使用特征工程，计算特征重要性
        if self.feature_engineering:
            importance_df = self.predictor.feature_importance(df)
            if importance_df is not None:
                self._plot_feature_importance(importance_df)
        
        logger.info(f"增强型LSTM模型训练完成: MSE={evaluation['mse']:.4f}, RMSE={evaluation['rmse']:.4f}, "
                  f"方向准确率={evaluation['direction_accuracy']:.2f}%")
        
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
            # 训练整个数据集以确保特征工程一致性
            self.train(df)
        
        # 创建预测列
        df['predicted_close'] = np.nan
        df['predicted_change'] = np.nan
        
        # 如果数据量不足，返回原始数据
        if len(df) <= self.sequence_length:
            logger.warning(f"数据量({len(df)})不足以进行预测(需要{self.sequence_length + 1}条)")
            return df
        
        # 确保'close'列存在
        if 'close' not in df.columns:
            logger.error("数据中缺少'close'列，无法计算指标")
            return df
        
        # 确保索引是日期时间类型
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                logger.warning(f"转换索引为日期时间类型失败: {str(e)}")
        
        # 设置迭代次数和重试次数，防止过多的失败操作
        max_retries = 3
        retry_count = 0
        success_count = 0
        error_count = 0
        
        # 使用滑动窗口进行预测
        for i in range(self.sequence_length, len(df)):
            tries = 0
            while tries < max_retries:
                try:
                    # 获取窗口数据
                    window_df = df.iloc[i-self.sequence_length:i].copy()
                    
                    # 进行预测 - 只预测1步
                    predicted_price = self.predictor.predict_next_day(window_df, n_steps=1)
                    
                    # 如果预测成功，填充预测值
                    if predicted_price and len(predicted_price) > 0:
                        # 填充预测值
                        df.loc[df.index[i], 'predicted_close'] = predicted_price[0]
                        
                        # 计算预测变化率
                        current_price = df.iloc[i]['close']
                        df.loc[df.index[i], 'predicted_change'] = (predicted_price[0] - current_price) / current_price
                        
                        success_count += 1
                        break  # 成功预测，退出重试循环
                    else:
                        # 预测失败但没有抛出异常，这很奇怪，尝试重试
                        logger.warning(f"第{i}个窗口预测返回空结果，尝试重试 ({tries+1}/{max_retries})")
                        tries += 1
                except Exception as e:
                    # 处理预测异常
                    logger.error(f"第{i}个窗口预测失败: {str(e)}")
                    tries += 1
                    if tries >= max_retries:
                        error_count += 1
                        # 已达到最大重试次数，继续下一个窗口
                        break
        
        # 执行额外的健全性检查 - 如果成功率低于50%，可能存在系统性问题
        if success_count + error_count > 0:
            success_rate = success_count / (success_count + error_count) * 100
            logger.info(f"预测完成: 成功={success_count}, 失败={error_count}, 成功率={success_rate:.1f}%")
            
            if success_rate < 50:
                logger.warning("预测成功率低于50%，可能存在特征不匹配或模型问题")
                
                # 如果全部预测都失败了，重新训练模型并尝试再次预测
                if success_count == 0 and retry_count == 0:
                    logger.warning("所有预测都失败，尝试重新训练模型并再次预测")
                    retry_count += 1
                    
                    # 重新训练模型
                    logger.info("正在使用当前数据重新训练模型...")
                    self.train(df)
                    
                    # 递归调用自身，但只尝试一次，防止无限循环
                    if retry_count <= 1:
                        return self.calculate_indicators(df)
        
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
        
        # 应用止损和止盈（如果设置）
        if self.stop_loss_pct is not None or self.take_profit_pct is not None:
            self._apply_risk_management(df)
        
        return df

    def _apply_risk_management(self, df):
        """
        应用风险管理：止损和止盈
        
        Args:
            df (pandas.DataFrame): 带有仓位的DataFrame
            
        Returns:
            pandas.DataFrame: 应用了风险管理的DataFrame
        """
        # 记录入场价格
        entry_price = None
        last_position = 0
        
        for i in range(1, len(df)):
            # 如果发生仓位变化，更新入场价格
            if df['position'].iloc[i] != last_position and df['position'].iloc[i] != 0:
                entry_price = df['close'].iloc[i]
            
            # 如果持有多头仓位
            if df['position'].iloc[i] == 1 and entry_price is not None:
                current_price = df['close'].iloc[i]
                price_change = (current_price - entry_price) / entry_price
                
                # 止损检查
                if self.stop_loss_pct is not None and price_change < -self.stop_loss_pct:
                    df.loc[df.index[i], 'position'] = 0
                    logger.info(f"触发止损: 日期={df.index[i]}, 入场价={entry_price:.2f}, 当前价={current_price:.2f}, 跌幅={price_change:.2%}")
                    entry_price = None
                
                # 止盈检查
                elif self.take_profit_pct is not None and price_change > self.take_profit_pct:
                    df.loc[df.index[i], 'position'] = 0
                    logger.info(f"触发止盈: 日期={df.index[i]}, 入场价={entry_price:.2f}, 当前价={current_price:.2f}, 涨幅={price_change:.2%}")
                    entry_price = None
            
            # 如果持有空头仓位
            elif df['position'].iloc[i] == -1 and entry_price is not None:
                current_price = df['close'].iloc[i]
                price_change = (entry_price - current_price) / entry_price  # 注意这里是反向的
                
                # 止损检查
                if self.stop_loss_pct is not None and price_change < -self.stop_loss_pct:
                    df.loc[df.index[i], 'position'] = 0
                    logger.info(f"触发止损: 日期={df.index[i]}, 入场价={entry_price:.2f}, 当前价={current_price:.2f}, 上涨={-price_change:.2%}")
                    entry_price = None
                
                # 止盈检查
                elif self.take_profit_pct is not None and price_change > self.take_profit_pct:
                    df.loc[df.index[i], 'position'] = 0
                    logger.info(f"触发止盈: 日期={df.index[i]}, 入场价={entry_price:.2f}, 当前价={current_price:.2f}, 下跌={price_change:.2%}")
                    entry_price = None
            
            # 如果当前无仓位，重置入场价格
            if df['position'].iloc[i] == 0:
                entry_price = None
            
            # 更新上一个仓位
            last_position = df['position'].iloc[i]
        
        return df

    def _plot_training_history(self, history):
        """
        绘制训练历史
        
        Args:
            history (dict): 训练历史记录
        """
        plt.figure(figsize=(12, 6))
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 第一个子图：损失曲线
        ax1.plot(history['train_loss'], label='训练损失')
        if 'val_loss' in history and history['val_loss']:
            ax1.plot(history['val_loss'], label='验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True)
        
        # 第二个子图：学习率曲线
        ax2.plot(history['lr'], label='学习率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('学习率')
        ax2.set_title('学习率变化')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        history_path = get_image_path("enhanced_lstm_training_history.png")
        plt.savefig(history_path)
        logger.info(f"训练历史图表已保存至: {history_path}")
        plt.close()

    def _plot_predictions(self, predictions, targets, dates=None):
        """
        绘制预测结果
        
        Args:
            predictions (numpy.ndarray): 预测值
            targets (numpy.ndarray): 真实值
            dates (pandas.DatetimeIndex, optional): 日期索引
        """
        plt.figure(figsize=(12, 6))
        
        if dates is not None and len(dates) == len(predictions):
            plt.plot(dates, targets, label='实际价格')
            plt.plot(dates, predictions, label='预测价格', linestyle='--')
            plt.xlabel('日期')
        else:
            plt.plot(targets, label='实际价格')
            plt.plot(predictions, label='预测价格', linestyle='--')
            plt.xlabel('时间步')
        
        plt.ylabel('价格')
        plt.title('增强型LSTM模型预测结果')
        plt.legend()
        plt.grid(True)
        
        # 添加误差信息
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        
        info_text = f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%'
        plt.figtext(0.02, 0.02, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存图表
        predict_path = get_image_path("enhanced_lstm_predictions.png")
        plt.savefig(predict_path)
        logger.info(f"预测结果图表已保存至: {predict_path}")
        plt.close()

    def _plot_feature_importance(self, importance_df):
        """
        绘制特征重要性
        
        Args:
            importance_df (pandas.DataFrame): 特征重要性数据框
        """
        plt.figure(figsize=(12, 8))
        
        # 只显示前20个特征
        n_features = min(20, len(importance_df))
        top_features = importance_df.head(n_features)
        
        # 创建彩色条形图，根据相关性正负着色
        colors = ['green' if c > 0 else 'red' for c in top_features['correlation']]
        plt.barh(range(n_features), top_features['importance'], color=colors, alpha=0.6)
        
        # 添加特征标签
        plt.yticks(range(n_features), top_features['feature'])
        
        plt.xlabel('重要性分数（绝对相关性）')
        plt.title('增强型LSTM模型特征重要性（绿色为正相关，红色为负相关）')
        plt.gca().invert_yaxis()  # 反转y轴，使最重要的特征在顶部
        
        plt.tight_layout()
        
        # 保存图表
        importance_path = get_image_path("enhanced_lstm_feature_importance.png")
        plt.savefig(importance_path)
        logger.info(f"特征重要性图表已保存至: {importance_path}")
        plt.close()

    def run(self, df, initial_capital=10000.0, commission=0.001, train_test_split=0.7, save_model=True):
        """
        运行策略：训练模型并生成交易信号

        Args:
            df (pandas.DataFrame): 包含OHLCV数据的DataFrame
            initial_capital (float): 初始资金
            commission (float): 手续费率，如0.001表示0.1%
            train_test_split (float): 训练集比例
            save_model (bool): 是否保存训练好的模型

        Returns:
            tuple: (结果DataFrame, 性能指标字典)
        """
        # 确保数据量足够
        if len(df) < self.sequence_length * 3:
            logger.error(f"数据量({len(df)})不足以训练和测试模型，需要至少{self.sequence_length * 3}条数据")
            return df, {}
        
        # 分割训练集和测试集
        train_size = int(len(df) * train_test_split)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size-self.sequence_length:]  # 重叠序列长度的数据
        
        logger.info(f"分割数据：训练集 {len(train_df)} 条，测试集 {len(test_df)} 条")
        
        # 训练模型
        self.train(train_df)
        
        # 生成交易信号
        result_df = self.generate_signals(test_df)
        
        # 保存模型（如果需要）
        if save_model and self._is_trained:
            model_path = self.predictor.save_model()
            logger.info(f"增强型LSTM模型已保存至: {model_path}")
        
        # 计算性能指标
        performance = self._calculate_performance_metrics(result_df, initial_capital, commission)
        
        # 返回测试集的结果和性能指标
        return result_df, performance
        
    def _calculate_performance_metrics(self, df, initial_capital, commission):
        """
        计算策略性能指标
        
        Args:
            df (pandas.DataFrame): 包含交易信号的DataFrame
            initial_capital (float): 初始资金
            commission (float): 手续费率
            
        Returns:
            dict: 性能指标字典
        """
        # 确保数据中有position列
        if 'position' not in df.columns:
            logger.error("数据中缺少position列，无法计算性能指标")
            return {}
            
        # 计算每日收益率
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        
        # 考虑手续费（仅在仓位变化时）
        df['position_change'] = df['position'].diff().fillna(0)
        df['commission_cost'] = abs(df['position_change']) * commission
        df['strategy_returns'] = df['strategy_returns'] - df['commission_cost']
        
        # 累计收益
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod()
        
        # 计算权益曲线
        df['equity_curve'] = initial_capital * df['cumulative_strategy_returns']
        
        # 计算高点和回撤
        df['previous_peaks'] = df['equity_curve'].cummax()
        df['drawdown'] = (df['equity_curve'] - df['previous_peaks']) / df['previous_peaks']
        
        # 性能指标
        total_return = df['cumulative_strategy_returns'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        
        # 计算夏普比率（假设无风险利率为0）
        daily_returns = df['strategy_returns'].mean()
        daily_volatility = df['strategy_returns'].std()
        sharpe_ratio = (daily_returns / daily_volatility) * (252 ** 0.5) if daily_volatility > 0 else 0
        
        # 计算最大回撤
        max_drawdown = df['drawdown'].min()
        
        # 计算卡尔玛比率
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown < 0 else float('inf')
        
        # 计算胜率
        winning_days = (df['strategy_returns'] > 0).sum()
        losing_days = (df['strategy_returns'] < 0).sum()
        win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0
        
        # 计算盈亏比
        avg_win = df.loc[df['strategy_returns'] > 0, 'strategy_returns'].mean() if winning_days > 0 else 0
        avg_loss = abs(df.loc[df['strategy_returns'] < 0, 'strategy_returns'].mean()) if losing_days > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # 返回性能指标字典
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'winning_trades': winning_days,
            'losing_trades': losing_days
        } 