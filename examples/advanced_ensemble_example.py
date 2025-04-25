"""
高级集成策略示例
展示如何使用改进的市场状态检测和自适应集成方法
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import traceback

# 确保可以导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_quant.strategies.technical.macd_strategy import MACDStrategy
from crypto_quant.strategies.ml_based.enhanced_lstm_strategy import EnhancedLSTMStrategy
from crypto_quant.strategies.hybrid.macd_lstm_hybrid_strategy import MACDLSTMHybridStrategy
from crypto_quant.strategies.hybrid.gradient_boosting_ensemble import GradientBoostingEnsemble
from crypto_quant.strategies.hybrid.neural_network_ensemble import NeuralNetworkEnsemble
from crypto_quant.indicators.technical_indicators import TechnicalIndicators
from crypto_quant.data.sources.binance_source import BinanceDataSource
from crypto_quant.utils.logger import logger, set_log_level
from crypto_quant.utils.font_helper import get_font_helper
from crypto_quant.risk_management.risk_manager import RiskManager

# 设置日志级别
set_log_level('INFO')

# 获取字体助手
font_helper = get_font_helper()

# 创建输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output/advanced_ensemble")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_data(symbol="BTCUSDT", interval="1h", start_date=None, end_date=None, use_small_dataset=False):
    """获取数据"""
    # 设置时间范围，默认获取近2年数据
    if start_date is None:
        if use_small_dataset:
            # 测试时使用较小的数据集，仅获取最近3个月数据
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        else:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    logger.info(f"获取{symbol}数据，时间范围: {start_date} 到 {end_date}")
    
    # 使用数据源获取数据
    data_source = BinanceDataSource()
    df = data_source.get_historical_data(
        symbol=symbol.replace('USDT', '/USDT'),  # 转换为BinanceDataSource格式
        interval=interval,
        start=start_date,
        end=end_date
    )
    
    logger.info(f"获取到 {len(df)} 条数据")
    return df


def analyze_market_regimes(df):
    """分析不同市场状态"""
    # 创建结果目录
    regime_dir = os.path.join(OUTPUT_DIR, "market_regimes")
    os.makedirs(regime_dir, exist_ok=True)
    
    # 识别市场状态
    logger.info("识别市场状态...")
    df['market_regime'] = 'unknown'
    
    # 逐行计算市场状态
    for i in range(50, len(df)):
        current_data = df.iloc[:i+1].copy()
        regime = TechnicalIndicators.identify_market_regime(current_data)
        df.loc[df.index[i], 'market_regime'] = regime
    
    # 统计各市场状态比例
    regime_counts = df['market_regime'].value_counts()
    logger.info("市场状态分布:")
    for regime, count in regime_counts.items():
        percentage = count / len(df) * 100
        logger.info(f"  {regime}: {count} ({percentage:.2f}%)")
    
    # 将市场状态转换为数值，便于机器学习处理
    regime_map = {
        'trending_volatile': 3,
        'ranging_volatile': 2,
        'trending_stable': 1,
        'ranging_stable': 0,
        'unknown': -1
    }
    df['market_regime_num'] = df['market_regime'].map(regime_map)
    
    # 绘制市场状态分布图
    plt.figure(figsize=(10, 6))
    regime_counts.plot(kind='bar')
    plt.title('市场状态分布')
    plt.tight_layout()
    plt.savefig(os.path.join(regime_dir, 'regime_distribution.png'))
    
    # 绘制市场状态随时间变化图
    plt.figure(figsize=(14, 7))
    
    # 绘制收盘价和市场状态
    ax1 = plt.subplot(211)
    ax1.plot(df.index, df['close'], 'b-')
    ax1.set_ylabel('价格')
    ax1.set_title('BTC价格与市场状态')
    
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(df.index, df['market_regime_num'], 'r-')
    ax2.set_ylabel('市场状态')
    ax2.set_yticks(list(regime_map.values()))
    ax2.set_yticklabels(list(regime_map.keys()))
    
    plt.tight_layout()
    plt.savefig(os.path.join(regime_dir, 'regime_time_series.png'))
    
    # 分析不同市场状态下的价格特性
    grouped = df.groupby('market_regime')
    regime_stats = grouped['close'].agg(['count', 'mean', 'std', 'min', 'max'])
    regime_stats['volatility'] = grouped.apply(lambda x: x['close'].pct_change().std())
    regime_stats['daily_range'] = grouped.apply(lambda x: ((x['high'] / x['low']) - 1).mean())
    
    logger.info("\n不同市场状态的价格特性:")
    logger.info(regime_stats)
    
    regime_stats.to_csv(os.path.join(regime_dir, 'regime_stats.csv'))
    
    return df


def compare_strategies(df):
    """比较不同策略的表现"""
    
    # 创建保存目录
    strategy_dir = os.path.join(OUTPUT_DIR, 'strategy_comparison')
    os.makedirs(strategy_dir, exist_ok=True)
    
    # 创建风险管理器
    risk_manager = RiskManager(max_drawdown=0.15, max_position_size=0.20)
    
    # 创建基础策略
    logger.info("创建基础策略...")
    macd_strategy = MACDStrategy()
    
    # LSTM参数
    lstm_strategy = EnhancedLSTMStrategy(
        sequence_length=20,
        prediction_threshold=0.01,
        hidden_dim=128,
        feature_engineering=True,
        use_attention=True
    )
    
    # 创建Expert混合策略
    logger.info("创建Expert混合策略...")
    expert_strategy = MACDLSTMHybridStrategy(
        # MACD参数
        macd_fast_period=12,
        macd_slow_period=26,
        macd_signal_period=9,
        
        # LSTM参数
        lstm_sequence_length=20,
        lstm_hidden_dim=128,
        lstm_feature_engineering=True,
        lstm_use_attention=True,
        
        # 混合参数
        ensemble_method='expert',
        ensemble_weights=(0.5, 0.5),
        market_regime_threshold=0.15,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        output_dir=strategy_dir
    )
    expert_strategy.risk_manager = risk_manager
    
    # 创建梯度提升集成策略
    logger.info("创建梯度提升集成策略...")
    base_strategies = [
        {'strategy': macd_strategy, 'name': 'macd'},
        {'strategy': lstm_strategy, 'name': 'lstm'}
    ]
    
    gb_model_path = os.path.join(strategy_dir, 'gb_model.joblib')
    gb_ensemble = GradientBoostingEnsemble(
        base_strategies=base_strategies,
        window_size=50,
        retrain_interval=100,
        min_train_samples=200,
        prediction_threshold=0.005,
        model_path=gb_model_path
    )
    
    # 检查是否需要重新训练梯度提升模型
    need_train_gb = not gb_ensemble._is_trained
    if need_train_gb:
        logger.warning("梯度提升模型加载失败或不存在，将在信号生成过程中重新训练")
    else:
        # 调试现有模型结构
        logger.info("检查梯度提升模型结构:")
        gb_ensemble.debug_model_structure()
    
    # 创建神经网络集成策略
    logger.info("创建神经网络集成策略...")
    nn_model_path = os.path.join(strategy_dir, 'nn_model.pt')
    nn_ensemble = NeuralNetworkEnsemble(
        base_strategies=base_strategies,
        window_size=50,
        retrain_interval=150,
        min_train_samples=250,
        prediction_threshold=0.005,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=30,  # 30轮训练，以加快演示
        model_path=nn_model_path,
        use_attention=True
    )
    
    # 检查是否需要重新训练神经网络模型
    need_train_nn = not nn_ensemble._is_trained
    if need_train_nn:
        logger.warning("神经网络模型加载失败或不存在，将在信号生成过程中重新训练")
    
    # 训练LSTM模型
    logger.info("准备LSTM模型...")
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size].copy()
    
    # 去除训练数据中可能导致问题的列
    numerical_columns = ['open', 'high', 'low', 'close', 'volume']
    train_columns = [col for col in train_df.columns if col in numerical_columns]
    train_df_cleaned = train_df[train_columns].copy()
    
    logger.info(f"使用数值型列进行训练: {train_columns}")
    lstm_strategy.train(train_df_cleaned)
    
    # 生成信号
    strategies_dict = {}
    valid_strategies = {}
    
    # 尝试为每个策略生成信号，并验证返回结果是否有效
    try:
        logger.info("生成MACD策略信号...")
        macd_df = macd_strategy.generate_signals(df)
        if macd_df is not None and isinstance(macd_df, pd.DataFrame) and 'position' in macd_df.columns:
            valid_strategies['MACD'] = macd_df
        else:
            logger.warning("MACD策略未返回有效的信号DataFrame")
    except Exception as e:
        logger.error(f"生成MACD策略信号时出错: {str(e)}")
    
    try:
        logger.info("生成LSTM策略信号...")
        lstm_df = lstm_strategy.generate_signals(df)
        if lstm_df is not None and isinstance(lstm_df, pd.DataFrame) and 'position' in lstm_df.columns:
            valid_strategies['LSTM'] = lstm_df
        else:
            logger.warning("LSTM策略未返回有效的信号DataFrame")
    except Exception as e:
        logger.error(f"生成LSTM策略信号时出错: {str(e)}")
    
    try:
        logger.info("生成Expert策略信号...")
        expert_df = expert_strategy.generate_signals(df)
        if expert_df is not None and isinstance(expert_df, pd.DataFrame) and 'position' in expert_df.columns:
            valid_strategies['Expert'] = expert_df
        else:
            logger.warning("Expert策略未返回有效的信号DataFrame")
    except Exception as e:
        logger.error(f"生成Expert策略信号时出错: {str(e)}")
    
    try:
        logger.info("生成梯度提升集成策略信号...")
        # 如果模型加载失败，需要预先训练
        if need_train_gb:
            train_size = int(len(df) * 0.8)
            logger.info(f"开始训练梯度提升模型，使用前{train_size}行数据...")
            gb_train_features = gb_ensemble.prepare_features(df.iloc[:train_size], df.iloc[:train_size].index[-1], is_training=True)
            
            # 检查特征
            logger.info(f"梯度提升训练特征形状: {gb_train_features.shape}")
            logger.info(f"特征列: {gb_train_features.columns.tolist()}")
            
            # 只使用数值列
            numeric_cols = gb_train_features.select_dtypes(include=['number']).columns
            gb_train_features = gb_train_features[numeric_cols]
            
            # 训练模型
            if 'future_returns' in gb_train_features.columns:
                gb_target = gb_train_features['future_returns']
                gb_ensemble.train(gb_train_features, gb_target)
            else:
                logger.error("找不到目标列'future_returns'，无法训练梯度提升模型")
        
        # 生成信号
        gb_df = gb_ensemble.generate_signals(df)
        if gb_df is not None and isinstance(gb_df, pd.DataFrame) and 'position' in gb_df.columns:
            valid_strategies['GradientBoosting'] = gb_df
        else:
            logger.warning("梯度提升集成策略未返回有效的信号DataFrame")
    except Exception as e:
        logger.error(f"生成梯度提升集成策略信号时出错: {str(e)}")
        traceback.print_exc()
    
    try:
        logger.info("生成神经网络集成策略信号...")
        # 如果模型加载失败，需要预先训练
        if need_train_nn:
            train_size = int(len(df) * 0.8)
            logger.info(f"开始训练神经网络模型，使用前{train_size}行数据...")
            nn_train_features = nn_ensemble.prepare_features(df.iloc[:train_size], df.iloc[:train_size].index[-1], is_training=True)
            
            # 检查特征
            logger.info(f"神经网络训练特征形状: {nn_train_features.shape}")
            logger.info(f"特征列: {nn_train_features.columns.tolist()}")
            
            # 只使用数值列
            numeric_cols = nn_train_features.select_dtypes(include=['number']).columns
            nn_train_features = nn_train_features[numeric_cols]
            
            # 训练模型
            if 'future_returns' in nn_train_features.columns:
                nn_target = nn_train_features['future_returns']
                nn_ensemble.train(nn_train_features, nn_target)
            else:
                logger.error("找不到目标列'future_returns'，无法训练神经网络模型")
        
        nn_df = nn_ensemble.generate_signals(df)
        if nn_df is not None and isinstance(nn_df, pd.DataFrame) and 'position' in nn_df.columns:
            valid_strategies['NeuralNetwork'] = nn_df
        else:
            logger.warning("神经网络集成策略未返回有效的信号DataFrame")
    except Exception as e:
        logger.error(f"生成神经网络集成策略信号时出错: {str(e)}")
        traceback.print_exc()
    
    # 检查是否有有效策略
    if not valid_strategies:
        logger.error("没有策略返回有效的信号DataFrame，无法进行比较")
        return {}, {}
    
    logger.info(f"有效策略总数: {len(valid_strategies)}")
    
    # 计算策略收益
    returns = {}
    initial_capital = 10000
    
    for name, strategy_df in valid_strategies.items():
        try:
            strategy_df['strategy_returns'] = strategy_df['position'].shift(1) * strategy_df['close'].pct_change()
            strategy_df['cumulative_returns'] = (1 + strategy_df['strategy_returns']).cumprod().fillna(1)
            strategy_df['equity_curve'] = initial_capital * strategy_df['cumulative_returns']
            
            # 计算性能指标
            returns[name] = {
                'final_return': strategy_df['cumulative_returns'].iloc[-1] - 1,
                'annualized_return': (strategy_df['cumulative_returns'].iloc[-1] ** (252 / len(strategy_df)) - 1),
                'sharpe_ratio': strategy_df['strategy_returns'].mean() / (strategy_df['strategy_returns'].std() + 1e-6) * np.sqrt(252),
                'max_drawdown': (strategy_df['cumulative_returns'] / strategy_df['cumulative_returns'].cummax() - 1).min(),
                'win_rate': (strategy_df['strategy_returns'] > 0).mean(),
                'trade_count': strategy_df['position'].diff().abs().sum() / 2
            }
            
            logger.info(f"\n{name}策略性能指标:")
            for metric, value in returns[name].items():
                logger.info(f"  {metric}: {value:.4f}")
        except Exception as e:
            logger.error(f"计算{name}策略性能指标时出错: {str(e)}")
            # 从有效策略中移除
            valid_strategies.pop(name, None)
            returns.pop(name, None)
    
    if not valid_strategies:
        logger.error("所有策略在计算性能指标时出错，无法生成绘图")
        return {}, {}
    
    # 绘制权益曲线
    try:
        plt.figure(figsize=(14, 7))
        for name, strategy_df in valid_strategies.items():
            if 'equity_curve' in strategy_df.columns:
                plt.plot(strategy_df.index, strategy_df['equity_curve'], label=name)
        
        plt.title('策略权益曲线比较')
        plt.xlabel('日期')
        plt.ylabel('权益')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(strategy_dir, 'equity_curves.png'))
    except Exception as e:
        logger.error(f"绘制权益曲线时出错: {str(e)}")
    
    # 创建性能对比表格
    try:
        if returns:
            performance_df = pd.DataFrame(returns).T
            performance_df.to_csv(os.path.join(strategy_dir, 'strategy_performance.csv'))
            
            # 显示性能图表
            metrics = ['final_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            for metric in metrics:
                plt.figure(figsize=(10, 6))
                performance_df[metric].plot(kind='bar')
                plt.title(f'策略{metric}对比')
                plt.tight_layout()
                plt.savefig(os.path.join(strategy_dir, f'{metric}_comparison.png'))
    except Exception as e:
        logger.error(f"创建性能对比表格时出错: {str(e)}")
    
    # 为高级集成策略创建特征重要性报告
    for ensemble_name, ensemble_model in [('GradientBoosting', gb_ensemble), ('NeuralNetwork', nn_ensemble)]:
        if ensemble_name in valid_strategies and hasattr(ensemble_model, '_is_trained') and ensemble_model._is_trained:
            ensemble_df = valid_strategies[ensemble_name]
            report_dir = os.path.join(strategy_dir, f'{ensemble_name.lower()}_ensemble')
            try:
                report = ensemble_model.create_ensemble_report(ensemble_df, report_dir)
                
                # 打印集成策略在不同市场状态下的表现
                if 'regime_performance' in report:
                    logger.info(f"\n{ensemble_name}策略在不同市场状态下的表现:")
                    regime_perf = report['regime_performance']
                    for regime in regime_perf.get('mean', {}).keys():
                        mean_return = regime_perf['mean'].get(regime, 0)
                        sharpe = regime_perf['sharpe'].get(regime, 0)
                        count = regime_perf['count'].get(regime, 0)
                        logger.info(f"  {regime}: 平均收益={mean_return:.6f}, 夏普比率={sharpe:.4f}, 样本数={count}")
            except Exception as e:
                logger.warning(f"创建{ensemble_name}集成报告时发生错误: {str(e)}")
    
    # 分析不同市场状态下各策略的表现
    market_performance = {}
    
    for name, strategy_df in valid_strategies.items():
        # 确保有市场状态列
        if 'market_regime' not in strategy_df.columns and 'market_regime' in df.columns:
            strategy_df['market_regime'] = df['market_regime']
        
        if 'market_regime' in strategy_df.columns and 'strategy_returns' in strategy_df.columns:
            # 按市场状态分组计算性能
            try:
                grouped = strategy_df.groupby('market_regime')['strategy_returns']
                
                regime_stats = grouped.agg(['mean', 'std', 'count'])
                regime_stats['sharpe'] = regime_stats['mean'] / (regime_stats['std'] + 1e-6) * np.sqrt(252)
                
                market_performance[name] = regime_stats
            except Exception as e:
                logger.warning(f"计算{name}策略在不同市场状态下的表现时出错: {str(e)}")
    
    # 创建市场状态性能对比报告
    if market_performance:
        logger.info("\n不同市场状态下各策略的表现:")
        for name, perf in market_performance.items():
            logger.info(f"\n{name}策略:")
            logger.info(perf)
            perf.to_csv(os.path.join(strategy_dir, f'{name}_regime_performance.csv'))
        
        # 绘制不同市场状态下的夏普比率对比
        regimes = list(set([regime for perf in market_performance.values() for regime in perf.index]))
        
        for regime in regimes:
            try:
                plt.figure(figsize=(10, 6))
                regime_sharpe = {name: perf.loc[regime, 'sharpe'] if regime in perf.index else 0 
                                for name, perf in market_performance.items()}
                
                pd.Series(regime_sharpe).plot(kind='bar')
                plt.title(f'{regime}市场状态下各策略夏普比率')
                plt.tight_layout()
                plt.savefig(os.path.join(strategy_dir, f'regime_{regime}_sharpe.png'))
            except Exception as e:
                logger.warning(f"绘制{regime}市场状态下的夏普比率对比图时出错: {str(e)}")
    
    return valid_strategies, returns


def main():
    """主函数"""
    try:
        # 获取数据 - 为了快速测试使用较小的数据集
        df = fetch_data(symbol="BTCUSDT", interval="1h", use_small_dataset=True)
        
        # 分析市场状态
        df = analyze_market_regimes(df)
        
        # 比较不同策略
        strategies, performance = compare_strategies(df)
        
        logger.info("示例运行完成，结果已保存至: " + OUTPUT_DIR)
    except Exception as e:
        logger.error(f"运行示例时发生错误: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 