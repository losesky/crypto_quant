#!/usr/bin/env python
"""
参数优化器，用于优化策略参数和模型超参数
支持网格搜索和贝叶斯优化
"""
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Callable

try:
    # 尝试导入optuna进行贝叶斯优化
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from ..backtesting.engine import BacktestEngine
from ..utils.logger import logger
from ..utils.output_helper import get_image_path  # 导入get_image_path函数


class ParameterOptimizer:
    """
    参数优化器类，用于优化策略参数
    
    支持两种优化方式：
    1. 网格搜索：搜索所有可能的参数组合
    2. 贝叶斯优化：使用optuna库进行贝叶斯优化（如果可用）
    
    Attributes:
        strategy_class: 策略类
        param_grid: 参数网格，格式为 {参数名: 参数值列表}
        data: 用于回测的数据
        initial_capital: 初始资金
        commission: 手续费率
        metric: 优化目标指标，可选 'sharpe_ratio', 'calmar_ratio', 'annual_return', 'max_drawdown'(取负值)
        method: 优化方法，可选 'grid_search', 'bayesian'
        trials: 贝叶斯优化的试验次数
    """
    
    VALID_METRICS = ['sharpe_ratio', 'calmar_ratio', 'annual_return', 'max_drawdown']
    
    def __init__(
        self,
        strategy_class,
        param_grid: Dict[str, List],
        data: pd.DataFrame,
        initial_capital: float = 10000,
        commission: float = 0.001,
        metric: str = 'sharpe_ratio',
        method: str = 'grid_search',
        trials: int = 100
    ):
        """
        初始化参数优化器
        
        Args:
            strategy_class: 策略类
            param_grid: 参数网格，格式为 {参数名: 参数值列表}
            data: 用于回测的数据
            initial_capital: 初始资金
            commission: 手续费率
            metric: 优化目标指标，可选 'sharpe_ratio', 'calmar_ratio', 'annual_return', 'max_drawdown'(取负值)
            method: 优化方法，可选 'grid_search', 'bayesian'
            trials: 贝叶斯优化的试验次数
        """
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        
        # 验证指标
        if metric not in self.VALID_METRICS:
            raise ValueError(f"Invalid metric: {metric}. Valid metrics are: {self.VALID_METRICS}")
        self.metric = metric
        
        # 验证方法
        if method not in ['grid_search', 'bayesian']:
            raise ValueError("Method must be 'grid_search' or 'bayesian'")
        
        # 如果选择贝叶斯优化但optuna不可用，则回退到网格搜索
        if method == 'bayesian' and not OPTUNA_AVAILABLE:
            logger.warning("Optuna is not available, falling back to grid search")
            method = 'grid_search'
            
        self.method = method
        self.trials = trials
        
        # 结果存储
        self.results = None
        self.best_params = None
        self.best_score = None
        
    def _generate_param_combinations(self) -> List[Dict]:
        """
        生成所有参数组合
        
        Returns:
            参数组合列表
        """
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _evaluate_params(self, params: Dict) -> float:
        """
        评估一组参数的性能
        
        Args:
            params: 参数字典
            
        Returns:
            评估得分
        """
        # 创建策略实例
        strategy = self.strategy_class(**params)
        
        # 创建回测引擎
        engine = BacktestEngine(
            data=self.data.copy(),
            strategy=strategy,
            initial_capital=self.initial_capital,
            commission=self.commission
        )
        
        # 运行回测
        engine.run()
        
        # 获取性能指标
        performance = engine.performance
        
        # 根据优化目标返回得分
        if self.metric == 'max_drawdown':
            # 对最大回撤取负值，使其成为最大化目标
            return -performance.get('max_drawdown', float('inf'))
        else:
            return performance.get(self.metric, float('-inf'))
    
    def grid_search(self) -> Tuple[Dict, float, pd.DataFrame]:
        """
        执行网格搜索
        
        Returns:
            最佳参数，最佳得分，结果数据框
        """
        combinations = self._generate_param_combinations()
        total_combinations = len(combinations)
        
        logger.info(f"执行网格搜索，共有 {total_combinations} 种参数组合")
        
        if total_combinations > 1000:
            logger.warning(f"参数组合数量过多 ({total_combinations})，可能需要较长时间")
        
        results = []
        
        # 使用tqdm显示进度条
        for params in tqdm(combinations, desc="Parameter Optimization"):
            score = self._evaluate_params(params)
            
            # 记录结果
            result = params.copy()
            result[self.metric] = score
            results.append(result)
        
        # 转换为数据框
        results_df = pd.DataFrame(results)
        
        # 查找最佳参数组合
        if self.metric == 'max_drawdown':
            # 最大回撤是负值，越小越好
            best_idx = results_df[self.metric].idxmax()
        else:
            # 其他指标是越大越好
            best_idx = results_df[self.metric].idxmax()
        
        best_params = results_df.iloc[best_idx].drop(self.metric).to_dict()
        best_score = results_df.iloc[best_idx][self.metric]
        
        return best_params, best_score, results_df
    
    def bayesian_optimization(self) -> Tuple[Dict, float, pd.DataFrame]:
        """
        执行贝叶斯优化
        
        Returns:
            最佳参数，最佳得分，结果数据框
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization")
        
        logger.info(f"执行贝叶斯优化，目标指标: {self.metric}，试验次数: {self.trials}")
        
        # 定义目标函数
        def objective(trial):
            params = {}
            
            # 为每个参数创建合适的建议
            for param_name, param_values in self.param_grid.items():
                if isinstance(param_values, range):
                    # 整数参数
                    params[param_name] = trial.suggest_int(
                        param_name,
                        min(param_values),
                        max(param_values),
                        step=param_values.step if hasattr(param_values, 'step') else 1
                    )
                elif all(isinstance(x, int) for x in param_values):
                    # 整数列表
                    params[param_name] = trial.suggest_int(
                        param_name,
                        min(param_values),
                        max(param_values)
                    )
                elif all(isinstance(x, float) for x in param_values):
                    # 浮点数列表
                    params[param_name] = trial.suggest_float(
                        param_name,
                        min(param_values),
                        max(param_values)
                    )
                else:
                    # 分类参数
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_values
                    )
            
            # 评估参数
            return self._evaluate_params(params)
        
        # 创建研究对象
        if self.metric == 'max_drawdown':
            # 最大回撤是负值，越小越好
            direction = 'maximize'  # 因为我们对最大回撤取了负值
        else:
            # 其他指标是越大越好
            direction = 'maximize'
            
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=self.trials)
        
        # 获取最佳参数
        best_params = study.best_params
        best_score = study.best_value
        
        # 收集所有试验结果
        trials_df = pd.DataFrame([
            {**t.params, self.metric: t.value}
            for t in study.trials
        ])
        
        return best_params, best_score, trials_df
    
    def run(self) -> Dict:
        """
        运行参数优化
        
        Returns:
            最佳参数
        """
        logger.info(f"开始参数优化，方法: {self.method}，优化目标: {self.metric}")
        
        # 根据选择的方法执行优化
        if self.method == 'grid_search':
            self.best_params, self.best_score, self.results = self.grid_search()
        elif self.method == 'bayesian':
            self.best_params, self.best_score, self.results = self.bayesian_optimization()
        
        logger.info(f"参数优化完成，最佳参数: {self.best_params}")
        logger.info(f"最佳得分 ({self.metric}): {self.best_score}")
        
        return self.best_params
    
    def plot_results(self, top_n: int = 10) -> None:
        """
        绘制优化结果
        
        Args:
            top_n: 显示前N个最佳结果
        """
        if self.results is None:
            raise ValueError("No optimization results available. Run optimization first.")
        
        # 根据优化目标对结果排序
        if self.metric == 'max_drawdown':
            # 最大回撤是负值，越小越好
            sorted_results = self.results.sort_values(by=self.metric, ascending=False)
        else:
            # 其他指标是越大越好
            sorted_results = self.results.sort_values(by=self.metric, ascending=False)
        
        # 获取前N个结果
        top_results = sorted_results.head(top_n)
        
        # 创建一个图表网格
        n_params = len(self.param_grid)
        fig_width = 12
        fig_height = 4 * n_params
        
        fig, axes = plt.subplots(n_params, 1, figsize=(fig_width, fig_height))
        if n_params == 1:
            axes = [axes]  # 确保axes始终是一个列表
        
        # 为每个参数创建散点图
        for i, param_name in enumerate(self.param_grid.keys()):
            sns.scatterplot(
                data=self.results,
                x=param_name,
                y=self.metric,
                ax=axes[i]
            )
            
            # 标记最佳点
            best_value = self.best_params[param_name]
            best_score = self.best_score
            axes[i].plot(best_value, best_score, 'ro', markersize=10, label='Best')
            
            axes[i].set_title(f'Impact of {param_name} on {self.metric}')
            axes[i].grid(True)
            axes[i].legend()
        
        plt.tight_layout()
        # 使用output_helper保存图像
        output_path = get_image_path('parameter_optimization_results.png')
        plt.savefig(output_path)
        logger.info(f"参数优化结果图表已保存至: {output_path}")
        plt.show()
        
        # 绘制热力图（如果有两个或更多参数）
        if n_params >= 2:
            param_pairs = list(itertools.combinations(self.param_grid.keys(), 2))
            n_pairs = len(param_pairs)
            
            # 创建一个新的图表网格
            fig_height = 4 * ((n_pairs + 1) // 2)  # 每行两个图表
            fig, axes = plt.subplots(
                (n_pairs + 1) // 2, 2, 
                figsize=(fig_width, fig_height)
            )
            axes = axes.flatten()
            
            # 为每对参数创建热力图
            for i, (param1, param2) in enumerate(param_pairs):
                if i < len(axes):
                    # 创建透视表
                    pivot = self.results.pivot_table(
                        index=param1,
                        columns=param2,
                        values=self.metric,
                        aggfunc='mean'
                    )
                    
                    # 绘制热力图
                    sns.heatmap(
                        pivot,
                        annot=True,
                        cmap='viridis',
                        ax=axes[i],
                        fmt='.2f'
                    )
                    
                    axes[i].set_title(f'{param1} vs {param2} ({self.metric})')
            
            # 隐藏空白子图
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            # 使用output_helper保存图像
            heatmap_path = get_image_path('parameter_optimization_heatmap.png')
            plt.savefig(heatmap_path)
            logger.info(f"参数优化热力图已保存至: {heatmap_path}")
            plt.show()
            
        # 打印前N个最佳参数组合
        print("\n前{}个最佳参数组合:".format(top_n))
        for i, row in top_results.iterrows():
            params_str = ", ".join([f"{k}={v}" for k, v in row.drop(self.metric).items()])
            score_str = f"{self.metric}={row[self.metric]:.4f}"
            print(f"{i+1}. {params_str} -> {score_str}") 