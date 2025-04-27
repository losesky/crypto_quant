#!/usr/bin/env python
"""
字体帮助模块 - 用于解决matplotlib中文字体显示问题
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from pathlib import Path
import warnings
import logging
import matplotlib.font_manager as fm
import contextlib
import io
import numpy as np

from .logger import logger

logger = logging.getLogger(__name__)

class FontHelper:
    """
    帮助处理中文字体的类，用于matplotlib可视化
    """
    def __init__(self):
        """初始化字体助手"""
        self.chinese_font = None
        self.font_file = None
        self.has_chinese_font = False
        # 首先禁用字体警告
        self._suppress_font_warnings()
        # 然后初始化字体支持
        self.init_font_support()
    
    def _suppress_font_warnings(self):
        """禁用matplotlib字体相关警告"""
        # 设置matplotlib的日志级别为ERROR，以抑制font相关的warning
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        # 使用warnings模块过滤字体相关警告
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        warnings.filterwarnings("ignore", message="findfont.*")
        warnings.filterwarnings("ignore", message="Glyph.*missing from current font")
        warnings.filterwarnings("ignore", message=".*Generic family.*not found because.*")
        warnings.filterwarnings("ignore", message=".*PL UMing CN.*")
        warnings.filterwarnings("ignore", message=".*family.*not found.*")
        
        # 拦截sys.stderr的输出以阻止字体相关警告
        class FontWarningFilter:
            def __init__(self, original_stderr):
                self.original_stderr = original_stderr
            
            def write(self, message):
                # 过滤掉字体相关的警告信息
                if ("findfont" not in message and 
                    "font" not in message.lower() and 
                    "cannot find font" not in message.lower() and
                    "family" not in message.lower() and
                    "not found" not in message.lower()):
                    self.original_stderr.write(message)
            
            def flush(self):
                self.original_stderr.flush()
        
        # 应用过滤器到sys.stderr
        # 只在非交互式环境中应用该过滤器，因为在交互式环境中可能会影响调试
        if not hasattr(sys, 'ps1'):  # 检查是否在交互式环境
            sys.stderr = FontWarningFilter(sys.stderr)
        
        # 设置matplotlib后端（如果不需要交互式图形界面）
        try:
            mpl.use('Agg')  # 使用非交互式后端
        except Exception as e:
            logger.warning(f"设置matplotlib后端失败: {e}")
        
        # 设置matplotlib日志级别为ERROR，屏蔽INFO及以下级别的日志
        mpl.set_loglevel("error")
        
        # 重定向matplotlib内部的字体搜索日志
        original_findfont = fm.findfont
        
        def silent_findfont(*args, **kwargs):
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    return original_findfont(*args, **kwargs)
        
        fm.findfont = silent_findfont
        
        # 设置numpy错误处理方式，抑制相关警告
        np.seterr(all='ignore')
        
        logger.info("已禁用matplotlib字体警告")
    
    def init_font_support(self):
        """初始化字体支持"""
        # 抑制字体警告
        self._suppress_font_warnings()
        
        # 寻找中文字体
        self.chinese_font = self.find_chinese_font_file()
        
        # 如果找到了中文字体，设置has_chinese_font为True
        self.has_chinese_font = (self.chinese_font is not None)
        
        if self.has_chinese_font:
            logger.info(f"找到中文字体: {self.chinese_font}")
            try:
                # 给matplotlib设置中文字体
                plt.rcParams['font.family'] = ['sans-serif']
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                
                # 测试字体
                _, ax = plt.subplots(figsize=(2, 2))
                ax.set_title("测试")
                ax.set_xlabel("测试")
                ax.set_ylabel("测试")
                self.apply_font_to_figure()
                plt.close()
                
                logger.info("中文字体设置成功")
            except Exception as e:
                logger.warning(f"设置中文字体时出错: {str(e)}")
                self.has_chinese_font = False
        else:
            logger.warning("未找到中文字体，将使用默认字体")
    
    def find_chinese_font_file(self):
        """寻找可用的中文字体文件"""
        # 常见的中文字体名称
        chinese_fonts = [
            'SimHei', 'Microsoft YaHei', 'STSong', 'STKaiti', 'STFangsong', 'SimSun', 'SimKai',
            'FangSong', 'KaiTi', 'Heiti SC', 'STHeiti', 'STHeiti Light', 'STXihei', 'STZhongsong',
            'STFangsong', 'STSong', 'STCaiyun', 'STHupo', 'STLiti', 'STXingkai', 'STXinwei',
            'WenQuanYi Micro Hei', 'Source Han Sans CN', 'Source Han Serif CN', 'Noto Sans CJK SC', 
            'Noto Serif CJK SC', 'Hiragino Sans GB', 'AR PL UMing CN', 'AR PL UKai CN',
            'DFKai-SB', 'Arial Unicode MS', 'Droid Sans Fallback'
        ]
        
        # 查找字体
        for font_name in chinese_fonts:
            try:
                font_path = fm.findfont(font_name, fallback_to_default=False)
                if font_path and font_path != fm.findfont('sans-serif'):
                    logger.info(f"找到中文字体: {font_name} 在 {font_path}")
                    return font_path
            except Exception as e:
                pass
        
        # 查找系统已安装的所有字体
        all_fonts = fm.findSystemFonts(fontpaths=None)
        for font_path in all_fonts:
            try:
                font_name = fm.FontProperties(fname=font_path).get_name()
                if any(chinese_name in font_name for chinese_name in 
                        ['Hei', 'Kai', 'Song', 'Yuan', 'Ming', 'Fang', 'Yi']):
                    logger.info(f"找到可能的中文字体: {font_name} 在 {font_path}")
                    return font_path
            except Exception as e:
                pass
        
        return None
    
    def apply_font_to_figure(self, fig=None):
        """将中文字体应用到整个图表
        
        Args:
            fig: 可选的matplotlib图形对象，如果不提供则使用当前活动的图形
        """
        # 使用提供的图形或当前活动图形
        if fig is None:
            fig = plt.gcf()
        
        # 如果没有找到中文字体，尝试设置全局中文支持
        if not self.has_chinese_font:
            try:
                # 尝试使用matplotlib内置的中文支持
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 
                                                  'Bitstream Vera Sans', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                logger.info("尝试使用matplotlib内置的中文支持")
                
                # 验证是否设置成功
                test_fig, test_ax = plt.subplots(figsize=(2, 2))
                test_ax.set_title("测试")
                plt.close(test_fig)
                
                # 如果能执行到这里，说明没有抛出异常，可以标记为成功
                self.has_chinese_font = True
                logger.info("成功应用内置中文字体支持")
            except Exception as e:
                logger.warning(f"应用内置中文字体支持失败: {str(e)}")
                
                # 尝试使用通用字体
                try:
                    # 查找系统中的通用字体
                    font_path = fm.findfont('sans-serif')
                    if font_path:
                        self.chinese_font = font_path
                        self.has_chinese_font = True
                        logger.info(f"使用通用字体: {font_path}")
                except Exception as e:
                    logger.warning(f"无法找到通用字体: {str(e)}")
        
        # 如果有中文字体，应用字体
        if self.has_chinese_font and self.chinese_font:
            try:
                for ax in fig.get_axes():
                    # 设置标题和标签的字体
                    if ax.get_title():
                        ax.set_title(ax.get_title(), fontproperties=FontProperties(fname=self.chinese_font))
                    
                    # X轴标签
                    if ax.get_xlabel():
                        ax.set_xlabel(ax.get_xlabel(), fontproperties=FontProperties(fname=self.chinese_font))
                    
                    # Y轴标签
                    if ax.get_ylabel():
                        ax.set_ylabel(ax.get_ylabel(), fontproperties=FontProperties(fname=self.chinese_font))
                    
                    # 图例
                    legend = ax.get_legend()
                    if legend:
                        for text in legend.get_texts():
                            text.set_fontproperties(FontProperties(fname=self.chinese_font))
                            
                # 设置主标题字体
                if fig._suptitle:
                    fig._suptitle.set_fontproperties(FontProperties(fname=self.chinese_font))
            except Exception as e:
                logger.warning(f"应用字体到图表失败: {str(e)}")
        else:
            # 如果没有中文字体，使用英文替代
            logger.warning("没有找到可用的中文字体，将使用英文标题和标签")
            for ax in fig.get_axes():
                # 将中文标题替换为英文标题（如果需要）
                current_title = ax.get_title()
                if current_title:
                    # 检测是否包含中文字符并替换为英文
                    if any('\u4e00' <= ch <= '\u9fff' for ch in current_title):
                        # 这里可以添加中英文映射关系
                        mappings = {
                            # 基础图表标题
                            '策略回测结果': 'Strategy Backtest Results',
                            '价格': 'Price',
                            '日期': 'Date',
                            '不同策略组合方法的性能比较': 'Performance Comparison of Different Ensemble Methods',
                            '账户价值': 'Equity',
                            '混合策略': 'Hybrid Strategy',
                            '比特币市场状态分类': 'Bitcoin Market Regime Classification',
                            
                            # 回测图表相关
                            '价格与交易信号': 'Price and Trading Signals',
                            '权益曲线': 'Equity Curve',
                            '回撤': 'Drawdown',
                            '仓位变化': 'Position Changes',
                            '买入': 'Buy',
                            '卖出': 'Sell',
                            '信号': 'Signals',
                            '持仓': 'Position',
                            '策略': 'Strategy',
                            '基准': 'Benchmark',
                            '回撤 %': 'Drawdown %',
                            '仓位': 'Position',
                            
                            # 市场状态相关
                            '强上升趋势': 'Strong Uptrend',
                            '强下降趋势': 'Strong Downtrend',
                            '高波动区间': 'Volatile Range',
                            '低波动区间': 'Tight Range',
                            '市场状态': 'Market State',
                            '市场状态分布': 'Market State Distribution',
                            
                            # LSTM模型相关
                            '模型训练曲线': 'Model Training Curve',
                            '损失函数': 'Loss Function',
                            '训练集': 'Training Set',
                            '验证集': 'Validation Set',
                            '测试集': 'Test Set',
                            '预测结果': 'Prediction Results',
                            '实际值': 'Actual Values',
                            '预测值': 'Predicted Values',
                            '注意力权重': 'Attention Weights',
                            '特征重要性': 'Feature Importance',
                            
                            # 风险管理相关
                            '风险管理性能': 'Risk Management Performance',
                            '无风控': 'No Risk Control',
                            '固定止损': 'Fixed Stop Loss',
                            '跟踪止损': 'Trailing Stop',
                            '动态风险': 'Dynamic Risk',
                            '最大回撤': 'Maximum Drawdown',
                            '夏普比率': 'Sharpe Ratio',
                            '卡玛比率': 'Calmar Ratio',
                            
                            # 参数优化相关
                            '参数优化结果': 'Parameter Optimization Results',
                            '优化表面': 'Optimization Surface',
                            '参数分布': 'Parameter Distribution',
                            '最优参数': 'Optimal Parameters',
                            
                            # 特征工程相关
                            '特征相关性': 'Feature Correlation',
                            '特征分布': 'Feature Distribution',
                            '主成分分析': 'Principal Component Analysis',
                            '累积解释方差': 'Cumulative Explained Variance',
                            
                            # 数据处理相关
                            '原始数据': 'Raw Data',
                            '处理后数据': 'Processed Data',
                            '缺失值填充': 'Missing Value Imputation',
                            '异常值检测': 'Outlier Detection',
                            '数据分布': 'Data Distribution'
                        }
                        
                        if current_title in mappings:
                            ax.set_title(mappings[current_title])
                        else:
                            # 对于未定义映射的中文标题，尝试移除中文字符
                            logger.warning(f"未找到标题 '{current_title}' 的映射，尝试替换为英文...")
                            # 保留英文部分和数字、标点符号
                            english_title = ''.join([c for c in current_title if not '\u4e00' <= c <= '\u9fff'])
                            if english_title.strip():
                                ax.set_title(english_title.strip())
                            else:
                                ax.set_title("Chart")  # 默认标题
                
                # 同样处理X轴和Y轴标签
                current_xlabel = ax.get_xlabel()
                if current_xlabel and any('\u4e00' <= ch <= '\u9fff' for ch in current_xlabel):
                    if current_xlabel in mappings:
                        ax.set_xlabel(mappings[current_xlabel])
                        
                current_ylabel = ax.get_ylabel()
                if current_ylabel and any('\u4e00' <= ch <= '\u9fff' for ch in current_ylabel):
                    if current_ylabel in mappings:
                        ax.set_ylabel(mappings[current_ylabel])
                        
            # 处理图表主标题
            if fig._suptitle and any('\u4e00' <= ch <= '\u9fff' for ch in fig._suptitle.get_text()):
                suptitle_text = fig._suptitle.get_text()
                if suptitle_text in mappings:
                    fig.suptitle(mappings[suptitle_text])
                else:
                    # 对于未映射的中文标题，尝试移除中文字符
                    english_title = ''.join([c for c in suptitle_text if not '\u4e00' <= c <= '\u9fff'])
                    if english_title.strip():
                        fig.suptitle(english_title.strip())
                    else:
                        # 针对特殊格式的中文标题，如"混合策略 - VOTE"
                        if "-" in suptitle_text:
                            parts = suptitle_text.split("-")
                            if len(parts) >= 2:
                                chinese_part = parts[0].strip()
                                english_part = ''.join([c for c in parts[1] if not '\u4e00' <= c <= '\u9fff']).strip()
                                
                                if chinese_part in mappings:
                                    fig.suptitle(f"{mappings[chinese_part]} - {english_part}")
                                elif english_part:
                                    fig.suptitle(english_part)
                                else:
                                    fig.suptitle("Strategy Analysis")
                        else:
                            fig.suptitle("Strategy Analysis")
            
            # 处理子图的xlabel和ylabel共有的文本
            # 许多子图可能共享同一个标签，如"日期"、"价格"等
            common_xlabels = {}
            common_ylabels = {}
            
            # 收集所有子图的标签
            for ax in fig.get_axes():
                xlabel = ax.get_xlabel()
                if xlabel and any('\u4e00' <= ch <= '\u9fff' for ch in xlabel):
                    if xlabel in mappings:
                        common_xlabels[xlabel] = mappings[xlabel]
                    else:
                        english_label = ''.join([c for c in xlabel if not '\u4e00' <= c <= '\u9fff'])
                        if english_label.strip():
                            common_xlabels[xlabel] = english_label.strip()
                        else:
                            common_xlabels[xlabel] = "Value"
                
                ylabel = ax.get_ylabel()
                if ylabel and any('\u4e00' <= ch <= '\u9fff' for ch in ylabel):
                    if ylabel in mappings:
                        common_ylabels[ylabel] = mappings[ylabel]
                    else:
                        english_label = ''.join([c for c in ylabel if not '\u4e00' <= c <= '\u9fff'])
                        if english_label.strip():
                            common_ylabels[ylabel] = english_label.strip()
                        else:
                            common_ylabels[ylabel] = "Value"
            
            # 应用收集到的标签翻译
            for ax in fig.get_axes():
                xlabel = ax.get_xlabel()
                if xlabel and any('\u4e00' <= ch <= '\u9fff' for ch in xlabel):
                    if xlabel in common_xlabels:
                        ax.set_xlabel(common_xlabels[xlabel])
                
                ylabel = ax.get_ylabel()
                if ylabel and any('\u4e00' <= ch <= '\u9fff' for ch in ylabel):
                    if ylabel in common_ylabels:
                        ax.set_ylabel(common_ylabels[ylabel])
                        
            # 处理图例中的中文
            for ax in fig.get_axes():
                legend = ax.get_legend()
                if legend:
                    for text in legend.get_texts():
                        text_str = text.get_text()
                        if any('\u4e00' <= ch <= '\u9fff' for ch in text_str):
                            if text_str in mappings:
                                text.set_text(mappings[text_str])
                            else:
                                # 对于未映射的中文文本，尝试移除中文字符
                                english_text = ''.join([c for c in text_str if not '\u4e00' <= c <= '\u9fff'])
                                if english_text.strip():
                                    text.set_text(english_text.strip())
                                else:
                                    # 对于完全是中文的文本，尝试一些常见图例项的翻译
                                    if "策略" in text_str:
                                        text.set_text("Strategy")
                                    elif "基准" in text_str or "买入持有" in text_str:
                                        text.set_text("Benchmark")
                                    elif "买入" in text_str:
                                        text.set_text("Buy")
                                    elif "卖出" in text_str:
                                        text.set_text("Sell")
                                    elif "信号" in text_str:
                                        text.set_text("Signal")
                                    else:
                                        text.set_text("Value")
    
    def set_chinese_title(self, ax, title):
        """设置带中文字体的标题"""
        if self.has_chinese_font:
            ax.set_title(title, fontproperties=FontProperties(fname=self.chinese_font))
        else:
            ax.set_title(title)
    
    def set_chinese_label(self, ax, xlabel=None, ylabel=None):
        """设置带中文字体的坐标轴标签"""
        if self.has_chinese_font:
            if xlabel:
                ax.set_xlabel(xlabel, fontproperties=FontProperties(fname=self.chinese_font))
            if ylabel:
                ax.set_ylabel(ylabel, fontproperties=FontProperties(fname=self.chinese_font))
        else:
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
    
    def set_chinese_legend(self, ax):
        """设置带中文字体的图例"""
        if self.has_chinese_font:
            ax.legend(prop=FontProperties(fname=self.chinese_font))
        else:
            ax.legend()
    
    def get_label(self, zh_label, en_label):
        """
        根据是否有中文字体支持，选择相应的标签文本
        
        Args:
            zh_label: 中文标签
            en_label: 英文标签
            
        Returns:
            str: 根据系统字体支持返回适当的标签
        """
        if self.has_chinese_font:
            return zh_label
        return en_label

# 实例化一个全局字体助手对象
font_helper = FontHelper()

def get_font_helper():
    """获取字体助手实例"""
    return font_helper

def install_chinese_fonts():
    """
    尝试在系统中安装中文字体。在WSL或Linux环境中运行。
    如果需要管理员权限，会请求用户输入密码。
    
    Returns:
        bool: 安装是否成功
    """
    import subprocess
    import platform
    import os
    
    # 检测操作系统
    system = platform.system().lower()
    
    if system == 'linux':
        try:
            # 检查是否是WSL环境
            with open('/proc/version', 'r') as f:
                is_wsl = 'microsoft' in f.read().lower()
        except:
            is_wsl = False
            
        font_packages = [
            'fonts-wqy-microhei',
            'fonts-wqy-zenhei',
            'xfonts-wqy',
            'fonts-noto-cjk'
        ]
        
        try:
            # 先更新包列表
            update_cmd = ['sudo', 'apt', 'update']
            update_process = subprocess.run(update_cmd, capture_output=True, text=True)
            if update_process.returncode != 0:
                logger.error(f"更新软件源失败: {update_process.stderr}")
                return False
                
            # 安装字体包
            install_cmd = ['sudo', 'apt', 'install', '-y'] + font_packages
            install_process = subprocess.run(install_cmd, capture_output=True, text=True)
            
            if install_process.returncode != 0:
                logger.error(f"安装字体包失败: {install_process.stderr}")
                return False
                
            # 刷新字体缓存
            fc_cache_cmd = ['sudo', 'fc-cache', '-f', '-v']
            fc_process = subprocess.run(fc_cache_cmd, capture_output=True, text=True)
            
            if fc_process.returncode != 0:
                logger.error(f"刷新字体缓存失败: {fc_process.stderr}")
                return False
                
            logger.info("成功安装中文字体并刷新缓存")
            
            # 重新初始化字体助手
            font_helper.init_font_support()
            
            return font_helper.has_chinese_font
        
        except Exception as e:
            logger.error(f"安装中文字体时出错: {str(e)}")
            return False
    
    elif system == 'windows':
        logger.info("Windows系统请手动安装中文字体，如微软雅黑、宋体等")
        return False
        
    elif system == 'darwin':  # macOS
        try:
            # macOS通常预装中文字体，但可以尝试安装额外的字体
            homebrew_cmd = ['brew', 'install', '--cask', 'font-wqy-microhei', 'font-wqy-zenhei']
            
            # 检查是否安装了Homebrew
            which_brew = subprocess.run(['which', 'brew'], capture_output=True, text=True)
            if which_brew.returncode != 0:
                logger.warning("在macOS上未找到Homebrew，请手动安装中文字体")
                return False
                
            brew_process = subprocess.run(homebrew_cmd, capture_output=True, text=True)
            if brew_process.returncode != 0:
                logger.warning(f"安装中文字体失败: {brew_process.stderr}")
                return False
                
            # 刷新字体缓存
            fc_cache_cmd = ['fc-cache', '-f', '-v']
            subprocess.run(fc_cache_cmd, capture_output=True, text=True)
            
            # 重新初始化字体助手
            font_helper.init_font_support()
            
            return font_helper.has_chinese_font
            
        except Exception as e:
            logger.error(f"安装中文字体时出错: {str(e)}")
            return False
    
    else:
        logger.warning(f"不支持的操作系统: {system}")
        return False

def generate_install_fonts_cmd():
    """
    生成一个用于安装中文字体的命令行脚本
    
    Returns:
        str: 包含安装命令的shell脚本内容
    """
    script = """#!/bin/bash
# 安装中文字体的自动化脚本
# 适用于Ubuntu/Debian系统和WSL

set -e

echo "开始安装中文字体包..."

# 检查是否有sudo权限
if ! command -v sudo &> /dev/null; then
    echo "错误: 此脚本需要sudo权限才能安装字体包"
    exit 1
fi

# 更新包索引
echo "更新软件包索引..."
sudo apt update

# 安装常用中文字体包
echo "安装中文字体包..."
sudo apt install -y fonts-wqy-microhei fonts-wqy-zenhei xfonts-wqy fonts-noto-cjk

# 刷新字体缓存
echo "刷新字体缓存..."
sudo fc-cache -f -v

# 检查字体是否成功安装
echo "检查字体是否成功安装..."
fc-list :lang=zh

echo "中文字体安装完成！"
echo "请重启您的Python应用程序，使字体更改生效。"
"""
    return script

def create_font_install_script(path="install_chinese_fonts.sh"):
    """
    创建安装中文字体的脚本文件
    
    Args:
        path (str): 脚本文件保存路径
        
    Returns:
        str: 脚本文件的完整路径
    """
    import os
    
    script_content = generate_install_fonts_cmd()
    
    # 获取完整路径
    if not os.path.isabs(path):
        # 如果不是绝对路径，使用当前工作目录
        cwd = os.getcwd()
        full_path = os.path.join(cwd, path)
    else:
        full_path = path
    
    # 写入脚本文件
    try:
        with open(full_path, 'w') as f:
            f.write(script_content)
        
        # 添加执行权限
        os.chmod(full_path, 0o755)
        
        logger.info(f"中文字体安装脚本已创建: {full_path}")
        logger.info(f"请在终端中运行: bash {full_path}")
        
        return full_path
    except Exception as e:
        logger.error(f"创建字体安装脚本时出错: {str(e)}")
        return None 