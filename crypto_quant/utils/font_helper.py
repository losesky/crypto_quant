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
        # 查找中文字体
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                self.chinese_font = self.find_chinese_font_file()
        
        if self.chinese_font:
            self.has_chinese_font = True
            # 设置matplotlib字体
            self.apply_font_to_figure()
            logger.info(f"已成功加载中文字体: {self.chinese_font}")
        else:
            self.has_chinese_font = False
            logger.warning("未找到中文字体，将使用系统默认字体，中文可能无法正确显示")
    
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
    
    def apply_font_to_figure(self):
        """将中文字体应用到整个图表"""
        if not self.has_chinese_font:
            return
        
        # 应用字体到所有子图
        for ax in plt.gcf().get_axes():
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