"""
图像生成调试工具模块

提供专门用于调试图像生成过程的工具函数
"""
import os
import sys
import time
import traceback
from datetime import datetime
from loguru import logger as main_logger

# 强制matplotlib使用Agg后端，确保在无GUI环境中也能正确保存图片
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 创建图像调试专用logger
image_logger = main_logger.bind(name="image_debug")

def setup_image_debug_logger(log_dir=None):
    """
    设置图像调试专用日志记录器
    
    Args:
        log_dir (str, optional): 日志目录。默认为None，会在当前脚本所在目录创建logs子目录。
    """
    # 确保日志目录存在
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_log_file = os.path.join(log_dir, f"image_debug_{timestamp}.log")
    
    # 移除所有已有的处理程序
    image_logger.remove()
    
    # 添加文件处理程序，只记录图像生成相关的日志
    image_logger.add(
        image_log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {function}:{line} | {message}",
        level="DEBUG",
        backtrace=True,
        diagnose=True,
        enqueue=True  # 使用队列，避免日志记录阻塞主线程
    )
    
    # 添加一个标准错误处理程序，方便直接查看
    image_logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | 图像调试 | {level} | {function}:{line} | {message}",
        level="DEBUG",
        filter=lambda record: record["name"] == "image_debug"
    )
    
    image_logger.info(f"图像调试日志已初始化，日志文件: {image_log_file}")
    return image_logger

def trace_image_save(fig, filepath, dpi=100):
    """
    跟踪图像保存过程，处理异常并记录详细信息
    
    Args:
        fig: matplotlib图形对象
        filepath (str): 保存路径
        dpi (int): 图像DPI
        
    Returns:
        bool: 保存是否成功
    """
    image_logger.info(f"开始保存图像到: {filepath}")
    
    # 记录当前工作目录
    image_logger.debug(f"当前工作目录: {os.getcwd()}")
    
    # 检查目录是否存在，不存在则创建
    save_dir = os.path.dirname(os.path.abspath(filepath))
    image_logger.debug(f"图像保存目录: {save_dir}")
    
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
            image_logger.debug(f"创建保存目录: {save_dir}")
        except Exception as e:
            image_logger.error(f"创建目录失败: {str(e)}")
            image_logger.error(traceback.format_exc())
            return False
    
    # 检查目录权限
    if not os.access(save_dir, os.W_OK):
        image_logger.error(f"目录没有写入权限: {save_dir}")
        try:
            os.chmod(save_dir, 0o755)
            image_logger.debug(f"尝试修改目录权限: {save_dir}")
            if not os.access(save_dir, os.W_OK):
                image_logger.error(f"修改权限后仍然无法写入: {save_dir}")
                return False
        except Exception as e:
            image_logger.error(f"修改目录权限失败: {str(e)}")
            image_logger.error(traceback.format_exc())
            return False
    
    # 检查图形对象
    if fig is None:
        image_logger.error("图形对象为None，无法保存")
        return False
    
    # 检查保存格式
    file_ext = os.path.splitext(filepath)[1].lower()
    if not file_ext:
        image_logger.warning(f"文件没有扩展名，将自动添加.png: {filepath}")
        filepath = f"{filepath}.png"
    elif file_ext not in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']:
        image_logger.warning(f"不常见的图像格式: {file_ext}，可能不被支持")
    
    # 尝试保存图像
    try:
        start_time = time.time()
        fig.savefig(filepath, dpi=dpi)
        end_time = time.time()
        
        # 验证文件是否成功保存
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            image_logger.info(f"图像保存成功: {filepath}, 大小: {file_size} 字节, 耗时: {end_time-start_time:.3f}秒")
            return True
        else:
            image_logger.error(f"图像保存失败: 文件不存在: {filepath}")
            return False
    except Exception as e:
        image_logger.error(f"保存图像时异常: {str(e)}")
        image_logger.error(traceback.format_exc())
        
        # 尝试使用不同的后端再次保存
        try:
            image_logger.debug("尝试使用Agg后端重新保存")
            plt.switch_backend('Agg')
            fig.savefig(filepath, dpi=dpi)
            
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                image_logger.info(f"使用Agg后端成功保存图像: {filepath}, 大小: {file_size} 字节")
                return True
            else:
                image_logger.error("使用Agg后端保存仍然失败")
                return False
        except Exception as e2:
            image_logger.error(f"使用Agg后端保存时异常: {str(e2)}")
            image_logger.error(traceback.format_exc())
            return False

def debug_figure(fig, title="调试图形"):
    """
    记录图形对象的信息到日志中
    
    Args:
        fig: matplotlib图形对象
        title (str): 标识该图形的标题
    """
    if fig is None:
        image_logger.error(f"{title}: 图形对象为None")
        return

    # 记录基本信息
    image_logger.debug(f"{title}: 图形对象信息")
    image_logger.debug(f"  - 类型: {type(fig)}")
    image_logger.debug(f"  - 画布大小: {fig.get_size_inches()} 英寸")
    image_logger.debug(f"  - DPI: {fig.get_dpi()}")
    image_logger.debug(f"  - 后端: {plt.get_backend()}")
    
    # 记录轴信息
    axes_count = len(fig.get_axes())
    image_logger.debug(f"  - 轴数量: {axes_count}")
    
    # 记录图形内容
    for i, ax in enumerate(fig.get_axes()):
        image_logger.debug(f"  - 轴 #{i+1}:")
        image_logger.debug(f"    - 标题: {ax.get_title()}")
        image_logger.debug(f"    - x轴标签: {ax.get_xlabel()}")
        image_logger.debug(f"    - y轴标签: {ax.get_ylabel()}")
        image_logger.debug(f"    - 图例: {[text.get_text() for text in ax.get_legend().get_texts()] if ax.get_legend() else '无图例'}")
        image_logger.debug(f"    - 数据集数量: {len(ax.get_lines())}")

# 初始化图像调试日志
setup_image_debug_logger() 