#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
输出路径助手模块
提供统一管理图表和报告输出路径的功能
"""
import os
import datetime

# 基础目录
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
DEFAULT_IMAGES_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'images')
DEFAULT_REPORTS_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'reports')
DEFAULT_DATA_DIR = os.path.join(DEFAULT_OUTPUT_DIR, 'data')


def ensure_dir_exists(dir_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def get_image_path(filename, subdirectory=None):
    """
    获取图表文件的保存路径
    
    Args:
        filename (str): 文件名
        subdirectory (str, optional): 子目录名。默认为None，使用当前日期作为子目录。
    
    Returns:
        str: 完整的文件路径
    """
    if subdirectory is None:
        subdirectory = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # 确保有.png扩展名
    if not filename.endswith('.png'):
        filename = f"{filename}.png"
    
    img_dir = os.path.join(DEFAULT_IMAGES_DIR, subdirectory)
    ensure_dir_exists(img_dir)
    
    return os.path.join(img_dir, filename)


def get_report_path(filename, subdirectory=None):
    """
    获取报告文件的保存路径
    
    Args:
        filename (str): 文件名
        subdirectory (str, optional): 子目录名。默认为None，使用当前日期作为子目录。
    
    Returns:
        str: 完整的文件路径
    """
    if subdirectory is None:
        subdirectory = datetime.datetime.now().strftime('%Y-%m-%d')
    
    report_dir = os.path.join(DEFAULT_REPORTS_DIR, subdirectory)
    ensure_dir_exists(report_dir)
    
    return os.path.join(report_dir, filename)


def get_data_path(filename, subdirectory=None):
    """
    获取数据文件的保存路径
    
    Args:
        filename (str): 文件名
        subdirectory (str, optional): 子目录名。默认为None，使用当前日期作为子目录。
    
    Returns:
        str: 完整的文件路径
    """
    if subdirectory is None:
        subdirectory = datetime.datetime.now().strftime('%Y-%m-%d')
    
    data_dir = os.path.join(DEFAULT_DATA_DIR, subdirectory)
    ensure_dir_exists(data_dir)
    
    return os.path.join(data_dir, filename)


def clean_old_outputs(days=30):
    """
    清理超过指定天数的旧输出文件
    
    Args:
        days (int): 保留的天数，默认30天
    
    Returns:
        int: 清理的文件数量
    """
    import shutil
    from datetime import datetime, timedelta
    
    cutoff_date = datetime.now() - timedelta(days=days)
    count = 0
    
    # 清理图表目录
    for dir_name in os.listdir(DEFAULT_IMAGES_DIR):
        dir_path = os.path.join(DEFAULT_IMAGES_DIR, dir_name)
        try:
            if os.path.isdir(dir_path):
                dir_date = datetime.strptime(dir_name, '%Y-%m-%d')
                if dir_date < cutoff_date:
                    shutil.rmtree(dir_path)
                    count += 1
        except (ValueError, OSError):
            # 跳过非日期格式目录或无法删除的目录
            pass
    
    # 清理报告目录
    for dir_name in os.listdir(DEFAULT_REPORTS_DIR):
        dir_path = os.path.join(DEFAULT_REPORTS_DIR, dir_name)
        try:
            if os.path.isdir(dir_path):
                dir_date = datetime.strptime(dir_name, '%Y-%m-%d')
                if dir_date < cutoff_date:
                    shutil.rmtree(dir_path)
                    count += 1
        except (ValueError, OSError):
            # 跳过非日期格式目录或无法删除的目录
            pass
    
    # 清理数据目录
    for dir_name in os.listdir(DEFAULT_DATA_DIR):
        dir_path = os.path.join(DEFAULT_DATA_DIR, dir_name)
        try:
            if os.path.isdir(dir_path):
                dir_date = datetime.strptime(dir_name, '%Y-%m-%d')
                if dir_date < cutoff_date:
                    shutil.rmtree(dir_path)
                    count += 1
        except (ValueError, OSError):
            # 跳过非日期格式目录或无法删除的目录
            pass
    
    return count 