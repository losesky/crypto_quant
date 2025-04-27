#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
字体问题修复工具

此脚本检查所有示例脚本和其他图表生成代码中的字体问题，并提供修复方案。
"""

import os
import sys
import logging
import glob
import re
import argparse
from typing import List, Dict, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入字体助手模块
from crypto_quant.utils.font_helper import (
    get_font_helper, 
    install_chinese_fonts, 
    create_font_install_script
)
from crypto_quant.utils.logger import logger

# 配置日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def find_python_files(directory: str, pattern: str = "*.py") -> List[str]:
    """
    在指定目录中查找符合模式的Python文件
    
    Args:
        directory: 要搜索的目录
        pattern: 文件匹配模式 (默认 "*.py")
        
    Returns:
        List[str]: 匹配文件的路径列表
    """
    return glob.glob(os.path.join(directory, pattern))

def analyze_file(file_path: str) -> Dict:
    """
    分析文件中潜在的字体问题
    
    Args:
        file_path: 要分析的文件路径
        
    Returns:
        Dict: 分析结果
    """
    result = {
        'file': file_path,
        'imports_matplotlib': False,
        'uses_chinese_characters': False,
        'uses_font_helper': False,
        'direct_plt_calls': 0,
        'issues': [],
        'fixes_needed': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否导入了matplotlib
        result['imports_matplotlib'] = bool(re.search(r'import\s+matplotlib|from\s+matplotlib', content))
        
        # 检查是否使用了中文字符
        result['uses_chinese_characters'] = bool(re.search(r'[\u4e00-\u9fff]', content))
        
        # 检查是否使用了字体助手
        result['uses_font_helper'] = 'font_helper' in content
        
        # 查找直接调用plt方法的地方
        plt_calls = re.findall(r'plt\.(title|xlabel|ylabel|suptitle|text|legend|figtext)\s*\(\s*[\'"][\u4e00-\u9fff]+', content)
        result['direct_plt_calls'] = len(plt_calls)
        
        # 检查潜在问题
        if result['imports_matplotlib'] and result['uses_chinese_characters']:
            if not result['uses_font_helper']:
                result['issues'].append("使用了中文字符但未导入字体助手模块")
                result['fixes_needed'].append("添加 'from crypto_quant.utils.font_helper import get_font_helper'")
                result['fixes_needed'].append("添加 'font_helper = get_font_helper()'")
            
            if result['direct_plt_calls'] > 0:
                result['issues'].append(f"存在{result['direct_plt_calls']}处直接使用plt设置中文标题/标签的情况")
                result['fixes_needed'].append("使用font_helper.set_chinese_title, font_helper.set_chinese_label等方法")
                result['fixes_needed'].append("或在保存图表前调用font_helper.apply_font_to_figure(plt.gcf())")
    
    except Exception as e:
        result['issues'].append(f"分析文件时出错: {str(e)}")
    
    return result

def check_font_support() -> Dict:
    """
    检查当前系统的中文字体支持情况
    
    Returns:
        Dict: 检查结果
    """
    font_helper = get_font_helper()
    
    result = {
        'has_chinese_font': font_helper.has_chinese_font,
        'chinese_font': font_helper.chinese_font,
        'system_info': {}
    }
    
    # 获取系统信息
    import platform
    result['system_info']['platform'] = platform.system()
    result['system_info']['release'] = platform.release()
    
    # 检查是否是WSL
    is_wsl = False
    try:
        with open('/proc/version', 'r') as f:
            is_wsl = 'microsoft' in f.read().lower()
    except:
        pass
    
    result['system_info']['is_wsl'] = is_wsl
    
    # 列出可用的中文字体
    import subprocess
    try:
        fc_list = subprocess.run(['fc-list', ':lang=zh'], capture_output=True, text=True)
        available_fonts = fc_list.stdout.strip().split('\n')
        if available_fonts and available_fonts[0]:
            result['available_chinese_fonts'] = available_fonts
        else:
            result['available_chinese_fonts'] = []
    except:
        result['available_chinese_fonts'] = []
    
    return result

def fix_matplotlib_import(file_path: str) -> bool:
    """
    修复文件中缺少的matplotlib配置
    
    Args:
        file_path: 要修复的文件路径
        
    Returns:
        bool: 是否成功修复
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.readlines()
        
        # 查找matplotlib导入的位置
        mpl_import_index = -1
        plt_import_index = -1
        for i, line in enumerate(content):
            if re.search(r'import\s+matplotlib', line):
                mpl_import_index = i
            if re.search(r'import\s+matplotlib.pyplot', line) or re.search(r'from\s+matplotlib\s+import\s+pyplot', line):
                plt_import_index = i
        
        # 如果有matplotlib导入，添加字体配置
        if plt_import_index >= 0:
            # 查找最后一个导入语句的位置
            last_import_index = plt_import_index
            for i in range(plt_import_index + 1, len(content)):
                if re.search(r'^import\s+|^from\s+', content[i]):
                    last_import_index = i
                elif not content[i].strip() or content[i].strip().startswith('#'):
                    continue
                else:
                    break
            
            # 在导入后添加字体助手导入
            font_helper_import = "\n# 导入字体助手\nfrom crypto_quant.utils.font_helper import get_font_helper\nfont_helper = get_font_helper()\n"
            
            # 检查是否已经导入了字体助手
            if 'font_helper' not in ''.join(content):
                content.insert(last_import_index + 1, font_helper_import)
                
                # 写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(content)
                
                logger.info(f"已修复文件 {file_path} 中的字体导入")
                return True
        
        return False
    
    except Exception as e:
        logger.error(f"修复文件 {file_path} 中的字体导入时出错: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='字体问题修复工具')
    parser.add_argument('--check', action='store_true', help='只检查问题，不修复')
    parser.add_argument('--fix', action='store_true', help='修复检测到的问题')
    parser.add_argument('--install', action='store_true', help='尝试安装中文字体')
    parser.add_argument('--create-script', action='store_true', help='创建安装中文字体的脚本')
    
    args = parser.parse_args()
    
    # 检查字体支持
    logger.info("检查系统字体支持...")
    font_support = check_font_support()
    
    if font_support['has_chinese_font']:
        logger.info(f"系统中已找到中文字体: {font_support['chinese_font']}")
    else:
        logger.warning("系统中未找到中文字体！")
        if args.install:
            logger.info("尝试安装中文字体...")
            if install_chinese_fonts():
                logger.info("中文字体安装成功！")
            else:
                logger.error("中文字体安装失败。")
                if args.create_script:
                    script_path = create_font_install_script()
                    if script_path:
                        logger.info(f"已创建字体安装脚本: {script_path}")
                        logger.info(f"请在终端中运行: bash {script_path}")
    
    if args.check or args.fix:
        # 查找和分析文件
        logger.info("查找和分析项目文件...")
        
        # 查找examples目录下的Python文件
        examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
        example_files = find_python_files(examples_dir)
        
        # 查找scripts目录下的Python文件
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        script_files = find_python_files(scripts_dir)
        
        all_files = example_files + script_files
        
        # 分析每个文件
        results = []
        for file_path in all_files:
            logger.info(f"分析文件: {os.path.basename(file_path)}")
            result = analyze_file(file_path)
            results.append(result)
            
            if result['issues']:
                for issue in result['issues']:
                    logger.warning(f"  - {issue}")
                    
                for fix in result['fixes_needed']:
                    logger.info(f"  - 建议修复: {fix}")
            
            # 如果需要修复
            if args.fix and result['issues'] and not result['uses_font_helper']:
                logger.info(f"尝试修复文件: {os.path.basename(file_path)}")
                if fix_matplotlib_import(file_path):
                    logger.info(f"  - 已修复字体导入")
        
        # 汇总结果
        files_with_issues = [r for r in results if r['issues']]
        
        logger.info("\n===== 分析结果汇总 =====")
        logger.info(f"总计分析了 {len(results)} 个文件")
        logger.info(f"发现 {len(files_with_issues)} 个文件存在字体问题")
        
        if files_with_issues:
            logger.info("\n需要注意的文件:")
            for result in files_with_issues:
                logger.info(f"  - {result['file']}")

if __name__ == "__main__":
    main() 