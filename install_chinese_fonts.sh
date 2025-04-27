#!/bin/bash
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
