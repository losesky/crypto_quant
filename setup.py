#!/usr/bin/env python
"""
比特币量化交易框架安装脚本
"""
from setuptools import setup, find_packages
import os

# 读取requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# 读取README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="crypto_quant",
    version="0.1.0",
    author="losesky",
    author_email="losesky@gmail.com",
    description="比特币量化交易框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/losesky/crypto_quant",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "crypto-quant-api=crypto_quant.api.rest.api_service:start_api_server",
        ],
    },
) 