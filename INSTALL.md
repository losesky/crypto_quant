# 比特币量化交易框架安装指南

本文档提供详细的安装和配置指南，帮助您成功部署比特币量化交易框架。

## 系统要求

- Python 3.8 或更高版本
- 适用于 Linux、macOS 或 Windows 系统
- 2GB 以上内存（建议4GB以上）
- 对于机器学习模块，建议使用支持CUDA的GPU

## 安装步骤

### 1. 克隆代码仓库

```bash
git clone https://github.com/losesky/crypto_quant.git
cd crypto_quant
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用 venv（Python 内置）
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

或者使用 Conda:

```bash
# 使用 Conda
conda create -n crypto_quant python=3.8
conda activate crypto_quant
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

对于带GPU的机器学习支持，请安装相应的PyTorch版本：

```bash
# CUDA 11.3
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### 4. 配置环境变量

复制环境变量模板并编辑：

```bash
cp .env.example .env
```

使用您喜欢的文本编辑器编辑 `.env` 文件，填入必要的 API 密钥和配置信息：

```
# 编辑 .env 文件
# Linux/macOS
nano .env
# Windows
notepad .env
```

### 5. ClickHouse 数据库配置

#### 安装 ClickHouse（可选，如果您已有ClickHouse服务器可跳过）

**Ubuntu/Debian:**

```bash
sudo apt-get install apt-transport-https ca-certificates dirmngr
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv E0C56BD4

echo "deb https://packages.clickhouse.com/deb stable main" | sudo tee \
    /etc/apt/sources.list.d/clickhouse.list
sudo apt-get update

sudo apt-get install -y clickhouse-server clickhouse-client

sudo service clickhouse-server start
```

**macOS (使用 Homebrew):**

```bash
brew install clickhouse
brew services start clickhouse
```

**Docker:**

```bash
docker run -d --name clickhouse-server -p 9000:9000 -p 8123:8123 yandex/clickhouse-server
```

#### 配置数据库

```bash
clickhouse-client --password
```

创建数据库：

```sql
CREATE DATABASE IF NOT EXISTS crypto_quant;
```

### 6. 测试安装

运行示例脚本以验证安装：

```bash
# 运行数据处理示例
python examples/data_processing_example.py

# 运行特征工程示例
python examples/feature_engineering_example.py
```

### 7. 安装为开发模式（可选）

如果您想参与开发或进行定制化修改，可以以开发模式安装：

```bash
pip install -e .
```

## 常见问题

### Q: 安装依赖时出现错误？

A: 尝试更新 pip 然后重新安装：
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Q: ClickHouse 连接失败？

A: 检查 `.env` 文件中的配置是否正确，确保 ClickHouse 服务正在运行：
```bash
sudo service clickhouse-server status  # Linux
brew services list  # macOS
```

### Q: 缺少 CUDA 支持？

A: 如果您不需要GPU支持，可以使用CPU版本的PyTorch：
```bash
pip install torch==1.10.0
```

## 下一步

成功安装后，请查看 `README.md` 文件了解更多使用信息，或查看 `examples` 目录下的示例脚本开始使用。

如需运行 API 服务：

```bash
python run_api_server.py
```

API 文档可访问：http://localhost:8000/docs 