# 工具模块

此目录包含了在量化交易框架中使用的各种实用工具类和函数。

## 字体辅助模块 (font_helper.py)

### 解决中文字体显示问题

在生成图表时，如果使用中文标题、标签或图例，可能会遇到字体显示问题，例如方块乱码。这是因为系统可能缺少相应的中文字体，或者matplotlib无法正确加载中文字体。

### 使用方法

1. 在绘图代码中导入字体助手模块：

```python
from crypto_quant.utils.font_helper import get_font_helper

# 获取全局字体助手实例
font_helper = get_font_helper()
```

2. 使用字体助手设置中文标题和标签：

```python
# 设置标题
font_helper.set_chinese_title(ax, '策略回测结果')

# 设置坐标轴标签
font_helper.set_chinese_label(ax, xlabel='日期', ylabel='价格')

# 设置图例
font_helper.set_chinese_legend(ax)
```

3. 或者，对整个图表进行批量处理：

```python
# 创建图表，设置标题等...
fig, ax = plt.subplots()
ax.set_title('策略回测结果')
ax.set_xlabel('日期')
ax.set_ylabel('价格')

# 保存前应用字体
font_helper.apply_font_to_figure(fig)

# 保存图表
plt.savefig('策略回测结果.png')
```

4. 获取中英文标签：

```python
# 根据是否有中文字体，自动选择中文或英文标签
title = font_helper.get_label('策略回测结果', 'Strategy Backtest Results')
```

### 字体问题诊断与修复

项目提供了一个辅助脚本来检查和修复字体问题：

```bash
# 检查项目中的字体问题
python scripts/fix_font_issues.py --check

# 自动修复项目中的字体问题
python scripts/fix_font_issues.py --fix

# 安装中文字体（Linux/WSL）
python scripts/fix_font_issues.py --install

# 创建字体安装脚本（如果自动安装失败）
python scripts/fix_font_issues.py --create-script
```

### WSL/Linux系统中安装中文字体

如果使用WSL或Linux系统，可以通过以下命令安装中文字体：

```bash
# 安装字体
sudo apt update
sudo apt install -y fonts-wqy-microhei fonts-wqy-zenhei xfonts-wqy fonts-noto-cjk

# 刷新字体缓存
sudo fc-cache -f -v

# 检查安装的中文字体
fc-list :lang=zh
```

也可以使用字体助手中的辅助函数：

```python
from crypto_quant.utils.font_helper import install_chinese_fonts, create_font_install_script

# 方法1：直接安装
install_chinese_fonts()

# 方法2：创建安装脚本
script_path = create_font_install_script()
print(f"请在终端中运行: bash {script_path}")
```

### 常见问题

1. **乱码问题**：即使安装了中文字体，也可能出现乱码。这通常是因为matplotlib无法找到或加载字体。解决方法是使用`font_helper.apply_font_to_figure()`来处理整个图表。

2. **字体找不到**：如果字体助手无法找到中文字体，它会自动将中文标题和标签替换为英文版本。你可以在`font_helper.py`的映射字典中添加更多中英文映射。

3. **性能问题**：在大量图表生成的情况下，每次设置字体可能会影响性能。可以考虑一次性设置matplotlib的全局字体：

```python
import matplotlib.pyplot as plt
from crypto_quant.utils.font_helper import get_font_helper

font_helper = get_font_helper()
if font_helper.has_chinese_font:
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
``` 