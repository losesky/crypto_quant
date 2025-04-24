# 输出目录 (Output Directory)

本目录用于存储由系统生成的各类输出文件，如图表、报告和数据导出。

## 目录结构

- `images/`: 存储所有生成的图表和可视化文件
- `reports/`: 存储生成的报告、分析结果和导出的数据

## 使用说明

在代码中，使用 `output_helper` 模块来获取正确的路径用于保存文件：

```python
from crypto_quant.utils.output_helper import get_image_path, get_report_path

# 保存图像
plt.savefig(get_image_path("my_chart.png"))

# 保存报告
with open(get_report_path("analysis_report.html"), "w") as f:
    f.write(report_content)
```

这样可以确保所有输出文件都存储在适当的位置，而不是直接保存到项目根目录。 