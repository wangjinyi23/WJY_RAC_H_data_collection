import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# 数据
x = [50, 100, 150, 200, 250]

scale_100 = [201598, 155352, 125274, 118481, 102019]
scale_200 = [341263, 253317, 199945, 173951, 153414]
scale_300 = [454893, 324668, 265080, 217787, 194638]
scale_400 = [562640, 401141, 312892, 271208, 235359]
scale_500 = [651090, 467823, 372596, 307470, 265577]

# 创建图形
fig, ax = plt.subplots(figsize=(6.5, 4.5))  # IEEE 期刊常用大小 (6.5 x 4.5 英寸)

# 画网格线，IEEE 推荐使用淡色虚线网格
ax.grid(True, linestyle=':', linewidth=1, alpha=0.7)

# 绘制折线图
ax.plot(x, scale_100, color='b', label='Scale = 100', marker='o', linestyle='-', linewidth=2.5, markersize=6)
ax.plot(x, scale_200, color='g', label='Scale = 200', marker='s', linestyle='--', linewidth=2.5, markersize=6)
ax.plot(x, scale_300, color='r', label='Scale = 300', marker='^', linestyle='-.', linewidth=2.5, markersize=6)
ax.plot(x, scale_400, color='c', label='Scale = 400', marker='D', linestyle=':', linewidth=2.5, markersize=6)
ax.plot(x, scale_500, color='m', label='Scale = 500', marker='v', linestyle='-', linewidth=2.5, markersize=6)

# 设置横轴刻度
ax.set_xticks(x)

# 设置标题和轴标签
plt.xlabel("Communication Range (m)", fontsize=14, fontname="serif")
plt.ylabel("Total Energy Consumption (kJ)", fontsize=14, fontname="serif", labelpad=12)
ax.yaxis.set_label_coords(-0.09, 0.5)  # 将标签向右移

# 设置纵轴数字格式（单位 kJ）
def fmt(x, pos):
    return f'{x * 1e-3:.0f}'  # 只保留整数，IEEE 期刊风格

ax.yaxis.set_major_formatter(FuncFormatter(fmt))

# 设置图例格式，IEEE 期刊建议不带边框或者加黑色边框
legend = plt.legend(fontsize=12, loc='best', frameon=True)
legend.get_frame().set_edgecolor("black")  # IEEE 推荐黑色边框

# 调整刻度字体
ax.tick_params(axis='both', which='major', labelsize=12)

# 紧凑布局
plt.tight_layout()

# 保存 IEEE 期刊格式的高质量图片
save_path = 'C:/Users/13780/Desktop/实验相关/论文图片/IEEE_传感器通信范围对总能耗影响.png'
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 高分辨率适合论文排版

# 显示图形
plt.show()
