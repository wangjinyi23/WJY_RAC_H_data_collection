import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# 数据
x = [100, 200, 300, 400, 500]
Cmax_values = [0.2, 0.4, 0.6, 0.8, 1.0]

Cmax_0_2 = [463,1131,1942,2807,3628]
Cmax_0_4 = [874,2019,3539,5270,6097]
Cmax_0_6 = [1190,2879,4983,7451,9734]
Cmax_0_8 = [1759,3994,7032,10191,13630]
Cmax_1 = [1889,4444,7828,11307,14993]

# 创建图形
fig, ax = plt.subplots(figsize=(6, 4))  # 设置图像大小为6x4英寸

# 绘制折线
ax.plot(x, Cmax_0_2, color='b', label='Cmax=0.2', marker='o', linestyle='-', linewidth=2)
ax.plot(x, Cmax_0_4, color='g', label='Cmax=0.4', marker='s', linestyle='--', linewidth=2)
ax.plot(x, Cmax_0_6, color='r', label='Cmax=0.6', marker='^', linestyle='-.', linewidth=2)
ax.plot(x, Cmax_0_8, color='c', label='Cmax=0.8', marker='D', linestyle=':', linewidth=2)
ax.plot(x, Cmax_1, color='m', label='Cmax=1', marker='v', linestyle='-', linewidth=2)

# 设置横轴刻度
ax.set_xticks(x)

# 添加标题和标签
plt.title("", fontsize=14)
plt.xlabel("Network Scale", fontsize=12)
plt.ylabel("Redundant Data Volume(GB)", fontsize=12)

# 设置纵轴数字格式，将数字缩小为原来的千分之一，并显示单位为千焦 (kJ)
def fmt(x, pos):
    return f'{x * 1e-3:.0f}'  # 保留一位小数


ax.yaxis.set_major_formatter(FuncFormatter(fmt))

# 显示图例
plt.legend(fontsize=10)



# 显示图形
save_path = 'C:/Users/13780/Desktop/实验相关/论文图片/不同规模下的冗余数据量.png'  # 修改为你希望保存的路径
plt.tight_layout()
plt.savefig(save_path, dpi=300)  # 保存为高分辨率图片

plt.show()
