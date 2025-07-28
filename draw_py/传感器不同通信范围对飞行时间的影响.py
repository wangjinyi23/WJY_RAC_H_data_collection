import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 数据
x = [50, 100, 150, 200, 250]
nodes_scales = [100, 200, 300, 400, 500]

scale_100 = [461, 364, 288, 323, 277]
scale_200 = [624, 452, 365, 357, 324]
scale_300 = [677, 414, 418, 340, 331]
scale_400 = [741, 536, 452, 396, 359]
scale_500 = [758, 573, 485, 404, 370]

# 创建图形
fig, ax = plt.subplots(figsize=(6.5, 4.5))  # IEEE 期刊常用尺寸

# 画网格线 (使用淡色虚线)
ax.grid(True, linestyle=':', linewidth=1, alpha=0.7)

# 绘制折线图
ax.plot(x, scale_100, color='blue', label='Scale=100', marker='o', linestyle='-', linewidth=2.5, markersize=6)
ax.plot(x, scale_200, color='green', label='Scale=200', marker='s', linestyle='--', linewidth=2.5, markersize=6)
ax.plot(x, scale_300, color='red', label='Scale=300', marker='^', linestyle='-.', linewidth=2.5, markersize=6)
ax.plot(x, scale_400, color='cyan', label='Scale=400', marker='D', linestyle=':', linewidth=2.5, markersize=6)
ax.plot(x, scale_500, color='magenta', label='Scale=500', marker='v', linestyle='-', linewidth=2.5, markersize=6)

# 设置横轴刻度
ax.set_xticks(x)

# 设置轴标签 (使用 serif 字体，符合 IEEE 期刊风格)
plt.xlabel("Communication Range (m)", fontsize=14, fontname="serif")
plt.ylabel("Flight Time (s)", fontsize=14, fontname="serif", labelpad=12)
ax.yaxis.set_label_coords(-0.08, 0.5)  # 将标签向右移

# 设置纵轴格式
def fmt(x, pos):
    return f'{x:.0f}'

ax.yaxis.set_major_formatter(FuncFormatter(fmt))

# 设置图例并修改边框颜色
legend = plt.legend(fontsize=12, loc='upper right')
legend.get_frame().set_edgecolor('black')  # 黑色边框

# 调整刻度字体
ax.tick_params(axis='both', which='major', labelsize=12)

# 紧凑布局
plt.tight_layout()

# 保存并显示图形
save_path = 'C:/Users/13780/Desktop/实验相关/论文图片/IEEE_传感器通信范围对飞行时间的影响.png'  # 修改为你希望保存的路径
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 高分辨率适合论文排版

# 显示图形
plt.show()
