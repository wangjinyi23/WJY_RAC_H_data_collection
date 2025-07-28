import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# cmax的值
x = [0.2, 0.4, 0.6, 0.8, 1]

# 各算法的飞行距离 (m)
ours = [8075.4, 8363.82, 8267.11, 8330.61, 8596.29]
GA = [8509.42, 8808.22, 8515.21, 8961.56, 9632.65]
learning_greedy = [9134.08, 9293.04, 9093.82, 8855.43, 9576.26]
SOM = [8357.28, 8935.33, 8692.75, 8943.33, 9205.97]
ACO = [8461.39, 9330.34, 9068.47, 9051.14, 9601.06]

# 创建图形
fig, ax = plt.subplots(figsize=(6.5, 4.5))  # IEEE 期刊常用大小 (6.5 x 4.5 英寸)

# 画网格线
ax.grid(True, linestyle=':', linewidth=1, alpha=0.7)

# 绘制折线图
ax.plot(x, ours, color='b', label='Ours', marker='o', linestyle='-', linewidth=2.5, markersize=6)
ax.plot(x, GA, color='g', label='GA', marker='s', linestyle='--', linewidth=2.5, markersize=6)
ax.plot(x, learning_greedy, color='r', label='Learning+Greedy', marker='^', linestyle='-.', linewidth=2.5, markersize=6)
ax.plot(x, SOM, color='c', label='SOM', marker='D', linestyle=':', linewidth=2.5, markersize=6)
ax.plot(x, ACO, color='m', label='ACO', marker='v', linestyle='-', linewidth=2.5, markersize=6)

# 设置 x 轴刻度
ax.set_xticks(x)

# 设置轴标签
plt.xlabel("Different $C_{max}$", fontsize=14, fontname="serif")
plt.ylabel("Flight Distance (km)", fontsize=14, fontname="serif", labelpad=12)
ax.yaxis.set_label_coords(-0.07, 0.5)  # 将标签向右移
# 设置纵轴数字格式（单位 km）
def fmt(x, pos):
    return f'{x * 1e-3:.1f}'  # 以千米 (km) 为单位，保留1位小数

ax.yaxis.set_major_formatter(FuncFormatter(fmt))

# **优化图例**
legend = plt.legend(
    fontsize=11,  # 调小字体，避免遮挡
    loc='upper center',  # 置于顶部中央
    bbox_to_anchor=(0.5, 1.02),  # 向上微调
    ncol=2,  # 3 列布局，节省空间
    frameon=True
)
legend.get_frame().set_edgecolor("black")  # IEEE 推荐黑色边框

# 调整刻度字体
ax.tick_params(axis='both', which='major', labelsize=12)

# 紧凑布局，防止图例溢出
plt.tight_layout()

# 保存高质量图片
save_path = 'C:/Users/13780/Desktop/实验相关/论文图片/IEEE_不同搜索算法在不同cmax下的影响.png'
plt.savefig(save_path, dpi=600, bbox_inches='tight')

# 显示图形
plt.show()
