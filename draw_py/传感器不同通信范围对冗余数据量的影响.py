import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 数据
x = [50, 100, 150, 200, 250]
Cmax_values = [0.2, 0.4, 0.6, 0.8, 1.0]

Cmax_0_2 = [215, 463, 604, 715, 758]
Cmax_0_4 = [378, 874, 1092, 1244, 1419]
Cmax_0_6 = [553, 1190, 1590, 1968, 2022]
Cmax_0_8 = [696, 1759, 2297, 2542, 2752]
Cmax_1 = [764, 1889, 2532, 2813, 3173]

# 创建图形
fig, ax = plt.subplots(figsize=(6.5, 4.5))  # IEEE 期刊常用尺寸

# 画网格线 (IEEE 推荐使用淡色虚线)
ax.grid(True, linestyle=':', linewidth=1, alpha=0.7)

# 绘制折线图
ax.plot(x, Cmax_0_2, color='b', label='$C_{max}=0.2$', marker='o', linestyle='-', linewidth=2.5, markersize=6)
ax.plot(x, Cmax_0_4, color='g', label='$C_{max}=0.4$', marker='s', linestyle='--', linewidth=2.5, markersize=6)
ax.plot(x, Cmax_0_6, color='r', label='$C_{max}=0.6$', marker='^', linestyle='-.', linewidth=2.5, markersize=6)
ax.plot(x, Cmax_0_8, color='c', label='$C_{max}=0.8$', marker='D', linestyle=':', linewidth=2.5, markersize=6)
ax.plot(x, Cmax_1, color='m', label='$C_{max}=1.0$', marker='v', linestyle='-', linewidth=2.5, markersize=6)

# 设置横轴刻度
ax.set_xticks(x)

# 设置轴标签 (IEEE 期刊推荐使用 serif 字体)
plt.xlabel("Communication Range (m)", fontsize=14, fontname="serif")
plt.ylabel("Redundant Data Volume (GB)", fontsize=14, fontname="serif", labelpad=12)
ax.yaxis.set_label_coords(-0.07, 0.5)  # 将标签向右移

# 设置纵轴数字格式（单位 GB）
def fmt(x, pos):
    return f'{x * 1e-3:.1f}'  # 以 GB 为单位，保留 1 位小数

ax.yaxis.set_major_formatter(FuncFormatter(fmt))

# 设置图例格式，IEEE 推荐无边框或黑色边框
legend = plt.legend(fontsize=12, loc='best', frameon=True)
legend.get_frame().set_edgecolor("black")  # IEEE 推荐黑色边框
# 调整刻度字体
ax.tick_params(axis='both', which='major', labelsize=12)

# 紧凑布局
plt.tight_layout()

# 保存 IEEE 期刊格式的高质量图片
save_path = 'C:/Users/13780/Desktop/实验相关/论文图片/IEEE_通信范围对冗余数据量的影响.png'
plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 600 dpi 高分辨率适合论文排版

# 显示图形
plt.show()
