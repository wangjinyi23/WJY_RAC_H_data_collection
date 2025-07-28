import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 数据
x = [100, 200, 300, 400, 500]  # 不同的网络规模
# 各种搜索算法的结果
greedy_search = [30947,40430,45140,49507,53787]
beam_search = [30369,39107,43227,46945,50498]
mcts = [30305,38878,42705,46097,49255]

# 创建图形
fig, ax = plt.subplots(figsize=(6.5, 4.5))  # IEEE 期刊常用尺寸

# 画网格线 (IEEE 推荐使用淡色虚线)
ax.grid(True, linestyle=':', linewidth=1, alpha=0.7)

# 绘制折线图
ax.plot(x, greedy_search, color='b', label='Greedy', marker='o', linestyle='-', linewidth=2.5, markersize=6)
ax.plot(x, beam_search, color='g', label='Beam', marker='s', linestyle='--', linewidth=2.5, markersize=6)
ax.plot(x, mcts, color='r', label='MCTS', marker='^', linestyle='-.', linewidth=2.5, markersize=6)

# 设置横轴刻度
ax.set_xticks(x)

# 设置轴标签 (IEEE 期刊推荐使用 serif 字体)
plt.xlabel("Number of Sensors", fontsize=14, fontname="serif")
ax.set_ylabel("Flight Energy Consumption(kJ)", fontsize=14, fontname="serif", labelpad=12)
ax.yaxis.set_label_coords(-0.06, 0.5)  # 将标签向右移

# 设置纵轴数字格式（单位 GB）
def fmt(x, pos):
    return f'{x * 1e-3:.0f}'  # 以 GB 为单位，保留 1 位小数

ax.yaxis.set_major_formatter(FuncFormatter(fmt))

# 设置图例格式，IEEE 推荐无边框或黑色边框
legend = plt.legend(fontsize=12, loc='best', frameon=True)
legend.get_frame().set_edgecolor("black")  # IEEE 推荐黑色边框

# 调整刻度字体
ax.tick_params(axis='both', which='major', labelsize=12)

# 紧凑布局
plt.tight_layout()

# 保存 IEEE 期刊格式的高质量图片
#save_path = 'C:/Users/13780/Desktop/实验相关/论文图片/Different_Search_Energy_Consumption.png'  # 修改为你希望保存的路径
#plt.savefig(save_path, dpi=600, bbox_inches='tight')  # 600 dpi 高分辨率适合论文排版

# 显示图形
plt.show()
