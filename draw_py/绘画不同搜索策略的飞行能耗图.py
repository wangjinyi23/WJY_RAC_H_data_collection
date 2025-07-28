import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import FuncFormatter


# 数据
x = [100, 200, 300, 400, 500]  # 不同的飞行距离（或其他变量）
ga_search = [30947, 40430, 44140, 49507, 53787]
beam_search = [30369, 39107, 43227, 46945, 50498]
mcts = [30305, 38878, 42705, 46097, 49255]

# 创建主图
# fig, ax = plt.subplots()
# 创建图形并设置相同的图像大小
fig, ax = plt.subplots(figsize=(6, 4))  # 设置图像大小为6x4英寸
ax.plot(x, ga_search, color='b', label='Greedy', marker='o', linestyle='-', linewidth=2)
ax.plot(x, beam_search, color='g', label='Beam', marker='s', linestyle='--', linewidth=2)
ax.plot(x, mcts, color='r', label='MCTS', marker='^', linestyle='-.', linewidth=2)

# 添加标题和坐标轴标签
# ax.set_title('Comparison of Flight Energy Consumption for Different Search Strategies', fontsize=14)
ax.set_xlabel('Network Scale', fontsize=12)
ax.set_ylabel('Flight Energy Consumption(kJ)', fontsize=10)

# 设置纵坐标范围
ax.set_ylim(30000, 55000)
ax.set_xlim(100, 500)

# 设置横轴刻度
ax.set_xticks(x)

# 添加图例
ax.legend()

# 在主图上绘制矩形框，表示放大区域
x1, x2, y1, y2 = 110, 150, 32000, 35000
rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='gray', facecolor='none', linestyle='--')
ax.add_patch(rect)

# 创建插图
axins = inset_axes(ax, width="40%", height="40%", loc='lower center', bbox_to_anchor=(0.25, 0.1, 1, 1), bbox_transform=ax.transAxes)
axins.plot(x, ga_search, color='b', label='GA_search', marker='o', linestyle='-', linewidth=2)
axins.plot(x, beam_search, color='g', label='Beam_search', marker='s', linestyle='--', linewidth=2)
axins.plot(x, mcts, color='r', label='MCTS', marker='^', linestyle='-.', linewidth=2)
def fmt(x, pos):
    return f'{x * 1e-3:.0f}'  # 不显示小数点后的零

ax.yaxis.set_major_formatter(FuncFormatter(fmt))

# 设置放大区域的范围
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# 隐藏放大图的坐标轴
axins.set_xticks([])
axins.set_yticks([])

# 使用连接线连接主图与插图
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", linestyle="--")

# 显示图形
# 保存图形
save_path = 'C:/Users/13780/Desktop/实验相关/论文图片/Different_Search_Energy_Consumption.png'  # 修改为你希望保存的路径
plt.tight_layout()
plt.savefig(save_path, dpi=300)  # 保存为高分辨率图片
plt.show()