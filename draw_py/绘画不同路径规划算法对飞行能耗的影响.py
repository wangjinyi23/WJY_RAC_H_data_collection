# import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter
# # 数据
# x = [100, 200, 300, 400, 500]  # 不同的飞行距离（或其他变量）
# ours = [32569, 39130, 41811, 44997, 47730]
# ga = [33360, 43043, 44762, 48871, 51346]
# learning_greedy = [33738, 41668, 45404, 50681, 50739]
# som = [32569, 39915, 44865, 47851, 51243]
# aco = [32867, 42086, 43393, 45587, 51862]
#
# # 创建图形
# fig, ax = plt.subplots(figsize=(6, 4))
#
# # 绘制折线
# ax.plot(x, ours, color='b', label='Ours', marker='o', linestyle='-', linewidth=2)
# ax.plot(x, ga, color='g', label='GA', marker='s', linestyle='--', linewidth=2)
# ax.plot(x, learning_greedy, color='r', label='Learning+Greedy', marker='^', linestyle='-.', linewidth=2)
# ax.plot(x, som, color='c', label='SOM', marker='D', linestyle=':', linewidth=2)
# ax.plot(x, aco, color='m', label='ACO', marker='v', linestyle='-', linewidth=2)
#
# # 添加标题和坐标轴标签
# # ax.set_title('Energy Consumption Comparison for Different Strategies', fontsize=12, weight='bold')
# ax.set_xlabel('Network Scale', fontsize=10)
# ax.set_ylabel('Energy Consumption (kJ)', fontsize=10)
# # 设置横轴刻度
# ax.set_xticks(x)
# # 设置坐标轴范围
# ax.set_ylim(30000, 55000)
# ax.set_xlim(100, 500)
#
#
# # 设置纵轴数字格式，将数字缩小为原来的千分之一，并显示单位为千焦 (kJ)
# def fmt(x, pos):
#     return f'{x * 1e-3:.0f}'  # 不显示小数点后的零
#
# ax.yaxis.set_major_formatter(FuncFormatter(fmt))
# # 添加图例
# ax.legend()
#
# # 显示图形
# # 保存图形
# save_path = 'C:/Users/13780/Desktop/实验相关/论文图片/Different_Tra_Energy_Consumption.png'  # 修改为你希望保存的路径
# plt.tight_layout()
# plt.savefig(save_path, dpi=300)  # 保存为高分辨率图片
# plt.show()


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# cmax的值
x = [100, 200, 300, 400, 500]

# 飞行能耗
ours = [32569, 39130, 41811, 44997, 47730]
ga = [33360, 43043, 44762, 48871, 51346]
learning_greedy = [33738, 41668, 45404, 50681, 50739]
som = [32569, 39915, 44865, 47851, 51243]
aco = [32867, 42086, 43393, 45587, 51862]

# 创建图形
fig, ax = plt.subplots(figsize=(6.5, 4.5))  # IEEE 期刊常用大小 (6.5 x 4.5 英寸)

# 画网格线
ax.grid(True, linestyle=':', linewidth=1, alpha=0.7)

# 绘制折线图
ax.plot(x, ours, color='b', label='Ours', marker='o', linestyle='-', linewidth=2.5, markersize=6)
ax.plot(x, ga, color='g', label='GA', marker='s', linestyle='--', linewidth=2.5, markersize=6)
ax.plot(x, learning_greedy, color='r', label='Learning+Greedy', marker='^', linestyle='-.', linewidth=2.5, markersize=6)
ax.plot(x, som, color='c', label='SOM', marker='D', linestyle=':', linewidth=2.5, markersize=6)
ax.plot(x, aco, color='m', label='ACO', marker='v', linestyle='-', linewidth=2.5, markersize=6)

# 设置 x 轴刻度
ax.set_xticks(x)

# 设置轴标签
plt.xlabel("Number of Sensors", fontsize=14, fontname="serif")
plt.ylabel("Flight Energy Consumption(kJ)", fontsize=14, fontname="serif", labelpad=12)
ax.yaxis.set_label_coords(-0.09, 0.5)  # 将标签向右移
# 设置纵轴数字格式（单位 km）
def fmt(x, pos):
    return f'{x * 1e-3:.1f}'  # 以千米 (km) 为单位，保留1位小数

ax.yaxis.set_major_formatter(FuncFormatter(fmt))

# 设置图例格式，IEEE 推荐无边框或黑色边框
legend = plt.legend(fontsize=12, loc='best', frameon=True)
legend.get_frame().set_edgecolor("black")  # IEEE 推荐黑色边框
# 调整刻度字体
ax.tick_params(axis='both', which='major', labelsize=12)


# 紧凑布局，防止图例溢出
plt.tight_layout()

# 保存高质量图片
save_path = 'C:/Users/13780/Desktop/实验相关/论文图片/Different_Tra_Energy_Consumption.png'
plt.savefig(save_path, dpi=600, bbox_inches='tight')

# 显示图形
plt.show()
