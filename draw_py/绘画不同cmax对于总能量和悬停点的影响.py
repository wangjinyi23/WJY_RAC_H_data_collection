import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# 数据
cmax_values = [0.2, 0.4, 0.6, 0.8, 1.0]  # 不同的Cmax值
energy_consumption = [
    [205597, 190576, 182744, 161408, 154577],  # 100节点
    [538048, 482913, 431326, 362717, 332020],  # 300节点
    [864678, 779534, 650850, 514534, 468144]   # 500节点
]
hover_points = [
    [39, 40, 45, 39, 39],  # 100节点
    [70, 78, 75, 74, 76],  # 300节点
    [98, 104, 106, 102, 111]  # 500节点
]
network_scales = [100, 300, 500]  # 网络规模
bar_colors = ['#6a5acd', '#ff7f50', '#ffc0cb']  # 不同网络规模的柱状图颜色

line_colors = ['#32cd32', '#ff6347', '#1e90ff']  # 不同网络规模的折线颜色
bar_width = 0.25  # 柱状图宽度

# x轴位置调整
x_positions = np.arange(len(cmax_values))  # 基础x轴位置
x_offsets = [-bar_width, 0, bar_width]  # 不同网络规模的偏移量

# 创建图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制柱状图
for i, (scale, energy) in enumerate(zip(network_scales, energy_consumption)):
    ax1.bar(
        x_positions + x_offsets[i],  # 每个柱子的位置
        energy,  # 数据
        width=bar_width,
        color=bar_colors[i],
        label=f'Scale={scale}',
        edgecolor='black',  # 黑色边框
        linewidth=1.5  # 边框线宽
    )

# 设置x轴和y轴（左侧）
ax1.set_xlabel('Different $C_{max}$', fontsize=14, fontname="serif")
ax1.set_ylabel('Total Energy Consumption(kJ)', fontsize=14, fontname="serif",labelpad=12)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(cmax_values)

# 添加方格背景
ax1.grid(True, which='both', axis='both', linestyle=':', color='gray', alpha=0.5)

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制折线图
for i, (scale, hover) in enumerate(zip(network_scales, hover_points)):
    ax2.plot(
        x_positions + x_offsets[i],  # x轴为Cmax值，调整偏移量以对齐柱状图
        hover,  # y轴为悬停点数量
        color=line_colors[i],  # 使用不同的颜色
        marker=['o', 's', '^'][i],
        linestyle=['-', '--', '-.'][i],
        linewidth=2,  # 增加折线宽度
        markersize=6,  # 增加标记点大小
        label=f'Scale={scale}'
    )

# 设置y轴（右侧）
ax2.set_ylabel('Hover Points Count', fontsize=12)

# 设置纵轴数字格式，将数字缩小为原来的千分之一，并显示单位为千焦 (kJ)
def fmt(x, pos):
    return f'{x * 1e-3:.0f}'  # 不显示小数点后的零

ax1.yaxis.set_major_formatter(FuncFormatter(fmt))

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines + lines2, labels + labels2, loc='center left', bbox_to_anchor=(0.78, 0.74), title="Network Scale")

# 调整布局
plt.tight_layout()

# 保存图形
save_path = 'C:/Users/13780/Desktop/实验相关/论文图片/Different_Cmax_Scale_modified.png'  # 修改为你希望保存的路径
plt.savefig(save_path, dpi=300)  # 保存为高分辨率图片

# 显示图形
plt.show()
