import matplotlib.pyplot as plt

# 飞行速度
speeds = [10, 12, 14, 16, 18, 20]

# 各种算法的飞行时间
ours = [859.6, 716.3, 614, 537.3, 477.5, 429.8]
GA = [940.2, 818.7, 670, 622, 535, 478]
learning_greedy = [957.6, 798, 684, 598.5, 532, 469]
SOM = [920.6, 767, 657.5, 575, 511.4, 460]
ACO = [960.1, 800, 685.8, 600, 533.4, 489]

# 绘制折线图
plt.figure(figsize=(8, 6))

plt.plot(speeds, ours, label='Ours', marker='o', linestyle='-', color='b')
plt.plot(speeds, GA, label='GA', marker='o', linestyle='-', color='g')
plt.plot(speeds, learning_greedy, label='Learning+Greedy', marker='o', linestyle='-', color='r')
plt.plot(speeds, SOM, label='SOM', marker='o', linestyle='-', color='c')
plt.plot(speeds, ACO, label='ACO', marker='o', linestyle='-', color='m')

# 设置标题和标签
plt.title('Flight Time vs Speed')
plt.xlabel('Speed (km/h)')
plt.ylabel('Flight Time (s)')

# 添加网格和图例
plt.grid(True)
plt.legend()

# 显示图表
plt.show()
