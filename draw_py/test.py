import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# 设置随机种子，保证可复现
np.random.seed(42)

# 参数定义
num_sensors = 30  # 传感器节点总数
num_clusters = 5   # 集群数量
H = 30            # 无人机悬停高度
city_size = 100   # 城市区域大小 (100x100)

# 生成传感器节点坐标 (x, y)
sensor_positions = np.random.rand(num_sensors, 2) * city_size

# 生成城市建筑物位置和高度
num_buildings = 10
building_positions = np.random.rand(num_buildings, 2) * city_size
building_heights = np.random.randint(10, 50, num_buildings)  # 设定不同高度的建筑物

# 使用 K-Means 进行聚类，确定主传感器位置
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(sensor_positions)
cluster_centers = kmeans.cluster_centers_  # 主传感器节点位置

# 计算悬停点 (UAV hover points)，位于主传感器节点上方 H 处
hover_positions = np.hstack((cluster_centers, np.full((num_clusters, 1), H)))

# 计算 UAV 飞行路径（TSP 近似解）
distance_matrix = cdist(hover_positions[:, :2], hover_positions[:, :2])
G = nx.complete_graph(num_clusters)
for i in range(num_clusters):
    for j in range(i + 1, num_clusters):
        G[i][j]['weight'] = distance_matrix[i, j]

tsp_path = list(nx.approximation.traveling_salesman_problem(G, cycle=True))
tsp_hover_positions = hover_positions[tsp_path]

# 3D 画图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制建筑物
for i in range(num_buildings):
    x, y = building_positions[i]
    h = building_heights[i]
    ax.bar3d(x, y, 0, 10, 10, h, color='gray', alpha=0.5)

# 绘制传感器节点
ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1], 0, c=labels, cmap='tab10', marker='o', label='Sensor Nodes')

# 绘制主传感器节点
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 0, c='red', marker='D', s=100, label='Cluster Heads')

# 绘制无人机悬停点
ax.scatter(hover_positions[:, 0], hover_positions[:, 1], hover_positions[:, 2], c='blue', marker='^', s=150, label='UAV Hover Points')

# 绘制无人机路径
ax.plot(tsp_hover_positions[:, 0], tsp_hover_positions[:, 1], tsp_hover_positions[:, 2], c='black', linestyle='--', linewidth=2, label='UAV Path')

# 绘制无人机到主传感器的垂直线
for i in range(num_clusters):
    ax.plot([hover_positions[i, 0], cluster_centers[i, 0]],
            [hover_positions[i, 1], cluster_centers[i, 1]],
            [hover_positions[i, 2], 0], 'gray', linestyle=':', linewidth=1)

# 设置图例和标签
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Altitude')
ax.set_title('UAV-Assisted IoT Data Collection in Smart City')

plt.legend()
plt.show()
