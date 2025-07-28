import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def calculate_path_length(file_path, traversal_order):
    """
    计算给定遍历顺序的路径长度。

    参数：
    - file_path: str，Excel 文件路径。
    - traversal_order: list，遍历顺序的节点索引。

    返回：
    - total_length: float，总路径长度。
    """
    # 读取 Excel 文件
    data = pd.read_excel(file_path)

    # 提取坐标列并转换为二维数组
    coordinates = data['Coordinates'].apply(lambda x: eval(x))  # 将字符串元组转为实际元组
    coordinates = np.array(list(coordinates))

    # 使用给定的遍历顺序计算路径长度
    total_length = 0
    for i in range(len(traversal_order) - 1):
        node_a = traversal_order[i]
        node_b = traversal_order[i + 1]
        # 计算欧几里得距离
        distance = np.linalg.norm(coordinates[node_a] - coordinates[node_b])
        total_length += distance

    return total_length

def optimize_tour(file_path, initial_order):
    """
    使用贪婪算法优化初始路径。

    参数：
    - file_path: str，Excel 文件路径。
    - initial_order: list，初始遍历顺序。

    返回：
    - optimized_order: list，优化后的遍历顺序。
    - optimized_length: float，优化后的路径长度。
    """
    # 读取 Excel 文件
    data = pd.read_excel(file_path)
    coordinates = data['Coordinates'].apply(lambda x: eval(x))  # 将字符串元组转为实际元组
    coordinates = np.array(list(coordinates))

    num_nodes = len(initial_order)
    optimized_order = [initial_order[0]]  # 从初始节点开始

    # 构建距离矩阵
    dist_matrix = cdist(coordinates, coordinates)

    # 剩余节点集合
    remaining_nodes = set(initial_order[1:])

    # 贪婪选择最近的节点
    current_node = initial_order[0]
    while remaining_nodes:
        next_node = min(remaining_nodes, key=lambda x: dist_matrix[current_node][x])
        optimized_order.append(next_node)
        remaining_nodes.remove(next_node)
        current_node = next_node

    # 返回到起始节点形成闭环
    optimized_order.append(initial_order[0])
    optimized_length = calculate_path_length(file_path, optimized_order)

    return optimized_order, optimized_length

def plot_tour(file_path, traversal_order):
    """
    绘制节点遍历路径。

    参数：
    - file_path: str，Excel 文件路径。
    - traversal_order: list，遍历顺序的节点索引。
    """
    # 读取 Excel 文件
    data = pd.read_excel(file_path)
    coordinates = data['Coordinates'].apply(lambda x: eval(x))  # 将字符串元组转为实际元组
    coordinates = np.array(list(coordinates))

    # 提取遍历路径的坐标
    tour_coordinates = coordinates[traversal_order]

    # 绘制路径
    plt.figure(figsize=(10, 6))
    plt.plot(tour_coordinates[:, 0], tour_coordinates[:, 1], marker='o', linestyle='-', color='b', label='Path')
    for i, coord in enumerate(tour_coordinates):
        plt.text(coord[0], coord[1], str(traversal_order[i]), fontsize=8, color='red')

    plt.title('Optimized Tour Path')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 文件路径
    file_path = "points_with_data_100_nodes.xlsx"

    # 初始遍历顺序
    initial_order = [0, 19, 14, 32, 1, 30, 4, 22, 3, 2, 33, 28, 12, 38, 5, 18, 36, 10, 35, 20, 9, 15, 17, 24, 21, 25, 11, 26, 6, 29, 37, 27, 34, 7, 8, 13, 16, 31, 23]
    # 计算初始路径长度
    initial_length = calculate_path_length(file_path, initial_order)
    print(f"初始路径长度: {initial_length:.2f}")

    # 优化路径
    optimized_order, optimized_length = optimize_tour(file_path, initial_order)
    print(f"优化后的遍历顺序: {optimized_order}")
    print(f"优化后的路径长度: {optimized_length:.2f}")

    # 绘制优化后的路径
    plot_tour(file_path, optimized_order)
