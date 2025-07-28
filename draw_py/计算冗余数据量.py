import numpy as np
import pandas as pd

# 加载邻接矩阵
neighbors_matrix = np.load('neighbors_matrix_with_cij_500_nodes.npy')

# 主节点集合
main_nodes = [5, 16, 19, 22, 23, 26, 31, 39, 48, 49, 54, 82, 85, 87, 104, 105, 106, 109, 126, 127, 128, 131, 141, 148, 161, 162, 166, 167, 171, 184, 212, 228, 230, 249, 251, 259, 263, 271, 276, 302, 308, 311, 317, 325, 339, 348, 349, 359, 371, 375, 377, 389, 405, 416, 448, 0, 4, 10, 44, 56, 66, 86, 163, 264, 266, 337, 376, 433, 463]

# 提取坐标和数据量的函数
def extract_coordinates_and_volumes(file_path):
    # 读取 Excel 文件
    data = pd.read_excel(file_path)

    # 初始化存储坐标和数据量的数组
    coordinates = []
    data_volumes = []

    # 遍历每一行数据
    for index, row in data.iterrows():
        # 读取坐标列并将字符串形式的坐标转换为整数型元组
        coordinate = eval(row['Coordinates'])  # 转换为元组 (int, int)
        volume = row['Data Volume (MB)']  # 读取数据量列

        # 添加到对应数组
        coordinates.append(coordinate)
        data_volumes.append(volume)

    # 将坐标转换为二维数组（整型）
    coordinates = np.array(coordinates, dtype=int)

    return coordinates, data_volumes


# 提取坐标和数据量
file_path = "points_with_data_500_nodes.xlsx"  # 替换为你的文件路径
nodes, data_volumes = extract_coordinates_and_volumes(file_path)

# 计算总冗余数据量
total_redundant_data = 0

# 遍历主节点集合中的每对主节点和附属节点
for i in main_nodes:
    for j in range(i + 1, len(neighbors_matrix)):  # 避免计算节点与自己之间的冗余，并确保i < j
        # 计算冗余数据量 = min(数据量_i, 数据量_j) * pij
        min_data_volume = min(data_volumes[i], data_volumes[j])
        redundant_data = min_data_volume * neighbors_matrix[i][j]
        total_redundant_data += redundant_data

# 输出总冗余数据量
print(f"Total Redundant Data: {total_redundant_data}")
