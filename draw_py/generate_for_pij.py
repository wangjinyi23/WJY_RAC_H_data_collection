
import pandas as pd
import ast
import numpy as np
from scipy.spatial import distance_matrix
from generate_for_cij import generate_single_c_ij
# 设置随机种子

# np.random.seed(42)  # 42 是一个示例种子，你可以选择其他整数
# # 生成500个随机点，每个点的坐标范围在0到1000
# def generate_random_points_with_data(num_points=500, range_min=1, range_max=1000, data_min=10, data_max=100):
#     # 生成点的坐标
#     points = np.random.randint(range_min, range_max + 1, size=(num_points, 2))
#
#     # 生成每个点的数据量，单位为MB
#     data_volumes = np.random.randint(data_min, data_max + 1, size=num_points)
#
#     # 将点和对应的数据量组合起来
#     points_with_data = [(tuple(points[i]), data_volumes[i]) for i in range(num_points)]
#
#     return points_with_data
# def extract_coordinates(points_with_data):
#     # 提取坐标点
#     coordinates = np.array([point for point, _ in points_with_data])
#     return coordinates
#
#
def calculate_neighbors_matrix(points, comm_range):
    # 计算距离矩阵
    dist_matrix = distance_matrix(points, points)

    # 初始化邻居矩阵  初始值全部填充为0
    num_points = len(points)
    neighbors_matrix = np.zeros((num_points, num_points), dtype=float)

    # 填充邻居矩阵
    for i in range(num_points):
        # 获取与点 i 之间距离小于或等于 comm_range 的点的索引
        neighbors_i = np.where(dist_matrix[i] <= comm_range)[0]
        # 不包括自己
        neighbors_i = neighbors_i[neighbors_i != i]
        # 将邻居位置的值设为 -1
        neighbors_matrix[i, neighbors_i] = -1

    return neighbors_matrix
#
# # 生成随机点及其数据
# points_with_data = generate_random_points_with_data()
#
# print(points_with_data)
#
# # 将数据转换为 DataFrame
# df = pd.DataFrame(points_with_data, columns=['Coordinates', 'Data Volume (MB)'])
#
# # 保存到 Excel 文件
# df.to_excel('points_with_data_500_nodes.xlsx', index=False)


# ==========================================额外增加的代码，用来寻找不同覆盖半径下每个节点的邻居节点
# 读取 Excel 文件
# file_path = 'points_with_data_100_nodes.xlsx'
# df = pd.read_excel(file_path)
#
# # 提取坐标列
# coordinates_column = df['Coordinates']
#
# # 解析坐标数据，将字符串 "(x, y)" 转换为元组 (x, y)
# points = [ast.literal_eval(coord) for coord in coordinates_column]
# neighbors_matrix = calculate_neighbors_matrix(points, 250)  # 通信范围100米
#
# # 保存 neighbors_matrix 到文件
# np.save('neighbors_matrix_100_nodes_r_250.npy', neighbors_matrix)
#==================================================
#

#
# # 从
# # 提取坐标点
# points = extract_coordinates(points_with_data)
# # 计算每个点的邻居集合
# neighbors_matrix = calculate_neighbors_matrix(points, 100)  # 通信范围100米
#
# # 保存 neighbors_matrix 到文件
# np.save('neighbors_matrix_500_nodes_r_150.npy', neighbors_matrix)
#
# print("邻居矩阵已保存到 'neighbors_matrix.npy'")





#上述是关于随机生成点以及计算每个点的邻居集合生成一个N*N的数组，数组的元素-1代表是邻居关系
#=========================================================================
#下边代码是生成cij矩阵的


import numpy as np
from scipy.spatial import distance_matrix

# 定义C(i,j)的值和相应的概率

# # ==============================cmax=1
# c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# probabilities = [0.03, 0.07, 0.10, 0.13, 0.07, 0.13, 0.15, 0.18, 0.10, 0.04]

#==============================cmax=0.8
# c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# probabilities = [0.0475, 0.0875, 0.1175, 0.1475, 0.0875, 0.1475, 0.1675, 0.1975]

# ===========================cmax=0.6
# c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# probabilities = [0.11, 0.148, 0.178, 0.208, 0.148, 0.208]


#==============================cmax=0.4
# c_values = [0.1, 0.2, 0.3, 0.4]
# probabilities = [0.199, 0.237, 0.267, 0.297]
#
# ==================================cmax=0.2
# c_values = [0.1, 0.2]
# probabilities = [0.481, 0.519]
# # #
# # #
# # # # 随机产生一个C(i,j)值
# def generate_single_c_ij():
#     return np.random.choice(c_values, p=probabilities)
#
# # 读取 neighbors_matrix 从文件
# neighbors_matrix = np.load('neighbors_matrix_100_nodes_r_250.npy')
#
# # 遍历 neighbors_matrix，替换 -1 的值
# for i in range(neighbors_matrix.shape[0]):
#     for j in range(neighbors_matrix.shape[1]):
#         if neighbors_matrix[i, j] == -1:
#             # 生成随机的 C(i,j) 值
#             random_c_ij = generate_single_c_ij()
#
#             # 打印生成的随机 C(i,j) 值
#             print(f"生成的随机 C(i,j) 值: {random_c_ij}")
#
#             # 替换 -1
#             neighbors_matrix[i, j] = random_c_ij
#
#             # 打印替换后的矩阵值，确保它们一致
#             print(f"替换后的值: {neighbors_matrix[i, j]}")
# # 保存更新后的 neighbors_matrix 到文件
# np.save('neighbors_matrix_with_cij_cmax_1_100_nodes_r_250.npy', neighbors_matrix)
# # #
# # #
#
#
# # # 打印更新后的 neighbors_matrix
# # print("更新后的邻居矩阵（部分示例）：")
# # print(neighbors_matrix)
#








# #上述代码生成的是cij矩阵
# #=========================================================
# #接下来的代码是生成pij代码
# #
import pandas as pd
import numpy as np
#
# # 读取 Excel 文件
df = pd.read_excel('points_with_data_100_nodes.xlsx')
# 读取 updated_neighbors_matrix.npy 文件
neighbors_matrix = np.load('neighbors_matrix_with_cij_cmax_1_100_nodes_r_250.npy')

# 获取矩阵的形状
rows, cols = neighbors_matrix.shape

# 根据索引取出坐标点及其对应的数据量
def get_data_volume_by_index(index):
    # 提取坐标和对应的数据量
    coordinates = df.iloc[index]['Coordinates']
    data_volume = df.iloc[index]['Data Volume (MB)']

    return coordinates, data_volume

a,b=get_data_volume_by_index(0)
print(b)

# 遍历矩阵中的每一个元素
for i in range(rows):
    index = i
    _, data_volume_i = get_data_volume_by_index(index)

    print("===========================================")

    for j in range(cols):
        value = neighbors_matrix[i, j]
        _, data_volume_j = get_data_volume_by_index(j)
        data_volume = min(data_volume_i,data_volume_j)

        # 更新矩阵中的值：乘以对应的 data_volume

        neighbors_matrix[i, j] = value * data_volume
        print(neighbors_matrix[i, j])

# 保存更新后的邻居矩阵
np.save('_neighbors_matrix_with_pij_cmax_1_100_nodes_r_250.npy', neighbors_matrix)
