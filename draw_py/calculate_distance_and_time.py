import numpy as np
import pandas as pd
from networkx.classes import nodes



r=100  #无人机覆盖半径
a=9.61  #环境常数
b=0.16  #环境常数
d_m_k=0 #任意两点之间的距离
h=50    #无人机飞行高度

#任意两点之间的Los概率
#PLos_m_k = 1 / ( 1 + a * np.exp(-b *  [180 / np.pi * np.arcsin(h / d_m_k) - a]))


#空对地路径损耗
β_l=3
fc=2e9    #载波频率
c=3 * (10 ** 8) #光速
#L_ploss_dmk = β_l * ((4 * np.pi * fc *d_m_k / c)**2)
σ_2=-110    #噪声功率
β=-60     #参考距离d0=1m处的通道增益
Pm=0.1   #发射功率
B=1e7    #系统带宽

#给定传感器的发射功率Pm,可计算任意两点之间传输数据的速率
#R_mk=B * np.log2(1+β * Pm / L_ploss_dmk * σ_2)

#每个点的数据长度1M
data_length=1e6


#计算两个点之间的距离
def euclidean_distance(node1, node2):
    return np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

#计算一个点和其他任何点之间的距离（在其他点上方50米处）
def find_with_distances(nodes):
    distance = {}
    for i, node1 in enumerate(nodes):
        mk_distances = {}
        for j, node2 in enumerate(nodes):
            mk_distances[j] = np.sqrt(euclidean_distance(node1, node2)**2+50*50)
        distance[i] =mk_distances
    return distance

#计算传输时间
def find_with_upload_time(distance, data_volumes):
    upload_time = {}
    for i in range(len(distance)):
        mk_upload_time = {}
        # 获取当前节点的数据量
        data_length = data_volumes[i]  # 使用data_volumes中的数据量
        for j in range(len(distance)):
            # 计算路径损失（L_ploss_dmk）
            L_ploss_dmk = β_l * ((4 * np.pi * fc * distance[i][j] / c) ** 2)
            # 计算传输速率 (R_mk)
            R_mk = B * np.log2(1 + β * Pm / (L_ploss_dmk * σ_2))
            # 计算上传时间 (upload_time)
            mk_upload_time[j] = data_length / R_mk
        upload_time[i] = mk_upload_time
    return upload_time



# 保存数据到Excel表格
def save_to_excel(upload_time_set):
    # 将upload_time_set转换为DataFrame
    df = pd.DataFrame(upload_time_set)
    # 将DataFrame写入Excel表格
    df.to_excel('upload_time_set_500_lunwen.xlsx', index=False)


"""提取points_with_data_100_nodes.xls等的节点坐标"""

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
        volume = row['Data Volume (MB)']

        # 添加到对应数组
        coordinates.append(coordinate)
        data_volumes.append(volume)

    # 将坐标转换为二维数组（整型）
    coordinates = np.array(coordinates, dtype=int)

    return coordinates, data_volumes




# 检查是否有保存的数据文件，如果有则加载，否则生成新的随机点
# try:
#     df = pd.read_excel('random_points_50.xlsx')
#     nodes = df.values  # 从 Excel 表格中加载点的坐标
# except FileNotFoundError:
#     num_nodes = 500
#     nodes = np.random.rand(num_nodes, 2) * 2000  # 随机生成 100 个点的坐标
#     df = pd.DataFrame(nodes, columns=['X', 'Y'])
#     df.to_excel('random_points_50.xlsx', index=False)  # 将生成的点保存到 Excel 表格中
file_path = "points_with_data_500_nodes.xlsx"  # 替换为你的文件路径
#
# # 调用函数提取数据
nodes, data_volumes = extract_coordinates_and_volumes(file_path)



#距离
distance_set=find_with_distances(nodes)

upload_time_set=find_with_upload_time(distance_set,data_volumes)

# 调用保存函数
save_to_excel(upload_time_set)

#打印每个结点到其他点的距离
for i, distance in distance_set.items():
    print(f"Node {i} diatance: {distance}")



#打印每个结点到悬停点传输数据所需要的时间
for i, upload_time in upload_time_set.items():
    print(f"Node {i} neighbor_transmission_time: {upload_time}")\




















