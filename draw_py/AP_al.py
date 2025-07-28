from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd





k=6
dataLen=500
# 读取之前保存的随机点坐标
# try:
#     df = pd.read_excel('random_points_500.xlsx')
#     nodes = df.values
# except FileNotFoundError:
#     print("随机点坐标文件不存在，请先运行随机生成点的代码。")
#     exit()

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




def euclidean_distance(node1, node2):
    """计算两个点之间的欧氏距离"""
    return np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

# 找到每个点的邻居集合（不含结点自身）
def find_neighbors_without_self(nodes, radius):
    """计算每个点的邻居集合，不含结点自身"""
    neighbors_false = []
    for i, node1 in enumerate(nodes):
        neighbor_indices_false = []
        for j, node2 in enumerate(nodes):
            if i != j and euclidean_distance(node1, node2) <= radius:
                neighbor_indices_false.append(j)
        neighbors_false.append(neighbor_indices_false)
    return neighbors_false


# 找到每个点的邻居集合（包含结点自身）
def find_neighbors_with_self(nodes, radius):
    """计算每个点的邻居集合，包含结点自身"""
    neighbors_true = []
    for i, node1 in enumerate(nodes):
        neighbor_indices_true = []
        for j, node2 in enumerate(nodes):
            if euclidean_distance(node1, node2) <= radius:
                neighbor_indices_true.append(j)
        neighbors_true.append(neighbor_indices_true)
    return neighbors_true


file_path = "points_with_data_500_nodes.xlsx"  # 替换为你的文件路径
#
# # 调用函数提取数据
nodes, data_volumes = extract_coordinates_and_volumes(file_path)

# 不包含自身结点的邻居集合
neighbors_false=find_neighbors_without_self(nodes,100)
# 包含自身结点的集合
neighbors_true=find_neighbors_with_self(nodes,100)






# Smk=−ζmk− κ, m = k
# Smk=−ζmk, m != k
# 其中−ζmk代表从节点m到结点k上悬停点传输数据所需要的时间
def cal_simi():
    ##这个数据集的相似度矩阵，最终是二维数组
    simi = []
    # 读取Excel文件，设置header和index_col参数为None
    df = pd.read_excel('upload_time_set_500_lunwen.xlsx', header=None, index_col=None)

    # 将DataFrame转换为二维数组
    data_array = df.to_numpy()

    # 逐个打印二维数组中的每个元素
    for i in range(len(data_array)):
        temp = []
        for j in range(len(data_array[i])):
            if i != j:
                # 如果行号和列号不相等，则执行以下操作
                # 采用负的欧式距离计算相似度
                s = -data_array[i][j]
            else:
                # 如果行号和列号相等，则执行以下操作
                s = -data_array[i][j] - k
            temp.append(s)
        simi.append(temp)

    return simi


'''
第三步：计算吸引度矩阵，即R
       公式1：r(n+1) =s(n)-(s(n)+a(n))-->简化写法，具体参见上图公式
       公式2：r(n+1)=(1-λ)*r(n+1)+λ*r(n)
'''


##初始化R矩阵  创建一个大小为 dataLen × dataLen 的二维数组，并将所有元素初始化为 0，然后将该数组返回
def init_R(dataLen):
    R = [[0] * dataLen for j in range(dataLen)]
    return R

#初始化A矩阵  创建一个大小为 dataLen × dataLen 的二维数组，并将所有元素初始化为 0
def init_A(dataLen):
    A = [[0] * dataLen for j in range(dataLen)]
    return A

##迭代更新R矩阵
def iter_update_R(dataLen, R, A, simi):
    old_r = 0  ##更新前的某个r值
    lam = 0.5  ##阻尼系数,用于算法收敛
    ##此循环更新R矩阵
    for m in range(dataLen):
        for k in range(dataLen):
            old_r = R[m][k]

            max1 = A[m][0] + simi[m][0]  #最大值的初始值
            for j in neighbors_true[m]:
                if j != k:
                    if A[m][j] + simi[m][j] > max1:
                        max1 = A[m][j] + simi[m][j]
            ##更新后的R[i][k]值
            R[m][k] = simi[m][k] - max1
            ##带入阻尼系数重新更新
            R[m][k] = (1 - lam) * R[m][k] + lam * old_r

    print("max_r:" + str(np.max(R)))
    # print(np.min(R))
    return R


'''
    第四步：计算归属度矩阵，即A
'''


##迭代更新A矩阵
def iter_update_A(dataLen, R, A):
    old_a = 0  ##更新前的某个a值
    lam = 0.5  ##阻尼系数,用于算法收敛
    ##此循环更新A矩阵
    for m in range(dataLen):
        for k in range(dataLen):
            old_a = A[m][k]
            # m==k的情况下计算amk
            if m == k:
                max3 = R[0][k]  ##注意初始值的设置


                # 在结点k的邻居集合（不包含结点k）中进行计算
                for j in neighbors_false[k]:
                    if R[j][k]>0:
                        max3 += R[j][k]
                    else:
                        max3 += 0

                A[m][k] = max3
                ##带入阻尼系数更新A值
                A[m][k] = (1 - lam) * A[m][k] + lam * old_a

            # m!=k的情况下计算amk
            else:
                max4 = R[0][k]  ##注意初始值的设置
                for j in neighbors_false[k]:
                    ##上图公式中的i!=k 的求和部分
                    if j != m:
                        if R[j][k] > 0:
                            max4 += R[j][k]
                        else:
                            max4 += 0

                ##上图公式中的min部分
                if R[k][k] + max4 > 0:
                    A[m][k] = 0
                else:
                    A[m][k] = R[k][k] + max4

                ##带入阻尼系数更新A值
                A[m][k] = (1 - lam) * A[m][k] + lam * old_a
    print("max_a:" + str(np.max(A)))
    # print(np.min(A))
    return A


'''
   第5步：计算聚类中心
'''


##计算聚类中心
def cal_cls_center(dataLen, simi, R, A):
    ##进行聚类，不断迭代直到预设的迭代次数或者判断comp_cnt次后聚类中心不再变化
    max_iter = 300  ##最大迭代次数
    curr_iter = 0  ##当前迭代次数
    max_comp = 20  ##最大比较次数
    curr_comp = 0  ##当前比较次数
    class_cen = []  ##聚类中心列表，存储的是数据点在nodes中的索引
    while True:
        ##计算R矩阵
        R = iter_update_R(dataLen, R, A, simi)
        ##计算A矩阵
        A = iter_update_A(dataLen, R, A)
        ##开始计算聚类中心
        for k in range(dataLen):
            if R[k][k] + A[k][k] > 0:
                if k not in class_cen:
                    class_cen.append(k)
                else:
                    curr_comp += 1
        curr_iter += 1
        print(curr_iter)
        if curr_iter >= max_iter or curr_comp > max_comp:
            break
    return class_cen

if __name__=='__main__':
    # # 初始化R、A矩阵
    R = init_R(dataLen)
    A = init_A(dataLen)
    ##计算相似度
    simi = cal_simi()
    ##输出聚类中心
    class_cen = cal_cls_center(dataLen, simi, R, A)
    ##根据聚类中心划分数据
    print(class_cen)
    print(len(class_cen))
    plt.figure(figsize=(10, 10))
# 绘制所有点
#     plt.scatter(nodes[:, 0], nodes[:, 1], color='blue')
#
#     # 确定需要用红点标出的索引
#     red_indices=[5, 16, 19, 22, 23, 26, 31, 39, 48, 49, 54, 82, 85, 87, 104, 105, 106, 109, 126, 127, 128, 131, 141, 148, 161, 162, 166, 167, 171, 184, 212, 228, 230, 249, 251, 259, 263, 271, 276, 302, 308, 311, 317, 325, 339, 348, 349, 359, 371, 375, 377, 389, 405, 416, 448, 0, 4, 10, 44, 56, 66, 86, 163, 264, 266, 337, 376, 433, 463]
#     # 绘制红点
#     plt.scatter(nodes[red_indices, 0], nodes[red_indices, 1], color='red')
#
#     # 显示图例和标题
#     plt.legend(['All Nodes', 'Selected Nodes'], loc='upper left')
#     plt.title('Nodes Plot')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#
#     # 显示图形
#     plt.show()
