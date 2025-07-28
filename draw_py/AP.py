from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#
# '''
# 第一步：生成测试数据
#     1.生成实际中心为centers的测试样本300个，
#     2.Xn是包含150个(x,y)点的二维数组
#     3.labels_true为其对应的真是类别标签
# '''
#
#
#

try:
    df = pd.read_excel('random_points_500.xlsx')
    nodes = df.values
except FileNotFoundError:
    print("随机点坐标文件不存在，请先运行随机生成点的代码。")
    exit()




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


# 不包含自身结点的邻居集合
neighbors_false=find_neighbors_without_self(nodes,100)
# 包含自身结点的集合
neighbors_true=find_neighbors_with_self(nodes,100)


'''
第二步：计算相似度矩阵
'''
# 相似矩阵simi[] 里面的元素Smk与传输时间相关
def cal_simi(nodes):
    ##这个数据集的相似度矩阵，最终是二维数组
    simi = []
    for m in nodes:
        ##每个数字与所有数字的相似度列表，即矩阵中的一行
        temp = []
        for n in nodes:
            ##采用负的欧式距离计算相似度
            s = -(np.sqrt((m[0] - n[0]) ** 2 + (m[1] - n[1]) ** 2))
            temp.append(s)
        simi.append(temp)

    ##设置参考度，即对角线的值，一般为最小值或者中值
    # p = np.min(simi)   ##11个中心
    p = np.max(simi)  ##14个中心
    # p =-20  ##5个中心

    for i in range(dataLen):
        simi[i][i] = p
    return simi


'''
第三步：计算吸引度矩阵，即R
       公式1：r(n+1) =s(n)-(s(n)+a(n))-->简化写法，具体参见上图公式
       公式2：r(n+1)=(1-λ)*r(n+1)+λ*r(n)
'''


##初始化R矩阵、A矩阵
def init_R(dataLen):
    R = [[0] * dataLen for j in range(dataLen)]
    return R


def init_A(dataLen):
    A = [[0] * dataLen for j in range(dataLen)]
    return A


##迭代更新R矩阵
def iter_update_R(dataLen, R, A, simi):
    old_r = 0  ##更新前的某个r值
    lam = 0.5  ##阻尼系数,用于算法收敛
    ##此循环更新R矩阵
    for i in range(dataLen):
        for k in range(dataLen):
            old_r = R[i][k]
            if i != k:
                max1 = A[i][0] + R[i][0]  ##注意初始值的设置
                for j in range(dataLen):
                    if j != k:
                        if A[i][j] + R[i][j] > max1:
                            max1 = A[i][j] + R[i][j]
                ##更新后的R[i][k]值
                R[i][k] = simi[i][k] - max1
                ##带入阻尼系数重新更新
                R[i][k] = (1 - lam) * R[i][k] + lam * old_r
            else:
                max2 = simi[i][0]  ##注意初始值的设置
                for j in range(dataLen):
                    if j != k:
                        if simi[i][j] > max2:
                            max2 = simi[i][j]
                ##更新后的R[i][k]值
                R[i][k] = simi[i][k] - max2
                ##带入阻尼系数重新更新
                R[i][k] = (1 - lam) * R[i][k] + lam * old_r
    print("max_r:" + str(np.max(R)))
    # print(np.min(R))
    return R




'''
    第四步：计算归属度矩阵，即A
'''


#迭代更新A矩阵
def iter_update_A(dataLen, R, A):
    old_a = 0  ##更新前的某个a值
    lam = 0.5  ##阻尼系数,用于算法收敛
    ##此循环更新A矩阵
    for i in range(dataLen):
        for k in range(dataLen):
            old_a = A[i][k]
            if i == k:
                max3 = R[0][k]  ##注意初始值的设置
                for j in range(dataLen):
                    if j != k:
                        if R[j][k] > 0:
                            max3 += R[j][k]
                        else:
                            max3 += 0
                A[i][k] = max3
                ##带入阻尼系数更新A值
                A[i][k] = (1 - lam) * A[i][k] + lam * old_a
            else:
                max4 = R[0][k]  ##注意初始值的设置
                for j in range(dataLen):
                    ##上图公式中的i!=k 的求和部分
                    if j != k and j != i:
                        if R[j][k] > 0:
                            max4 += R[j][k]
                        else:
                            max4 += 0

                ##上图公式中的min部分
                if R[k][k] + max4 > 0:
                    A[i][k] = 0
                else:
                    A[i][k] = R[k][k] + max4

                ##带入阻尼系数更新A值
                A[i][k] = (1 - lam) * A[i][k] + lam * old_a
    print("max_a:" + str(np.max(A)))
    # print(np.min(A))
    return A



'''
   第5步：计算聚类中心
'''


##计算聚类中心
def cal_cls_center(dataLen, simi, R, A):
    ##进行聚类，不断迭代直到预设的迭代次数或者判断comp_cnt次后聚类中心不再变化
    max_iter = 100  ##最大迭代次数
    curr_iter = 0  ##当前迭代次数
    max_comp = 20  ##最大比较次数
    curr_comp = 0  ##当前比较次数
    class_cen = []  ##聚类中心列表，存储的是数据点在Xn中的索引
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


if __name__ == '__main__':
    ##初始化数据
    # Xn, dataLen = init_sample()
    # ##初始化R、A矩阵
    # dataLen = 500
    # R = init_R(dataLen)
    # A = init_A(dataLen)
    # ##计算相似度
    # simi = cal_simi(nodes)
    # ##输出聚类中心
    # class_cen = cal_cls_center(dataLen, simi, R, A)
    # # for i in class_cen:
    # #    print(str(i)+":"+str(nodes[i]))
    # print(class_cen)



    #
    # """画图"""
    # 创建一个1000x1000像素大小的图像
    plt.figure(figsize=(10, 10))
    # 绘制所有点
    plt.scatter(nodes[:, 0], nodes[:, 1], color='blue')

    # 确定需要用红点标出的索引
    red_indices = [28, 112, 167, 235, 298, 354, 1, 6, 8, 9, 13, 16, 23, 45, 59, 60, 10, 14, 25, 39, 42, 61, 105, 106, 125, 130]



    # 绘制红点
    plt.scatter(nodes[red_indices, 0], nodes[red_indices, 1], color='red')

    # 显示图例和标题
    plt.legend(['All Nodes', 'Selected Nodes'], loc='upper left')
    plt.title('Nodes Plot')
    plt.xlabel('X')
    plt.ylabel('Y')



    # 显示图形
    plt.show()



