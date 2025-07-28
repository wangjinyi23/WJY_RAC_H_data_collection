import numpy as np

# 加载数据矩阵
p = np.load('F:/learning/postgraduate/研一/实验/Affinity-Propagation-main/_neighbors_matrix_with_pij_200_nodes.npy')

# 主节点集合
M =  {2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 24, 25, 27, 30, 33, 35, 37, 40, 41, 42, 44, 47, 49, 55, 57, 58, 59, 62, 65, 70, 72, 73, 79, 81, 82, 84, 86, 88, 93, 94, 100, 107, 108, 109, 110, 111, 112, 115, 119, 121, 123, 124, 127, 128, 143, 144, 159, 168, 174, 177, 185, 187}

# 计算剩余从节点集合
V = set(range(1, 101))  # 节点范围为 1 到 500
S = V - M  # 从节点集合

# 字典存储主节点和其连接的从节点
master_to_slaves = {master: [] for master in M}

# 对每个从节点选择一个主节点
for j in S:
    max_p_value = -1
    selected_master = None

    for i in M:
        if p[i-1, j-1] > max_p_value:  # 寻找使得 p[i, j] 最大的主节点
            max_p_value = p[i-1, j-1]
            selected_master = i

    # 将从节点分配给找到的主节点
    if selected_master is not None:
        master_to_slaves[selected_master].append(j)





# 输出主节点及其连接的从节点字典
print(master_to_slaves)

