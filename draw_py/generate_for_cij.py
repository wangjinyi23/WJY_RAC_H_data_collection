import numpy as np
from scipy.spatial import distance_matrix
# 定义C(i,j)的值和相应的概率
c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
probabilities = [0.03, 0.07, 0.10, 0.13, 0.07, 0.13, 0.15, 0.18, 0.10, 0.04]

# 随机产生一个C(i,j)值
def generate_single_c_ij():
    return np.random.choice(c_values, p=probabilities)

# 生成一个随机值
random_c_ij = generate_single_c_ij()



