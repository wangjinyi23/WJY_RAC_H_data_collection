import numpy as np

def greedy_search(prob_heatmap, start_node=0):
    """
    使用贪心算法在概率热图中搜索遍历顺序。

    参数：
    - prob_heatmap: 2D numpy 数组，表示节点之间属于最优解的概率矩阵。
    - start_node: 起始节点的索引（默认为 0）。

    返回：
    - tour: 一个列表，表示按照贪心算法生成的节点遍历顺序。
    """
    num_nodes = prob_heatmap.shape[0]
    visited = [False] * num_nodes  # 用于标记节点是否已经访问
    tour = [start_node]  # 初始化遍历路径
    visited[start_node] = True  # 标记起始节点为已访问

    current_node = start_node
    for _ in range(num_nodes - 1):
        # 获取当前节点的所有邻居的概率，并屏蔽已访问节点
        probabilities = prob_heatmap[current_node]
        probabilities[visited] = -1  # 将已访问节点的概率设为 -1，避免被选中

        # 选择概率最高的下一个节点
        next_node = np.argmax(probabilities)
        tour.append(next_node)  # 添加到遍历路径
        visited[next_node] = True  # 标记节点为已访问
        current_node = next_node  # 更新当前节点

    return tour

# 从文件中读取概率热图
def read_prob_heatmap_from_file(file_path):
    """
    从指定的文件中读取概率热图。

    参数：
    - file_path: txt文件的路径。

    返回：
    - prob_heatmap: 2D numpy 数组，表示概率热图。
    """
    return np.loadtxt(file_path)

# 示例使用
if __name__ == "__main__":
    file_path = "D:/learning/postgraduate/实验/heatmap/tsp111_1/heatmaptsp111_0_greedy.txt"  # 指定热图文件路径

    # 从文件中读取热图
    prob_heatmap = read_prob_heatmap_from_file(file_path)

    start_node = 0  # 起始节点索引
    tour = greedy_search(prob_heatmap, start_node)
    print("生成的遍历顺序:", tour)
