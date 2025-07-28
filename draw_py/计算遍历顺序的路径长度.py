import numpy as np
import matplotlib.pyplot as plt

def calculate_path_length(coordinates, traversal_order):
    """
    计算给定遍历顺序的路径长度。

    参数：
    - coordinates: list，包含节点坐标的列表。
    - traversal_order: list，遍历顺序的节点索引。

    返回：
    - total_length: float，总路径长度。
    """
    # 转换为二维数组
    coordinates = np.array(coordinates)

    # 计算路径长度
    total_length = 0
    for i in range(len(traversal_order) - 1):
        node_a = traversal_order[i]
        node_b = traversal_order[i + 1]
        # 计算欧氏距离
        distance = np.linalg.norm(coordinates[node_a] - coordinates[node_b])
        total_length += distance

    return total_length

def plot_traversal_path(coordinates, traversal_order):
    """
    绘制节点遍历的路径图。

    参数：
    - coordinates: list，包含节点坐标的列表。
    - traversal_order: list，遍历顺序的节点索引。
    """
    # 转换为二维数组
    coordinates = np.array(coordinates)

    # 获取遍历顺序的坐标
    path = coordinates[traversal_order]

    # 绘制节点和路径
    plt.figure(figsize=(10, 8))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', label='节点', marker='o')
    plt.plot(path[:, 0], path[:, 1], c='red', label='遍历路径', marker='x')

    # 标注节点索引
    for i, (x, y) in enumerate(coordinates):
        plt.text(x, y, str(i), fontsize=9, ha='right')

    # 设置标题和标签
    plt.title("TSP 遍历路径图")
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.legend()

    # 显示图形
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 节点坐标，以字符串形式存储
    input_data = """
    103 436
    107 72
    701 21
    615 122
    467 215
    770 344
    192 956
    561 475
    682 476
    700 976
    876 567
    244 832
    841 167
    601 316
    14 242
    565 898
    455 428
    509 776
    872 388
    2 390
    772 822
    402 730
    556 162
    271 456
    462 727
    252 702
    338 879
    53 792
    922 217
    65 857
    344 129
    499 593
    135 201
    930 33
    99 684
    872 726
    987 547
    15 858
    743 241
    """

    # 解析字符串为坐标列表
    coordinates = [list(map(int, line.split())) for line in input_data.strip().split("\n")]

    # 遍历顺序
    traversal_order = [0, 19, 14, 32, 1, 30, 4, 22, 3, 2, 33, 28, 12, 38, 5, 18, 36, 10, 35, 20, 9, 15, 17, 24, 21, 25, 11, 26, 6, 29, 37,27,34,31, 7,8,13,16,23,0]

    # 计算路径长度
    path_length = calculate_path_length(coordinates, traversal_order)
    print(f"总路径长度: {path_length:.2f}")

    # 绘制路径图
    plot_traversal_path(coordinates, traversal_order)
