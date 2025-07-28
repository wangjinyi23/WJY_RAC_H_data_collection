import pandas as pd

# 需要提取的行号（从1开始）
row_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 21, 22, 23, 25, 27, 28, 29, 30, 33, 37, 41, 42, 47, 56, 61, 63, 66, 69, 70, 79, 81, 92, 93, 133, 141, 152, 157, 180, 248, 342, 375, 400, 403, 430, 452

]



# 读取Excel文件
file_path = "points_with_data_500_nodes.xlsx"  # 替换为实际文件路径
df = pd.read_excel(file_path)

# 提取对应行的坐标
coordinates = df.iloc[[i - 1 for i in row_indices], 0]  # 转换为0索引

# 打印结果，去掉括号和逗号，用空格分隔
print("提取的坐标点：")
for coord in coordinates:
    x, y = map(int, coord.strip("()").split(","))
    print(f"{x} {y}")



# ============================================================下面计算有多少个元素
# # 定义集合
# data ={1, 3, 4, 5, 6, 7, 10, 11, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 28, 30, 32, 33, 35, 37, 38, 39, 41, 44, 46, 48, 49, 50, 51, 55, 56, 57, 58, 59, 60, 61, 62, 66, 68, 71, 72, 74, 75, 78, 79, 80, 84, 88, 91, 92, 96, 102, 103, 105, 107, 109, 110, 112, 115, 116, 129, 133, 136, 140, 141, 145, 146, 149, 157, 161, 163, 165, 166, 167, 172, 182, 192, 198, 199, 203, 215, 235, 244, 245, 253, 255, 275, 278, 287, 288, 290, 293, 307, 310, 311, 315, 319, 320, 339, 356, 405, 434, 435, 446, 452, 466, 487}
#
# count = len(data)
#
# # 输出结果
# print(f"集合中共有 {count} 个元素。")






# =====================================================下面是进行坐标的转换
# def convert_graph_coordinates(points):
#     """
#     Converts the coordinates of a graph G0 to a new graph G00.
#
#     Parameters:
#     points (list of tuples): List of (x, y) coordinates representing the vertices of G0.
#
#     Returns:
#     list of tuples: List of (x_new, y_new) coordinates representing the vertices of G00.
#     """
#     if not points:
#         raise ValueError("The points list is empty.")
#
#     # Extract x and y coordinates from the points
#     x_coords = [point[0] for point in points]
#     y_coords = [point[1] for point in points]
#
#     # Calculate xmin, xmax, ymin, ymax
#     xmin, xmax = min(x_coords), max(x_coords)
#     ymin, ymax = min(y_coords), max(y_coords)
#
#     # Calculate the amplification factor
#     s = 1 / max(xmax - xmin, ymax - ymin)
#
#     # Convert coordinates
#     new_points = []
#     for x, y in points:
#         x_new = s * (x - xmin)
#         y_new = s * (y - ymin)
#         new_points.append((x_new, y_new))
#
#     return new_points
#
# # Input points
# original_points = [
#     (103, 436), (861, 271), (107, 72), (701, 21), (467, 215), (331, 459), (100, 872), (664, 131), (662, 309), (770, 344), (492, 414), (192, 956), (277, 161), (748, 857), (700, 976), (783, 190), (876, 567), (244, 832), (367, 956), (509, 776), (872, 388), (2, 390), (566, 106), (772, 822), (402, 730), (271, 456), (462, 727), (252, 702), (53, 792), (41, 157), (648, 472), (499, 593), (490, 231), (135, 201), (930, 33), (99, 684), (872, 726), (987, 547), (554, 892)
# ]
#
# # Convert the coordinates
# converted_points = convert_graph_coordinates(original_points)
#
# # Print the results in one line with higher precision
# print("Converted Points:")
# print(" ".join(f"{point[0]:.16f} {point[1]:.16f}" for point in converted_points))





















#
# def format_coordinates(input_coordinates):
#     """
#     Formats a list of coordinates from space-separated values to a Python list of tuples.
#
#     Parameters:
#     input_coordinates (str): Multi-line string of coordinates in "x y" format.
#
#     Returns:
#     str: Formatted string in the desired list of tuples format.
#     """
#     lines = input_coordinates.strip().split("\n")
#     points = [(int(x), int(y)) for line in lines for x, y in [line.split()]]
#     formatted_output = "[\n    " + ", ".join(f"({x}, {y})" for x, y in points) + "\n]"
#     return formatted_output
#

# Input coordinates
input_data = """
103 436
861 271
107 72
701 21
467 215
331 459
100 872
664 131
662 309
770 344
492 414
192 956
277 161
748 857
700 976
783 190
876 567
244 832
367 956
509 776
872 388
2 390
566 106
772 822
402 730
271 456
462 727
252 702
53 792
41 157
648 472
499 593
490 231
135 201
930 33
99 684
872 726
987 547
554 892
"""
#
# # Format and print the result
# formatted_coordinates = format_coordinates(input_data)
# print(formatted_coordinates)

























#
# # 输入数据，模拟命令行输入的格式
# input_data = """
#  0  6 11 39 46  1 73 22 10 65 59 19  5 38 72 42 64 55 13 68 51 71  9 24
#  44 45  4 66 27  3  2 62 61 36 15 37 58  8 20 75 26 41 49 17 16 60 54 48
#  30 67 69 52 14 63 12 28 34 50 21 25 31 29 43 33 32 74 18 23 53  7 40 35
#  56 70 47 57
# """
#
# # 将输入数据转换为数字列表
# numbers = list(map(int, input_data.split()))
#
# # 将每个元素加一
# updated_numbers = [num + 1 for num in numbers]
#
# # 格式化输出结果，每行输出最多10个数字
# print("Updated Numbers:")
# for i in range(0, len(updated_numbers), 10):
#     print(" ".join(map(str, updated_numbers[i:i+10])))












#
# import matplotlib.pyplot as plt
#
# # 输入的节点坐标
# coordinates = [
#     0.091470, 0.442960, 0.870504, 0.273381, 0.465570, 0.215827, 0.325797, 0.466598, 0.076053, 0.378212, 0.668037,
#     0.129496, 0.813977, 0.391572, 0.562179, 0.483042, 0.705036, 0.997945, 0.885920, 0.577595, 0.236382, 0.849949,
#     0.504625, 0.129496, 0.484070, 0.836588, 0.650565, 0.016444, 0.849949, 0.166495, 0.267215, 0.393628, 0.603289,
#     0.319630, 0.000000, 0.243577, 0.566290, 0.917780, 0.335046, 0.089414, 0.362795, 0.977390, 0.197328, 0.078109,
#     0.475848, 0.717369, 0.193217, 0.979445, 0.872559, 0.833505, 0.244604, 0.716341, 0.040082, 0.808839, 0.770812,
#     0.188078, 0.376156, 0.501542, 0.651593, 0.479959, 0.050360, 0.137718, 0.375128, 0.789311, 0.027749, 0.023638,
#     0.848921, 0.796506, 0.941418, 0.028777, 0.034943, 0.511819, 0.733813, 0.822199, 0.087359, 0.697842, 1.000000,
#     0.557040, 0.973279, 0.754368, 0.775951, 0.000000, 0.209661, 0.511819, 0.001028, 0.876670, 0.576567, 0.882837,
#     0.749229, 0.242549
# ]
#
# # 输出的顺序
# output_order = [
#     1, 5, 18, 31, 33, 22, 20, 12, 3, 16, 42, 4, 29, 8, 30, 17, 45, 28, 6, 14, 41, 35, 15, 2, 7, 10, 39, 40, 25,
#     34, 37, 9, 19, 44, 13, 23, 32, 26, 11, 21, 24, 43, 27, 38, 36, 1
# ]
#
# # 将坐标重新排列成(x, y)的格式
# points = [(coordinates[i], coordinates[i+1]) for i in range(0, len(coordinates), 2)]
#
# # 根据output_order获取节点的连接顺序
# ordered_points = [points[i-1] for i in output_order]
#
# # 提取x和y坐标
# x_vals = [point[0] for point in ordered_points]
# y_vals = [point[1] for point in ordered_points]
#
# # 绘制图形
# plt.figure(figsize=(8, 6))
# plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', markersize=6)
#
# # 添加节点的标签
# for i, (x, y) in enumerate(ordered_points):
#     plt.text(x, y, str(i+1), fontsize=9, ha='right', color='red')
#
# # 设置图形的标题和标签
# plt.title("Node Connections Following the Output Order")
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
#
# # 显示图形
# plt.grid(True)
# plt.show()
#

