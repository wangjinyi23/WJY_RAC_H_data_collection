import pandas as pd
import matplotlib.pyplot as plt

# 读取上传的 Excel 文件
file_path = 'F:\learning\postgraduate\研一\实验\Affinity-Propagation-main\points_with_data_500_nodes.xlsx'
df = pd.read_excel(file_path)

# 主节点列表
main_nodes = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 33, 38, 39, 41, 42, 45, 48, 50, 53, 54, 55, 56, 57, 59, 60, 62, 63, 64, 65, 66, 67, 68, 70, 71, 74, 75, 78, 79, 82, 84, 88, 91, 92, 96, 100, 102, 103, 105, 107, 109, 110, 111, 112, 115, 116, 122, 136, 141, 143, 157, 174, 179, 186, 187, 191, 192, 195, 203, 214, 215, 218, 240, 275, 288, 299, 306, 307, 310, 319, 321, 337, 368, 380, 404, 405, 412, 425, 434, 456, 481, 487, 489, 500}

# 提取坐标和数据量
df['Coordinates'] = df['Coordinates'].apply(lambda x: eval(x))
df['Data Volume'] = df['Data Volume (MB)']  # 假设你的数据量在这一列

# 创建一个字典来累积主节点的数据量
main_node_data_volume = {}

# 创建图形
plt.figure(figsize=(10, 10))

# 遍历每一行，判断节点是否为主节点，并使用不同颜色绘制
for index, row in df.iterrows():
    x, y = row['Coordinates']
    data_volume = row['Data Volume']

    # 检查是否为主节点
    if index + 1 in main_nodes:
        plt.scatter(x, y, color='red', label='Main Node' if index == 0 else "")
        # 累积主节点的数据量
        if index + 1 in main_node_data_volume:
            main_node_data_volume[index + 1] += data_volume
        else:
            main_node_data_volume[index + 1] = data_volume
    else:
        plt.scatter(x, y, color='blue', label='Other Node' if index == 0 else "")

# 打印每个主节点的累积数据量，并计算总数据量
total_volume = 0
print("主节点及其累积的数据量：")
for node, volume in main_node_data_volume.items():
    print(f"节点 {node}: {volume} MB")
    total_volume += volume  # 累加所有主节点的数据量

# 打印总数据量
print(f"所有主节点的总数据量: {total_volume} MB")


# 添加标签和显示图例
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Coordinates Plot (Red: Main Nodes, Blue: Other Nodes)')
plt.legend(loc='upper right')

# 显示图形
plt.grid(True)
plt.show()
