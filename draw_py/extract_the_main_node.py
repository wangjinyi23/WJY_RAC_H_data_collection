import pandas as pd

# 读取 Excel 文件
file_path = 'F:\\learning\\postgraduate\\研一\\实验\\Affinity-Propagation-main\\points_with_data_100_nodes.xlsx'
df = pd.read_excel(file_path)

# 主节点列表
main_nodes = {1, 3, 4, 5, 6, 12, 15, 20, 22, 23, 27, 28, 32, 34, 35, 37, 40, 41, 45, 46, 48, 50, 51, 55, 56, 57, 60, 61, 62, 67, 69, 72, 78, 80, 84, 85, 86, 95, 99}

# 提取坐标和数据量
df['Coordinates'] = df['Coordinates'].apply(lambda x: eval(x))
df['Data Volume'] = df['Data Volume (MB)']  # 假设你的数据量在这一列

# 创建一个表示节点编号的新列
df['Node Number'] = df.index + 1

# 只保留主节点的坐标数据
main_nodes_data = df[df['Node Number'].isin(main_nodes)][['Coordinates', 'Data Volume']]

# 打印主节点的坐标及其数据量
print(main_nodes_data)

# 如果需要将主节点数据另存为 Excel 文件
output_file = 'main_nodes_coordinates_100.xlsx'
main_nodes_data.to_excel(output_file, index=False)
print(f"主节点的坐标数据已保存到 {output_file}")
