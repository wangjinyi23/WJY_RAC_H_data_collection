import pandas as pd
import elkai

# 读取 Excel 文件
file_path = 'main_nodes_coordinates_100.xlsx'
df = pd.read_excel(file_path)

# 提取坐标
coordinates = df['Coordinates'].apply(lambda x: eval(x)).to_dict()
coordinates = {f'city{i+1}': coord for i, coord in enumerate(coordinates.values())}

# 使用 elkai 求解 TSP
cities = elkai.Coordinates2D(coordinates)
solution = cities.solve_tsp()

# 提取和打印序号
sequence_numbers = [name.replace('city', '') for name in solution]
print("访问顺序:", sequence_numbers)
