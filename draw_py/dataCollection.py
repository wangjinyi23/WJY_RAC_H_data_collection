import numpy as np
import random
import math





# 设置随机种子
np.random.seed(42)  # 42 是一个示例种子，你可以选择其他整数

# 生成500个随机点，每个点的坐标范围在0到1000
def generate_random_points_with_data(num_points=500, range_min=0, range_max=1000, data_min=100, data_max=1000):
    # 生成点的坐标
    points = np.random.randint(range_min, range_max + 1, size=(num_points, 2))

    # 生成每个点的数据量，单位为MB
    data_volumes = np.random.randint(data_min, data_max + 1, size=num_points)

    # 将点和对应的数据量组合起来
    points_with_data = [(tuple(points[i]), data_volumes[i]) for i in range(num_points)]

    return points_with_data





class GA(object):
    def __init__(self, num_city, num_total, iteration, data):
        self.num_city = num_city
        self.num_total = num_total
        self.scores = []
        self.iteration = iteration
        self.location = data
        self.ga_choose_ratio = 0.2
        self.mutate_ratio = 0.05
        # 计算距离矩阵
        self.dis_mat = self.compute_dis_mat(num_city, data)
        self.fruits = self.greedy_init(self.dis_mat, num_total, num_city)
        # 显示初始化后的最佳路径
        scores = self.compute_adp(self.fruits)
        sort_index = np.argsort(-scores)
        init_best = self.fruits[sort_index[0]]
        init_best = self.location[init_best]

        # 存储每个iteration的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [1. / scores[sort_index[0]]]

    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x
                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        return result

    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    def compute_pathlen(self, path, dis_mat):
        result = dis_mat[path[0]][path[-1]]
        for i in range(len(path) - 1):
            result += dis_mat[path[i]][path[i + 1]]
        return result

    def compute_adp(self, fruits):
        adp = []
        for fruit in fruits:
            length = self.compute_pathlen(fruit, self.dis_mat)
            adp.append(1.0 / length)
        return np.array(adp)

    def ga_cross(self, x, y):
        len_ = len(x)
        assert len(x) == len(y)
        path_list = [t for t in range(len_)]
        order = list(random.sample(path_list, 2))
        order.sort()
        start, end = order

        tmp = x[start:end]
        x_conflict_index = [y.index(sub) for sub in tmp if not (y.index(sub) >= start and y.index(sub) < end)]
        y_conflict_index = [x.index(sub) for sub in y[start:end] if not (x.index(sub) >= start and x.index(sub) < end)]

        tmp = x[start:end].copy()
        x[start:end] = y[start:end]
        y[start:end] = tmp

        for i, j in zip(x_conflict_index, y_conflict_index):
            y[i], x[j] = x[j], y[i]

        return list(x), list(y)

    def ga_parent(self, scores, ga_choose_ratio):
        sort_index = np.argsort(-scores).copy()
        sort_index = sort_index[0:int(ga_choose_ratio * len(sort_index))]
        parents = [self.fruits[index] for index in sort_index]
        parents_score = [scores[index] for index in sort_index]
        return parents, parents_score

    def ga_choose(self, genes_score, genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(genes_choose[index1]), list(genes_choose[index2])

    def ga_mutate(self, gene):
        path_list = [t for t in range(len(gene))]
        order = list(random.sample(path_list, 2))
        start, end = min(order), max(order)
        gene[start:end] = gene[start:end][::-1]
        return list(gene)

    def ga(self):
        scores = self.compute_adp(self.fruits)
        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]
        fruits = parents.copy()
        while len(fruits) < self.num_total:
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.ga_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.ga_mutate(gene_y_new)
            x_adp = 1. / self.compute_pathlen(gene_x_new, self.dis_mat)
            y_adp = 1. / self.compute_pathlen(gene_y_new, self.dis_mat)
            if x_adp > y_adp and gene_x_new not in fruits:
                fruits.append(gene_x_new)
            elif x_adp <= y_adp and gene_y_new not in fruits:
                fruits.append(gene_y_new)

        self.fruits = fruits
        return tmp_best_one, tmp_best_score

    def run(self):
        BEST_LIST = None
        best_score = -math.inf
        self.best_record = []
        for i in range(1, self.iteration + 1):
            tmp_best_one, tmp_best_score = self.ga()
            self.iter_x.append(i)
            self.iter_y.append(1. / tmp_best_score)
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                BEST_LIST = tmp_best_one
            self.best_record.append(1. / best_score)
            # print(f"Iteration {i}, Best Path Length: {1. / best_score}")
        return self.location[BEST_LIST], 1. / best_score



# 用于计算路径长度-----后续用精确求解器进行求解
def TSP(path):
    data = np.array(path)
    Best, Best_path = math.inf, None

    model = GA(num_city=data.shape[0], num_total=25, iteration=500, data=data.copy())
    path_tsp, path_len = model.run()

    if path_len < Best:
        Best = path_len
        Best_path = path

    return path_len


#对监控区域进行划分，即将监控区域划分成一个一个的小正方形
def partition_region(width, height, d):
    """# 将监测区域划分成M个相同的小正方形
    # width：监测区域矩形的宽度
    # height：监测区域矩形的高度
    # d：要划分的小正方形的边长
    """
    # 计算在x和y方向上的小正方形数量
    num_squares_x = int(np.ceil(width / d))
    num_squares_y = int(np.ceil(height / d))

    # 存储小正方形中心的坐标
    centers = []

    for i in range(num_squares_x):
        for j in range(num_squares_y):
            # 计算每个小正方形的中心坐标，中心坐标根据传参d进行计算
            center_x = i * d + d / 2
            center_y = j * d + d / 2

            # 对边界区域进行处理，如果中心点在矩形区域内，添加到列表中
            if center_x <= width and center_y <= height:
                centers.append((center_x, center_y))

    return centers


def is_within_radius(point, center, radius):
    """
    判断传感器节点是否在小正方形中心的覆盖半径内
    point: 传感器节点的坐标
    center: 小正方形中心的坐标
    radius: 覆盖半径
    """
    # 计算点和中心之间的欧几里得距离
    distance = np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
    return distance <= radius

def find_covered_sensors(points_with_data, centers, radius):
    """
    计算每个小正方形中心能够覆盖的传感器节点
    points_with_data: 传感器节点的列表，包含坐标和数据量
    centers: 小正方形中心的列表
    radius: 小正方形的覆盖半径
    """
    coverage_map = {}

    for center in centers:
        covered_points = []
        for point, data in points_with_data:
            if is_within_radius(point, center, radius):
                covered_points.append((point, data))
        coverage_map[center] = covered_points

    return coverage_map




# 更新后的覆盖关系
def remove_covered_sensors(coverage_map, chosen_center):
    """
    删除覆盖集合中指定小正方形中心及其覆盖的传感器节点。

    参数:
    coverage_map (dict): 每个小正方形的中心位置与无人机能够覆盖的传感器节点集合一一对应的字典
    chosen_center (tuple): 选择的作为路径点的小正方形中心

    返回:
    dict: 删除指定中心及其覆盖传感器节点后的字典
    """
    if chosen_center not in coverage_map:
        return coverage_map

    # 获取选择中心覆盖的传感器节点
    covered_sensors = set(coverage_map[chosen_center])

    # 删除指定的中心及其覆盖传感器节点
    del coverage_map[chosen_center]

    # 从其他中心的覆盖传感器节点中删除被覆盖的传感器
    for center, sensors in coverage_map.items():
        coverage_map[center] = [sensor for sensor in sensors if sensor not in covered_sensors]

    return coverage_map



# 权重计算公式
def calculate_rho(V_total_data, v, t_total, eta_h, TSP_current, TSP_previous, eta_t):
    """
    参数：
    V_total_data：某个悬停点收集的数据总量
    v：无人机的飞行速度
    t_total：某个悬停点的悬停时间
    eta_h：悬停时的能量消耗速率
    TSP_current：加入某点后的路径长度
    TSP_previous：加入某点前的路径长度
    eta_t：无人机飞行时的能量消耗速率
    """
    # 计算分子
    numerator = V_total_data * v
    # 计算分母
    denominator = t_total * eta_h * v + (TSP_current - TSP_previous) * eta_t

    # 计算rho(s_j)
    rho = numerator / denominator
    return rho




# 计算在一个潜在悬停点收集数据时所需要的时间。在每次选择下一个悬停点之后的计算中，需要先根据更新后的覆盖关系 再计算每个悬停点的悬停时间和收集的数据量
def hover_time_at_center(center, coverage_map, data_amounts, B):
    """
    计算在指定的小正方形中心位置悬停收集数据的时间和总数据量。

    参数:
    center (tuple): 指定的小正方形中心位置，例如 (cx, cy)
    coverage_map (dict): 每个小正方形中心位置与其能够覆盖的传感器节点集合，例如 {(cx1, cy1): [(x1, y1), (x2, y2), ...], ...}
    data_amounts (dict): 每个传感器节点的数据量，例如 {(x1, y1): Dv1, (x2, y2): Dv2, ...}
    B (float): 数据传输速率

    返回:
    float: 在指定中心位置悬停的时间
    float: 在指定中心位置收集的数据总量
    """
    if center not in coverage_map:
        return 0, 0

    sensors = coverage_map[center]

    # 找到覆盖范围内传感器的最大数据量
    if sensors:
        max_data = max(data_amounts[sensor] for sensor in sensors)
        total_data = sum(data_amounts[sensor] for sensor in sensors)
    else:
        max_data = 0
        total_data=0

    # 悬停时间是最大数据量除以传输速率
    hover_time = max_data / B if max_data > 0 else 0
    return hover_time, total_data




def uav_path_planning(coverage_map,data_amounts):
    """
    覆盖区域具有重叠的全数据收集
    参数：
    sensor_positions：聚合传感器节点的集合
    data_amounts：每个聚合传感器节点的数据量
    width：区域的宽度
    height：区域的高度
    d：小正方形的边长
    coverage_radius：无人机的覆盖半径
    B：数据传输速率
    eta_h：悬停时的能量消耗速率
    eta_t：无人机飞行时的能量消耗速率
    v：无人机的飞行速度
    energy_constraint：能量限制
    输出：找到一条路径，使得能量受限的无人机尽可能多的收集数据
    """


    # 无人机的初始位置，可以进行适当的调整
    S0 = [(0, 0)]
    # 将S0复制到path中作为路径的起点
    path = S0.copy()
    j = 1
    TSP_Sj_minus_1 = 0  # 初始路径长度为0
    # # 调用find_covered_sensors 函数，计算在每个潜在悬停点无人机能够覆盖的传感器节点
    # coverage_map = find_covered_sensors(sensor_positions, S, coverage_radius)

    #总的收集的数据量
    data=0
    #总的悬停时间
    total_hover_time=0

    while True:
        # 计算当前路径的能量消耗
        ##？？？？？？？？？？？-----------------------------------------需要对energy_sum重新进行计算，在每次计算悬停时间时需要对覆盖关系进行更新
        #energy_sum = sum(hover_time_at_center(p, coverage_map, data_amounts, B)[0] * eta_h for p in path) + TSP(path) * eta_t / v
        energy_sum = total_hover_time * eta_h + TSP(path) * eta_t / v

        if energy_sum > energy_constraint:
            break

        max_rho = -np.inf
        next_hover_point = None
        hover_time_next = 0
        total_data_next = 0


        #选择一个比率最大的悬停点
        for sj in S:
            if sj not in path:
                hover_time, total_data = hover_time_at_center(sj, coverage_map, data_amounts, B)
                TSP_current = TSP(path + [sj])
                rho_sj = calculate_rho(total_data, v, hover_time, TSP_current, TSP_Sj_minus_1, eta_h, eta_t)

                if rho_sj > max_rho:
                    max_rho = rho_sj
                    next_hover_point = sj
                    hover_time_next = hover_time
                    total_data_next = total_data



        if next_hover_point:
            # 检查加入该点后的能量消耗
            # 更新收集的数据量和悬停时间
            data += total_data_next
            total_hover_time += hover_time_next

            energy_check = total_hover_time *eta_h + TSP(path + [next_hover_point]) * eta_t / v
            if energy_check <= energy_constraint:
                path.append(next_hover_point)
                TSP_Sj_minus_1 = TSP(path)
                # 从S中删除该悬停点
                S.remove(next_hover_point)
                # 更新覆盖关系
                coverage_map = remove_covered_sensors(coverage_map, next_hover_point)
                j += 1
            else:
                data -= total_data_next
                total_hover_time -= hover_time_next
                break
        else:
            break

    return path,data

# 全局变量
v = 1  # 无人机的飞行速度
eta_h = 150  # 悬停时的能量消耗速率
eta_t = 100  # 飞行时的能量消耗速率
coverage_radius = 49  # 无人机的覆盖半径
width = 1000  # 监测区域矩形的宽度
height = 1000  # 监测区域矩形的高度
d = 10  # 小正方形的边长
B = 150  # 数据传输速率
energy_constraint = 300000  # 能量约束

# 调用函数生成点  生成的点范围在0-1000，并且每个点的数据量在100MB-1000MB之间随机生成
random_points_with_data = generate_random_points_with_data()
# 将生成的数据转化为字典，以坐标为键，数据量为值
data_amounts = {point: data for point, data in random_points_with_data}

# 调用partition_region 函数将监测区域划分成M个相同的小正方形，返回小正方形的中心坐标列表
S = partition_region(width, height, d)

# 调用 find_covered_points 函数，计算每个小正方形中心能够覆盖的传感器节点
coverage_map = find_covered_sensors(random_points_with_data, S, coverage_radius)



#
path_planning,data=uav_path_planning(coverage_map,data_amounts)


print(path_planning)
print("==================================")
print(data)





# # 示例输出：打印每个小正方形中心及其覆盖的传感器节点数量
# for center, covered_points in covered_points_by_center.items():
#     print(f"中心点 {center} 覆盖的传感器节点数: {len(covered_points)}")










