




## 以调试好-----------------------------------

import random
import math
import numpy as np
import matplotlib.pyplot as plt


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
            print(f"Iteration {i}, Best Path Length: {1. / best_score}")
        return self.location[BEST_LIST], 1. / best_score


# 定义城市的坐标
cities = [
    (64, 96), (80, 39), (69, 23), (72, 42)
    ]



data = np.array(cities)
Best, Best_path = math.inf, None

model = GA(num_city=data.shape[0], num_total=40, iteration=500, data=data.copy())
path, path_len = model.run()

# 打印最优路径长度
if path_len < Best:
    Best = path_len
    Best_path = path

print(f"最优路径长度: {path_len}")
# print(path)




