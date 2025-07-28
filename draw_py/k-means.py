import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def generate_sensor_nodes(num_nodes, area_size):
    """
    随机生成传感器节点的坐标。
    """
    return np.random.rand(num_nodes, 2) * area_size


def filter_clusters_by_communication_range(sensor_nodes, labels, communication_range):
    """
    过滤不满足通信范围约束的聚类。
    """
    clusters = {}
    for label in np.unique(labels):
        cluster_indices = np.where(labels == label)[0]
        cluster_nodes = sensor_nodes[cluster_indices]

        # 检查是否满足通信范围约束
        valid = True
        for i in range(len(cluster_nodes)):
            for j in range(i + 1, len(cluster_nodes)):
                distance = np.linalg.norm(cluster_nodes[i] - cluster_nodes[j])
                if distance > communication_range:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            clusters[label] = cluster_indices.tolist()

    # 如果没有任何有效聚类，返回空字典
    return clusters


def assign_unclassified_nodes(sensor_nodes, valid_clusters, communication_range):
    """
    将未分类节点重新分配到最近的有效聚类，或者作为单独的聚类。
    """
    if not valid_clusters:
        # 如果没有任何有效聚类，为每个节点创建单独的聚类
        valid_clusters = {i: [i] for i in range(len(sensor_nodes))}
        return valid_clusters

    valid_indices = [idx for indices in valid_clusters.values() for idx in indices]
    unclassified_indices = np.setdiff1d(range(len(sensor_nodes)), valid_indices)
    unclassified_nodes = sensor_nodes[unclassified_indices]

    # 遍历每个未分类节点，尝试分配到最近的有效聚类
    for idx, node in zip(unclassified_indices, unclassified_nodes):
        assigned = False
        for cluster_id, cluster_indices in valid_clusters.items():
            cluster_nodes = sensor_nodes[cluster_indices]
            distances = np.linalg.norm(cluster_nodes - node, axis=1)
            if np.min(distances) <= communication_range:
                valid_clusters[cluster_id].append(idx)
                assigned = True
                break

        # 如果无法分配，将其作为新聚类
        if not assigned:
            new_cluster_id = max(valid_clusters.keys()) + 1 if valid_clusters else 0
            valid_clusters[new_cluster_id] = [idx]

    return valid_clusters


def find_main_sensor(sensor_nodes, cluster_indices):
    """
    为每个聚类选择主传感器节点（聚类质心最近的节点）。
    """
    cluster_nodes = sensor_nodes[cluster_indices]
    centroid = np.mean(cluster_nodes, axis=0)
    distances = np.linalg.norm(cluster_nodes - centroid, axis=1)
    main_sensor_index = cluster_indices[np.argmin(distances)]
    return main_sensor_index


def plot_clusters(sensor_nodes, valid_clusters, main_sensors):
    """
    可视化聚类结果。
    """
    plt.figure(figsize=(8, 8))
    colors = plt.cm.get_cmap("tab10", len(valid_clusters))

    # 绘制有效聚类
    for cluster_id, indices in valid_clusters.items():
        cluster_nodes = sensor_nodes[indices]
        plt.scatter(cluster_nodes[:, 0], cluster_nodes[:, 1], label=f"Cluster {cluster_id}", color=colors(cluster_id))
        # 标记主传感器节点
        main_sensor = sensor_nodes[main_sensors[cluster_id]]
        plt.scatter(main_sensor[0], main_sensor[1], color="black", marker="x", s=100)

    plt.title("K-means Clustering with Communication Range Constraints")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    num_nodes = 100  # 传感器节点数量
    area_size = 1000.0  # 传感器部署区域的边长
    k_clusters = 39  # 聚类数量
    communication_range = 100.0  # 传感器的通信范围

    # 1. 随机生成传感器节点
    sensor_nodes = generate_sensor_nodes(num_nodes, area_size)

    # 2. 使用 K-means 聚类
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    labels = kmeans.fit_predict(sensor_nodes)

    # 3. 过滤不满足通信范围的聚类
    valid_clusters = filter_clusters_by_communication_range(sensor_nodes, labels, communication_range)

    # 4. 将未分类节点重新分配
    valid_clusters = assign_unclassified_nodes(sensor_nodes, valid_clusters, communication_range)

    # 5. 找到每个聚类的主传感器节点
    main_sensors = {}
    for cluster_id, cluster_indices in valid_clusters.items():
        main_sensors[cluster_id] = find_main_sensor(sensor_nodes, cluster_indices)

    # 6. 输出每个聚类的主传感器节点
    print("Main sensors for each cluster:")
    for cluster_id, main_sensor_index in main_sensors.items():
        print(f"Cluster {cluster_id}: Node {main_sensor_index} at {sensor_nodes[main_sensor_index]}")

    # 7. 可视化聚类结果
    plot_clusters(sensor_nodes, valid_clusters, main_sensors)


if __name__ == "__main__":
    import os
    os.environ["OMP_NUM_THREADS"] = "1"  # 避免 KMeans 的内存泄漏问题
    main()
