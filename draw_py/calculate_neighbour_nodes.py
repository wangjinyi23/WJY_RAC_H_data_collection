import numpy as np
import matplotlib.pyplot as plt


transmission_rate = 1 * 10 ** 6  # Transmission rate in bps (1 Mbps)
data_packet_length = 1 * 10 ** 9  # Data packet length in bits

def euclidean_distance(node1, node2):
    return np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

def find_neighbors_with_distances(nodes, radius):
    neighbors = {}
    for i, node1 in enumerate(nodes):
        neighbor_distances = {}
        for j, node2 in enumerate(nodes):
            if euclidean_distance(node1, node2) <= radius:
                neighbor_distances[j] = np.sqrt(euclidean_distance(node1, node2)**2+50*50)
        neighbors[i] =neighbor_distances
    return neighbors



#计算传输时间

def find_neighbors_with_trasmission_time(nodes, radius):
    transmission_time = {}
    for i, node1 in enumerate(nodes):
        neighbor_transmission_time = {}
        for j, node2 in enumerate(nodes):
            if euclidean_distance(node1, node2) <= radius:
                distance_uav_node=np.sqrt(euclidean_distance(node1, node2)**2+50*50)
                neighbor_transmission_time[j] =data_packet_length/transmission_rate/distance_uav_node
        transmission_time[i] = neighbor_transmission_time
    return transmission_time



# 在100*100的区域中随机产生100个点
num_nodes = 100
nodes = np.random.rand(num_nodes, 2) * 2000  # Random positions in a 100x100m plane

# 无人机覆盖范围半径
coverage_radius = 1000  # Adjust this value as needed

# Find neighbors for each node with distances
neighbor_sets_with_distances = find_neighbors_with_distances(nodes, coverage_radius)

#计算每个节点的邻居节点集合到该节点上方50m处传输数据所需要的时间
neighbors_sets_with_trasmission_time = find_neighbors_with_trasmission_time(nodes, coverage_radius)


# #打印每个节点的邻居节点
# for i, neighbor_indices in neighbor_sets.items():
#     print(f"Node {i} neighbors: {neighbor_indices}")


# Print distances for each node's neighbors
for i, neighbor_distances in neighbor_sets_with_distances.items():
    print(f"Node {i} neighbor distances: {neighbor_distances}")


#打印每个结点到悬停点传输数据所需要的时间
for i, neighbor_transmission_time in neighbors_sets_with_trasmission_time.items():
    print(f"Node {i} neighbor_transmission_time: {neighbor_transmission_time}")



# Plot the nodes
plt.figure(figsize=(8, 8))
plt.scatter(nodes[:, 0], nodes[:, 1], color='blue', label='Nodes')

# Plot the neighbors for each node
for i, neighbor_distances in neighbor_sets_with_distances.items():
    for neighbor_index, distance in neighbor_distances.items():
        plt.plot([nodes[i, 0], nodes[neighbor_index, 0]], [nodes[i, 1], nodes[neighbor_index, 1]], color='red', linestyle='--')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Nodes and Their Neighbors')
plt.legend()
plt.grid(True)
plt.show()
