#Floyd最短路径算法,原文如下
# Python program for Floyd Warshall Algorithm to compute the shortest paths
# between any pair of B-rep faces on the face adjacency graph and take the number
# of edges on this path as the shortest distance.
# Define infinity as a large enough value. This value will be
# used for vertices not connected to each other

def floyd_warshall(graph):
    """
    return：
    - result: 三维矩阵，存储最短路径的边索引
    """
    graph.edge_index = {edge: idx for idx, edge in enumerate(graph.edges)}
    V = len(graph.nodes)
    INF = float('inf')
    dist = [[INF] * V for _ in range(V)]
    next_edge = [[[] for _ in range(V)] for _ in range(V)]  # 存储边序列

    # 初始化直接相连的边
    for u, v in graph.edges:
        if graph.has_edge(u, v):
            dist[u][v] = 1  # 邻接边权重为1（每边算作1步）
            next_edge[u][v] = [(u, v)]  # 直接连接的边序列

    # Floyd-Warshall核心逻辑
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_edge[i][j] = next_edge[i][k] + next_edge[k][j]  # 合并路径

    # 找出最长路径的长度
    max_length = 0
    for i in range(V):
        for j in range(V):
            path = next_edge[i][j]
            max_length = max(max_length, len(path))

    result = [[[-1] * max_length for _ in range(V)] for _ in range(V)]

    for i in range(V):
        for j in range(V):
            path = next_edge[i][j]
            for k, edge in enumerate(path):
                if k < max_length:
                    result[i][j][k] = graph.edge_index[edge]

    return result
