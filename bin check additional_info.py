from dgl.data.utils import load_graphs


# 加载图数据
graphfile = load_graphs("00000121_p.bin", [0])  # glist will be [g1]
graph = graphfile[0][0]
additional_info=graphfile[1]

# 打印图的基本信息
print(f"Number of nodes: {graph.num_nodes()}")
print(f"Number of edges: {graph.num_edges()}")

# 打印节点和边的特征
print("Node features:")
for key, value in graph.ndata.items():
    print(f"{key}: {value}")
    print(f"{key}: {value.shape}")

print("Edge features:")
for key, value in graph.edata.items():
    print(f"{key}: {value}")
    print(f"{key}: {value.shape}")

# 打印节点和边的ID
print("Node IDs:")
print(graph.nodes())

print("Edge IDs:")
print(graph.edges())

# 打印邻接矩阵
print("Adjacency matrix:")
print(graph.adjacency_matrix().to_dense())

# 打印每个节点的入度和出度
print("In-degrees:")
print(graph.in_degrees())

print("Out-degrees:")
print(graph.out_degrees())

# 打印标签信息
print("\nLabel information:")
if "commands_primitive" in additional_info:
    print(f"commands_primitive: {additional_info['commands_primitive']}")
else:
    print("commands_primitive not found in label_dict.")

if "args_primitive" in additional_info:
    print(f"args_primitive: {additional_info['args_primitive']}")
else:
    print("args_primitive not found in label_dict.")

if "commands_feature" in additional_info:
    print(f"commands_feature: {additional_info['commands_feature']}")
else:
    print("commands_feature not found in label_dict.")

if "args_feature" in additional_info:
    print(f"args_feature: {additional_info['args_feature']}")
else:
    print("args_feature not found in label_dict.")

if "edges_path" in additional_info:
    print(f"edges_path: {additional_info['edges_path']}")
    print(f"edges_path: {additional_info['edges_path'].shape}")
else:
    print("edges_path not found in label_dict.")

if "spatial_pos" in additional_info:
    print(f"spatial_pos: {additional_info['spatial_pos']}")
    print(f"spatial_pos: {additional_info['spatial_pos'].shape}")
else:
    print("spatial_pos not found in label_dict.")

if "d2_distance" in additional_info:
    print(f"d2_distance: {additional_info['d2_distance']}")
    print(f"d2_distance: {additional_info['d2_distance'].shape}")

else:
    print("d2_distance not found in label_dict.")

if "angle_distance" in additional_info:
    print(f"angle_distance: {additional_info['angle_distance']}")
    print(f"angle_distance: {additional_info['angle_distance'].shape}")
else:
    print("angle_distance not found in label_dict.")