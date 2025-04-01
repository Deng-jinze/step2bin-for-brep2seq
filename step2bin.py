import argparse
import math
import pathlib
import dgl
import numpy as np
import torch
from OCC.Core.STEPControl import STEPControl_Reader
from dgl.backend.backend import dtype
from occwl.graph import face_adjacency
from occwl.io import load_step
from occwl.uvgrid import ugrid, uvgrid
from tqdm import tqdm
from multiprocessing.pool import Pool
from itertools import repeat
from feature.face_type import recognise_face_type
from feature.face_area import compute_face_area
#from feature.special_pos import ask_face_centroid
from feature.d2_distance import calculate_d2_distance
from feature.angle_distance import calculate_angle
from feature.edge_path import floyd_warshall
from feature.get_face import get_faces_from_solid
import signal


def build_graph(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    # Build face adjacency graph with B-rep entities as node and edge features
    # 构建一个以边界表示（B-rep）实体作为节点和边特征的面邻接图
    # solid 是一个表示三维实体的对象，face_adjacency 函数用于根据这个实体构建面邻接图
    graph = face_adjacency(solid)
    #初始化对应的空列表，用于存储特征
    graph_face_feat = []
    graph_edge_feat = []
    face_areas = []
    face_types = []

    for face_idx in graph.nodes:
        # 根据节点索引从图中获取对应的面对象
        face = graph.nodes[face_idx]["face"]
        # 使用 uvgrid 函数计算面上的点的 UV 网格，method="point" 表示计算点的坐标
        # num_u 和 num_v 分别指定了 UV 网格在 u 和 v 方向上的采样点数
        points = uvgrid(
            face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        # 计算面上的法线的 UV 网格
        normals = uvgrid(
            face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        # 计算面上的可见性状态的 UV 网格
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        # 创建一个掩码数组，用于标记哪些点是在内部或边界上
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # 沿着最后一个维度将点坐标、法线和掩码数组连接起来，形成面的特征张量
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        # 将当前面的特征添加到列表中
        graph_face_feat.append(face_feat)
    # 将存储面特征的列表转换为 numpy 数组
    graph_face_feat = np.asarray((graph_face_feat),dtype=np.float32)

    # 初始化一个空列表，用于存储每条边的特征
    # 遍历图中的每条边
    for edge_idx in graph.edges:
        # 根据边的索引从图中获取对应的边对象
        edge = graph.edges[edge_idx]["edge"]
        # 检查边是否有曲线表示，如果没有，则跳过这条边
        if not edge.has_curve():
            continue
        # 使用 ugrid 函数计算边上的点的 U 网格，method="point" 表示计算点的坐标
        # num_u 指定了 U 网格在 u 方向上的采样点数
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        # 计算边上的切线的 U 网格
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        #根据数据集中bin文件的读取内容添加的，原文中没有说明
        specialnum=np.full((points.shape[0], 1), 1.5708,dtype=np.float32)
        # 沿着最后一个维度将点坐标和切线数组连接起来，形成边的特征张量
        edge_feat = np.concatenate((points, tangents,specialnum), axis=-1)
        # 将当前边的特征添加到列表中
        graph_edge_feat.append(edge_feat)
    # 将存储边特征的列表转换为 numpy 数组
    graph_edge_feat = np.asarray((graph_edge_feat),dtype=np.float32)

    # Convert face-adj graph to DGL format
    # 将图中的边转换为列表
    edges = list(graph.edges)
    # 提取边的源节点索引
    src = [e[0] for e in edges]
    # 提取边的目标节点索引
    dst = [e[1] for e in edges]
    # 使用 DGL 库创建一个图对象，指定边的源节点和目标节点索引，以及节点的数量
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    # 将面的特征数组转换为 PyTorch 张量，并赋值给 DGL 图的节点数据
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    # 将边的特征数组转换为 PyTorch 张量，并赋值给 DGL 图的边数据
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)

    # 计算face_area&face_type
    # 获取底层的 OCC 形状
    occ_shape = solid.topods_shape()
    faces = get_faces_from_solid(occ_shape)
    #faces=solid.faces()
    for i, face in enumerate(faces):
        # 计算面的面积
        face_area = compute_face_area(face)
        face_area=round(face_area)
        face_areas.append(face_area)
        # 获取面的类型
        face_type = recognise_face_type(face)
        face_types.append(face_type)
    # 转换格式
    face_areas_np = np.asarray(face_areas,dtype=np.int32)
    face_types_np = np.asarray(face_types,dtype=np.int32)
    # graph.ndata["y"] 通常存储面的面积信息
    dgl_graph.ndata["y"] = torch.from_numpy(face_areas_np)  # face_area[num_nodes]
    # graph.ndata["z"] 通常存储面的类型信息
    dgl_graph.ndata["z"] = torch.from_numpy(face_types_np)  # face_type[num_nodes]

    # 计算d2_distance&angle_distance
    face_points_list = []  # for d2_distance
    face_centroids = []  # for angle_distance
    for face_idx in graph.nodes:
        face = graph.nodes[face_idx]["face"]
        points = uvgrid(
            face, method="point", num_u=512, num_v=512
        )
        flattened_points = points.reshape(-1, 3)
        face_points_list.append(flattened_points)
        centroid = np.mean(points.reshape(-1, 3), axis=0)
        face_centroids.append(centroid)
    num_nodes = len(face_points_list)
    #d2_distance
    angle_distance_array=np.zeros((num_nodes, num_nodes, 32),dtype=np.float32)
    d2_distance_array = np.zeros((num_nodes, num_nodes, 32),dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            d2_dist = calculate_d2_distance(face_points_list[i], face_points_list[j])
            d2_distance_array[i, j, :] = d2_dist
            # angle_distance
            for k in range(num_nodes):
                angle_dist = calculate_angle(face_centroids[i], face_centroids[j], face_centroids[k])
                angle_distance_array[i, j, :] = angle_dist/math.pi
    #angle_distance
    #angle_distance_array = get_angle_distance_matrix(face_centroids, num_samples=64)

    # 计算最短路径
    edge_path= floyd_warshall(graph)
    edge_path_np=np.asarray((edge_path),dtype=np.int32)

    #计算位置信息
    spatial_pos=np.zeros((num_nodes, num_nodes),dtype=np.int32)
    spatial_pos_np=np.asarray(spatial_pos)
    #spatial_pos算不出来了，先全部置0

    # 构建附加信息字典
    additional_info = {
        "edges_path": torch.from_numpy(edge_path_np),
        # edge_input[num_nodes, num_nodes, max_dist, 1, U_grid, pnt_feature]
        "spatial_pos":torch.from_numpy(spatial_pos_np),
        # spatial_pos[num_nodes, num_nodes]
        "d2_distance":torch.from_numpy(d2_distance_array),
        # d2_distance[num_nodes, num_nodes, 32]
        "angle_distance":torch.from_numpy(angle_distance_array)
        # angle_distance[num_nodes, num_nodes, 32]
    }
    return dgl_graph,additional_info

def process_one_file(arguments):
    fn, args = arguments
    fn_stem = fn.stem
    output_path = pathlib.Path(args.output)
    solid = load_step(fn)[0]  # Assume there's one solid per file
    dgl_graph, additional_info = build_graph(  # 接收build_graph函数的两个返回值
        solid, args.curv_u_samples, args.surf_u_samples, args.surf_v_samples
    )
    # 将附加信息添加到图对象的属性字典中
    # 保存图对象
    dgl.data.utils.save_graphs(str(output_path.joinpath(fn_stem + ".bin")), [dgl_graph],additional_info)

def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

#处理所有文件
def process(args):
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    step_files = list(input_path.glob("*.st*p"))
    # for fn in tqdm(step_files):
    #     process_one_file(fn, args)
    pool = Pool(processes=args.num_processes, initializer=initializer)
    try:
        results = list(tqdm(pool.imap(process_one_file, zip(step_files, repeat(args))), total=len(step_files)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    print(f"Processed {len(results)} files.")

def main():
    parser = argparse.ArgumentParser(
        "Convert solid models to face-adjacency graphs with UV-grid features"
    )
    parser.add_argument("input", type=str, help="Input folder of STEP files")
    parser.add_argument("output", type=str, help="Output folder of DGL graph BIN files")
    parser.add_argument(
        "--curv_u_samples", type=int, default=5, help="Number of samples on each curve"
    )
    parser.add_argument(
        "--surf_u_samples",
        type=int,
        default=5,
        help="Number of samples on each surface along the u-direction",
    )
    parser.add_argument(
        "--surf_v_samples",
        type=int,
        default=5,
        help="Number of samples on each surface along the v-direction",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes to use",
    )
    args = parser.parse_args()
    process(args)

if __name__ == "__main__":
    main()