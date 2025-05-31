import os
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import scipy.sparse

def compute_adjacency_matrix_from_weather5k(data_dir, save_path_adj="adjacency_matrix.npy",save_path_loc=None, normalize=True):
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    latitudes, longitudes = [], []

    print("🔍 提取站点坐标中...")
    for file in tqdm(csv_files):
        try:
            df = pd.read_csv(file, nrows=1)
            latitudes.append(df['LATITUDE'].iloc[0])
            longitudes.append(df['LONGITUDE'].iloc[0])
        except Exception as e:
            print(f"⚠️ 跳过文件 {file}: {e}")

    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)
    n = len(latitudes)

    # 保存站点坐标
    coordinates = np.stack((longitudes, latitudes), axis=1)  # Shape: (N, 2), [lon, lat]
    np.save(save_path_loc, coordinates)
    
    print(f"✅ 站点坐标保存到: {save_path_loc}")
    print("📏 计算 Haversine 距离矩阵...")
    distance_matrix = np.zeros((n, n))
    for i in tqdm(range(n)):
        for j in range(n):
            distance_matrix[i, j] = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])

    print("🔗 构造邻接矩阵（距离倒数）...")
    epsilon = 1e-5
    adjacency_matrix = 1 / (distance_matrix + epsilon)

    if normalize:
        print("📐 正在归一化邻接矩阵...")
        D = np.diag(np.sum(adjacency_matrix, axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        adjacency_matrix = D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt
    adjacency_matrix = adjacency_matrix.astype(np.float32)

    np.save(save_path_adj, adjacency_matrix)
    print(f"✅ 邻接矩阵保存到: {save_path_adj}")
    return adjacency_matrix


def compute_sparse_adjacency_matrix_from_weather5k_2(
    data_dir_1,
    data_dir_2,
    save_path_adj="adjacency_matrix_sparse.npz",
    save_path_loc=None,
    k=100,
    normalize=True
):
    def load_coordinates(data_dir):
        latitudes, longitudes = [], []
        npy_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        for file in tqdm(npy_files, desc=f"读取 {data_dir}"):
            try:
                data = np.load(file, allow_pickle=True).item()
                latitudes.append(data['lat'])
                longitudes.append(data['lon'])
            except Exception as e:
                print(f"⚠️ 跳过文件: {file} - {e}")
        return np.array(latitudes), np.array(longitudes)

    # 加载两个目录的坐标数据
    print("📍 正在加载所有站点坐标...")
    lat1, lon1 = load_coordinates(data_dir_1)
    lat2, lon2 = load_coordinates(data_dir_2)

    latitudes = np.concatenate([lat1, lat2])
    longitudes = np.concatenate([lon1, lon2])
    coordinates = np.stack((longitudes, latitudes), axis=1)  # [lon, lat]

    # 保存坐标
    if save_path_loc:
        np.save(save_path_loc, coordinates)
        print(f"✅ 坐标保存到: {save_path_loc}")

    # 使用 sklearn 构建 KNN 邻接矩阵（Haversine距离）
    print(f"🔗 正在构建 K={k} 稀疏邻接矩阵...")
    coords_rad = np.radians(coordinates[:, [1, 0]])  # lat, lon in radians
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='haversine').fit(coords_rad)
    distances, indices = nbrs.kneighbors(coords_rad)
    R = 6371
    distances *= R  # 将弧度转换为公里

    # 构建稀疏邻接矩阵
    row, col, data = [], [], []
    for i in range(len(coordinates)):
        for j in range(1, k + 1):  # 跳过自己
            row.append(i)
            col.append(indices[i][j])
            data.append(1 / (distances[i][j] + 1e-5))  # 倒数作为权重

    A = scipy.sparse.coo_matrix((data, (row, col)), shape=(len(coordinates), len(coordinates)))
    A = A + A.T  # 对称化，确保无向图
    A = A.tocsr()

    if normalize:
        print("📐 正在归一化邻接矩阵...")
        degrees = np.array(A.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1e-10  # 防止除零
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
        D_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
        A = D_inv_sqrt @ A @ D_inv_sqrt  # 对称归一化

    # 保存稀疏矩阵
    scipy.sparse.save_npz(save_path_adj, A)
    print(f"✅ 稀疏邻接矩阵保存到: {save_path_adj}")
    return A
