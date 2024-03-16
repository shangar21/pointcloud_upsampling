import numpy as np
import open3d as o3d
from tqdm import tqdm
import smoothing
from visualize import visualize_numpy_pointcloud_o3d
import time

def estimate_normals(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    return normals

def knn_smoothing(point_cloud_sample, new_pts, k=5, visualize=False):
    start = time.time()
    point_cloud_sample = smoothing.nearest_neighbours_smoothing(point_cloud_sample, new_pts, k=k)
    end = time.time()
    print("Knn smoothing Time: ", end - start)
    print("Knn smoothing shape: ", point_cloud_sample.shape)
    if visualize:
        visualize_numpy_pointcloud_o3d(point_cloud_sample)
    return point_cloud_sample

def bilateral_smoothing(point_cloud_sample, new_pts, k=30, sigma_d=0.1, sigma_n=0.1, n_iter=1, visualize=False):
    start = time.time()
    point_cloud_sample = smoothing.bilateral_smoothing(point_cloud_sample, new_pts, k=k, sigma_d=sigma_d, sigma_n=sigma_n, n_iter=n_iter)
    end = time.time()
    print("Bilateral smoothing Time: ", end - start)
    print("Bilateral smoothing shape: ", point_cloud_sample.shape)
    if visualize:
        visualize_numpy_pointcloud_o3d(point_cloud_sample)
    return point_cloud_sample

