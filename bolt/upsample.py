import numpy as np
import matplotlib.pyplot as plt
import sys
import bolt.octree as octree
import open3d as o3d
import bolt.smoothing as smoothing
import time
import bolt.distances as distances
from bolt.visualize import visualize_numpy_pointcloud_o3d, visualize_point_cloud, visualize_o3d_pointcloud_o3d
from tqdm import tqdm
from bolt.utils import knn_smoothing, bilateral_smoothing
import subprocess

def initial_sample(point_cloud_path, n, visualize=False):
    point_cloud = np.load(point_cloud_path)
    point_cloud_sample = point_cloud[np.random.choice(point_cloud.shape[0], 1024, replace=False), :]
    if visualize:
        visualize_numpy_pointcloud_o3d(point_cloud_sample)
    return point_cloud_sample

def octree_initial(point_cloud_sample, N=1, visualize=False):
    oct = octree.Octree()
    oct.create(point_cloud_sample)
    new_pts = []
    for _ in range(N):
        for point in tqdm(point_cloud_sample):
            parent = oct.deepest_parent(point)
            parent_dims = np.array([parent.width, parent.height, parent.depth])
            direction = np.random.randint(0, 8)
            direction = octree.DIRECTION_TO_COORDINATES[direction]
            magnitude = np.array([np.random.uniform(low=0, high=i) for i in parent_dims])
            new_pts.append(point + np.multiply(magnitude, direction))
            oct.insert(oct.tree, point + np.multiply(magnitude, direction))
    new_pts = np.array(new_pts)
    if visualize:
        visualize_numpy_pointcloud_o3d(np.vstack((new_pts, point_cloud_sample)))
    return new_pts

def random_sampling(point_cloud, n, visualize=False):
    max_x, max_y, max_z = np.max(point_cloud, axis=0)
    min_x, min_y, min_z = np.min(point_cloud, axis=0)
    np.random.seed(0)
    new_points = np.array([[np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y), np.random.uniform(min_z, max_z)] for _ in range(n)])
    if visualize:
        visualize_numpy_pointcloud_o3d(np.vstack((new_points, point_cloud)))
    return new_points

def bolt(point_cloud_source, **kwargs):
    n_initial_sample = kwargs.get("n_initial_sample", 1024)
    k_nearest_neighbours = kwargs.get("k_nearest_neighbours", 30)
    sigma_d = kwargs.get("sigma_d", 0.1)
    sigma_n = kwargs.get("sigma_n", 0.1)
    n_iter = kwargs.get("n_iter", 1)
    n_final_sample = kwargs.get("n_final_sample", 2048)
    visualize = kwargs.get("visualize", False)
    smoothing_type = kwargs.get("smoothing_type", "bilateral")
    random_sample = kwargs.get("random_sample", False)

    point_cloud_sample = initial_sample(point_cloud_source, n_initial_sample, visualize=visualize)

    if random_sample:
        new_pts = random_sampling(point_cloud_sample, n_final_sample - n_initial_sample, visualize=visualize)
    else:
        N = int(np.ceil(n_final_sample / n_initial_sample)) - 1
        new_pts = octree_initial(point_cloud_sample, N=N, visualize=visualize)

    if smoothing_type == "knn":
        point_cloud_sample = knn_smoothing(point_cloud_sample, new_pts, k=k_nearest_neighbours, visualize=visualize)
    elif smoothing_type == "bilateral":
        point_cloud_sample = bilateral_smoothing(point_cloud_sample, new_pts, k=k_nearest_neighbours, sigma_d=sigma_d, sigma_n=sigma_n, n_iter=n_iter, visualize=visualize)

    return point_cloud_sample
