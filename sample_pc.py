import numpy as np
import matplotlib.pyplot as plt
import sys
import octree
import open3d as o3d
import smoothing
import time
import distances
from visualize import visualize_numpy_pointcloud_o3d, visualize_point_cloud, visualize_o3d_pointcloud_o3d
import argparse
from tqdm import tqdm
from utils import knn_smoothing, bilateral_smoothing
import subprocess

'''
todo:
- add chamfer distance
- add hausdorff distances
- CHANGE BILATERAL FILTERING TO ONLY SMOOTH THE POINTS THAT ARE NOT IN THE ORIGINAL POINT CLOUD!!!

'''

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

def mls_sampling(point_cloud_sample_path, k, polynomial_order, output_path, visualize=False):
    subprocess.run(["pcl_src/build/mls", point_cloud_sample_path, str(k), str(polynomial_order), output_path])
    point_cloud_sample = o3d.io.read_point_cloud(output_path)
    if visualize:
        visualize_o3d_pointcloud_o3d(point_cloud_sample)
    point_cloud_sample = np.asarray(point_cloud_sample.points)
    print("MLS shape: ", point_cloud_sample.shape)
    return point_cloud_sample 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--point_cloud_source", type=str, help="Path to the point cloud")
    parser.add_argument("-n", "--n_initial_sample", type=int, help="Number of initial samples", default=1024)
    parser.add_argument("-k", "--k_nearest_neighbours", type=int, help="Number of nearest neighbours", default=30)
    parser.add_argument("-d", "--sigma_d", type=float, help="Sigma_d", default=0.1)
    parser.add_argument("-r", "--sigma_n", type=float, help="Sigma_n", default=0.1)
    parser.add_argument("-i", "--n_iter", type=int, help="Number of iterations", default=1)
    parser.add_argument("-N", "--n_final_sample", type=int, help="Number of final samples", default=2048)
    parser.add_argument("-v", "--visualize", help="Visualize the point cloud", action="store_true")
    parser.add_argument("-t", "--smoothing_type", type=str, help="Type of smoothing", default="bilateral")
    parser.add_argument("-m", "--metric", type=str, help="Type of metric", default="chamfer")
    parser.add_argument("--random_sampling", help="Random sampling", action="store_true")
    args = parser.parse_args()


    point_cloud_sample = initial_sample(args.point_cloud_source, args.n_initial_sample, visualize=args.visualize)
    o3d.io.write_point_cloud("tmp/pointcloud.pcd", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud_sample)), format="pcd")
    N = int(np.ceil(args.n_final_sample / args.n_initial_sample)) - 1

    if args.random_sampling:
        new_pts = random_sampling(point_cloud_sample, args.n_final_sample - args.n_initial_sample, visualize=args.visualize)
    else:
        new_pts = octree_initial(point_cloud_sample, N=N, visualize=args.visualize)

    if args.smoothing_type == "knn":
        point_cloud_sample = knn_smoothing(point_cloud_sample, new_pts, k=args.k_nearest_neighbours, visualize=args.visualize)
    elif args.smoothing_type == "bilateral":
        point_cloud_sample = bilateral_smoothing(point_cloud_sample, new_pts, k=args.k_nearest_neighbours, sigma_d=args.sigma_d, sigma_n=args.sigma_n, n_iter=args.n_iter, visualize=args.visualize)
    else:
        print("Invalid smoothing type, please choose either 'knn' or 'bilateral'")
        sys.exit(1)
    
    mls_sampling("tmp/pointcloud.pcd", 20, 2, "tmp/mls.pcd", visualize=args.visualize)

