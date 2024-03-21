import torch
from model import BoltNet 
import torch.optim as optim
from loss import BoltLoss
import sys
import argparse
import numpy as np
from model import BoltNet
import open3d as o3d

def initial_sample(point_cloud_path, n, visualize=False):
    point_cloud = np.load(point_cloud_path)
    point_cloud_sample = point_cloud[np.random.choice(point_cloud.shape[0], 1024, replace=False), :]
    normals = estimate_normals(point_cloud_sample)
    return point_cloud_sample, normals

def estimate_normals(point_cloud, k=30):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    normals = np.asarray(pcd.normals)
    return normals

def voxelize(point_cloud, voxel_size=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    voxels = voxel_grid.get_voxels()
    indices = np.stack(list(vx.grid_index for vx in voxels))
    x, y, z = indices[:, 0], indices[:, 1], indices[:, 2]
    voxel_tensor = torch.zeros((np.max(x) + 1, np.max(y) + 1, np.max(z) + 1), dtype=torch.float32)
    for i in range(indices.shape[0]):
        voxel_tensor[x[i], y[i], z[i]] = 1
    return voxel_tensor

if __name__ == '__main__':
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
    parser.add_argument("--mls", help="Use mls sampling", action="store_true")
    parser.add_argument("-o", "--output_path", type=str, help="Output path", default="./bolt_upsampled.npy")
    parser.add_argument("-p", "--model_path", type=str, help="Path to the model", default="./model.pth")
    args = parser.parse_args()
    
    kwargs = vars(args)
    source_path = kwargs.pop('point_cloud_source')
    pc_sample, normals = initial_sample(source_path, kwargs.pop('n_initial_sample'), kwargs.pop('visualize'))
    voxel_grid = voxelize(pc_sample)
    
   
