import torch
import numpy as np
import open3d as o3d
import os

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

class PointCloudDataLoader(torch.utils.data.Dataset):
    def __init__(self, point_cloud_path):
        self.point_cloud_path = point_cloud_path
        self.classes = [i for i in os.listdir(point_cloud_path) if i != 'train']
        self.pc = []
        for i in self.classes:
            self.pc += [os.path.join(point_cloud_path, i, 'train', j) for j in os.listdir(os.path.join(point_cloud_path, i, 'train'))]

    def num_classes(self):
        return len(self.classes)
        
    def __len__(self):
        return len(self.pc)
   
    def __getitem__(self, idx):
        point_cloud_sample, normals = initial_sample(self.pc[idx], 1024, False)
        t = torch.from_numpy(point_cloud_sample).float()
        return t
