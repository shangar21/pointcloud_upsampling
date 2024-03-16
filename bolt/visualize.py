import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def visualize_point_cloud(point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    plt.show()

def visualize_numpy_pointcloud_o3d(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd])

def visualize_o3d_pointcloud_o3d(point_cloud):
    o3d.visualization.draw_geometries([point_cloud])


