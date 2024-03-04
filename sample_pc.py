import numpy as np
import matplotlib.pyplot as plt
import sys
import octree
import open3d as o3d
import smoothing
import time

def visualize_point_cloud(point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    plt.show()

def visualize_numpy_pointcloud_o3d(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd])

def estimate_normals(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    normals = np.asarray(pcd.estimate_normals())
    return normals 

def estimate_normals(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    return normals

point_cloud_path = sys.argv[1]
point_cloud = np.load(point_cloud_path)

# sample 1024 random points
point_cloud_sample = point_cloud[np.random.choice(point_cloud.shape[0], 1024, replace=False), :]
#visualize_numpy_pointcloud_o3d(point_cloud_sample)

oct = octree.Octree()
oct.create(point_cloud_sample)
N = 3
for i in range(N):
    for point in point_cloud_sample:
        parent = oct.deepest_parent(point)
        parent_dims = np.array([parent.width, parent.height, parent.depth])
        direction = np.random.randint(0, 8)
        direction = octree.DIRECTION_TO_COORDINATES[direction]
        magnitude = np.array([np.random.uniform(low=0, high=i) for i in parent_dims])
        oct.insert(oct.tree, point + np.multiply(magnitude, direction))


visualize_numpy_pointcloud_o3d(oct.get_points())

start = time.time()
point_cloud_sample = smoothing.nearest_neighbours_smoothing(np.array(oct.get_points()), k=30)
end = time.time()
print("KNN smoothing Time: ", end - start)
visualize_numpy_pointcloud_o3d(point_cloud_sample)

#point_cloud_sample = smoothing.moving_least_squares_smoothing(np.array(oct.get_points()), k=30, degree=2)
#visualize_numpy_pointcloud_o3d(point_cloud_sample)

start = time.time()
point_cloud_sample = smoothing.bilateral_smoothing(np.array(oct.get_points()), k=30, sigma_d=0.1, sigma_n=0.1)
end = time.time()
print("Bilateral smoothing Time: ", end - start)
print("Bilateral smoothing shape: ", point_cloud_sample.shape)
visualize_numpy_pointcloud_o3d(point_cloud_sample)
