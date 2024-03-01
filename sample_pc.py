import numpy as np
import matplotlib.pyplot as plt
import sys
import octree
import open3d as o3d

def visualize_point_cloud(point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    plt.show()

def visualize_numpy_pointcloud_o3d(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd])

point_cloud_path = sys.argv[1]
point_cloud = np.load(point_cloud_path)

print("Point cloud shape: ", point_cloud.shape)

# sample 1024 random points
point_cloud_sample = point_cloud[np.random.choice(point_cloud.shape[0], 1024, replace=False), :]
visualize_numpy_pointcloud_o3d(point_cloud_sample)
print("Point cloud sample shape: ", point_cloud_sample.shape)

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
#        oct.insert(oct.tree, parent.center + np.multiply(parent_dims / 2, direction))

new_points = np.array(oct.get_points())
print("New points shape: ", new_points.shape)
visualize_numpy_pointcloud_o3d(new_points)
