import numpy as np
import matplotlib.pyplot as plt
import sys
import octree
import open3d as o3d
import smoothing
import time
import distances

'''
todo:
- add chamfer distance
- add hausdorff distances
- CHANGE BILATERAL FILTERING TO ONLY SMOOTH THE POINTS THAT ARE NOT IN THE ORIGINAL POINT CLOUD!!!

'''

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
visualize_numpy_pointcloud_o3d(point_cloud_sample)
#visualize_numpy_pointcloud_o3d(point_cloud_sample)

oct = octree.Octree()
oct.create(point_cloud_sample)
N = 2
new_pts = []
for i in range(N):
    for point in point_cloud_sample:
        parent = oct.deepest_parent(point)
        parent_dims = np.array([parent.width, parent.height, parent.depth])
        direction = np.random.randint(0, 8)
        direction = octree.DIRECTION_TO_COORDINATES[direction]
        magnitude = np.array([np.random.uniform(low=0, high=i) for i in parent_dims])
        new_pts.append(point + np.multiply(magnitude, direction))
        oct.insert(oct.tree, point + np.multiply(magnitude, direction))

new_pts = np.array(new_pts)

# Comute chamfer distance
point_cloud_gt = point_cloud[np.random.choice(point_cloud.shape[0], 2048, replace=False), :]
distance = distances.chamfer_distance(point_cloud_sample, point_cloud_gt)
print(distance)

# to do:
# modify the bilateral smoothing to only smooth the points that are not in the original point cloud !!!!!!

start = time.time()
point_cloud_sample = smoothing.bilateral_smoothing(point_cloud_sample, new_pts, k=30, sigma_d=0.1, sigma_n=0.1)
end = time.time()
print("Bilateral smoothing Time: ", end - start)
print("Bilateral smoothing shape: ", point_cloud_sample.shape)
visualize_numpy_pointcloud_o3d(point_cloud_sample)


