import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree 
import open3d as o3d
import bolt.bilateral_smooth_src as bss

def nearest_neighbours_smoothing(point_cloud, new_points, k=5):
    """
    This function performs nearest neighbours smoothing on a point cloud.
    """
    # create KDTree from the point cloud
    tree = KDTree(point_cloud)
        
    # Iterate over each point in the point cloud
    for i in tqdm(range(new_points.shape[0])):
        # get the k nearest neighbours
        indices = tree.query(new_points[i].reshape(1, -1), k=k+1)[1][0]
        # average the k nearest neighbours
        new_points[i] = np.mean(point_cloud[indices], axis=0)

    return np.vstack((point_cloud, new_points))

def regression_plane(neighbours):
    mean = np.mean(neighbours, axis=0)
    centered_neighbours = neighbours - mean
    covariance_matrix = np.cov(centered_neighbours, rowvar=False)
    ev, evec = np.linalg.eig(covariance_matrix)
    normal = evec[:, np.argmin(ev)]
    return mean, normal

def bilateral_smoothing(point_cloud, new_points, k=5, sigma_d=0.1, sigma_n=0.1, n_iter=1):
    # create KDTree from the point cloud
    tree = KDTree(point_cloud)
    # Iterate over each point in the point cloud
    for _ in range(n_iter):
        for i in tqdm(range(new_points.shape[0])):
            # get the k nearest neighbours
            indices = tree.query(new_points[i].reshape(1, -1), k=k+1)[1][0]
            neighbours = point_cloud[indices]
            mean, normal = regression_plane(neighbours)
            sum_of_weights = 0
            delta_p = 0
            for q in neighbours:
                d_d = np.linalg.norm(q - new_points[i])
                d_n = np.dot(normal, q - new_points[i])
                w = np.exp(-d_d**2/(2*sigma_d**2) - d_n**2/(2*sigma_n**2))
                delta_p = delta_p + (w * d_n)
                sum_of_weights = sum_of_weights + w
            new_points[i] = new_points[i] + ((delta_p / sum_of_weights ) * normal)
    return np.vstack((point_cloud, new_points))


def bilateral_smooth_cpp(point_cloud, new_points, k=5, sigma_d=0.1, sigma_n=0.1, n_iter=1):
    print(point_cloud)
    result = bss.bilateral_smooth(point_cloud, new_points, k, sigma_d, sigma_n, n_iter)
    return np.vstack((point_cloud, result))
