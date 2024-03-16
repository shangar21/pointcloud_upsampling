import numpy as np
from tqdm import tqdm
from scipy.spatial import KDTree 
import open3d as o3d

def nearest_neighbours_smoothing(point_cloud, k=5):
    """
    This function performs nearest neighbours smoothing on a point cloud.
    """
    # create KDTree from the point cloud
    tree = KDTree(point_cloud)
        
    # Iterate over each point in the point cloud
    for i in tqdm(range(point_cloud.shape[0])):
        # get the k nearest neighbours
        indices = tree.query(point_cloud[i].reshape(1, -1), k=k+1)[1][0]
        indices = indices[1:]
        # average the k nearest neighbours
        point_cloud[i] = np.mean(point_cloud[indices], axis=0)

    return point_cloud

def fit_polynomial(neighbours, point, degree=2, target_dim=3, num_coefficients=6):
    target_dim = target_dim - 1
    x = neighbours[:, [i for i in range(neighbours.shape[1]) if i != target_dim]]
    y = neighbours[:, target_dim]
    A = np.zeros((x.shape[0], num_coefficients))
    b = np.zeros((x.shape[0], 1))
    for i in range(x.shape[0]):
        A[i] = [x[i, 0]**2, x[i, 1]**2, x[i, 0]*x[i, 1], x[i, 0], x[i, 1], 1]
        b[i] = y[i]
    p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    point = point[:, [i for i in range(point.shape[1]) if i != target_dim]]
    return np.dot([point[0, 0]**2, point[0, 1]**2, point[0, 0]*point[0, 1], point[0, 0], point[0, 1], 1], p)

def fit_plane(neighbours, point):
    # create the A matrix
    A = np.ones((neighbours.shape[0], 3))
    A[:, :2] = neighbours[:, :2]
    # create the b matrix
    b = neighbours[:, 2].reshape(-1, 1)
    # solve the least squares problem
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    x = x.reshape(-1)
    point = point.reshape(-1)
    # return the value of the plane at the point
    return x[0]*point[0] + x[1]*point[1] + x[2]

def moving_least_squares_smoothing(point_cloud, k=5, degree=2):
    tree = KDTree(point_cloud)
    for i in tqdm(range(point_cloud.shape[0])):
        indices = tree.query(point_cloud[i].reshape(1, -1), k=k+1)[1][0]
        indices = indices[1:]
        point = point_cloud[i].reshape(1, -1)
        #point_cloud[i, 2] = fit_plane(point_cloud[indices], point)
        point_cloud[i, 2] = fit_polynomial(point_cloud[indices], point, degree=degree)
    return point_cloud

def regression_plane(neighbours):
    mean = np.mean(neighbours, axis=0)
    centered_neighbours = neighbours - mean
    covariance_matrix = np.cov(centered_neighbours, rowvar=False)
    ev, evec = np.linalg.eig(covariance_matrix)
    normal = evec[:, np.argmin(ev)]
    return mean, normal

def bilateral_smoothing(point_cloud, new_points, k=5, sigma_d=0.1, sigma_n=0.1):
    # create KDTree from the point cloud
    tree = KDTree(point_cloud)
    # Iterate over each point in the point cloud
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

  
