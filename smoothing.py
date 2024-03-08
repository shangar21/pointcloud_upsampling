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
    """
    This function fits a plane to the k nearest neighbours of a point.
    """
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

#
#def get_polynomial_surface(points, coefs, degree=2):
#    """
#    This function returns the value of the polynomial surface at the points.
#    """
#    poly = PolynomialFeatures(degree=degree)
#    X = points[:, [0, 1]]
#    X_ = poly.fit_transform(X)
#    poly_vals = np.dot(X_, coefs).reshape(-1, 1)
#    return np.hstack((points[:, [0,1]], poly_vals))
#
#
#def fit_point(points, point, degree=2, target_dim=3):
#    """
#    This function fits a polynomial to the points and returns the value of the polynomial at the point.
#    """
#    # create polynomial features
#    target_dim -= 1
#    poly = PolynomialFeatures(degree=degree)
#    X = points[:, [i for i in range(3) if i != target_dim]]
#    y = points[:, target_dim]
#    X = poly.fit_transform(X)
#    # fit the polynomial
#    model = LinearRegression().fit(X, y)
#    # find closest point to polynomial
#    coefs = model.coef_
#    surface = get_polynomial_surface(points, coefs, degree)
#    idx = np.argmin(np.linalg.norm(point - surface, axis=1))
#    return surface[idx]

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

def estimate_curvature(normals, knn_indices, knn_distances, k=5):
    """
    This function estimates the curvature of a point cloud.
    """
    point_idx = knn_indices[0]
    knn_indices = knn_indices[1:]
    knn_distances = knn_distances[1:]
    weights = 1 / ((knn_distances + 1e-6)**2)
    norm_mul = np.multiply(normals[knn_indices], normals[point_idx])
    norm_normals = np.linalg.norm(normals[knn_indices], axis=1)
    norm_point_normal = np.linalg.norm(normals[point_idx])
    cos_theta = np.sum(norm_mul, axis=1) / (norm_normals * norm_point_normal)
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)
    curvature = np.sum(weights * theta) / np.sum(weights)
    return curvature

def bilateral_smoothing(point_cloud, k=5, sigma_d=0.1, sigma_n=0.1):
    """
    This function performs bilaterial filtering on a point cloud.
    """
    # create KDTree from the point cloud
    tree = KDTree(point_cloud)
    # Iterate over each point in the point cloud
    for i in tqdm(range(point_cloud.shape[0])):
        # get the k nearest neighbours
        indices = tree.query(point_cloud[i].reshape(1, -1), k=k+1)[1][0]
        indices = indices[1:]
        neighbours = point_cloud[indices]
        mean, normal = regression_plane(neighbours)
        sum_of_weights = 0
        delta_p = 0
        for q in neighbours:
            d_d = np.linalg.norm(q - point_cloud[i])
            d_n = np.dot(normal, q - point_cloud[i])
            w = np.exp(-d_d**2/(2*sigma_d**2) - d_n**2/(2*sigma_n**2))
            delta_p = delta_p + (w * d_n)
            sum_of_weights = sum_of_weights + w
        point_cloud[i] = point_cloud[i] + ((delta_p / sum_of_weights ) * normal)
    return point_cloud

def estimate_normals(point_cloud):
  """Estimates normals for each point using PCA."""
  point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
  return point_cloud.normals 

def detect_edges(normals, threshold=0.8):
  """Detects points likely on edges based on normal deviation."""
  edge_flags = []
  normals = np.array(normals)
  for i in range(len(normals)):
    neighbor_normals = o3d.geometry.KDTreeFlann(normals).search_knn_vector_3d(normals[i], 10)[1][1:]
    average_normal = np.mean(neighbor_normals, axis=0)
    angle = np.arccos(np.dot(normals[i], average_normal) / (np.linalg.norm(normals[i]) * np.linalg.norm(average_normal)))
    edge_flags.append(angle > threshold)
  return edge_flags

def get_neighbors(point, point_cloud, k=10):
  """Finds k nearest neighbors for a given point."""
  neighbors, _ = o3d.geometry.KDTree(point_cloud).search_knn(point, k)
  return neighbors

def estimate_surface(points):
  """Estimates a local surface using radial basis functions (RBF) interpolation."""
  # Convert points to numpy array for easier RBF fitting
  points_np = np.array(points)
  # Extract coordinates
  coords = points_np[:, :3]
  # Extract features (optional, can be set to 1 for all points)
  features = points_np[:, 3:]

  # RBF interpolation (replace with your preferred method)
  rbf = o3d.registration.RBFInterpolator()
  rbf.set_epsilon(0.1)  # Adjust epsilon for smoothness control
  rbf.fit(coords, features)

  def surface_function(query_point):
    return rbf.evaluate(query_point)

  return surface_function

def sample_on_surface(surface_function, point, max_distance=0.1):
  """Samples a new point on the estimated local surface."""
  # Sample a random direction within a sphere
  direction = np.random.randn(3)
  direction /= np.linalg.norm(direction)

  # Sample a point along the random direction within a limited distance
  new_point = point + direction * max_distance

  # Project the new point onto the estimated surface
  projected_point = surface_function(new_point)

  return projected_point

def edge_aware_resampling(point_cloud, target_density):
  """
  Performs edge-aware resampling on a point cloud.

  Args:
      point_cloud: Input point cloud (Open3D.geometry.PointCloud object).
      target_density: Target number of points in the upsampled cloud.

  Returns:
      Upsampled point cloud (Open3D.geometry.PointCloud object).
  """
  points = point_cloud.points
  normals = estimate_normals(point_cloud)
  edge_flags = detect_edges(normals)

  away_points = [point for point, flag in zip(points, edge_flags) if not flag]

  while len(away_points) < target_density:
    new_points = []
    for point in away_points:
      neighbors = get_neighbors(point, points)
      surface_function = estimate_surface(neighbors)
      new_point = sample_on_surface(surface_function, point)
      new_points.append(new_point)
    away_points.extend(new_points)

  return o3d.geometry.PointCloud(np.array(away_points))

   
