import numpy as np
from scipy.spatial import KDTree 
from tqdm import tqdm

def chamfer_distance(p, q):
    p_tree = KDTree(p)
    q_tree = KDTree(q)
    p_to_q, _ = p_tree.query(q)
    q_to_p, _ = q_tree.query(p)
    return np.mean(p_to_q) + np.mean(q_to_p)

def directed_hausdorff_distance(p, q):
    min_q = np.array([np.inf, np.inf, np.inf])
    for i in q:
        if np.linalg.norm(i - p) < np.linalg.norm(min_q - p):
            min_q = i

    max_p = np.array([0, 0, 0])
    for i in p:
        if np.linalg.norm(i - max_p) > np.linalg.norm(max_p - q):
            max_p = i

    return np.linalg.norm(max_p - min_q)

def hausdorff_distance(p, q):
    h_p = directed_hausdorff_distance(p, q)
    h_q = directed_hausdorff_distance(q, p)
    return max(h_p, h_q)

