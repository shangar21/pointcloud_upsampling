import numpy as np
from scipy.spatial import KDTree 
from tqdm import tqdm

def chamfer_distance(p, q):
    p_tree = KDTree(p)
    q_tree = KDTree(q)
    p_to_q, _ = p_tree.query(q)
    q_to_p, _ = q_tree.query(p)
    return np.mean(p_to_q) + np.mean(q_to_p)

