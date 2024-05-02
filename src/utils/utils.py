import torch
import numpy as np

def compute_dihedral(p): 
    """http://stackoverflow.com/q/20305272/1128289"""
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array(
        [v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.arctan2(y, x)

def get_log_normal(x):
    normal = torch.distributions.normal.Normal(loc=0, scale=1)
    return normal.log_prob(x)

def get_dist_matrix(x):
    dist_matrix = torch.cdist(x, x)
    return dist_matrix