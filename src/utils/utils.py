import torch
import numpy as np

def compute_dihedral(p): 
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array(
        [v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])

    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.arctan2(y, x)

def get_log_likelihood(diff, std):
    print(diff.size(2) * diff.size(3))
    D = diff.size(2) * diff.size(3)
    log_det_cov = diff.size(3) * std.log().sum()
    exp_term = -0.5 * torch.sum(diff.square()/std, dim=(2, 3))
    normalization_term = -0.5 * (D * torch.log(torch.tensor(2 * torch.pi)) + log_det_cov)
    log_likelihood = normalization_term + exp_term
    return log_likelihood