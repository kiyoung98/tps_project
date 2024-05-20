import torch

def pairwise_dist(x):
    dist_matrix = torch.cdist(x, x)
    return dist_matrix

def compute_dihedral(positions):
    v = positions[:, :, :-1] - positions[:, :, 1:]
    v0 = - v[:, :, 0]
    v1 = v[:, :, 2]
    v2 = v[:, :, 1]
    
    s0 = torch.sum(v0 * v2, dim=-1, keepdim=True) / torch.sum(v2 * v2, dim=-1, keepdim=True)
    s1 = torch.sum(v1 * v2, dim=-1, keepdim=True) / torch.sum(v2 * v2, dim=-1, keepdim=True)

    v0 = v0 - s0 * v2
    v1 = v1 - s1 * v2

    v0 = v0 / torch.norm(v0, dim=-1, keepdim=True)
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)

    x = torch.sum(v0 * v1, dim=-1)
    v3 = torch.cross(v0, v2, dim=-1)
    y = torch.sum(v3 * v1, dim=-1)
    return torch.atan2(y, x)

def get_log_likelihood(diff, std):
    D = diff.size(2) * diff.size(3)
    log_det_cov = diff.size(3) * std.log().sum()
    exp_term = -0.5 * torch.sum(diff.square()/std, dim=(2, 3))
    normalization_term = -0.5 * (D * torch.log(torch.tensor(2 * torch.pi)) + log_det_cov)
    log_likelihood = normalization_term + exp_term
    return log_likelihood