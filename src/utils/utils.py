import torch


def pairwise_dist(x):
    dist_matrix = torch.cdist(x, x)
    return dist_matrix


def kabsch(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD, in a batched manner.
    :param P: A BxNx3 matrix of points
    :param Q: A BxNx3 matrix of points
    :return: Aligned P, and the RMSD.
    """

    # Compute centroids
    centroid_P = torch.mean(P, dim=1, keepdims=True)  # Bx1x3
    centroid_Q = torch.mean(Q, dim=1, keepdims=True)  # Bx1x3

    # Center the points
    p = P - centroid_P  # BxNx3
    q = Q - centroid_Q  # BxNx3

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(1, 2), q)  # Bx3x3

    # SVD
    U, S, Vt = torch.linalg.svd(H)  # Bx3x3

    # Validate right-handed coordinate system
    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))  # B
    flip = d < 0.0
    if flip.any().item():
        Vt[flip, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    aligned_P = torch.matmul(p, R.transpose(1, 2)) + centroid_Q

    return aligned_P


def compute_dihedral(positions):
    v = positions[:, :, :-1] - positions[:, :, 1:]
    v0 = -v[:, :, 0]
    v1 = v[:, :, 2]
    v2 = v[:, :, 1]

    s0 = torch.sum(v0 * v2, dim=-1, keepdim=True) / torch.sum(
        v2 * v2, dim=-1, keepdim=True
    )
    s1 = torch.sum(v1 * v2, dim=-1, keepdim=True) / torch.sum(
        v2 * v2, dim=-1, keepdim=True
    )

    v0 = v0 - s0 * v2
    v1 = v1 - s1 * v2

    v0 = v0 / torch.norm(v0, dim=-1, keepdim=True)
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)

    x = torch.sum(v0 * v1, dim=-1)
    v3 = torch.cross(v0, v2, dim=-1)
    y = torch.sum(v3 * v1, dim=-1)
    return torch.atan2(y, x)


def chignolin_h_bond(positions):
    asp3od1_thr6og = torch.norm(positions[:, :, 36] - positions[:, :, 76], dim=-1)
    asp3od2_thr6og = torch.norm(positions[:, :, 37] - positions[:, :, 76], dim=-1)
    asp3n_thr8o = torch.norm(positions[:, :, 30] - positions[:, :, 95], dim=-1)

    # find pair wise minimum between dists (36, 37) and angle corresponding to the minimum for hydrogen bond indexed by 81
    asp3od_thr6og = torch.min(asp3od1_thr6og, asp3od2_thr6og)
    # asp3od_thr6og = torch.where(
    #     asp3od1_thr6og < asp3od2_thr6og, asp3od_thr6og, asp3od2_thr6og
    # )

    # asp3od_thr6og_angle = angle_between_vectors(
    #     positions[:, :, 36] - positions[:, :, 81],
    #     positions[:, :, 81] - positions[:, :, 76],
    # )
    # asp3n_thr8o_angle = angle_between_vectors(
    #     positions[:, :, 30] - positions[:, :, 38],
    #     positions[:, :, 38] - positions[:, :, 95],
    # )
    return asp3od_thr6og, asp3n_thr8o  # , angle


def poly_handed(positions):
    c1 = positions[:, :, 16]
    c2 = positions[:, :, 18]
    c3 = positions[:, :, 30]
    c4 = positions[:, :, 44]

    ab = c2 - c1
    cd = c4 - c3
    ef = (c3 + c4) / 2 - (c1 + c2) / 2
    h = ef * torch.cross(ab, cd, dim=-1)
    h = h.sum(dim=-1) / (
        torch.norm(ab, dim=-1) * torch.norm(cd, dim=-1) * torch.norm(ef, dim=-1)
    )
    return h


def angle_between_vectors(v1, v2):
    unit_v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    unit_v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
    dot_product = torch.sum(unit_v1 * unit_v2, dim=-1)
    angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
    return angle * (180.0 / torch.pi)


def compute_s_dist(x, y):
    x_dist = torch.cdist(x, x) + torch.eye(x.shape[-2], device=x.device).unsqueeze(0)
    y_dist = torch.cdist(y, y) + torch.eye(y.shape[-2], device=y.device).unsqueeze(0)
    log_dist = x_dist.log() - y_dist.log()

    x_gyr = compute_gyration(x)
    y_gyr = compute_gyration(y)

    gyr_dist = torch.linalg.slogdet(x_gyr)[1] - torch.linalg.slogdet(y_gyr)[1]

    s_dist = log_dist.square().sum(dim=(-2, -1)) / 2 + gyr_dist.square()
    return s_dist


def compute_gyration(x):
    center = torch.mean(x, dim=-2, keepdim=True)
    centered_x = x - center
    gyration_tensors = torch.einsum("lni,lnj->lij", centered_x, centered_x)
    return gyration_tensors
