import torch
import numpy as np

from .utils import compute_dihedral, get_dist_matrix

def expected_pairwise_distance(last_position, target_position):
    last_dist_matrix = get_dist_matrix(last_position)
    target_dist_matrix = get_dist_matrix(target_position)
    
    epd = torch.mean((last_dist_matrix-target_dist_matrix)**2).item()
    return 1000*epd


def target_hit_percentage(last_position, target_position):
    last_position = last_position.detach().cpu().numpy()
    target_position = target_position.detach().cpu().numpy()
    
    hit = 0
    
    angle_2 = [1, 6, 8, 14]
    angle_1 = [6, 8, 14, 16]

    target_psi = compute_dihedral(target_position[0, 0, angle_1, :])
    target_phi = compute_dihedral(target_position[0, 0, angle_2, :])
    
    for i in range(last_position.shape[0]):
        psi = compute_dihedral(last_position[i, 0, angle_1, :])
        phi = compute_dihedral(last_position[i, 0, angle_2, :])
        psi_dist = min(abs(psi-target_psi), abs(psi-target_psi+2*np.pi), abs(psi-target_psi-2*np.pi))
        phi_dist = min(abs(phi-target_phi), abs(phi-target_phi+2*np.pi), abs(phi-target_phi-2*np.pi))
        if psi_dist < 0.75 and phi_dist < 0.75:
            hit += 1
    
    thp = int(100*hit/last_position.shape[0])
    return thp


def energy_transition_point(last_position, target_position, potentials):
    last_position = last_position.detach().cpu().numpy()
    target_position = target_position.detach().cpu().numpy()
    
    hit = 0
    etp = 0
    
    angle_2 = [1, 6, 8, 14]
    angle_1 = [6, 8, 14, 16]

    target_psi = compute_dihedral(target_position[0, 0, angle_1, :])
    target_phi = compute_dihedral(target_position[0, 0, angle_2, :])
    
    for i in range(last_position.shape[0]):
        psi = compute_dihedral(last_position[i, 0, angle_1, :])
        phi = compute_dihedral(last_position[i, 0, angle_2, :])
        psi_dist = min(abs(psi-target_psi), abs(psi-target_psi+2*np.pi), abs(psi-target_psi-2*np.pi))
        phi_dist = min(abs(phi-target_phi), abs(phi-target_phi+2*np.pi), abs(phi-target_phi-2*np.pi))
        if psi_dist < 0.75 and phi_dist < 0.75:
            etp += potentials[i].max()
            hit += 1
    
    etp = etp.item() / hit if hit > 0 else None
    return etp