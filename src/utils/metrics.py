import torch
import numpy as np

from .utils import compute_dihedral

def expected_pairwise_distance(agent, last_position, target_position):
    last_dist_matrix = agent.dist(last_position)
    target_dist_matrix = agent.dist(target_position)
    
    epd = torch.mean((last_dist_matrix-target_dist_matrix)**2).item()
    return 1000*epd

def expected_pairwise_scaled_distance(agent, last_position, target_position):
    last_dist_matrix = agent.scaled_dist(last_position)
    target_dist_matrix = agent.scaled_dist(target_position)
    
    epsd = torch.mean((last_dist_matrix-target_dist_matrix)**2).item()
    return 1000*epsd

def expected_pairwise_coulomb_distance(agent, last_position, target_position):
    last_dist_matrix = agent.coulomb(last_position)
    target_dist_matrix = agent.coulomb(target_position)
    
    epcd = torch.mean((last_dist_matrix-target_dist_matrix)**2).item()
    return 1000*epcd

def target_hit_percentage(last_position, target_position):
    last_position = last_position.detach().cpu().numpy()
    target_position = target_position.detach().cpu().numpy()
    
    hit = 0
    
    angle_2 = [1, 6, 8, 14]
    angle_1 = [6, 8, 14, 16]

    target_psi = compute_dihedral(target_position[0, angle_1, :])
    target_phi = compute_dihedral(target_position[0, angle_2, :])
    
    for i in range(last_position.shape[0]):
        psi = compute_dihedral(last_position[i, angle_1, :])
        phi = compute_dihedral(last_position[i, angle_2, :])
        psi_dist = min(abs(psi-target_psi), abs(psi-target_psi+2*np.pi), abs(psi-target_psi-2*np.pi))
        phi_dist = min(abs(phi-target_phi), abs(phi-target_phi+2*np.pi), abs(phi-target_phi-2*np.pi))
        if psi_dist < 0.75 and phi_dist < 0.75:
            hit += 1
    
    thp = int(100*hit/last_position.shape[0])
    return thp


def energy_point(last_position, target_position, potentials, last_idx):
    last_position = last_position.detach().cpu().numpy()
    target_position = target_position.detach().cpu().numpy()
    
    hit = 0
    etp = 0
    efp = 0
    
    angle_2 = [1, 6, 8, 14]
    angle_1 = [6, 8, 14, 16]

    target_psi = compute_dihedral(target_position[0, angle_1, :])
    target_phi = compute_dihedral(target_position[0, angle_2, :])
    
    for i in range(last_position.shape[0]):
        psi = compute_dihedral(last_position[i, angle_1, :])
        phi = compute_dihedral(last_position[i, angle_2, :])
        psi_dist = min(abs(psi-target_psi), abs(psi-target_psi+2*np.pi), abs(psi-target_psi-2*np.pi))
        phi_dist = min(abs(phi-target_phi), abs(phi-target_phi+2*np.pi), abs(phi-target_phi-2*np.pi))
        if psi_dist < 0.75 and phi_dist < 0.75:
            etp += potentials[i][:last_idx[i]].max()
            efp += potentials[i][last_idx[i]]
            hit += 1
    
    etp = etp.item() / hit if hit > 0 else None
    efp = efp.item() / hit if hit > 0 else None
    return etp, efp