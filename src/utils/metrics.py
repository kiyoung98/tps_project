import torch

class Metric:
    def __init__(self, args, md):
        self.num_particles = md.num_particles
        self.eye = torch.eye(self.num_particles, device=args.device).unsqueeze(0)
        self.charge_matrix = torch.tensor(md.charge_matrix, dtype=torch.float, device=args.device).unsqueeze(0)
        self.covalent_radii_matrix = torch.tensor(md.covalent_radii_matrix, dtype=torch.float, device=args.device).unsqueeze(0)

        if args.molecule == 'alanine':
            self.angle_2 = torch.tensor([1, 6, 8, 14], dtype=torch.long, device=args.device)
            self.angle_1 = torch.tensor([6, 8, 14, 16], dtype=torch.long, device=args.device)

    def expected_pairwise_distance(self, last_position, target_position):
        last_dist_matrix = self.dist(last_position)
        target_dist_matrix = self.dist(target_position)
        
        pd = torch.mean((last_dist_matrix-target_dist_matrix)**2, dim=(1, 2))
        mean_pd, std_pd = pd.mean().item(), pd.std().item()
        return mean_pd, std_pd

    def expected_pairwise_scaled_distance(self, last_position, target_position):
        last_dist_matrix = self.scaled_dist(last_position)
        target_dist_matrix = self.scaled_dist(target_position)
        
        psd = torch.mean((last_dist_matrix-target_dist_matrix)**2, dim=(1, 2))
        mean_psd, std_psd = psd.mean().item(), psd.std().item()
        return mean_psd, std_psd

    def expected_pairwise_coulomb_distance(self, last_position, target_position):
        last_dist_matrix = self.coulomb(last_position)
        target_dist_matrix = self.coulomb(target_position)
        
        pcd = torch.mean((last_dist_matrix-target_dist_matrix)**2, dim=(1, 2))
        mean_pcd, std_pcd = pcd.mean().item(), pcd.std().item()
        return mean_pcd, std_pcd

    def alanine(self, positions, target_position, potentials):
        etps, efps, etp_idxs, efp_idxs = [], [], [], []

        target_psi = self.compute_dihedral(target_position[:, self.angle_1].unsqueeze(0))
        target_phi = self.compute_dihedral(target_position[:, self.angle_2].unsqueeze(0))

        psi = self.compute_dihedral(positions[:, :, self.angle_1])
        phi = self.compute_dihedral(positions[:, :, self.angle_2])

        hit_mask = (torch.abs(psi-target_psi) < 0.75) & (torch.abs(phi-target_phi) < 0.75)
        hit, hit_idxs = hit_mask.max(-1)

        thp = 100 * hit.sum().float() / hit.shape[0]

        for i, hit_idx in enumerate(hit_idxs):
            if hit_idx > 0:
                etp, idx = potentials[i][:hit_idx].max(0)
                etps.append(etp)
                etp_idxs.append(idx.item())

                efp = potentials[i][hit_idx]
                efps.append(efp)
                efp_idxs.append(hit_idx.item())

        if len(etps)>0:
            etps = torch.tensor(etps)
            efps = torch.tensor(efps)

            mean_etp = etps.mean().item()
            mean_efp = efps.mean().item()

            std_etp = etps.std().item()
            std_efp = efps.std().item() 

            mean_len = hit_idxs.float().mean().item() 
            std_len = hit_idxs.float().std().item() 
            return thp, hit_idxs, mean_len, std_len, mean_etp, std_etp, etps.cpu().numpy(), etp_idxs, mean_efp, std_efp, efps.cpu().numpy(), efp_idxs
    
        else:
            return thp, None, None, None, None, None, None, None, None, None, None, None
    
    # def energy_point(self, last_position, target_position, potentials, last_idx):
    #     etps, efps, etp_idxs, efp_idxs = [], [], [], []

    #     target_psi = compute_dihedral(target_position[0, self.angle_1, :])
    #     target_phi = compute_dihedral(target_position[0, self.angle_2, :])
        
    #     for i in range(last_position.shape[0]):
    #         psi = compute_dihedral(last_position[i, self.angle_1, :])
    #         phi = compute_dihedral(last_position[i, self.angle_2, :])
    #         psi_dist = min(abs(psi-target_psi), abs(psi-target_psi+2*np.pi), abs(psi-target_psi-2*np.pi))
    #         phi_dist = min(abs(phi-target_phi), abs(phi-target_phi+2*np.pi), abs(phi-target_phi-2*np.pi))
    #         if psi_dist < 0.75 and phi_dist < 0.75:
    #             etp, idx = potentials[i][:last_idx[i]].max(0)
    #             etps.append(etp)
    #             etp_idxs.append(idx.item())

    #             efp = potentials[i][last_idx[i]]
    #             efps.append(efp)
    #             efp_idxs.append(last_idx[i].item())
        
    #     if len(etps)>0:
    #         etps = torch.tensor(etps)
    #         efps = torch.tensor(efps)

    #         mean_etp = etps.mean().item()
    #         mean_efp = efps.mean().item()

    #         std_etp = etps.std().item()
    #         std_efp = efps.std().item()     
    #         return mean_etp, std_etp, etps.cpu().numpy(), etp_idxs, mean_efp, std_efp, efps.cpu().numpy(), efp_idxs
    
    #     else:
    #         return None, None, etps, etp_idxs, None, None, efps, efp_idxs
        
    def effective_sample_size(self, log_likelihood, log_reward):
        likelihood = log_likelihood.exp()
        reward = log_reward.exp()
        weight = reward / likelihood
        ess = weight.sum().square() / weight.square().sum()
        return ess

    def dist(self, x):
        dist_matrix = torch.cdist(x, x)
        return dist_matrix

    def scaled_dist(self, x):
        dist_matrix = torch.cdist(x, x) + self.eye
        scaled_dist_matrix = torch.exp(-1.7*(dist_matrix-self.covalent_radii_matrix)/self.covalent_radii_matrix) + 0.01 * self.covalent_radii_matrix / dist_matrix
        return scaled_dist_matrix

    def coulomb(self, x):
        dist_matrix = torch.cdist(x, x) + self.eye
        coulomb_matrix = self.charge_matrix / dist_matrix
        return coulomb_matrix    

    def compute_dihedral(self, positions):
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