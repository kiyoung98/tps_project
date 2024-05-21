import torch
from .utils import *
import openmm.unit as unit
from torch.distributions import Normal

class Metric:
    def __init__(self, args, md):
        self.num_particles = md.num_particles
        self.eye = torch.eye(self.num_particles, device=args.device).unsqueeze(0)
        self.charge_matrix = torch.tensor(md.charge_matrix, dtype=torch.float, device=args.device).unsqueeze(0)
        self.std = torch.tensor(md.std.value_in_unit(unit.nanometer/unit.femtosecond), dtype=torch.float, device=args.device)

        self.normal = Normal(0, self.std)

        if args.molecule == 'alanine':
            self.angle_2 = torch.tensor([1, 6, 8, 14], dtype=torch.long, device=args.device)
            self.angle_1 = torch.tensor([6, 8, 14, 16], dtype=torch.long, device=args.device)

    def expected_pairwise_distance(self, last_position, target_position):
        last_pd = pairwise_dist(last_position)
        target_pd = pairwise_dist(target_position)
        
        pd = (last_pd-target_pd).square().mean((1, 2))
        mean_pd, std_pd = pd.mean().item(), pd.std().item()
        return mean_pd, std_pd

    def expected_pairwise_coulomb_distance(self, last_position, target_position):
        last_pcd = self.coulomb(last_position)
        target_pcd = self.coulomb(target_position)
        
        pcd = (last_pcd-target_pcd).square().mean((1, 2))
        mean_pcd, std_pcd = pcd.mean().item(), pcd.std().item()
        return mean_pcd, std_pcd

    def coulomb(self, x):
        dist_matrix = torch.cdist(x, x) + self.eye
        coulomb_matrix = self.charge_matrix / dist_matrix
        return coulomb_matrix

    def log_likelihood(self, diff):
        log_likelihood = self.normal.log_prob(diff).sum((2, 3)).mean(1)
        mean_ll, std_ll = log_likelihood.mean().item(), log_likelihood.std().item()
        return mean_ll, std_ll

    def cv_metrics(self, last_position, target_position, potentials, last_idx):
        etps, efps, etp_idxs, efp_idxs = [], [], [], []

        target_psi = compute_dihedral(target_position[:, self.angle_1])
        target_phi = compute_dihedral(target_position[:, self.angle_2])

        psi = compute_dihedral(last_position[:, self.angle_1])
        phi = compute_dihedral(last_position[:, self.angle_2])

        hit = (torch.abs(psi-target_psi) < 0.75) & (torch.abs(phi-target_phi) < 0.75)
        hit = hit.squeeze().float()

        thp = 100 * hit.sum() / hit.shape[0]

        for i, j in enumerate(last_idx):
            if hit[i] > 0.5:
                etp, idx = potentials[i][:j].max(0)
                etps.append(etp)
                etp_idxs.append(idx.item())

                efp = potentials[i][j]
                efps.append(efp)
                efp_idxs.append(j.item())

        if len(etps)>0:
            etps = torch.tensor(etps)
            efps = torch.tensor(efps)

            mean_etp = etps.mean().item()
            mean_efp = efps.mean().item()

            std_etp = etps.std().item()
            std_efp = efps.std().item()

            return thp, mean_etp, std_etp, mean_efp, std_efp
        else:
            return thp, None, None, None, None
        