import torch
import numpy as np
import mdtraj as md
from tqdm import tqdm
import matplotlib.pyplot as plt

from .utils import compute_dihedral

class AlaninePotential():
    def __init__(self):
        super().__init__()
        self.open_file()

    def open_file(self):
        file = "./src/utils/vacuum.dat"

        with open(file) as f:
            lines = f.readlines()

        dims = [90, 90]

        self.locations = torch.zeros((int(dims[0]), int(dims[1]), 2))
        self.data = torch.zeros((int(dims[0]), int(dims[1])))

        i = 0
        for line in lines[1:]:
            # if line == '  \n':
            #     psi = psi + 1
            #     phi = 0
            #     continue
            splits = line[0:-1].split(" ")
            vals = [y for y in splits if y != '']

            x = float(vals[0])
            y = float(vals[1])
            val = float(vals[-1])

            self.locations[i // 90, i % 90, :] = torch.tensor([x, y])
            self.data[i // 90, i % 90] = (val)  # / 503.)
            i = i + 1

    def potential(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp, loc.double(), p=2)
        index = distances.argmin(dim=1)

        x = torch.div(index, self.locations.shape[0], rounding_mode='trunc')  # index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        z = self.data[x, y]
        return z

def plot_paths_alanine(dir_path, positions, target_position, last_idx):
    positions = positions.detach().cpu().numpy()
    target_position = target_position.detach().cpu().numpy()
    
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])

    angle_2 = [1, 6, 8, 14]
    angle_1 = [6, 8, 14, 16]

    potential = AlaninePotential()
    xs = np.arange(-np.pi, np.pi + .1, .1)
    ys = np.arange(-np.pi, np.pi + .1, .1)
    x, y = np.meshgrid(xs, ys)
    inp = torch.tensor(np.array([x, y])).view(2, -1).T

    z = potential.potential(inp)
    z = z.view(y.shape[0], y.shape[1])

    plt.contourf(xs, ys, z, levels=100, zorder=0)

    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1. * i / positions.shape[0]) for i in range(positions.shape[0])])

    psis_start = []
    phis_start = []

    for i in range(positions.shape[0]):
        psis_start.append(compute_dihedral(positions[i, 0, angle_1, :]))
        phis_start.append(compute_dihedral(positions[i, 0, angle_2, :]))

        psi = []
        phi = []
        for j in range(last_idx[i]):
            psi.append(compute_dihedral(positions[i, j, angle_1, :]))
            phi.append(compute_dihedral(positions[i, j, angle_2, :]))
        ax.plot(phi, psi, marker='o', linestyle='None', markersize=2, alpha=1.)

    ax.scatter(phis_start, psis_start, edgecolors='black', c='w', zorder=100, s=100, marker='*')
    
    psis_target = []
    phis_target = []
    psis_target.append(compute_dihedral(target_position[0, angle_1, :]))
    phis_target.append(compute_dihedral(target_position[0, angle_2, :]))
    ax.scatter(phis_target, psis_target, edgecolors='w', c='w', zorder=100, s=10)

    plt.xlabel('phi')
    plt.ylabel('psi')
    plt.show()
    plt.savefig(f'{dir_path}/paths.png')
    plt.close()
    return fig

def plot_path(dir_path, positions, target_position, last_idx):
    positions = positions.detach().cpu().numpy()
    target_position = target_position.detach().cpu().numpy()

    angle_2 = [1, 6, 8, 14]
    angle_1 = [6, 8, 14, 16]
    
    for i in range(positions.shape[0]):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])
        potential = AlaninePotential()
        xs = np.arange(-np.pi, np.pi + .1, .1)
        ys = np.arange(-np.pi, np.pi + .1, .1)
        x, y = np.meshgrid(xs, ys)
        inp = torch.tensor(np.array([x, y])).view(2, -1).T

        z = potential.potential(inp)
        z = z.view(y.shape[0], y.shape[1])

        plt.contourf(xs, ys, z, levels=100, zorder=0)
        
        psis_start = []
        phis_start = []

        psis_start.append(compute_dihedral(positions[i, 0, angle_1, :]))
        phis_start.append(compute_dihedral(positions[i, 0, angle_2, :]))

        psi = []
        phi = []
        for j in range(last_idx[i]):
            psi.append(compute_dihedral(positions[i, j, angle_1, :]))
            phi.append(compute_dihedral(positions[i, j, angle_2, :]))
        ax.plot(phi, psi, marker='o', linestyle='None', markersize=2, alpha=1.)

        ax.scatter(phis_start, psis_start, edgecolors='black', c='w', zorder=100, s=100, marker='*')
        
        psis_target = []
        phis_target = []
        psis_target.append(compute_dihedral(target_position[0, angle_1, :]))
        phis_target.append(compute_dihedral(target_position[0, angle_2, :]))
        ax.scatter(phis_target, psis_target, edgecolors='w', c='w', zorder=100, s=10)

        plt.xlabel('phi')
        plt.ylabel('psi')
        plt.show()
        plt.savefig(f'{dir_path}/paths_{i}.png')
        plt.close()

def plot_3D_view(dir_path, start_file, positions, last_idx):
    positions = positions.detach().cpu().numpy()
    for i in tqdm(range(positions.shape[0])):
        if last_idx[i] > 0:
            for j in range(last_idx[i]):
                traj = md.load_pdb(start_file)
                traj.xyz = positions[i, j]
                
                if j == 0:
                    trajs = traj
                else:
                    trajs = trajs.join(traj)
            trajs.save(f'{dir_path}/3D_view_{i}.h5')
    
def plot_potential(dir_path, potentials, log_reward, last_idx):
    potentials = potentials.detach().cpu().numpy()
    for i in range(potentials.shape[0]):
        if last_idx[i] > 0:
            plt.plot(potentials[i][:last_idx[i]], label=f"Sample {i}: log reward {log_reward[i]:.4f}")
            plt.xlabel('Time (fs)')
            plt.ylabel("Potential Energy (kJ/mol)")
            plt.legend()
            plt.show()
            plt.savefig(f'{dir_path}/potential_{i}.png')
            plt.close()

def plot_potentials(dir_path, rollout, potentials, log_reward, last_idx):
    potentials = potentials.detach().cpu().numpy()
    fig = plt.figure(figsize=(20, 5))
    for i in range(potentials.shape[0]):
        if last_idx[i] > 0:
            plt.plot(potentials[i][:last_idx[i]], label=f"Sample {i}: log reward {log_reward[i]:.4f}")
            
    plt.xlabel('Time (fs)')
    plt.ylabel("Potential Energy (kJ/mol)")
    plt.legend()
    plt.show()
    plt.savefig(f'{dir_path}/potential_rollout_{rollout}.png')
    plt.close()
    return fig

def plot_etps(dir_path, rollout, etps, etp_idxs):
    fig = plt.figure(figsize=(20, 5))
    plt.scatter(etp_idxs, etps)
    plt.xlabel('Time (fs)')
    plt.ylabel("Potential Energy (kJ/mol)")
    plt.legend()
    plt.show()
    plt.savefig(f'{dir_path}/etps_rollout_{rollout}.png')
    plt.close()
    return fig

def plot_efps(dir_path, rollout, efps, efp_idxs):
    fig = plt.figure(figsize=(20, 5))
    plt.scatter(efp_idxs, efps)
    plt.xlabel('Time (fs)')
    plt.ylabel("Potential Energy (kJ/mol)")
    plt.legend()
    plt.show()
    plt.savefig(f'{dir_path}/efps_rollout_{rollout}.png')
    plt.close()
    return fig