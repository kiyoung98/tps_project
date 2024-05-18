import torch
import numpy as np
import matplotlib.pyplot as plt

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

def plot_paths_alanine(dir_path, rollout, positions, target_position, last_idx):
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
    plt.savefig(f'{dir_path}/paths_{rollout}.png')
    plt.close()
    return fig