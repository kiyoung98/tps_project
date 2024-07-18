import torch
import numpy as np
import mdtraj as md
from tqdm import tqdm
import matplotlib.pyplot as plt
from .utils import *


class AlaninePotential:
    def __init__(self):
        super().__init__()
        self.open_file()

    def open_file(self):
        file = "./src/utils/alanine.dat"

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
            vals = [y for y in splits if y != ""]

            x = float(vals[0])
            y = float(vals[1])
            val = float(vals[-1])

            self.locations[i // 90, i % 90, :] = torch.tensor([x, y])
            self.data[i // 90, i % 90] = val  # / 503.)
            i = i + 1

    def potential(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp, loc.double(), p=2)
        index = distances.argmin(dim=1)

        x = torch.div(
            index, self.locations.shape[0], rounding_mode="trunc"
        )  # index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        z = self.data[x, y]
        return z


class HistidinePotential:  # TODO: Make histidine.dat for 4 torsion angles
    def __init__(self):
        super().__init__()
        self.open_file()

    def open_file(self):
        file = "./src/utils/histidine.dat"

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
            vals = [y for y in splits if y != ""]

            x = float(vals[0])
            y = float(vals[1])
            val = float(vals[-1])

            self.locations[i // 90, i % 90, :] = torch.tensor([x, y])
            self.data[i // 90, i % 90] = val  # / 503.)
            i = i + 1

    def potential(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp, loc.double(), p=2)
        index = distances.argmin(dim=1)

        x = torch.div(
            index, self.locations.shape[0], rounding_mode="trunc"
        )  # index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        z = self.data[x, y]
        return z


def plot_paths_alanine(save_dir, rollout, positions, target_position, last_idx):
    angle_1 = [6, 8, 14, 16]
    angle_2 = [1, 6, 8, 14]

    psi = compute_dihedral(positions[:, :, angle_1, :]).detach().cpu().numpy()
    phi = compute_dihedral(positions[:, :, angle_2, :]).detach().cpu().numpy()
    target_psi = (
        compute_dihedral(target_position[:, angle_1, :].unsqueeze(0))
        .detach()
        .cpu()
        .numpy()
    )
    target_phi = (
        compute_dihedral(target_position[:, angle_2, :].unsqueeze(0))
        .detach()
        .cpu()
        .numpy()
    )

    plt.clf()
    plt.close()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi, np.pi])

    potential = AlaninePotential()
    xs = np.arange(-np.pi, np.pi + 0.1, 0.1)
    ys = np.arange(-np.pi, np.pi + 0.1, 0.1)
    x, y = np.meshgrid(xs, ys)
    inp = torch.tensor(np.array([x, y])).view(2, -1).T

    z = potential.potential(inp)
    z = z.view(y.shape[0], y.shape[1])

    plt.contourf(xs, ys, z, levels=100, zorder=0)

    cm = plt.get_cmap("gist_rainbow")
    ax.set_prop_cycle(
        color=[cm(1.0 * i / positions.shape[0]) for i in range(positions.shape[0])]
    )

    for i in range(positions.shape[0]):
        ax.plot(
            phi[i, : last_idx[i] + 1],
            psi[i, : last_idx[i] + 1],
            marker="o",
            linestyle="None",
            markersize=2,
            alpha=1.0,
        )

    ax.scatter(
        phi[:1, 0], psi[:1, 0], edgecolors="black", c="w", zorder=100, s=100, marker="*"
    )

    ax.scatter(
        target_phi[:1, 0], target_psi[:1, 0], edgecolors="w", c="w", zorder=100, s=10
    )

    plt.xlabel("phi")
    plt.ylabel("psi")
    plt.show()
    plt.savefig(f"{save_dir}/paths/{rollout}.png")
    plt.close()
    return fig


def plot_paths_chignolin(save_dir, rollout, positions, last_idx):
    asp3od_thr6og, asp3n_thr8o = chignolin_h_bond(positions)
    asp3od_thr6og = asp3od_thr6og.detach().cpu().numpy()
    asp3n_thr8o = asp3n_thr8o.detach().cpu().numpy()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    cm = plt.get_cmap("gist_rainbow")
    ax.set_prop_cycle(
        color=[cm(1.0 * i / positions.shape[0]) for i in range(positions.shape[0])]
    )

    plt.xlim([0, 1.5])
    plt.ylim([0, 2])
    for i in range(positions.shape[0]):
        if last_idx[i] > 0:
            ax.plot(
                asp3od_thr6og[i][: last_idx[i] + 1],
                asp3n_thr8o[i][: last_idx[i] + 1],
                marker="o",
                linestyle="None",
                markersize=2,
                alpha=1.0,
            )

    plt.plot([0, 0.35], [0, 0], color="black")
    plt.plot([0.35, 0.35], [0, 0.35], color="black")
    plt.plot([0.35, 0], [0.35, 0.35], color="black")
    plt.plot([0, 0], [0.35, 0], color="black")

    plt.xlabel("Asp3OD-Thr6OG")
    plt.ylabel("Asp3N-Thr8O")
    plt.show()
    plt.savefig(f"{save_dir}/paths/{rollout}.png")
    plt.close()
    return fig


def plot_hands(save_dir, rollout, positions, last_idx):
    handed = poly_handed(positions)
    handed = handed.detach().cpu().numpy()

    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(111)
    plt.ylim([-1, 1])

    cm = plt.get_cmap("gist_rainbow")
    ax.set_prop_cycle(
        color=[cm(1.0 * i / positions.shape[0]) for i in range(positions.shape[0])]
    )

    for i in range(positions.shape[0]):
        if last_idx[i] > 0:
            ax.plot(
                handed[i][: last_idx[i] + 1],
                marker="o",
                linestyle="None",
                markersize=2,
                alpha=1.0,
            )

    plt.xlabel("Time (fs)")
    plt.ylabel("Handedness")
    plt.show()
    plt.savefig(f"{save_dir}/paths/{rollout}.png")
    plt.close()
    return fig


def plot_potentials(save_dir, rollout, potentials, last_idx):
    potentials = potentials.detach().cpu().numpy()
    fig = plt.figure(figsize=(20, 5))
    for i in range(potentials.shape[0]):
        if last_idx[i] > 0:
            plt.plot(potentials[i][: last_idx[i] + 1])

    plt.xlabel("Time (fs)")
    plt.ylabel("Potential Energy (kJ/mol)")
    plt.show()
    plt.savefig(f"{save_dir}/potentials/{rollout}.png")
    plt.close()
    return fig


def plot_etps(save_dir, rollout, etps, etp_idxs):
    fig = plt.figure(figsize=(20, 5))
    plt.scatter(etp_idxs, etps)
    plt.xlabel("Time (fs)")
    plt.ylabel("Potential Energy (kJ/mol)")
    plt.show()
    plt.savefig(f"{save_dir}/etps/{rollout}.png")
    plt.close()
    return fig


def plot_efps(save_dir, rollout, efps, efp_idxs):
    fig = plt.figure(figsize=(20, 5))
    plt.scatter(efp_idxs, efps)
    plt.xlabel("Time (fs)")
    plt.ylabel("Potential Energy (kJ/mol)")
    plt.show()
    plt.savefig(f"{save_dir}/efps/{rollout}.png")
    plt.close()
    return fig


def plot_path_alanine(save_dir, positions, target_position, last_idx):
    angle_1 = [6, 8, 14, 16]
    angle_2 = [1, 6, 8, 14]

    psi = compute_dihedral(positions[:, :, angle_1, :]).detach().cpu().numpy()
    phi = compute_dihedral(positions[:, :, angle_2, :]).detach().cpu().numpy()
    target_psi = (
        compute_dihedral(target_position[:, angle_1, :].unsqueeze(0))
        .detach()
        .cpu()
        .numpy()
    )
    target_phi = (
        compute_dihedral(target_position[:, angle_2, :].unsqueeze(0))
        .detach()
        .cpu()
        .numpy()
    )

    for i in range(positions.shape[0]):
        plt.clf()
        plt.close()
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])
        potential = AlaninePotential()
        xs = np.arange(-np.pi, np.pi + 0.1, 0.1)
        ys = np.arange(-np.pi, np.pi + 0.1, 0.1)
        x, y = np.meshgrid(xs, ys)
        inp = torch.tensor(np.array([x, y])).view(2, -1).T

        z = potential.potential(inp)
        z = z.view(y.shape[0], y.shape[1])

        plt.contourf(xs, ys, z, levels=100, zorder=0)

        ax.plot(
            phi[i, : last_idx[i] + 1],
            psi[i, : last_idx[i] + 1],
            marker="o",
            linestyle="None",
            markersize=2,
            alpha=1.0,
        )

        ax.scatter(
            phi[:1, 0],
            psi[:1, 0],
            edgecolors="black",
            c="w",
            zorder=100,
            s=100,
            marker="*",
        )

        ax.scatter(
            target_phi[:1, 0],
            target_psi[:1, 0],
            edgecolors="w",
            c="w",
            zorder=100,
            s=10,
        )

        plt.xlabel("phi")
        plt.ylabel("psi")
        plt.show()
        plt.savefig(f"{save_dir}/path/{i}.png")
        plt.close()


def plot_path_chignolin(save_dir, positions, last_idx):
    asp3od_thr6og, asp3n_thr8o = chignolin_h_bond(positions)
    asp3od_thr6og = asp3od_thr6og.detach().cpu().numpy()
    asp3n_thr8o = asp3n_thr8o.detach().cpu().numpy()

    for i in range(positions.shape[0]):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        plt.xlim([0, 1.5])
        plt.ylim([0, 2])
        if last_idx[i] > 0:
            ax.plot(
                asp3od_thr6og[i][: last_idx[i] + 1],
                asp3n_thr8o[i][: last_idx[i] + 1],
                marker="o",
                linestyle="None",
                markersize=2,
                alpha=1.0,
            )

            if (
                asp3od_thr6og[i][last_idx[i]] < 0.35
                and asp3n_thr8o[i][last_idx[i]] < 0.35
            ):
                np.save(
                    f"{save_dir}/path/{i}_36.npy",
                    asp3od_thr6og[i][: last_idx[i] + 1],
                )
                np.save(
                    f"{save_dir}/path/{i}_38.npy", asp3n_thr8o[i][: last_idx[i] + 1]
                )

        plt.plot([0, 0.35], [0, 0], color="black")
        plt.plot([0.35, 0.35], [0, 0.35], color="black")
        plt.plot([0.35, 0], [0.35, 0.35], color="black")
        plt.plot([0, 0], [0.35, 0], color="black")

        plt.xlabel("Asp3OD-Thr6OG")
        plt.ylabel("Asp3N-Thr8O")
        plt.show()
        plt.savefig(f"{save_dir}/path/{i}.png")
        plt.close()


def plot_hand(save_dir, positions, last_idx):
    handed = poly_handed(positions)
    handed = handed.detach().cpu().numpy()

    for i in range(handed.shape[0]):
        plt.clf()
        plt.close()
        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(111)
        plt.ylim([-1, 1])

        cm = plt.get_cmap("gist_rainbow")
        ax.set_prop_cycle(
            color=[cm(1.0 * i / positions.shape[0]) for i in range(positions.shape[0])]
        )

        ax.plot(
            handed[i][: last_idx[i] + 1],
            marker="o",
            linestyle="None",
            markersize=2,
            alpha=1.0,
        )

        np.save(f"{save_dir}/path/{i}.npy", handed[i][: last_idx[i] + 1])

        plt.xlabel("Time (fs)")
        plt.ylabel("Handedness")
        plt.show()
        plt.savefig(f"{save_dir}/path/{i}.png")
        plt.close()


def plot_3D_view(save_dir, start_file, positions, potentials, last_idx):
    positions = positions.detach().cpu().numpy()
    for i in tqdm(range(positions.shape[0]), desc="Plot 3D views"):
        if last_idx[i] > 0:
            traj = md.load_pdb(start_file)
            for j in [0, potentials[i].argmax(), last_idx[i]]:
                traj.xyz = positions[i, j]
                traj.save(f"{save_dir}/3D_views/{i}_{j}.pdb")

            for j in range(last_idx[i] + 1):
                traj.xyz = positions[i, j]

                if j == 0:
                    trajs = traj
                else:
                    trajs = trajs.join(traj)
            trajs.save(f"{save_dir}/3D_views/{i}.h5")


def plot_potential(save_dir, potentials, last_idx):
    potentials = potentials.detach().cpu().numpy()

    for i in tqdm(range(potentials.shape[0]), desc="Plot potentials"):
        if last_idx[i] > 0:
            plt.figure(figsize=(16, 2))
            pot = potentials[i][: last_idx[i] + 1]
            np.save(f"{save_dir}/potential/{i}.npy", pot)
            plt.plot(pot)
            plt.xlabel("Time (fs)")
            plt.ylabel("Potential Energy (kJ/mol)")
            plt.show()
            plt.savefig(f"{save_dir}/potential/{i}.png")
            plt.close()
