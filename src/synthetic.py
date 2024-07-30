import wandb
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import logging

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--type", default="train", type=str)
parser.add_argument("--device", default="cuda", type=str)

# Logger Config
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--project", default="synthetic", type=str)
parser.add_argument("--save_dir", default="results", type=str)
parser.add_argument("--date", default="date", type=str, help="Date of the training")
parser.add_argument(
    "--save_freq", default=10, type=int, help="Frequency of saving in  rollouts"
)

# Policy Config
parser.add_argument(
    "--force", action="store_true", help="Predict force otherwise potential"
)
parser.add_argument("--dist_feat", action="store_true")

# Sampling Config
parser.add_argument("--num_steps", default=1000, type=int, help="Length of paths")
parser.add_argument("--bias_scale", default=1, type=float, help="Scale factor of bias")
parser.add_argument(
    "--timestep", default=0.01, type=float, help="Timestep of integrator"
)
parser.add_argument("--sigma", default=1, type=float, help="Control reward of arrival")
parser.add_argument(
    "--num_samples", default=512, type=int, help="Number of paths to sample"
)
parser.add_argument(
    "--plot_samples", default=64, type=int, help="Number of paths to sample"
)
parser.add_argument(
    "--temperature", default=1200, type=float, help="Temperature for evaluation"
)

# Training Config
parser.add_argument("--start_temperature", default=4800, type=float)
parser.add_argument("--end_temperature", default=1200, type=float)
parser.add_argument(
    "--num_rollouts", default=100, type=int, help="Number of rollouts (or sampling)"
)
parser.add_argument(
    "--log_z_lr", default=1e-2, type=float, help="Learning rate of estimator for log Z"
)
parser.add_argument(
    "--policy_lr",
    default=1e-3,
    type=float,
    help="Learning rate of bias potential or force",
)
parser.add_argument(
    "--buffer_size",
    default=50000,
    type=int,
    help="Size of buffer which stores sampled paths",
)
parser.add_argument(
    "--trains_per_rollout",
    default=100,
    type=int,
    help="Number of training per rollout in a rollout",
)

args = parser.parse_args()

kB = 8.6173303e-5
kbT = kB * args.temperature
std = np.sqrt(2 * kbT * args.timestep)
normal = torch.distributions.Normal(0, std)
start_position = torch.tensor([-1.118, 0], dtype=torch.float32).to(args.device)
target_position = torch.tensor([1.118, 0], dtype=torch.float32).to(args.device)

std = np.sqrt(2 * kbT * args.timestep)

kbTs = (
    torch.linspace(args.start_temperature, args.end_temperature, args.num_rollouts) * kB
)
train_stds = torch.sqrt(2 * kbTs * args.timestep)


def system(pos):
    pos.requires_grad_(True)
    x = pos[:, 0]
    y = pos[:, 1]
    term_1 = 4 * (1 - x**2 - y**2) ** 2
    term_2 = 2 * (x**2 - 2) ** 2
    term_3 = ((x + y) ** 2 - 1) ** 2
    term_4 = ((x - y) ** 2 - 1) ** 2
    potential = (term_1 + term_2 + term_3 + term_4 - 2.0) / 6.0
    force = -torch.autograd.grad(potential.sum(), pos)[0]
    return potential, force


class NeuralBias(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.force:
            self.output_dim = 2
        else:
            self.output_dim = 1

        if args.dist_feat:
            self.input_dim = 3
        else:
            self.input_dim = 2

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, self.output_dim, bias=False),
        )

        self.log_z = nn.Parameter(torch.tensor(0.0))

        self.to(args.device)

    def forward(self, pos):
        if not args.force:
            pos.requires_grad = True
        if args.dist_feat:
            dist = torch.norm(pos - target_position, dim=-1, keepdim=True)
            pos_ = torch.cat([pos, dist], dim=-1)
        else:
            pos_ = pos

        out = self.mlp(pos_.reshape(-1, self.input_dim))

        if not args.force:
            f = -torch.autograd.grad(
                out.sum(), pos, create_graph=True, retain_graph=True
            )[0]
        else:
            f = out.view(*pos.shape)

        return f


class Logger:
    def __init__(self, args):
        self.type = args.type
        self.wandb = args.wandb
        self.save_freq = args.save_freq if args.type == "train" else 1

        self.best_ed = float("inf")
        self.plot_samples = args.plot_samples

        self.save_dir = os.path.join(
            args.save_dir, args.project, args.date, args.type, str(args.seed)
        )

        for name in ["paths", "policies"]:
            if not os.path.exists(f"{self.save_dir}/{name}"):
                os.makedirs(f"{self.save_dir}/{name}")

        # Logger basic configurations
        self.logger = logging.getLogger("tps")
        self.logger.setLevel(logging.INFO)

        # File handler
        log_file = args.type + ".log"
        log_file = os.path.join(self.save_dir, log_file)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

        for k, v in vars(args).items():
            self.logger.info(f"{k}: {v}")

    def info(self, message):
        if self.logger:
            self.logger.info(message)

    def log(
        self,
        policy,
        rollout,
        positions,
        potentials,
        log_likelihood,
        last_idx,
    ):
        ed, ed_std = self.expected_distance(positions, last_idx)
        ll, ll_std = self.log_likelihood(log_likelihood)
        thp, etp, efp, etp_std, efp_std = self.cv_metrics(
            positions, potentials, last_idx
        )
        len, std_len = last_idx.float().mean().item(), last_idx.float().std().item()

        # Log
        if self.type == "train":
            if ed < self.best_ed:
                self.best_ed = ed
                torch.save(policy.state_dict(), f"{self.save_dir}/policy.pt")

        if self.wandb:
            log = {
                "log_z": policy.log_z.item(),
                "ed": ed,
                "ll": ll,
                "thp": thp,
                "etp": etp,
                "efp": efp,
                "len": len,
                "ed_std": ed_std,
                "ll_std": ll_std,
                "etp_std": etp_std,
                "efp_std": efp_std,
                "std_len": std_len,
            }

            wandb.log(log, step=rollout)

        self.logger.info(f"log_z: {policy.log_z.item()}")
        self.logger.info(f"ed: {ed} ± {ed_std}")
        self.logger.info(f"ll: {ll} ± {ll_std}")
        self.logger.info(f"thp: {thp}")
        self.logger.info(f"etp: {etp} ± {etp_std}")
        self.logger.info(f"efp: {efp} ± {efp_std}")
        self.logger.info(f"len: {len} ± {std_len}")

        if rollout % self.save_freq == 0:
            torch.save(policy.state_dict(), f"{self.save_dir}/policies/{rollout}.pt")

            fig_paths = self.plot_paths(rollout, positions, last_idx)

            if self.wandb:
                log = {"paths": wandb.Image(fig_paths)}
                wandb.log(log, step=rollout)

    def expected_distance(self, positions, last_idx):
        last_position = positions[torch.arange(len(positions)), last_idx]
        dists = (last_position - target_position.unsqueeze(0)).square().mean((1))
        return dists.mean().item(), dists.std().item()

    def cv_metrics(self, positions, potentials, last_idx):
        etps, efps, etp_idxs, efp_idxs = [], [], [], []
        last_position = positions[torch.arange(len(positions)), last_idx]
        hits = (last_position - target_position.unsqueeze(0)).square().sum(
            1
        ).sqrt() < 0.5

        for i, hit_idx in enumerate(hits):
            if hit_idx:
                etp, idx = potentials[i][: last_idx[i] + 1].max(0)
                etps.append(etp)
                etp_idxs.append(idx.item())

                efp = potentials[i][last_idx[i]]
                efps.append(efp)
                efp_idxs.append(last_idx[i].item())

                # np.save(
                #     f"{self.save_dir}/paths/{i}.npy",
                #     positions[i].detach().cpu().numpy(),
                # )

        if len(etps) > 0:
            etps = torch.tensor(etps)
            efps = torch.tensor(efps)

            etp = etps.mean().item()
            efp = efps.mean().item()

            etp_std = etps.std().item()
            efp_std = efps.std().item()
        else:
            etp = None
            efp = None

            etp_std = None
            efp_std = None

        thp = 100 * hits.sum() / len(hits)
        return thp, etp, efp, etp_std, efp_std

    def log_likelihood(self, log_likelihood):
        return log_likelihood.mean().item(), log_likelihood.std().item()

    def plot_paths(self, rollout, positions, last_idx):
        fig, ax = plt.subplots(figsize=(7, 7))
        positions = positions[: self.plot_samples].detach().cpu().numpy()

        z_num = 100
        circle_size = 1200
        saddle_size = 2400

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        x = np.linspace(-1.5, 1.5, 400)
        y = np.linspace(-1.5, 1.5, 400)
        X, Y = np.meshgrid(x, y)

        term_1 = 4 * (1 - X**2 - Y**2) ** 2
        term_2 = 2 * (X**2 - 2) ** 2
        term_3 = ((X + Y) ** 2 - 1) ** 2
        term_4 = ((X - Y) ** 2 - 1) ** 2
        Z = (term_1 + term_2 + term_3 + term_4 - 2.0) / 6.0

        ax.contourf(X, Y, Z, levels=z_num, zorder=0, vmax=3)

        # Plot start and end positions
        ax.scatter(
            [start_position[0].item()],
            [start_position[1].item()],
            edgecolors="black",
            c="w",
            zorder=z_num,
            s=circle_size,
        )
        ax.scatter(
            [target_position[0].item()],
            [target_position[1].item()],
            edgecolors="black",
            c="w",
            zorder=z_num,
            s=circle_size,
        )

        saddle_points = [(0, 1), (0, -1)]
        for saddle in saddle_points:
            ax.scatter(
                saddle[0],
                saddle[1],
                edgecolors="black",
                c="w",
                zorder=z_num,
                s=saddle_size,
                marker="*",
            )

        cm = plt.get_cmap("gist_rainbow")

        ax.set_prop_cycle(
            color=[cm(1.0 * i / len(positions)) for i in range(len(positions))]
        )

        for i in range(len(positions)):
            ax.plot(
                positions[i, : last_idx[i], 0],
                positions[i, : last_idx[i], 1],
                marker="o",
                linestyle="None",
                markersize=2,
                alpha=1.0,
                zorder=z_num - 1,
            )

        # Plot basic configs
        ax.set_xlabel("x", fontsize=24, fontweight="medium")
        ax.set_ylabel("y", fontsize=24, fontweight="medium")
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        plt.tight_layout()

        plt.savefig(f"{self.save_dir}/paths/{rollout}.png")
        plt.close()

        return fig


class FlowNetAgent:
    def __init__(self, args):
        self.policy = NeuralBias(args)
        self.replay = ReplayBuffer(args)

    def sample(self, std, training=True):
        positions = torch.zeros(
            (args.num_samples, args.num_steps + 1, 2),
            device=args.device,
        )
        actions = torch.zeros(
            (args.num_samples, args.num_steps, 2),
            device=args.device,
        )
        noises = torch.normal(
            torch.zeros(
                (args.num_samples, args.num_steps, 2),
                device=args.device,
            ),
            torch.ones(
                (args.num_samples, args.num_steps, 2),
                device=args.device,
            ),
        )
        potentials = torch.zeros(
            (args.num_samples, args.num_steps + 1), device=args.device
        )

        potential = system(start_position.unsqueeze(0))[0]

        position = start_position.unsqueeze(0)
        positions[:, 0] = position
        potentials[:, 0] = potential

        for s in range(args.num_steps):
            noise = noises[:, s]
            bias = self.policy(position.detach()).squeeze().detach()
            # bias = torch.zeros_like(noise)
            potential, force = system(position)
            mean = position + force * args.timestep
            position = position + (force + bias) * args.timestep + std * noise
            positions[:, s + 1] = position
            potentials[:, s + 1] = potential
            actions[:, s] = position - mean

        log_md_reward = normal.log_prob(actions.detach())
        log_target_reward = (
            -0.5
            * torch.square((positions - target_position.view(1, 1, -1)) / std).mean(2)
            / args.sigma
        )
        log_target_reward, last_idx = log_target_reward.max(1)
        log_reward = log_md_reward.mean((1, 2)) + log_target_reward

        if training:
            self.replay.add((positions.detach(), actions.detach(), log_reward.detach()))

        log = {
            "positions": positions,
            "potentials": potentials,
            "log_likelihood": log_md_reward.sum(-1).mean(1),
            "last_idx": last_idx,
        }
        return log

    def train(self):
        optimizer = torch.optim.Adam(
            [
                {"params": [self.policy.log_z], "lr": args.log_z_lr},
                {"params": self.policy.mlp.parameters(), "lr": args.policy_lr},
            ]
        )

        positions, actions, log_reward = self.replay.sample()

        biases = self.policy(positions[:, :-1].detach())

        log_z = self.policy.log_z
        log_forward = normal.log_prob(actions - biases).mean((1, 2))

        loss = (log_z + log_forward - log_reward).square().mean()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        return loss.item()


class ReplayBuffer:
    def __init__(self, args):
        self.positions = torch.zeros(
            (args.buffer_size, args.num_steps + 1, 2),
            device=args.device,
        )
        self.actions = torch.zeros(
            (args.buffer_size, args.num_steps, 2), device=args.device
        )
        self.log_reward = torch.zeros(args.buffer_size, device=args.device)

        self.idx = 0

    def add(self, data):
        indices = torch.arange(self.idx, self.idx + args.num_samples) % args.buffer_size
        self.idx += args.num_samples

        self.positions[indices], self.actions[indices], self.log_reward[indices] = data

    def sample(self):
        indices = torch.randperm(min(self.idx, args.buffer_size))[: args.num_samples]
        return self.positions[indices], self.actions[indices], self.log_reward[indices]


if __name__ == "__main__":
    if args.wandb:
        wandb.init(project=args.project, config=args)
    torch.manual_seed(args.seed)
    agent = FlowNetAgent(args)
    logger = Logger(args)

    for rollout in range(args.num_rollouts):
        args.num_samples = 512
        log = agent.sample(train_stds[rollout])
        # logger.log(agent.policy, rollout, **log)

        args.num_samples = 64
        log = agent.sample(std, training=False)
        logger.log(agent.policy, rollout, **log)

        loss = 0
        for _ in range(args.trains_per_rollout):
            loss += agent.train()
        loss = loss / args.trains_per_rollout
        if args.wandb:
            wandb.log({"loss": loss}, step=rollout)
