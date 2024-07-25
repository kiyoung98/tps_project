import wandb
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--type", default="train", type=str)
parser.add_argument("--device", default="cuda", type=str)

# Logger Config
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--project", default="synthetic", type=str)
parser.add_argument("--model_path", default="", type=str)
parser.add_argument("--save_dir", default="results", type=str)
parser.add_argument("--date", default="date", type=str, help="Date of the training")

# Policy Config
parser.add_argument(
    "--force", action="store_true", help="Predict force otherwise potential"
)
parser.add_argument(
    "--log_z", default=0.0, type=float, help="Learning rate of estimator for log Z"
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
    "--temperature", default=1200, type=float, help="Temperature for evaluation"
)

# Training Config
parser.add_argument(
    "--start_temperature", default=4800, type=float, help="Temperature for training"
)
parser.add_argument(
    "--end_temperature", default=1200, type=float, help="Temperature for training"
)
parser.add_argument(
    "--num_rollouts", default=1000, type=int, help="Number of rollouts (or sampling)"
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

kB = 8.6173303e-5  # Boltzmann constant in eV/K
kbT = kB * args.temperature  # in eV

start_position = torch.tensor([-1.118, 0], dtype=torch.float32).to(args.device)
target_position = torch.tensor([1.1180, 0], dtype=torch.float32).to(args.device)

std = np.sqrt(2 * kbT * args.timestep)
normal = torch.distributions.Normal(0, std)

kbTs = (
    torch.linspace(args.start_temperature, args.end_temperature, args.num_rollouts) * kB
)
train_stds = torch.sqrt(2 * kbTs * args.timestep)


def system(pos, force=True):
    if force:
        pos.requires_grad_(True)
    x = pos[:, 0]
    y = pos[:, 1]
    term_1 = 4 * (1 - x**2 - y**2) ** 2
    term_2 = 2 * (x**2 - 2) ** 2
    term_3 = ((x + y) ** 2 - 1) ** 2
    term_4 = ((x - y) ** 2 - 1) ** 2
    potential = (term_1 + term_2 + term_3 + term_4 - 2.0) / 6.0
    if force:
        force = -torch.autograd.grad(potential.sum(), pos)[0]
        return potential, force
    else:
        return potential


def plot_system(pos):
    x, y = pos
    term_1 = 4 * (1 - x**2 - y**2) ** 2
    term_2 = 2 * (x**2 - 2) ** 2
    term_3 = ((x + y) ** 2 - 1) ** 2
    term_4 = ((x - y) ** 2 - 1) ** 2
    potential = (term_1 + term_2 + term_3 + term_4 - 2.0) / 6.0
    return potential


class Toy(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.force = args.force
        self.dist_feat = args.dist_feat

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

        self.log_z = nn.Parameter(torch.tensor(args.log_z))

        self.to(args.device)

    def forward(self, pos):
        if not self.force:
            pos.requires_grad = True
        if self.dist_feat:
            dist = torch.norm(pos - target_position, dim=-1, keepdim=True)
            pos_ = torch.cat([pos, dist], dim=-1)
        else:
            pos_ = pos

        out = self.mlp(pos_.reshape(-1, self.input_dim))

        if not self.force:
            f = -torch.autograd.grad(
                out.sum(), pos, create_graph=True, retain_graph=True
            )[0]
        else:
            f = out.view(*pos.shape)

        return f


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
        self.buffer_size = args.buffer_size
        self.num_samples = args.num_samples

    def add(self, data):
        indices = torch.arange(self.idx, self.idx + self.num_samples) % self.buffer_size
        self.idx += self.num_samples

        self.positions[indices], self.actions[indices], self.log_reward[indices] = data

    def sample(self):
        indices = torch.randperm(min(self.idx, self.buffer_size))[: self.num_samples]
        return self.positions[indices], self.actions[indices], self.log_reward[indices]


class FlowNetAgent:
    def __init__(self, args):
        self.policy = Toy(args)
        self.replay = ReplayBuffer(args)

    def sample(self, args, train_std, training=True):
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

        potential = system(start_position.unsqueeze(0), False)

        position = start_position.unsqueeze(0)
        positions[:, 0] = position
        potentials[:, 0] = potential

        for s in tqdm(range(args.num_steps), desc="Sampling"):
            noise = noises[:, s]
            bias = self.policy(position.detach()).squeeze().detach()
            potential, force = system(position)
            mean = position + force * args.timestep
            position = position + (force + bias) * args.timestep + train_std * noise
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
            "last_idx": last_idx,
            "positions": positions,
            "log_likelihood": log_md_reward.sum(-1).mean(1),
        }
        return log

    def train(self, args):
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


class Metric:
    def expected_distance(self, positions, last_idx):
        last_position = positions[torch.arange(len(positions)), last_idx]
        dists = (last_position - target_position.unsqueeze(0)).square().mean((1))
        return dists.mean().item(), dists.std().item()

    def target_hit_percentage(self, positions, last_idx):
        last_position = positions[torch.arange(len(positions)), last_idx]
        hits = (last_position - target_position.unsqueeze(0)).square().sum(
            1
        ).sqrt() < 0.5
        thp = 100 * hits.sum() / len(hits)
        return thp

    def log_likelihood(self, log_likelihood):
        return log_likelihood.mean().item(), log_likelihood.std().item()


def plot_paths(save_dir, rollout, positions, last_idx):
    fig = plt.figure(figsize=(10, 5))
    positions = positions.detach().cpu().numpy()

    x = np.linspace(-1.5, 1.5, 400)
    y = np.linspace(-1.5, 1.5, 400)
    X, Y = np.meshgrid(x, y)
    Z = plot_system((X, Y))

    plt.contour(X, Y, Z, levels=100, cmap="RdGy")

    for i in range(positions.shape[0]):
        plt.scatter(positions[i, : last_idx[i], 0], positions[i, : last_idx[i], 1], s=3)
        plt.plot(
            positions[i, : last_idx[i], 0],
            positions[i, : last_idx[i], 1],
            linewidth=1,
            alpha=0.5,
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"{save_dir}/{rollout}.png")
    plt.close()
    return fig


if __name__ == "__main__":
    if args.wandb:
        wandb.init(project=args.project, config=args)
    torch.manual_seed(args.seed)
    agent = FlowNetAgent(args)
    metric = Metric()
    save_dir = os.path.join(
        args.save_dir, args.project, args.date, args.type, str(args.seed)
    )
    for name in ["policy"]:
        if not os.path.exists(f"{save_dir}/{name}"):
            os.makedirs(f"{save_dir}/{name}")
    print("Start training")
    for rollout in range(args.num_rollouts):
        args.num_samples = 512
        log = agent.sample(args, train_stds[rollout])
        dist, dist_std = metric.expected_distance(log["positions"], log["last_idx"])
        ll, ll_std = metric.log_likelihood(log["log_likelihood"])
        thp = metric.target_hit_percentage(log["positions"], log["last_idx"])
        loss = 0
        for _ in tqdm(range(args.trains_per_rollout), desc="Training"):
            loss += agent.train(args)
        loss = loss / args.trains_per_rollout
        if rollout % 10 == 0 or rollout == args.num_rollouts - 1:
            torch.save(agent.policy.state_dict(), f"{save_dir}/policy/{rollout}.pt")
            train_paths = plot_paths(
                save_dir, rollout, log["positions"], log["last_idx"]
            )
            if args.wandb:
                L = {
                    "log_z": agent.policy.log_z.item(),
                    "dist": dist,
                    "dist_std": dist_std,
                    "ll": ll,
                    "ll_std": ll_std,
                    "thp": thp,
                    "loss": loss,
                    "train_paths": wandb.Image(train_paths),
                }
                wandb.log(L, step=rollout)

            args.num_samples = 32
            log = agent.sample(args, std, False)
            dist, dist_std = metric.expected_distance(log["positions"], log["last_idx"])
            ll, ll_std = metric.log_likelihood(log["log_likelihood"])
            thp = metric.target_hit_percentage(log["positions"], log["last_idx"])
            eval_paths = plot_paths(
                save_dir, f"eval_{rollout}", log["positions"], log["last_idx"]
            )

            if args.wandb:
                L = {
                    "eval_dist": dist,
                    "eval_dist_std": dist_std,
                    "eval_ll": ll,
                    "eval_ll_std": ll_std,
                    "eval_thp": thp,
                    "eval_paths": wandb.Image(eval_paths),
                }
                wandb.log(L, step=rollout)
        #     print(f"Train Rollout: {rollout}, ED: {dist}, THP: {thp}, LL: {ll}")
        # print(f"Rollout: {rollout}, Loss: {loss}")

    # args.num_samples = 32
    # log = agent.sample(args, std, False)
    # dist, dist_std = metric.expected_distance(log["positions"], log["last_idx"])
    # ll, ll_std = metric.log_likelihood(log["log_likelihood"])
    # thp = metric.target_hit_percentage(log["positions"], log["last_idx"])
    # print(f"Eval Rollout: {rollout}, ED: {dist}, THP: {thp}, LL: {ll}")
    # plot_paths(save_dir, f"eval_{rollout}", log["positions"], log["last_idx"])
