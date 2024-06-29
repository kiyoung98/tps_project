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
parser.add_argument("--project", default="2d", type=str)
parser.add_argument("--model_path", default="", type=str)
parser.add_argument("--save_dir", default="results", type=str)
parser.add_argument("--date", default="date", type=str, help="Date of the training")
parser.add_argument(
    "--save_freq", default=100, type=int, help="Frequency of saving in  rollouts"
)

# Policy Config
parser.add_argument(
    "--force", action="store_true", help="Predict force otherwise potential"
)

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
    "--train_temperature", default=1200, type=float, help="Temperature for training"
)
parser.add_argument(
    "--max_grad_norm", default=10, type=int, help="Maximum norm of gradient to clip"
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

A = torch.tensor([-1.118, 0], dtype=torch.float32).to(args.device)
B = torch.tensor([1.1180, 0], dtype=torch.float32).to(args.device)

std = np.sqrt(2 * kbT * args.timestep)
energy_shift = 0


def expected_distance(last_position):
    dists = (last_position - B.unsqueeze(0)).square().mean((1))
    mean_dist, std_dist = dists.mean().item(), dists.std().item()
    return mean_dist, std_dist


class Logger:
    def __init__(self, args):
        self.type = args.type
        self.wandb = args.wandb
        self.save_freq = args.save_freq if args.type == "train" else 1

        self.best_loss = float("inf")

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
        loss,
        policy,
        rollout,
        last_idx,
        positions,
        last_position,
    ):
        ed, std_ed = expected_distance(last_position)
        len, std_len = last_idx.float().mean().item(), last_idx.float().std().item()

        # Log
        if self.type == "train":
            self.logger.info(
                "-----------------------------------------------------------"
            )
            self.logger.info(f"Rollout: {rollout}")
            self.logger.info(f"loss: {loss}")
            if loss < self.best_loss:
                self.best_loss = loss
                torch.save(policy.state_dict(), f"{self.save_dir}/policy.pt")

        if self.wandb:
            log = {
                "loss": loss,
                "log_z": policy.log_z.item(),
                "ed": ed,
                "len": len,
            }

            wandb.log(log, step=rollout)

        self.logger.info(f"log_z: {policy.log_z.item()}")
        self.logger.info(f"ed: {ed}")
        self.logger.info(f"len: {len}")
        self.logger.info(f"std_ed: {std_ed}")
        self.logger.info(f"std_len: {std_len}")

        if rollout % self.save_freq == 0:
            torch.save(policy.state_dict(), f"{self.save_dir}/policies/{rollout}.pt")

            fig_paths = plot_paths(self.save_dir, rollout, positions, last_idx)

            if self.wandb:
                log = {"paths": wandb.Image(fig_paths)}
                wandb.log(log, step=rollout)


class FlowNetAgent:
    def __init__(self, args):
        self.policy = Toy(args)
        self.replay = ReplayBuffer(args)

    def sample(self, args):
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

        position = A.unsqueeze(0)
        positions[:, 0] = position

        for s in tqdm(range(args.num_steps), desc="Sampling"):
            noise = noises[:, s]
            bias = args.bias_scale * self.policy(position.detach()).squeeze().detach()
            force = -grad(position)
            mean = position + force * args.timestep
            position = position + (force + bias) * args.timestep + noise * std

            positions[:, s + 1] = position
            actions[:, s] = position - mean

        log_md_reward = -0.5 * torch.square(actions / std).mean((1, 2))
        log_target_reward = (
            -0.5
            * torch.square((B.view(1, 1, -1) - positions) / std).mean(2)
            / args.sigma
        )
        log_target_reward, last_idx = log_target_reward.max(1)
        log_reward = log_md_reward + log_target_reward

        if args.type == "train":
            self.replay.add((positions, actions, log_reward))

        log = {
            "last_idx": last_idx,
            "positions": positions,
            "last_position": positions[torch.arange(args.num_samples), last_idx],
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

        biases = args.bias_scale * self.policy(positions[:, :-1])

        log_z = self.policy.log_z
        log_forward = -0.5 * torch.square((biases - actions) / std).mean((1, 2))

        loss = (log_z + log_forward - log_reward).square().mean()

        loss.backward()

        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], args.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()
        return loss.item()


class Toy(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.force = args.force

        self.input_dim = 2
        # self.input_dim = 4

        if self.force:
            self.output_dim = 2
        else:
            self.output_dim = 1

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
        # feature1 = torch.exp(-torch.norm(pos - A, dim=-1))
        # feature2 = torch.exp(-torch.norm(pos - B, dim=-1))
        # a = torch.cat((pos, feature1, feature2), dim=-1)

        if not self.force:
            pos.requires_grad = True

        out = self.mlp(pos.reshape(-1, self.input_dim))

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


# Wei's potential energy function
def energy(pos):
    x, y = pos
    term_1 = energy_shift * y
    term_2 = 4 * (1 - x**2 - y**2) ** 2
    term_3 = 2 * (x**2 - 2) ** 2
    term_4 = ((x + y) ** 2 - 1) ** 2
    term_5 = ((x - y) ** 2 - 1) ** 2
    eng = term_1 + ((term_2 + term_3 + term_4 + term_5 - 2.0) / 6.0)
    return eng


# def energy_3d(pos):
#     x, y, z = pos
#     # 기존 2D 항목을 3D로 확장
#     term_1 = 4 * (1 - x**2 - y**2 - z**2) ** 2
#     term_2 = 2 * (x**2 - 2) ** 2
#     term_3 = ((x + y) ** 2 - 1) ** 2
#     term_4 = ((x - y) ** 2 - 1) ** 2
#     # z에 대한 새로운 항목 추가
#     term_5 = 2 * (z**2 - 2) ** 2
#     term_6 = ((x + z) ** 2 - 1) ** 2
#     term_7 = ((y + z) ** 2 - 1) ** 2
#     term_8 = ((x - z) ** 2 - 1) ** 2
#     term_9 = ((y - z) ** 2 - 1) ** 2
#     # 모든 항목을 결합하여 3D 에너지 계산
#     eng = ((term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9 - 3.0) / 9.0)
#     return eng


def grad(pos):
    x, y = pos[:, 0], pos[:, 1]
    grad = torch.zeros_like(pos)
    x_square = torch.square(x)
    y_square = torch.square(y)
    xy_square = torch.square(x + y)
    xminusy_square = torch.square(x - y)
    term1 = 1 - x_square - y_square

    grad[:, 0] = (
        1
        / 6.0
        * (
            -16 * torch.multiply(term1, x)
            + 8 * (x_square - 2) * x
            + 4 * (xy_square - 1) * (x + y)
            + 4 * (xminusy_square - 1) * (x - y)
        )
    )
    grad[:, 1] = energy_shift + 1 / 6.0 * (
        -16 * torch.multiply(term1, y)
        + 4 * (xy_square - 1) * (x + y)
        - 4 * (xminusy_square - 1) * (x - y)
    )
    return grad


def plot_paths(save_dir, rollout, positions, last_idx):
    fig = plt.figure(figsize=(10, 5))
    positions = positions.cpu().numpy()
    A_cpu = A.cpu().numpy()
    B_cpu = B.cpu().numpy()

    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    Z = energy((X, Y))

    plt.contour(X, Y, Z, levels=100, cmap="RdGy")

    for i in range(positions.shape[0]):
        plt.scatter(positions[i, : last_idx[i], 0], positions[i, : last_idx[i], 1], s=3)
        plt.plot(
            positions[i, : last_idx[i], 0],
            positions[i, : last_idx[i], 1],
            linewidth=1,
            alpha=0.5,
        )

    plt.scatter(
        [A_cpu[0]],
        [A_cpu[1]],
        zorder=1e3,
        s=100,
        c="b",
        marker="o",
        label="start state",
    )
    plt.scatter(
        [B_cpu[0]], [B_cpu[1]], zorder=1e3, s=100, c="b", marker="*", label="end state"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    # plt.title("Trajectory of state on 2D system")
    plt.legend()
    plt.savefig(f"{save_dir}/paths/{rollout}.png")
    plt.close()
    return fig


if __name__ == "__main__":
    if args.wandb:
        wandb.init(project=args.project, config=args)
    agent = FlowNetAgent(args)
    logger = Logger(args)

    if args.type == "eval":
        agent.policy.load_state_dict(torch.load(args.model_path))
        logger.info("Start evaluation")
        log = agent.sample(args)
        logger.log(0, agent.policy, 0, **log)
        logger.info("Finish evaluation")

    else:
        torch.manual_seed(args.seed)
        logger.info("Start training")
        for rollout in range(args.num_rollouts):
            print(f"Rollout: {rollout}")

            log = agent.sample(args)

            loss = 0
            for _ in tqdm(range(args.trains_per_rollout), desc="Training"):
                loss += agent.train(args)
            loss = loss / args.trains_per_rollout

            logger.log(loss, agent.policy, rollout, **log)
        logger.info("Finish training")
