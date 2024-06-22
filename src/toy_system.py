import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda"
# training parameters
num_samples = 128  # number of batches in training
force = False
buffer_size = 2048
bias_scale = 1

# simulation parameters
dt = 0.01  # discretized time step
sqrt_dt = np.sqrt(dt)

temp = 1200  # temperature in K
kB = 8.6173303e-5  # Boltzmann constant in eV/K
kbT = kB * temp  # in eV

mlp_lr = 1e-3
log_z_lr = 1e-2

T_deadline = 4
num_steps = int(T_deadline / dt)

A = torch.tensor([-1.11802943, -0.00285716], dtype=torch.float32).to(device)
B = torch.tensor([1.11802943, -0.00285716], dtype=torch.float32).to(device)

std = np.sqrt(2 * kbT * dt)
energy_shift = 0.05


class FlowNetAgent:
    def __init__(self):
        self.policy = Toy()
        self.replay = ReplayBuffer()

    def sample(self):
        positions = torch.zeros(
            (num_samples, num_steps + 1, 2),
            device=device,
        )
        actions = torch.zeros(
            (num_samples, num_steps, 2),
            device=device,
        )

        positions[:, 0] = position

        for s in tqdm(range(num_steps), desc="Sampling"):
            bias = bias_scale * self.policy(position.detach()).squeeze().detach()
            position, noise = step(position)
            positions[:, s + 1] = position
            actions[:, s] = bias + noise
        mds.reset()

        log_reward = -0.5 * torch.square(actions / std).mean((1, 2, 3))

        if type == "train":
            self.replay.add((positions, actions, log_reward))

    def train(self):
        log_z_optimizer = torch.optim.Adam([self.policy.log_z], lr=log_z_lr)
        mlp_optimizer = torch.optim.Adam(self.policy.mlp.parameters(), lr=mlp_lr)

        positions, actions, log_reward = self.replay.sample()

        biases = bias_scale * self.policy(positions[:, :-1])

        log_z = self.policy.log_z
        log_forward = -0.5 * torch.square((biases - actions) / std).mean((1, 2, 3))
        loss = (log_z + log_forward - log_reward).square().mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.log_z)
        torch.nn.utils.clip_grad_norm_(self.policy.mlp.parameters())

        mlp_optimizer.step()
        log_z_optimizer.step()
        mlp_optimizer.zero_grad()
        log_z_optimizer.zero_grad()
        return loss.item()


class Toy(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_dim = 2
        self.output_dim = 2 if force else 1

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2, bias=False),
        )

        self.log_z = nn.Parameter(torch.tensor(0.0))

        self.to(device)

    def forward(self, pos):
        if not force:
            pos.requires_grad = True

        out = self.mlp(pos.reshape(-1, self.input_dim))

        if not force:
            f = -torch.autograd.grad(
                out.sum(), pos, create_graph=True, retain_graph=True
            )[0]
        else:
            f = out.view(*pos.shape)

        return f


class ReplayBuffer:
    def __init__(self):
        self.positions = torch.zeros(
            (buffer_size, num_steps + 1, 2),
            device=device,
        )
        self.actions = torch.zeros((buffer_size, num_steps, 2), device=device)
        self.log_reward = torch.zeros(buffer_size, device=device)

        self.idx = 0
        self.buffer_size = buffer_size
        self.num_samples = num_samples

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


# Higher dimension (parallel) of computing gradient
def grad(pos):
    x, y = pos[:, 0], pos[:, 1]
    grad = torch.zeros((pos.shape[0], 2))
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


def step(position):
    noise = np.random.normal(size=(position.shape[0], 2))
    return position - grad(position) * dt + noise * std, noise


def save_traces(init_x, x_history_numpy, path):
    init_x = init_x.cpu().numpy()
    fig, ax = plt.subplots()

    plt.plot(
        [A.numpy()[0], B.numpy()[0]],
        [A.numpy()[1], B.numpy()[1]],
        "ro",
        label="local minima",
    )
    plt.plot(init_x[0], init_x[1], "go", label="starting point")

    plt.scatter(np.array(x_history_numpy)[:, 0], np.array(x_history_numpy)[:, 1], s=3)
    plt.plot(
        np.array(x_history_numpy)[:, 0],
        np.array(x_history_numpy)[:, 1],
        linewidth=1,
        alpha=0.5,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectory of points following the gradient and Brownian motion")
    plt.legend()
    plt.savefig(path)
    plt.close()
