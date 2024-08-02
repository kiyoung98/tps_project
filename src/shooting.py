import random
import time
from tqdm import tqdm
import torch
import numpy as np

temperature = 1200
timestep = 0.01
num_iterations = 1000
device = "cuda"

kB = 8.6173303e-5
kbT = kB * temperature
std = np.sqrt(2 * kbT * timestep)
normal = torch.distributions.Normal(0, std)
start_position = torch.tensor([-1.118, 0], dtype=torch.float32)
target_position = torch.tensor([1.118, 0], dtype=torch.float32)

std = np.sqrt(2 * kbT * timestep)


def system(pos):
    pos.requires_grad_(True)
    x = pos[0]
    y = pos[1]
    term_1 = 4 * (1 - x**2 - y**2) ** 2
    term_2 = 2 * (x**2 - 2) ** 2
    term_3 = ((x + y) ** 2 - 1) ** 2
    term_4 = ((x - y) ** 2 - 1) ** 2
    potential = (term_1 + term_2 + term_3 + term_4 - 2.0) / 6.0
    force = -torch.autograd.grad(potential.sum(), pos)[0]
    pos.requires_grad_(False)
    return potential, force


def hits(current_position, target_position, start_position):
    return (current_position - target_position).square().sum().sqrt() < 0.3 or (
        current_position - start_position
    ).square().sum().sqrt() < 0.3


def is_near(position, reference_position):
    return (position - reference_position).square().sum().sqrt() < 0.3


def two_way_shooting(
    initial_state, final_state, initial_path, num_iterations, timestep
):
    num_successes = 0
    path = initial_path.clone()
    cumulative_runtimes = []
    start = time.time()
    for i in tqdm(range(num_iterations)):
        shooting_point_index = random.randint(0, len(path) - 1)
        shooting_point = path[shooting_point_index]

        perturbed_shooting_point = (
            10 * normal.sample(shooting_point.shape) + shooting_point
        )

        forward_path = [perturbed_shooting_point]
        current_pos = perturbed_shooting_point.clone()
        while not hits(current_pos, final_state, initial_state):
            potential, force = system(current_pos)
            current_pos = (
                current_pos + force * timestep + normal.sample(current_pos.shape)
            )
            forward_path.append(current_pos.clone())

        backward_path = [perturbed_shooting_point]
        current_pos = perturbed_shooting_point.clone()
        while not hits(current_pos, final_state, initial_state):
            potential, force = system(current_pos)
            current_pos = (
                current_pos + force * timestep + normal.sample(current_pos.shape)
            )
            backward_path.append(current_pos.clone())

        new_path = torch.stack(
            backward_path[::-1] + forward_path[1:],
        )

        if not (
            (
                is_near(new_path[0], target_position)
                and is_near(new_path[-1], target_position)
            )
            or (
                is_near(new_path[0], start_position)
                and is_near(new_path[-1], start_position)
            )
        ):
            path = new_path
            end = time.time()
            cumulative_runtimes.append(end - start)
            np.save(f"paths/shooting/path_{i}.npy", path.numpy())
            num_successes += 1
            print(f"Total runtime: {end - start}")
            print(f"Succes rate: {num_successes / (i+1)}")

    return path


# initial_path = np.load("initial_path.npy")
# initial_path = torch.tensor(initial_path, dtype=torch.float32)
initial_path = torch.linspace(start_position[0], target_position[0], 1000)
initial_path = torch.stack([initial_path, torch.zeros_like(initial_path)], dim=-1).to(
    torch.float32
)

two_way_shooting(
    start_position, target_position, initial_path, num_iterations, timestep
)
