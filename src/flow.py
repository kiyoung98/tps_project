import torch
import random
from tqdm import tqdm

import proxy
from utils.utils import get_log_normal, get_dist_matrix

class FlowNetAgent:
    def __init__(self, args, md):
        self.std = md.std
        self.masses = md.masses
        self.v_scale = md.v_scale
        self.f_scale = md.f_scale
        self.num_particles = md.num_particles

        self.replay = ReplayBuffer(args)
        self.policy = getattr(proxy, args.molecule.title())(args, md)

    def sample(self, args, mds, target_position, temperature):
        positions = torch.zeros((args.num_samples, args.num_steps+1, self.num_particles, 3), device=args.device)
        potentials = torch.zeros(args.num_samples, args.num_steps+1, device=args.device)
        actions = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)

        position, _, _, potential = mds.report()
        
        positions[:, 0] = position
        potentials[:, 0] = potential

        mds.set_temperature(temperature)
        for s in tqdm(range(args.num_steps)):
            bias = args.bias_scale * self.policy(position.unsqueeze(1), target_position).squeeze().detach()
            mds.step(bias)

            next_position, velocity, force, potential = mds.report()

            # extract noise which openmm does not provide
            deterministic_velocities = self.v_scale * velocity + self.f_scale * force / self.masses.view(1, -1, 1)
            noise = (next_position - position) / args.timestep - deterministic_velocities

            position = next_position
            
            positions[:, s+1] = position
            potentials[:, s+1] = potential
            actions[:, s] = bias / self.masses.view(1, -1, 1) + noise / self.f_scale
        mds.reset()
        
        start_position = positions[0, 0].unsqueeze(0).unsqueeze(0)
        last_position = position.unsqueeze(1)

        mds.reset()

        last_dist_matrix = get_dist_matrix(last_position)
        target_dist_matrix = get_dist_matrix(target_position)

        sqrt_mass = torch.sqrt(self.masses)
        terminal_log_reward = get_log_normal(sqrt_mass.view(1, 1, -1)*(last_dist_matrix-target_dist_matrix)*sqrt_mass.view(1, -1, 1)/args.terminal_std).mean((1, 2))
        intermediate_log_reward = get_log_normal(actions/self.std.view(1, -1, 1)).mean((1, 2, 3))
        log_reward = intermediate_log_reward + terminal_log_reward

        self.replay.add((positions, actions, start_position, target_position, log_reward))

        log = {
            'positions': positions, 
            'start_position': start_position,
            'last_position': last_position, 
            'target_position': target_position, 
            'potentials': potentials,
            'terminal_log_reward': terminal_log_reward,
            'log_reward': log_reward,
        }
        return log


    def train(self, args):
        policy_optimizers = torch.optim.SGD(self.policy.parameters(), lr=args.learning_rate)

        positions, actions, start_position, target_position, log_reward = self.replay.sample()

        biases = args.bias_scale * self.policy(positions[:, :-1], target_position)
        biases = biases / self.masses.view(1, -1, 1)
        
        if args.loss == 'tb':
            log_z = self.policy.get_log_z(start_position, target_position)
            log_forward = get_log_normal((biases-actions)/self.std.view(1, -1, 1)).mean((1, 2, 3))
            loss = torch.mean((log_z+log_forward-log_reward)**2)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
        
        policy_optimizers.step()
        policy_optimizers.zero_grad()
        return loss
    

class ReplayBuffer:
    def __init__(self, args):
        self.buffer = []
        self.buffer_size = args.buffer_size

    def add(self, data):
        self.buffer.append(data)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample(self):
        idx = random.randrange(len(self))
        return self.buffer[idx]

    def __len__(self):
        return len(self.buffer)