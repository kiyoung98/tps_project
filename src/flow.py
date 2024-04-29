import torch
from tqdm import tqdm

import proxy
from utils.utils import get_log_normal, get_dist_matrix

class FlowNetAgent:
    def __init__(self, args, md):
        self.v_scale = torch.tensor(md.v_scale, dtype=torch.float, device=args.device)
        self.f_scale = torch.tensor(md.f_scale.value_in_unit(md.f_scale.unit), dtype=torch.float, device=args.device)
        self.std = torch.tensor(md.std.value_in_unit(md.std.unit), dtype=torch.float, device=args.device)
        self.masses = torch.tensor(md.masses.value_in_unit(md.masses.unit), dtype=torch.float, device=args.device)

        self.policy = getattr(proxy, args.molecule.title())(args, md)

        if args.type == 'train':
            self.replay = ReplayBuffer(args, md)

    def sample(self, args, mds, target_position, temperature, biased=True):
        position, velocity, force, potential = mds.set()
        positions, velocities, forces, potentials, biases = [position], [velocity], [force], [potential], []

        mds.set_temperature(temperature)
        for _ in tqdm(range(args.num_steps)):
            position = torch.tensor(position, dtype=torch.float, device=args.device)
            if biased:
                bias = args.bias_scale * self.policy(position, target_position).squeeze().detach()
            else:
                bias = torch.zeros_like(position)
            position, velocity, force, potential = mds.step(bias)
            positions.append(position); velocities.append(velocity); forces.append(force); potentials.append(potential); biases.append(bias)

        positions = torch.tensor(positions, dtype=torch.float, device=args.device).transpose(0, 1)
        velocities = torch.tensor(velocities, dtype=torch.float, device=args.device).transpose(0, 1)
        forces = torch.tensor(forces, dtype=torch.float, device=args.device).transpose(0, 1)
        potentials = torch.tensor(potentials, dtype=torch.float, device=args.device).transpose(0, 1)
        biases = torch.stack(biases).transpose(0, 1)

        # extract noise which openmm does not provide
        deterministic_velocities = self.v_scale * velocities[:, 1:] + self.f_scale * forces[:, 1:] / self.masses.view(-1, 1)
        noises = (positions[:, 1:] - positions[:, :-1]) / args.timestep - deterministic_velocities

        potentials[:, 1:] = potentials[:, 1:] - (biases*positions[:, 1:]).sum((-2, -1)) # subtract bias potential
        actions = biases / self.masses.view(-1, 1) + noises / self.f_scale

        dist_matrix = get_dist_matrix(positions.reshape(-1, *positions.shape[-2:]))
        target_dist_matrix = get_dist_matrix(target_position)
        log_target_reward, last_idx = get_log_normal((dist_matrix-target_dist_matrix)/args.target_std).mean((1, 2)).view(args.num_samples, -1).max(1)
        log_md_reward = get_log_normal(actions/self.std.view(-1, 1)).mean((1, 2, 3))
        log_reward = log_md_reward + log_target_reward

        if args.type == 'train':
            self.replay.add((positions.to(args.buffer_device), actions.to(args.buffer_device), target_position.expand(args.num_samples, -1, -1).to(args.buffer_device), log_reward.to(args.buffer_device)))

        log = {
            'positions': positions, 
            'last_position': positions[torch.arange(args.num_samples), last_idx],
            'target_position': target_position, 
            'potentials': potentials,
            'log_target_reward': log_target_reward,
            'log_reward': log_reward,
            'last_idx': last_idx,
        }
        return log


    def train(self, args):
        policy_optimizers = torch.optim.SGD(self.policy.parameters(), lr=args.learning_rate)

        positions, actions, target_position, log_reward = self.replay.sample()

        biases = args.bias_scale * self.policy(positions[:, :-1], target_position)
        biases = biases / self.masses.view(-1, 1)
        
        if args.loss == 'tb':
            log_z = self.policy.get_log_z(positions[:, :1], target_position)
            log_forward = get_log_normal((biases-actions)/self.std.view(-1, 1)).mean((1, 2, 3))
            loss = torch.mean((log_z+log_forward-log_reward)**2)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
        
        policy_optimizers.step()
        policy_optimizers.zero_grad()
        return loss.item()
    

class ReplayBuffer:
    def __init__(self, args, md):
        self.positions = torch.zeros((args.buffer_size, args.num_steps+1, md.num_particles, 3), device=args.buffer_device)
        self.actions = torch.zeros((args.buffer_size, args.num_steps, md.num_particles, 3), device=args.buffer_device)
        self.target_positions = torch.zeros((args.buffer_size, md.num_particles, 3), device=args.buffer_device)
        self.log_reward = torch.zeros(args.buffer_size, device=args.buffer_device)

        self.idx = 0
        self.device = args.device
        self.buffer_device = args.buffer_device
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.num_samples = args.num_samples
        self.replay_strategy = args.replay_strategy

    def add(self, data):
        indices = torch.arange(self.idx, self.idx+self.num_samples, device=self.buffer_device) % self.buffer_size
        if self.idx >= self.buffer_size and self.replay_strategy == 'top_k':
            indices = torch.argsort(self.log_reward)[:self.num_samples]
        self.idx += self.num_samples

        self.positions[indices], self.actions[indices], self.target_positions[indices], self.log_reward[indices] = data
            
    def sample(self):
        indices = torch.randperm(self.buffer_size)[:self.batch_size]

        return self.positions[indices].to(self.device), self.actions[indices].to(self.device), self.target_positions[indices].to(self.device), self.log_reward[indices].to(self.device)