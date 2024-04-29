import torch
import random
from tqdm import tqdm

import proxy
from utils.utils import get_log_normal, get_dist_matrix

class FlowNetAgent:
    def __init__(self, args, md):
        self.num_particles = md.num_particles
        self.policy = getattr(proxy, args.molecule.title())(args, md)

        if args.type == 'train':
            self.replay = ReplayBuffer(args, md)

    def sample(self, args, mds, biased=True):
        positions = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        potentials = torch.zeros(args.num_samples, args.num_steps, device=args.device)
        actions = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        noises = torch.normal(torch.zeros(args.num_samples, args.num_steps, self.num_particles, 3, device=args.device), torch.ones(args.num_samples, args.num_steps, self.num_particles, 3, device=args.device) * 0.2)

        for s in range(args.num_steps):
            position, potential = mds.report()
            noise = noises[:, s] if args.type == 'train' else 0
            if biased:
                bias = self.policy(position.unsqueeze(1)).squeeze().detach()
            else:
                bias = torch.zeros_like(position)
            action = bias + noise
            
            positions[:, s] = position
            potentials[:, s] = potential - (1000*bias*position).sum((-2, -1))
            actions[:, s] = action

            mds.step(action)

        mds.reset()

        dist_matrix = get_dist_matrix(positions.reshape(-1, *positions.shape[-2:]))
        target_dist_matrix = get_dist_matrix(mds.target_position)
        log_target_reward, last_idx = get_log_normal((dist_matrix-target_dist_matrix)/args.target_std).mean((1, 2)).view(args.num_samples, -1).max(1)
        log_md_reward = get_log_normal(actions/0.1).mean((1, 2, 3))
        log_reward = log_md_reward + log_target_reward

        if args.type == 'train':
            self.replay.add((positions, actions, log_reward))

        log = {
            'positions': positions, 
            'last_position': positions[torch.arange(args.num_samples), last_idx],
            'target_position': mds.target_position,
            'potentials': potentials,
            'log_target_reward': log_target_reward,
            'log_reward': log_reward,
            'last_idx': last_idx,
        }
        return log


    def train(self, args):
        policy_optimizers = torch.optim.SGD(self.policy.parameters(), lr=args.learning_rate)

        positions, actions, log_reward = self.replay.sample()

        biases = self.policy(positions)
        
        log_z = self.policy.log_z
        log_forward = get_log_normal((biases-actions)/0.1).mean((1, 2, 3))
        loss = torch.mean((log_z+log_forward-log_reward)**2)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
        
        policy_optimizers.step()
        policy_optimizers.zero_grad()
        return loss.item()
    

class ReplayBuffer:
    def __init__(self, args, md):
        self.positions = torch.zeros((args.buffer_size, args.num_steps, md.num_particles, 3), device=args.device)
        self.actions = torch.zeros((args.buffer_size, args.num_steps, md.num_particles, 3), device=args.device)
        self.target_positions = torch.zeros((args.buffer_size, md.num_particles, 3), device=args.device)
        self.log_reward = torch.zeros(args.buffer_size, device=args.device)

        self.idx = 0
        self.device = args.device
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.num_samples = args.num_samples
        self.replay_strategy = args.replay_strategy

    def add(self, data):
        indices = torch.arange(self.idx, self.idx+self.num_samples, device=self.device) % self.buffer_size
        if self.idx >= self.buffer_size and self.replay_strategy == 'top_k':
            indices = torch.argsort(self.log_reward)[:self.num_samples]
        self.idx += self.num_samples

        self.positions[indices], self.actions[indices], self.log_reward[indices] = data
            
    def sample(self):
        indices = torch.randperm(self.buffer_size)[:self.batch_size]
        return self.positions[indices], self.actions[indices], self.log_reward[indices]