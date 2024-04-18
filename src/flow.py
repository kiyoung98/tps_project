import torch
import random
from tqdm import tqdm
from collections import deque

import proxy
from utils.utils import get_log_normal, get_dist_matrix

class FlowNetAgent:
    def __init__(self, args, md):
        self.num_particles = md.num_particles
        self.replay = ReplayBuffer(args)
        self.policy = getattr(proxy, args.molecule.title())(args, md)

    def sample(self, args, mds, target_position, std):
        positions = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        potentials = torch.zeros(args.num_samples, args.num_steps, device=args.device)
        actions = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        noises = torch.normal(torch.zeros(args.num_samples, args.num_steps, self.num_particles, 3, device=args.device), torch.ones(args.num_samples, args.num_steps, self.num_particles, 3, device=args.device) * std)

        for s in tqdm(range(args.num_steps)):
            position, potential = mds.report()
            bias = self.policy(position.unsqueeze(1), target_position).squeeze().detach()
            noise = noises[:, s]
            action = bias + noise
            
            positions[:, s] = position
            potentials[:, s] = potential
            actions[:, s] = action

            mds.step(action)
        mds.reset()
        
        start_position = positions[0, 0].unsqueeze(0).unsqueeze(0)
        last_position = mds.report()[0].unsqueeze(1)

        self.replay.push((positions, actions, start_position, last_position, target_position))

        log = {
            'positions': positions, 
            'start_position': start_position,
            'last_position': last_position, 
            'target_position': target_position, 
            'potentials': potentials,
        }
        return log


    def train(self, args):
        policy_optimizers = torch.optim.SGD(self.policy.parameters(), lr=args.learning_rate)
        
        positions, actions, start_position, last_position, target_position = zip(self.replay.sample())

        biases = self.policy(positions, target_position)
        
        last_dist_matrix = get_dist_matrix(last_position)
        target_dist_matrix = get_dist_matrix(target_position)

        log_z = self.policy.get_log_z(start_position, target_position)
        log_forward = get_log_normal((biases-actions)/args.std).mean((1, 2, 3))
        log_reward = get_log_normal(actions/args.std).mean((1, 2, 3)) + get_log_normal((last_dist_matrix-target_dist_matrix)/args.terminal_std).mean((1, 2))
        
        loss = torch.mean((log_z+log_forward-log_reward)**2)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
        
        policy_optimizers.step()
        policy_optimizers.zero_grad()
        return loss.item()
    

class ReplayBuffer:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.buffer = deque(maxlen=args.buffer_size)

    def push(self, data):
        positions, actions, start_position, last_position, target_position = data
        for i in range(positions.shape[0]):
            self.buffer.append((positions[i], actions[i], start_position, last_position[i], target_position))

    def sample(self):
        batch_size = len(self) if len(self) < self.batch_size else self.batch_size
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)