import torch
import random
from tqdm import tqdm

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

        for s in range(args.num_steps):
            position, potential = mds.report()
            bias = self.policy(position.unsqueeze(1), target_position).squeeze().detach()
            noise = noises[:, s]
            action = bias + noise
            
            positions[:, s] = position
            potentials[:, s] = potential
            actions[:, s] = action

            mds.step(action)
        
        start_position = positions[0, 0].unsqueeze(0).unsqueeze(0)
        last_position = mds.report()[0].unsqueeze(1)

        mds.reset()

        last_dist_matrix = get_dist_matrix(last_position)
        target_dist_matrix = get_dist_matrix(target_position)
        terminal_reward = get_log_normal((last_dist_matrix-target_dist_matrix)/args.terminal_std).mean((1, 2))
        log_reward = get_log_normal(actions/args.std).mean((1, 2, 3)) + terminal_reward

        self.replay.add((positions, actions, noises, start_position, last_position, target_position, log_reward))

        log = {
            'positions': positions, 
            'start_position': start_position,
            'last_position': last_position, 
            'target_position': target_position, 
            'potentials': potentials,
            'terminal_reward': terminal_reward,
            'log_reward': log_reward,
        }
        return log


    def train(self, args):
        policy_optimizers = torch.optim.SGD(self.policy.parameters(), lr=args.learning_rate)

        positions, actions, noises, start_position, last_position, target_position, log_reward = self.replay.sample()

        # if args.hindsight and random.random() < 0.5: target_position = last_position

        biases = self.policy(positions, target_position)
        
        last_dist_matrix = get_dist_matrix(last_position)
        target_dist_matrix = get_dist_matrix(target_position)
        
        if args.loss == 'tb':
            log_z = self.policy.get_log_z(start_position, target_position)
            log_forward = get_log_normal((biases-actions)/args.std).mean((1, 2, 3))
            # log_reward = get_log_normal(actions/args.std).mean((1, 2, 3)) + get_log_normal((last_dist_matrix-target_dist_matrix)/args.terminal_std).mean((1, 2))
            loss = torch.mean((log_z+log_forward-log_reward)**2)
        elif args.loss == 'pice':
            costs = get_log_normal(noises/args.std).mean((1, 2, 3)) - get_log_normal(actions/args.std).mean((1, 2, 3)) - get_log_normal((last_dist_matrix-target_dist_matrix)/args.terminal_std).mean((1, 2))
            if args.molecule == "poly":
                importance = torch.softmax(-10000*costs, 0)
            elif args.molecule == "chignolin":
                importance = torch.softmax(-100*costs, 0)
            else:
                importances = torch.softmax(-costs, 0)
            match = - get_log_normal((biases-actions)/args.std).mean((1, 2, 3))
            loss = torch.sum(importances*match)
        
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