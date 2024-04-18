import torch
import random
from tqdm import tqdm

import proxy
from utils.utils import get_log_normal, get_dist_matrix

class GFlowNet:
    def __init__(self, args, md_info):
        self.num_particles = md_info.num_particles
        self.policy = getattr(proxy, args.molecule.title())(args, md_info)

    def sample(self, args, mds, target_position, std):
        positions = torch.zeros((args.num_samples, args.num_steps+1, self.num_particles, 3), device=args.device)
        potentials = torch.zeros(args.num_samples, args.num_steps+1, device=args.device)
        actions = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        noises = torch.normal(torch.zeros(args.num_samples, args.num_steps, self.num_particles, 3, device=args.device), torch.ones(args.num_samples, args.num_steps, self.num_particles, 3, device=args.device) * std)

        position, potential = mds.report()

        positions[:, 0] = position
        potentials[:, 0] = potential

        for s in tqdm(range(args.num_steps)):
            bias = self.policy(position.unsqueeze(1), target_position).squeeze().detach()
            noise = noises[:, s]

            action = bias + noise
            mds.step(action)

            next_position, potential = mds.report()

            position = next_position
            
            positions[:, s+1] = position
            potentials[:, s+1] = potential
            actions[:, s] = action
        mds.reset()

        log = {
            'positions': positions, 
            'start_position': positions[:1, :1],
            'last_position': position, 
            'target_position': target_position, 
            'potentials': potentials,
        }
        return (positions, actions, target_position), log


    def train(self, args, data):
        policy_optimizers = torch.optim.SGD(self.policy.parameters(), lr=args.learning_rate)
        
        positions, actions, target_position = data
        
        start_position, intermediate_positions, last_position = positions[:1, :1], positions[:, :-1], positions[:, -1:]
        
        if args.goal_conditioned and args.hindsight and random.random() < 0.5: 
            target_position = last_position

        biases = self.policy(intermediate_positions, target_position)
        
        last_dist_matrix = get_dist_matrix(last_position)
        target_dist_matrix = get_dist_matrix(target_position)

        log_z = self.policy.get_log_z(start_position, target_position)
        log_forward = get_log_normal((biases-actions)/args.std).mean((1, 2, 3))
        log_reward = get_log_normal(actions/args.std).mean((1, 2, 3)) + get_log_normal((last_dist_matrix-target_dist_matrix)/args.terminal_std).mean((1, 2))
        
        loss = torch.mean(log_z+log_forward-log_reward)**2
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
        
        policy_optimizers.step()
        policy_optimizers.zero_grad()
        return loss.item()