import torch
import random
from tqdm import tqdm

import proxy
from utils.utils import get_log_normal, get_dist_matrix

class GFlowNet:
    def __init__(self, args, md_info):
        self.std = md_info.std
        self.masses = md_info.masses
        self.v_scale = md_info.v_scale
        self.f_scale = md_info.f_scale
        self.num_particles = md_info.num_particles

        self.policy = getattr(proxy, args.molecule.title())(args, md_info)

    def sample(self, args, mds, target_position, temperature):
        positions = torch.zeros((args.num_samples, args.num_steps+1, self.num_particles, 3), device=args.device)
        potentials = torch.zeros(args.num_samples, args.num_steps+1, device=args.device)
        actions = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)

        position, _, _, potential = mds.report()
        
        positions[:, 0] = position
        potentials[:, 0] = potential

        target_position = target_position.unsqueeze(0).unsqueeze(0)

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
        
        positions, target_actions, target_position = data
        
        start_position, intermediate_positions, last_position = positions[:1, :1], positions[:, :-1], positions[:, -1:]
        
        if args.goal_conditioned and args.hindsight and random.random() < 0.5: 
            target_position = last_position

        biases = args.bias_scale * self.policy(intermediate_positions, target_position)
        actions = biases / self.masses.view(1, -1, 1)
        
        last_dist_matrix = get_dist_matrix(last_position)
        target_dist_matrix = get_dist_matrix(target_position)

        log_z = self.policy.get_log_z(start_position, target_position)
        log_forward = get_log_normal((actions-target_actions)/self.std.view(1, -1, 1)).mean((1, 2, 3))
        log_reward = get_log_normal(target_actions/self.std.view(1, -1, 1)).mean((1, 2, 3)) + get_log_normal((last_dist_matrix-target_dist_matrix)/args.terminal_std).mean((1, 2))
        
        loss = torch.mean((log_z+log_forward-log_reward)**2)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
        
        policy_optimizers.step()
        policy_optimizers.zero_grad()
        return loss.item()