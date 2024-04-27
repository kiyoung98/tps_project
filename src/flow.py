import torch
from tqdm import tqdm

import proxy
from utils.utils import get_log_normal, get_dist_matrix

class FlowNetAgent:
    def __init__(self, args, md):
        self.num_particles = md.num_particles

        self.v_scale = torch.tensor(md.v_scale, dtype=torch.float, device=args.device)
        self.f_scale = torch.tensor(md.f_scale.value_in_unit(md.f_scale.unit), dtype=torch.float, device=args.device)
        self.std = torch.tensor(md.std.value_in_unit(md.std.unit), dtype=torch.float, device=args.device)
        self.masses = torch.tensor(md.masses.value_in_unit(md.masses.unit), dtype=torch.float, device=args.device)

        self.policy = getattr(proxy, args.molecule.title())(args, md)

        if args.type == 'train':
            self.replay = ReplayBuffer(args)

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
        deterministic_velocities = self.v_scale * velocities[:, 1:] + self.f_scale * forces[:, 1:] / self.masses.view(1, 1, -1, 1)
        noises = (positions[:, 1:] - positions[:, :-1]) / args.timestep - deterministic_velocities

        potentials[:, 1:] = potentials[:, 1:] - (biases*positions[:, 1:]).sum((-2, -1)) # subtract bias potential
        actions = biases / self.masses.view(1, 1, -1, 1) + noises / self.f_scale

        last_position = positions[:, -1]

        last_dist_matrix = get_dist_matrix(last_position)
        target_dist_matrix = get_dist_matrix(target_position)

        terminal_log_reward = get_log_normal((last_dist_matrix-target_dist_matrix)/args.terminal_std).mean((1, 2))
        intermediate_log_reward = get_log_normal(actions/self.std.view(1, 1, -1, 1)).mean((1, 2, 3))
        log_reward = intermediate_log_reward + terminal_log_reward

        if args.type == 'train':
            self.replay.add((positions, actions, target_position.expand(*last_position.shape), log_reward))

        log = {
            'positions': positions, 
            'last_position': last_position, 
            'target_position': target_position, 
            'potentials': potentials,
            'terminal_log_reward': terminal_log_reward,
            'log_reward': log_reward,
        }
        return log


    def train(self, args):
        policy_optimizers = torch.optim.SGD(self.policy.parameters(), lr=args.learning_rate)

        positions, actions, target_position, log_reward = self.replay.sample()

        biases = args.bias_scale * self.policy(positions[:, :-1], target_position)
        biases = biases / self.masses.view(1, 1, -1, 1)
        
        if args.loss == 'tb':
            log_z = self.policy.get_log_z(positions[:, :1], target_position)
            log_forward = get_log_normal((biases-actions)/self.std.view(1, 1, -1, 1)).mean((1, 2, 3))
            loss = torch.mean((log_z+log_forward-log_reward)**2)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
        
        policy_optimizers.step()
        policy_optimizers.zero_grad()
        return loss.item()
    

class ReplayBuffer:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.replay_strategy = args.replay_strategy

    def add(self, data):
        if hasattr(self, 'positions'):
            positions, actions, target_positions, log_reward = data

            self.positions = torch.cat((self.positions, positions))
            self.actions = torch.cat((self.actions, actions))
            self.target_positions = torch.cat((self.target_positions, target_positions))
            self.log_reward = torch.cat((self.log_reward, log_reward))
            
            if self.replay_strategy == '':
                self.positions = self.positions[-self.buffer_size:]
                self.actions = self.actions[-self.buffer_size:]
                self.target_positions = self.target_positions[-self.buffer_size:]
                self.log_reward = self.log_reward[-self.buffer_size:]
            elif self.replay_strategy == 'top_k':
                sorted_indices = torch.argsort(self.log_reward, descending=True)[:self.buffer_size]

                self.positions = self.positions[sorted_indices]
                self.actions = self.actions[sorted_indices]
                self.target_positions = self.target_positions[sorted_indices]
                self.log_reward = self.log_reward[sorted_indices]
        else:
            self.positions, self.actions, self.target_positions, self.log_reward = data
            
    def sample(self):
        indices = torch.randperm(self.positions.shape[0])[:self.batch_size]
        return self.positions[indices], self.actions[indices], self.target_positions[indices], self.log_reward[indices]

    def __len__(self):
        return len(self.buffer)
