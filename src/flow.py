import torch
import proxy
import random

from tqdm import tqdm

class FlowNetAgent:
    def __init__(self, args, md):
        self.num_particles = md.num_particles
        self.eye = torch.eye(self.num_particles, device=args.device).unsqueeze(0)
        self.charge_matrix = torch.tensor(md.charge_matrix, device=args.device).unsqueeze(0)
        self.covalent_radii_matrix = torch.tensor(md.covalent_radii_matrix, device=args.device).unsqueeze(0)
        
        self.policy = getattr(proxy, args.molecule.title())(args, md)

        if args.type == 'train':
            self.replay = ReplayBuffer(args)

    def sample(self, args, mds, std):
        noises = torch.normal(torch.zeros(args.num_samples, args.num_steps, self.num_particles, 3, device=args.device), torch.ones(args.num_samples, args.num_steps, self.num_particles, 3, device=args.device))
        positions = torch.zeros((args.num_samples, args.num_steps+1, self.num_particles, 3), device=args.device)
        actions = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        potentials = torch.zeros(args.num_samples, args.num_steps+1, device=args.device)

        position, potential = mds.report()
        
        positions[:, 0] = position
        potentials[:, 0] = potential
        for s in tqdm(range(args.num_steps)):
            noise = noises[:, s]
            bias = args.bias_scale * self.policy(position).detach()
            action = bias + std * noise
            mds.step(action)

            position, potential = mds.report()

            positions[:, s+1] = position.detach()
            potentials[:, s+1] = potential - (1000*action*position).sum((-2, -1))
            actions[:, s] = action
        mds.reset()

        target_matrix = getattr(self, args.reward_matrix)(mds.target_position)

        if args.flexible:
            matrix = getattr(self, args.reward_matrix)(positions.reshape(-1, *positions.shape[-2:]))
            log_target_reward = (-1/2)*torch.square((matrix-target_matrix)/args.target_std).mean((1, 2))
            log_target_reward, last_idx = log_target_reward.view(args.num_samples, -1).max(1)
        else:
            matrix = getattr(self, args.reward_matrix)(positions[:, -1])
            last_idx = args.num_steps * torch.ones(args.num_samples, dtype=torch.long, device=args.device)
            log_target_reward = (-1/2)*torch.square((matrix-target_matrix)/args.target_std).mean((1, 2))
            
        log_md_reward = (-1/2)*torch.square(actions/args.std).mean((1, 2, 3))
        log_reward = log_md_reward + log_target_reward

        if args.type == 'train':
            self.replay.add((positions, actions, log_reward))

        log = {
            'positions': positions, 
            'last_position': positions[torch.arange(args.num_samples), last_idx],
            'target_position': mds.target_position,
            'potentials': potentials,
            'log_target_reward': log_target_reward,
            'log_md_reward': log_md_reward,
            'log_reward': log_reward,
            'last_idx': last_idx,
            'log_z': self.policy.log_z.item(),
        }
        return log

    def train(self, args):
        policy_optimizers = torch.optim.SGD(self.policy.parameters(), lr=args.learning_rate)

        positions, actions, log_reward = self.replay.sample()

        biases = args.bias_scale * self.policy(positions[:, :-1])
        
        log_z = self.policy.log_z
        log_forward = (-1/2)*torch.square((biases-actions)/args.std).mean((1, 2, 3))
        loss = torch.mean((log_z+log_forward-log_reward)**2)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
        
        policy_optimizers.step()
        policy_optimizers.zero_grad()
        return loss.item()

    def dist(self, x):
        dist_matrix = torch.cdist(x, x)
        return dist_matrix

    def scaled_dist(self, x):
        dist_matrix = torch.cdist(x, x) + self.eye
        scaled_dist_matrix = torch.exp(-1.7*(dist_matrix-self.covalent_radii_matrix)/self.covalent_radii_matrix) + 0.01 * self.covalent_radii_matrix / dist_matrix
        return scaled_dist_matrix * 3

    def coulomb(self, x):
        dist_matrix = torch.cdist(x, x) + self.eye
        coulomb_matrix = self.charge_matrix / dist_matrix
        return coulomb_matrix / 100

class ReplayBuffer:
    def __init__(self, args):
        self.buffer = []
        self.buffer_size = args.buffer_size

    def add(self, data):
        self.buffer.append(data)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample(self):
        idx = random.randrange(len(self.buffer))
        return self.buffer[idx]