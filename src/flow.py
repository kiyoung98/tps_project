import torch
import proxy

from tqdm import tqdm
from utils.utils import get_log_normal, get_dist_matrix

class FlowNetAgent:
    def __init__(self, args, md):
        self.num_particles = md.num_particles
        self.policy = getattr(proxy, args.molecule.title())(args, md)

        if args.type == 'train':
            self.replay = ReplayBuffer(args, md)

    def sample(self, args, mds):
        noises = torch.normal(torch.zeros(args.num_samples, args.num_steps, self.num_particles, 3, device=args.device), torch.ones(args.num_samples, args.num_steps, self.num_particles, 3, device=args.device))
        positions = torch.zeros((args.num_samples, args.num_steps+1, self.num_particles, 3), device=args.device)
        actions = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        potentials = torch.zeros(args.num_samples, args.num_steps+1, device=args.device)

        position, potential = mds.report()
        
        positions[:, 0] = position
        potentials[:, 0] = potential
        for s in tqdm(range(args.num_steps)):
            noise = noises[:, s] if args.type == 'train' else 0
            bias = args.bias_scale * self.policy(position).detach()
            action = bias + 2 * args.std * noise # We use tempered version of policy as sampler by multiplying 2
            mds.step(action)

            position, potential = mds.report()

            positions[:, s+1] = position.detach()
            potentials[:, s+1] = potential - (1000*action*position).sum((-2, -1))
            actions[:, s] = action
        mds.reset()

        target_dist_matrix = get_dist_matrix(mds.target_position)

        if args.flexible:
            dist_matrix = get_dist_matrix(positions.reshape(-1, *positions.shape[-2:]))
            log_target_reward, last_idx = get_log_normal((dist_matrix-target_dist_matrix)/args.target_std).mean((1, 2)).view(args.num_samples, -1).max(1)
        else:
            dist_matrix = get_dist_matrix(positions[:, -1])
            last_idx = args.num_steps * torch.ones(args.num_samples, dtype=torch.long, device=args.device)
            log_target_reward = get_log_normal((dist_matrix-target_dist_matrix)/args.target_std).mean((1, 2))
        log_md_reward = get_log_normal(actions/args.std).mean((1, 2, 3))
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

        biases = args.bias_scale * self.policy(positions[:, :-1])
        
        log_z = self.policy.log_z
        log_forward = get_log_normal((biases-actions)/args.std).mean((1, 2, 3))
        loss = torch.mean((log_z+log_forward-log_reward)**2)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
        
        policy_optimizers.step()
        policy_optimizers.zero_grad()
        return loss.item()
    

class ReplayBuffer:
    def __init__(self, args, md):
        self.positions = torch.zeros((args.buffer_size, args.num_steps+1, md.num_particles, 3), device=args.device)
        self.actions = torch.zeros((args.buffer_size, args.num_steps, md.num_particles, 3), device=args.device)
        self.log_reward = torch.zeros(args.buffer_size, device=args.device)

        self.idx = 0
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.num_samples = args.num_samples

    def add(self, data):
        indices = torch.arange(self.idx, self.idx+self.num_samples) % self.buffer_size
        self.positions[indices], self.actions[indices], self.log_reward[indices] = data
        self.idx += self.num_samples
            
    def sample(self):
        num_samples = min(self.idx, self.buffer_size)
        indices = torch.randperm(num_samples)[:self.batch_size]
        return self.positions[indices], self.actions[indices], self.log_reward[indices]