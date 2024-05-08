import torch
import proxy
from tqdm import tqdm
    

class FlowNetAgent:
    def __init__(self, args, md):
        self.num_particles = md.num_particles
        self.v_scale = torch.tensor(md.v_scale, dtype=torch.float, device=args.device)
        self.f_scale = torch.tensor(md.f_scale.value_in_unit(md.f_scale.unit), dtype=torch.float, device=args.device)
        self.std = torch.tensor(md.std.value_in_unit(md.std.unit), dtype=torch.float, device=args.device)
        self.masses = torch.tensor(md.masses.value_in_unit(md.masses.unit), dtype=torch.float, device=args.device).unsqueeze(-1)
        
        self.eye = torch.eye(self.num_particles, device=args.device).unsqueeze(0)
        self.charge_matrix = torch.tensor(md.charge_matrix, device=args.device).unsqueeze(0)
        self.covalent_radii_matrix = torch.tensor(md.covalent_radii_matrix, device=args.device).unsqueeze(0)

        if args.type == 'train':
            self.replay = ReplayBuffer(args, md)

        self.policy = getattr(proxy, args.molecule.title())(args, md)

    def sample(self, args, mds, temperature):
        positions = torch.zeros((args.num_samples, args.num_steps+1, self.num_particles, 3), device=args.device)
        actions = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        potentials = torch.zeros(args.num_samples, args.num_steps+1, device=args.device)
        
        position, _, _, potential = mds.report()
        
        positions[:, 0] = position
        potentials[:, 0] = potential

        mds.set_temperature(temperature)
        for s in tqdm(range(args.num_steps), desc='Sampling'):
            bias = args.bias_scale * self.policy(position.detach()).squeeze().detach()
            mds.step(bias)
            
            next_position, velocity, force, potential = mds.report()

            # extract noise which openmm does not provide
            noises = (next_position - position) - (self.v_scale * velocity + self.f_scale * force / self.masses)
            action = bias / self.masses + noises / self.f_scale

            positions[:, s+1] = next_position
            actions[:, s] = action
            potentials[:, s+1] = potential - (bias*next_position).sum((-2, -1))

            position = next_position
        mds.reset()
        
        target_matrix = getattr(self, args.reward_matrix)(mds.target_position)

        if args.flexible:
            matrix = getattr(self, args.reward_matrix)(positions.reshape(-1, *positions.shape[-2:]))
            if args.type == 'eval':
                log_target_reward = torch.zeros(args.num_samples*(args.num_steps+1), device=args.device)
                for i in range(args.num_samples):
                    log_target_reward[i*(args.num_steps+1):(i+1)*(args.num_steps+1)] = (-1/2)*torch.square((matrix[i*(args.num_steps+1):(i+1)*(args.num_steps+1)]-target_matrix)/args.target_std).mean((1, 2))
            else:
                log_target_reward = (-1/2)*torch.square((matrix-target_matrix)/args.target_std).mean((1, 2))
            log_target_reward, last_idx = log_target_reward.view(args.num_samples, -1).max(1)
        else:
            matrix = getattr(self, args.reward_matrix)(positions[:, -1])
            last_idx = args.num_steps * torch.ones(args.num_samples, dtype=torch.long, device=args.device)
            log_target_reward = (-1/2)*torch.square((matrix-target_matrix)/args.target_std).mean((1, 2))
            
        log_md_reward = (-1/2)*torch.square(actions/self.std).mean((1, 2, 3))
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

        biases = args.bias_scale * self.policy(positions[:, :-1]) / self.masses
        
        log_z = self.policy.log_z
        log_forward = (-1/2)*torch.square((biases-actions)/self.std).mean((1, 2, 3))
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
        return scaled_dist_matrix * 2

    def coulomb(self, x):
        dist_matrix = torch.cdist(x, x) + self.eye
        coulomb_matrix = self.charge_matrix / dist_matrix
        return coulomb_matrix / 100

class ReplayBuffer:
    def __init__(self, args, md):
        self.positions = torch.zeros((args.buffer_size, args.num_steps+1, md.num_particles, 3), device=args.device)
        self.actions = torch.zeros((args.buffer_size, args.num_steps, md.num_particles, 3), device=args.device)
        self.log_reward = torch.zeros(args.buffer_size, device=args.device)

        self.idx = 0
        self.buffer_size = args.buffer_size
        self.num_samples = args.num_samples

    def add(self, data):
        indices = torch.arange(self.idx, self.idx+self.num_samples) % self.buffer_size
        self.idx += self.num_samples

        self.positions[indices], self.actions[indices], self.log_reward[indices] = data
            
    def sample(self):
        indices = torch.randperm(min(self.idx, self.buffer_size))[:self.num_samples]
        return self.positions[indices], self.actions[indices], self.log_reward[indices]