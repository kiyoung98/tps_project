import torch
import proxy
from tqdm import tqdm
import openmm.unit as unit
from torch.distributions import Normal
    

class FlowNetAgent:
    def __init__(self, args, md):
        self.num_particles = md.num_particles
        self.v_scale = torch.tensor(md.v_scale, dtype=torch.float, device=args.device)
        self.f_scale = torch.tensor(md.f_scale.value_in_unit(unit.femtosecond), dtype=torch.float, device=args.device)
        self.std = torch.tensor(md.std.value_in_unit(unit.nanometer/unit.femtosecond), dtype=torch.float, device=args.device)
        self.masses = torch.tensor(md.masses.value_in_unit(md.masses.unit), dtype=torch.float, device=args.device).unsqueeze(-1)

        self.normal = Normal(0, self.std)
        self.eye = torch.eye(self.num_particles, device=args.device).unsqueeze(0)
        self.charge_matrix = torch.tensor(md.charge_matrix, dtype=torch.float, device=args.device).unsqueeze(0)
        # self.covalent_radii_matrix = torch.tensor(md.covalent_radii_matrix, dtype=torch.float, device=args.device).unsqueeze(0)

        if args.type == 'train':
            self.replay = ReplayBuffer(args, md)

        self.policy = getattr(proxy, args.molecule.title())(args, md)

    def sample(self, args, mds, temperature):
        positions = torch.zeros((args.num_samples, args.num_steps+1, self.num_particles, 3), device=args.device)
        actions = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        biases = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        noises = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        potentials = torch.zeros(args.num_samples, args.num_steps+1, device=args.device)
        
        position, _, _, potential = mds.report()
        
        positions[:, 0] = position
        potentials[:, 0] = potential

        mds.set_temperature(temperature)
        for s in tqdm(range(args.num_steps), desc='Sampling'):
            bias = args.bias_scale * self.policy(position.detach()).squeeze().detach() # kJ/(mol*nm)
            mds.step(bias)
            
            next_position, velocity, force, potential = mds.report()

            # extract noise which openmm does not provide
            noise = (next_position - position) - (self.v_scale * velocity + self.f_scale * force / self.masses)

            positions[:, s+1] = next_position
            potentials[:, s+1] = potential - (bias*next_position).sum((1, 2))

            position = next_position
            bias = 1e-6 * bias # kJ/(mol*nm) -> (da*nm)/fs**2
            action = self.f_scale * bias / self.masses + noise
            
            actions[:, s] = action
            biases[:, s] = bias
            noises[:, s] = noise

        mds.reset()
        
        target_matrix = getattr(self, args.reward_matrix)(mds.target_position)

        if args.flexible:
            # if args.type == 'eval':
            log_target_reward = torch.zeros(args.num_samples*(args.num_steps+1), device=args.device)
            for i in range(args.num_samples):
                matrix = getattr(self, args.reward_matrix)(positions[i])
                log_target_reward[i*(args.num_steps+1):(i+1)*(args.num_steps+1)] = - torch.square((matrix-target_matrix)/args.target_std).mean((1, 2))
            # else:
            #     matrix = getattr(self, args.reward_matrix)(positions.reshape(-1, *positions.shape[-2:]))
            #     log_target_reward = - torch.square((matrix-target_matrix)/args.target_std).mean((1, 2))
            log_target_reward, last_idx = log_target_reward.view(args.num_samples, -1).max(1)
        else:
            matrix = getattr(self, args.reward_matrix)(position)
            last_idx = args.num_steps * torch.ones(args.num_samples, dtype=torch.long, device=args.device)
            log_target_reward = - torch.square((matrix-target_matrix)/args.target_std).mean((1, 2))
            
        log_md_reward = self.normal.log_prob(actions).mean((1, 2, 3))
        log_reward = log_md_reward + log_target_reward

        log_likelihood = self.normal.log_prob(noises).mean((1, 2, 3))

        if args.type == 'train':
            self.replay.add((positions, actions, log_reward))
        
        log = {
            'positions': positions, 
            'biases': biases,
            'potentials': potentials,
            'last_idx': last_idx,
            'target_position': mds.target_position,
            'log_target_reward': log_target_reward,
            'log_md_reward': log_md_reward,
            'log_reward': log_reward,
            'log_likelihood': log_likelihood,
        }
        return log

    def train(self, args):
        mlp_optimizer = torch.optim.Adam(self.policy.mlp.parameters(), lr=args.mlp_lr)
        log_z_optimizer = torch.optim.Adam([self.policy.log_z], lr=args.log_z_lr)

        positions, actions, log_reward = self.replay.sample()

        biases = 1e-6 * args.bias_scale * self.policy(positions[:, :-1])
        biases = self.f_scale * biases / self.masses
        
        log_z = self.policy.log_z
        log_forward = self.normal.log_prob(biases-actions).mean((1, 2, 3))
        loss = torch.square(log_z+log_forward-log_reward).mean() 
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.mlp.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.policy.log_z, args.max_grad_norm)
        
        mlp_optimizer.step()
        log_z_optimizer.step()
        mlp_optimizer.zero_grad()
        log_z_optimizer.zero_grad()
        return loss.item()

    def dist(self, x):
        dist_matrix = torch.cdist(x, x)
        return dist_matrix

    # def scaled_dist(self, x):
    #     dist_matrix = torch.cdist(x, x) + self.eye
    #     scaled_dist_matrix = torch.exp(-1.7*(dist_matrix-self.covalent_radii_matrix)/self.covalent_radii_matrix) + 0.01 * self.covalent_radii_matrix / dist_matrix
    #     return scaled_dist_matrix

    def coulomb(self, x):
        dist_matrix = torch.cdist(x, x) + self.eye
        coulomb_matrix = self.charge_matrix / dist_matrix
        return coulomb_matrix

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