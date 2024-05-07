import torch
import wandb
import random
import numpy as np
from torch import nn
import openmm.unit as unit
from scipy.constants import physical_constants

import openmm as mm
from tqdm import tqdm
from openmm import app

from utils.utils import *
from utils.logging import Logger

import argparse

covalent_radii = {
    'H': 31 * unit.picometer / unit.nanometer,
    'C': 76 * unit.picometer / unit.nanometer,
    'N': 71 * unit.picometer / unit.nanometer,
    'O': 66 * unit.picometer / unit.nanometer,
}

nuclear_charge = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
}

parser = argparse.ArgumentParser()

# System Config
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--type', default='train', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--project', default='chignolin_mode', type=str)
parser.add_argument('--molecule', default='chignolin', type=str)

# Logger Config
parser.add_argument('--config', default="", type=str, help='Path to config file')
parser.add_argument('--logger', default=True, type=bool, help='Use system logger')
parser.add_argument('--date', default="test-run", type=str, help='Date of the training')
parser.add_argument('--save_freq', default=10, type=int, help='Frequency of saving in  rollouts')
parser.add_argument('--server', default="server", type=str, choices=["server", "cluster", "else"], help='Server we are using')

# Policy Config
parser.add_argument('--force', action='store_true', help='Predict force otherwise potential')

# Sampling Config
parser.add_argument('--reward_matrix', default='dist', type=str)
parser.add_argument('--bias_scale', default=7000, type=float, help='Scale factor of bias')
parser.add_argument('--num_samples', default=2, type=int, help='Number of paths to sample')
parser.add_argument('--flexible', action='store_true', help='Sample paths with flexible length')
parser.add_argument('--num_steps', default=5000, type=int, help='Number of steps in each path i.e. length of trajectory')
parser.add_argument('--target_std', default=0.1, type=float, help='Standard deviation of gaussian distribution w.r.t. dist matrix of position')

# Training Config
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--start_temperature', default=1200, type=float, help='Initial temperature of annealing schedule')
parser.add_argument('--end_temperature', default=1200, type=float, help='Final temperature of annealing schedule')
parser.add_argument('--num_rollouts', default=10000, type=int, help='Number of rollouts (or sampling)')
parser.add_argument('--trains_per_rollout', default=200, type=int, help='Number of training per rollout in a rollout')
parser.add_argument('--buffer_size', default=256, type=int, help='Size of buffer which stores sampled paths')
parser.add_argument('--max_grad_norm', default=10, type=int, help='Maximum norm of gradient to clip')

args = parser.parse_args()

class ChignolinDynamics:
    def __init__(self, state):
        super().__init__()
        self.start_file = f'./data/chignolin/{state}.pdb'

        self.temperature = 300 * unit.kelvin
        self.friction_coefficient = 1 / unit.picoseconds
        self.timestep = 1 * unit.femtoseconds

        self.pdb, self.integrator, self.simulation, self.external_force = self.setup()

        self.simulation.minimizeEnergy()
        self.position = self.report()[0]
        self.reset()

        self.num_particles = self.simulation.system.getNumParticles()

        self.v_scale, self.f_scale, self.masses, self.std = self.get_md_info()
        self.covalent_radii_matrix = self.get_covalent_radii_matrix()
        self.charge_matrix = self.get_charge_matrix()

    def setup(self):
        forcefield = app.ForceField('amber/protein.ff14SBonlysc.xml', 'implicit/gbn2.xml')
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005
        )
        external_force = mm.CustomExternalForce("fx*x+fy*y+fz*z")

        # creating the parameters
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])

        integrator = mm.LangevinIntegrator(
            self.temperature,  
            self.friction_coefficient,  
            self.timestep,
        ) 

        integrator.setConstraintTolerance(0.00001)

        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        return pdb, integrator, simulation, external_force

    def get_md_info(self):
        v_scale = np.exp(-self.timestep * self.friction_coefficient)
        f_scale = (1 - v_scale) / self.friction_coefficient

        masses = [self.simulation.system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(self.num_particles)]
        masses = unit.Quantity(np.array(masses), unit.dalton)
        
        unadjusted_variance = unit.BOLTZMANN_CONSTANT_kB * self.temperature * (1 - v_scale ** 2) / masses[:, None]
        std_SI_units = 1 / physical_constants['unified atomic mass unit'][0] * unadjusted_variance.value_in_unit(unit.joule / unit.dalton)
        std = unit.Quantity(np.sqrt(std_SI_units), unit.meter / unit.second)
        return v_scale, f_scale, masses, std

    def get_covalent_radii_matrix(self):
        covalent_radii_matrix = np.zeros((self.num_particles, self.num_particles))
        topology = self.pdb.getTopology()
        for i, atom_i in enumerate(topology.atoms()):
            for j, atom_j in enumerate(topology.atoms()):
                covalent_radii_matrix[i, j] = covalent_radii[atom_i.element.symbol] + covalent_radii[atom_j.element.symbol]
        return covalent_radii_matrix

    def get_charge_matrix(self):
        charge_matrix = np.zeros((self.num_particles, self.num_particles))
        topology = self.pdb.getTopology()
        for i, atom_i in enumerate(topology.atoms()):
            for j, atom_j in enumerate(topology.atoms()):
                if i == j:
                    charge_matrix[i, j] = 0.5 * nuclear_charge[atom_i.element.symbol]**(2.4)
                else:
                    charge_matrix[i, j] = nuclear_charge[atom_i.element.symbol] * nuclear_charge[atom_j.element.symbol]
        return charge_matrix

    def step(self, forces):
        for i in range(forces.shape[0]):
            self.external_force.setParticleParameters(i, i, forces[i])
        self.external_force.updateParametersInContext(self.simulation.context)
        self.simulation.step(1)

    def report(self):
        state = self.simulation.context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
        positions = state.getPositions().value_in_unit(unit.nanometer)
        velocities = state.getVelocities().value_in_unit(unit.nanometer/unit.femtosecond)
        forces = state.getForces().value_in_unit(unit.kilojoules/unit.mole/unit.nanometer)
        potentials = state.getPotentialEnergy().value_in_unit(unit.kilojoules/unit.mole)
        return positions, velocities, forces, potentials

    def reset(self):
        for i in range(len(self.position)):
            self.external_force.setParticleParameters(i, i, [0, 0, 0])
        self.external_force.updateParametersInContext(self.simulation.context)
        self.simulation.context.setPositions(self.position)
        self.simulation.context.setVelocitiesToTemperature(0)

    def set_temperature(self, temperature):
        self.integrator.setTemperature(temperature * unit.kelvin)

class MDs:
    def __init__(self, args):
        self.device = args.device
        self.num_samples = args.num_samples

        self.mds = self._init_mds()
        self.target_position = self._init_target_position()

    def _init_mds(self):
        print(f"Initialize dynamics starting at unfolded of chignolin")

        mds = []
        for _ in tqdm(range(self.num_samples)):
            md = ChignolinDynamics('unfolded')
            mds.append(md)
        return mds

    def _init_target_position(self):
        target_position = ChignolinDynamics('folded').position
        target_position = torch.tensor(target_position, dtype=torch.float, device=self.device).unsqueeze(0)
        return target_position

    def step(self, force):
        force = force.detach().cpu().numpy()
        for i in range(self.num_samples):
            self.mds[i].step(force[i])

    def report(self):
        positions, velocities, forces, potentials = [], [], [], []
        for i in range(self.num_samples):
            position, velocity, force, potential = self.mds[i].report()
            positions.append(position); velocities.append(velocity); forces.append(force); potentials.append(potential)
            
        positions = torch.tensor(positions, dtype=torch.float, device=self.device)
        velocities = torch.tensor(velocities, dtype=torch.float, device=self.device)
        forces = torch.tensor(forces, dtype=torch.float, device=self.device)
        potentials = torch.tensor(potentials, dtype=torch.float, device=self.device)
        return positions, velocities, forces, potentials
    
    def reset(self):
        for i in range(self.num_samples):
            self.mds[i].reset()

    def set_temperature(self, temperature):
        for i in range(self.num_samples):
            self.mds[i].set_temperature(temperature)


class ChignolinPolicy(nn.Module):
    def __init__(self, args, md):
        super().__init__()
        
        self.force = args.force

        self.num_particles = md.num_particles
        self.input_dim = md.num_particles*3
        self.output_dim = md.num_particles*3 if self.force else 1

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim, bias=False)
        )

        self.log_z = nn.Parameter(torch.tensor(0.))

        self.to(args.device)

    def forward(self, pos):
        if not self.force:
            pos.requires_grad = True
            
        out = self.mlp(pos.reshape(-1, self.input_dim))

        if not self.force:
            force = - torch.autograd.grad(out.sum(), pos, create_graph=True, retain_graph=True)[0]
        else:
            force = out.view(*pos.shape)
                
        return force
    

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

        self.replay = ReplayBuffer(args, md)
        self.policy = ChignolinPolicy(args, md)

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
            log_target_reward = (-1/2)*torch.square((matrix-target_matrix)/args.target_std).mean((1, 2))
            log_target_reward, last_idx = log_target_reward.view(args.num_samples, -1).max(1)
        else:
            matrix = getattr(self, args.reward_matrix)(positions[:, -1])
            last_idx = args.num_steps * torch.ones(args.num_samples, dtype=torch.long, device=args.device)
            log_target_reward = (-1/2)*torch.square((matrix-target_matrix)/args.target_std).mean((1, 2))
            
        log_md_reward = (-1/2)*torch.square(actions/self.std).mean((1, 2, 3))
        log_reward = log_md_reward + log_target_reward

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

if __name__ == '__main__':
    if args.wandb:
        wandb.init(
            project=args.project,
            config=args,
        )

    torch.manual_seed(args.seed)

    md = ChignolinDynamics('unfolded')

    mds = MDs(args)
    logger = Logger(args, md)
    agent = FlowNetAgent(args, md)

    annealing_schedule = torch.linspace(args.start_temperature, args.end_temperature, args.num_rollouts)
    for rollout in range(args.num_rollouts):
        print(f'Rollout: {rollout}')
        temperature = annealing_schedule[rollout]

        print('Sampling...')
        log = agent.sample(args, mds, temperature)

        print('Training...')
        loss = 0
        for _ in range(args.trains_per_rollout):
            loss += agent.train(args)
        
        loss = loss / args.trains_per_rollout

        logger.log(loss, agent, rollout, **log)

    
    if args.wandb:
        wandb.finish()
        wandb.init(project=args.project+"-eval", config=args)

    torch.manual_seed(random.randint())
    args.num_samples = 64
    args.type == 'eval'
    logger = Logger(args, md)
    log = agent.sample(args, mds, 300)
    logger.log(None, agent, 0, **log)