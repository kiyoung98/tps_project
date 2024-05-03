import torch
import random
import numpy as np
from torch import nn
import openmm.unit as unit
from scipy.constants import physical_constants

import openmm as mm
from tqdm import tqdm
from openmm import app

from src.utils.metrics import *
from src.utils.utils import *
from src.utils.plot import *

import argparse

parser = argparse.ArgumentParser()

# System Config
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)

# Policy Config
parser.add_argument('--force', action='store_true', help='Predict force otherwise potential')

# Sampling Config
parser.add_argument('--bias_scale', default=20, type=float, help='Scale factor of bias')
parser.add_argument('--num_samples', default=16, type=int, help='Number of paths to sample')
parser.add_argument('--flexible', action='store_true', help='Sample paths with flexible length')
parser.add_argument('--num_steps', default=500, type=int, help='Number of steps in each path i.e. length of trajectory')
parser.add_argument('--target_std', default=2, type=float, help='Standard deviation of gaussian distribution w.r.t. dist matrix of position')

# Training Config
parser.add_argument('--learning_rate', default=1e-2, type=float)
parser.add_argument('--num_rollouts', default=10000, type=int, help='Number of rollouts (or sampling)')
parser.add_argument('--trains_per_rollout', default=2000, type=int, help='Number of training per rollout in a rollout')
parser.add_argument('--buffer_size', default=100, type=int, help='Size of buffer which stores sampled paths')
parser.add_argument('--max_grad_norm', default=10, type=int, help='Maximum norm of gradient to clip')

args = parser.parse_args()

args.save_dir = f'results/{args.seed}'

class AlanineDynamics:
    def __init__(self, state):
        super().__init__()
        self.start_file = f'./data/alanine/{state}.pdb'

        self.temperature = 300 * unit.kelvin
        self.friction_coefficient = 1 / unit.picoseconds
        self.timestep = 1 * unit.femtoseconds

        self.integrator, self.simulation, self.external_force = self.setup()

        self.simulation.minimizeEnergy()
        self.position = self.report()[0]
        self.reset()

        self.num_particles = self.simulation.system.getNumParticles()
        self.v_scale, self.f_scale, self.masses, self.std = self.get_md_info()

    def setup(self):
        forcefield = app.ForceField('amber99sbildn.xml')
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
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

        return integrator, simulation, external_force

    def get_md_info(self):
        v_scale = np.exp(-self.timestep * self.friction_coefficient)
        f_scale = (1 - v_scale) / self.friction_coefficient

        masses = [self.simulation.system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(self.num_particles)]
        masses = unit.Quantity(np.array(masses), unit.dalton)
        
        unadjusted_variance = unit.BOLTZMANN_CONSTANT_kB * self.temperature * (1 - v_scale ** 2) / masses[:, None]
        std_SI_units = 1 / physical_constants['unified atomic mass unit'][0] * unadjusted_variance.value_in_unit(unit.joule / unit.dalton)
        std = unit.Quantity(np.sqrt(std_SI_units), unit.meter / unit.second)
        return v_scale, f_scale, masses, std

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
        print(f"Initialize dynamics starting at c5 of alanine")

        mds = []
        for _ in tqdm(range(self.num_samples)):
            md = AlanineDynamics('c5')
            mds.append(md)
        return mds

    def _init_target_position(self):
        target_position = AlanineDynamics('c7ax').position
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


class AlaninePolicy(nn.Module):
    def __init__(self, args, md):
        super().__init__()
        
        self.force = args.force

        self.num_particles = md.num_particles
        self.input_dim = md.num_particles*3
        self.output_dim = md.num_particles*3 if self.force else 1

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim, bias=False)
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

        self.policy = AlaninePolicy(args, md)

        self.replay = ReplayBuffer(args)

    def sample(self, args, mds):
        positions = torch.zeros((args.num_samples, args.num_steps+1, self.num_particles, 3), device=args.device)
        actions = torch.zeros((args.num_samples, args.num_steps, self.num_particles, 3), device=args.device)
        potentials = torch.zeros(args.num_samples, args.num_steps+1, device=args.device)
        
        position, _, _, potential = mds.report()
        
        positions[:, 0] = position
        potentials[:, 0] = potential

        for s in tqdm(range(args.num_steps), desc='Sampling'):
            bias = args.bias_scale * self.policy(position.detach()).squeeze().detach()
            mds.step(bias)
            
            next_position, velocity, force, potential = mds.report()

            # extract noise which openmm does not provide
            noises = (next_position - position) - (self.v_scale * velocity + self.f_scale * force / self.masses)
            action = self.f_scale * bias / self.masses + noises # TODO: 수치 찾기

            positions[:, s+1] = next_position
            actions[:, s] = action
            potentials[:, s+1] = potential - (bias*next_position).sum((-2, -1))

            position = next_position
        mds.reset()
        
        target_dist_matrix = get_dist_matrix(mds.target_position)

        dist_matrix = get_dist_matrix(positions[:, -1])
        last_idx = args.num_steps * torch.ones(args.num_samples, dtype=torch.long, device=args.device)
        log_target_reward = torch.square((dist_matrix-target_dist_matrix)).sum((1, 2))
        log_md_reward = torch.square(actions/self.std).sum((1, 2, 3))
        log_reward = log_md_reward + log_target_reward

        self.replay.add((positions, actions, log_reward))
        
        res = {
            'positions': positions, 
            'last_position': positions[torch.arange(args.num_samples), last_idx],
            'target_position': mds.target_position,
            'potentials': potentials,
            'log_target_reward': log_target_reward,
            'log_md_reward': log_md_reward,
            'log_reward': log_reward,
            'last_idx': last_idx,
        }
        return res


    def train(self, args):
        policy_optimizers = torch.optim.SGD(self.policy.parameters(), lr=args.learning_rate)

        positions, actions, log_reward = self.replay.sample()

        biases = args.bias_scale * self.policy(positions[:, :-1])
        biases = self.f_scale * biases / self.masses
        
        log_z = self.policy.log_z
        log_forward = torch.square((biases-actions)/self.std).sum((1, 2, 3))
        loss = torch.mean((log_z+log_forward-log_reward)**2)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
        
        policy_optimizers.step()
        policy_optimizers.zero_grad()
        return loss.item()

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


def log(loss, policy, rollout, positions, last_position, target_position, potentials, log_target_reward, log_md_reward, log_reward, last_idx):
    epd = expected_pairwise_distance(last_position, target_position)
    thp = target_hit_percentage(last_position, target_position)
    etp = energy_transition_point(last_position, target_position, potentials, last_idx)
    nll = -log_md_reward.mean().item()

    torch.save(policy.state_dict(), f'{args.save_dir}/policy.pt')
    if rollout % 10 == 0:
        res = f"Rollout: {rollout}, Loss: {loss:.4f}, EPD: {epd:.2f}, THP: {thp:.2f}, ETP: {str(etp)[:4]}, NLL: {nll:.2f}"
        plot_paths_alanine(args.save_dir, rollout, positions, target_position, last_idx, res)
        plot_potentials(args.save_dir, rollout, potentials, log_target_reward, log_reward, last_idx)

def plot(positions, target_position, potentials, log_target_reward, log_reward, last_idx, **kwargs):
    plot_potential(args.save_dir, potentials, log_target_reward, log_reward, last_idx)
    plot_3D_view(args.save_dir, './data/alanine/c5.pdb', positions, last_idx)
    plot_path(args.save_dir, positions, target_position, last_idx)


torch.manual_seed(args.seed)

mds = MDs(args)
agent = FlowNetAgent(args, mds.mds[0])

mds.set_temperature(1200)
for rollout in range(args.num_rollouts):    
    res = agent.sample(args, mds)

    loss = 0
    for _ in tqdm(range(args.trains_per_rollout), desc='Training'):
        loss += agent.train(args)
    
    loss = loss / args.trains_per_rollout

    log(loss, agent.policy, rollout, **res)
    
mds.set_temperature(300)
res = agent.sample(args, mds)
log(None, agent.policy, 0, **res)
plot(**res)