import wandb
import torch
import argparse
from tqdm import tqdm

import proxy
from dynamics.mds import MDs
from dynamics import dynamics
from utils.logging import Logger

parser = argparse.ArgumentParser()

# System Config
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--molecule', default='alanine', type=str)
parser.add_argument('--project', default='alanine_eval', type=str)

# Policy Config
parser.add_argument('--bias', action='store_true', help='Use bias in last layer')
parser.add_argument('--force', action='store_true', help='Model force otherwise potential')
parser.add_argument('--goal_conditioned', action='store_true', help='Receive target position')

# Sampling Config
parser.add_argument('--start_state', default='c5', type=str)
parser.add_argument('--end_state', default='c7ax', type=str)
parser.add_argument('--num_steps', default=500, type=int, help='Number of steps in each path i.e. length of trajectory')
parser.add_argument('--num_samples', default=16, type=int, help='Number of paths to sample')
parser.add_argument('--bias_scale', default=1000., type=float, help='Scale of bias which is the output of policy')
parser.add_argument('--timestep', default=1., type=float, help='Timestep (fs) of the langevin integrator')
parser.add_argument('--temperature', default=300., type=float, help='Temperature (K) of the langevin integrator which we want to evaluate')
parser.add_argument('--collision_rate', default=1., type=float, help='Collision Rate (ps) of the langevin integrator')

args = parser.parse_args()

if args.wandb:
    wandb.init(project=args.project, config=args)

if __name__ == '__main__':
    info = getattr(dynamics, args.molecule.title())(args, args.end_state)
    mds = MDs(args, args.start_state)

    logger = Logger(args, info)

    policy = getattr(proxy, args.molecule.title())(args, info)
    policy.load_state_dict(torch.load(f'results/{args.molecule}/policy.pt'))

    positions = torch.zeros((args.num_samples, args.num_steps+1, info.num_particles, 3), device=args.device)
    potentials = torch.zeros(args.num_samples, args.num_steps+1, device=args.device)

    position, _, _, potential = mds.report()
    
    positions[:, 0] = position
    potentials[:, 0] = potential

    target_position = info.position.unsqueeze(0).unsqueeze(0)

    print('Sampling:')
    for s in tqdm(range(args.num_steps)):
        bias = args.bias_scale * policy(position.unsqueeze(1), target_position).squeeze().detach()
        mds.step(bias)

        position, _, _, potential = mds.report()
        
        positions[:, s+1] = position
        potentials[:, s+1] = potential

    log = {
        'positions': positions, 
        'start_position': positions[:1, :1],
        'last_position': position, 
        'target_position': target_position, 
        'potentials': potentials,
    }

    logger.log(None, policy, args.start_state, args.end_state, 0, **log)
    logger.plot(**log)