import os
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
parser.add_argument('--type', default='eval', type=str)
parser.add_argument('--date', default='', type=str, help="Date of the training, uses the most recent if not provided")
parser.add_argument('--logger', default=True, type=bool, help='Use system logger')

# Policy Config
parser.add_argument('--force', action='store_true', help='Model force otherwise potential')
parser.add_argument('--goal_conditioned', action='store_true', help='Receive target position')

# Sampling Config
parser.add_argument('--start_state', default='c5', type=str)
parser.add_argument('--end_state', default='c7ax', type=str)
parser.add_argument('--num_steps', default=500, type=int, help='Number of steps in each path i.e. length of trajectory')
parser.add_argument('--num_samples', default=16, type=int, help='Number of paths to sample')
parser.add_argument('--temperature', default=300., type=float, help='Temperature (K) of the langevin integrator')

args = parser.parse_args()

if args.wandb:
    wandb.init(project=args.project, config=args)

if __name__ == '__main__':
    md = getattr(dynamics, args.molecule.title())(args, args.end_state)
    mds = MDs(args, args.start_state)
    target_position = torch.tensor(md.position, dtype=torch.float, device=args.device).unsqueeze(0).unsqueeze(0)

    logger = Logger(args, md)

    policy = getattr(proxy, args.molecule.title())(args, md)
    # if args.date == '':
    #     directory = f"results/{args.molecule}"
    #     folders = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    #     folders.sort(key=lambda x: os.path.getctime(x), reverse=True)
    #     if folders:
    #         args.date = os.path.basename(folders[0])
    #         logger.info(f"Using the most recent training date: {args.date}")
    #     else:
    #         raise ValueError(f"No folders found in {args.moleulce} directory")
    policy.load_state_dict(torch.load(f'results/{args.molecule}/{args.date}/policy.pt'))

    positions = torch.zeros((args.num_samples, args.num_steps, md.num_particles, 3), device=args.device)
    potentials = torch.zeros(args.num_samples, args.num_steps, device=args.device)

    print('Sampling:')
    for s in tqdm(range(args.num_steps)):
        position, potential = mds.report()
        
        positions[:, s] = position
        potentials[:, s] = potential
        bias = policy(position.unsqueeze(1), target_position).squeeze().detach()

        mds.step(bias)

    start_position = positions[0, 0].unsqueeze(0).unsqueeze(0)
    last_position = mds.report()[0].unsqueeze(1)

    log = {
        'positions': positions, 
        'start_position': start_position,
        'last_position': last_position, 
        'target_position': target_position, 
        'potentials': potentials,
        'date': args.date
    }

    logger.log(None, policy, args.start_state, args.end_state, 0, **log)
    logger.plot(**log)