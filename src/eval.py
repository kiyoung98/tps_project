import os
import wandb
import torch
import argparse
import proxy

from tqdm import tqdm
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
parser.add_argument('--date', default='', type=str, help="Date of the training")
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
parser.add_argument('--hindsight', action='store_true', help='Use hindsight replay proposed by https://arxiv.org/abs/1707.01495')


args = parser.parse_args()

if args.wandb:
    wandb.init(project=args.project, config=args)

if __name__ == '__main__':
    md = getattr(dynamics, args.molecule.title())(args, args.end_state)
    mds = MDs(args, args.start_state)
    target_position = torch.tensor(md.position, dtype=torch.float, device=args.device).unsqueeze(0).unsqueeze(0)

    logger = Logger(args, md)

    # Import policy model
    policy = getattr(proxy, args.molecule.title())(args, md)
    train_log_dir = f"results/{args.molecule}/{args.project}/{args.date}/train/{args.seed}"
    filename = "policy.pt"
    policy_file = f"{train_log_dir}/{filename}"
    if os.path.exists(policy_file):
        policy.load_state_dict(torch.load(policy_file))
    else:
        raise FileNotFoundError("Policy checkpoint not found")
    
    # Sampling and obtain results for evaluation (positions, potentials)
    positions = torch.zeros((args.num_samples, args.num_steps, md.num_particles, 3), device=args.device)
    potentials = torch.zeros(args.num_samples, args.num_steps, device=args.device)
    pbar = tqdm(
        range(args.num_steps),
        total=args.num_steps,
        desc="Sampling using trainend policy..."
    )
    for s in pbar:
        position, potential = mds.report()
        
        positions[:, s] = position
        potentials[:, s] = potential
        bias = policy(position.unsqueeze(1), target_position).squeeze().detach()

        mds.step(bias)

    start_position = positions[0, 0].unsqueeze(0).unsqueeze(0)
    last_position = mds.report()[0].unsqueeze(1)

    logger.info(f"Sampling done..!")
    
    logger.info(f"Evaluating results...")
    log = {
        'positions': positions, 
        'start_position': start_position,
        'last_position': last_position, 
        'target_position': target_position, 
        'potentials': potentials,
        'log_reward': 0,
        'terminal_reward': 0,
    }
    logger.log(None, policy, args.start_state, args.end_state, 0, **log)
    logger.plot(**log)
    logger.info(f"Evaluation done..!")