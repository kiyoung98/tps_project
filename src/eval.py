import os
import wandb
import torch
import argparse

from dynamics.mds import MDs
from dynamics import dynamics
from flow import FlowNetAgent
from utils.logging import Logger

parser = argparse.ArgumentParser()

# System Config
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--type', default='eval', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--project', default='alanine', type=str)
parser.add_argument('--molecule', default='alanine', type=str)
parser.add_argument('--date', default='', type=str, help="Date of the training")
parser.add_argument('--logger', default=True, type=bool, help='Use system logger')

# Policy Config
parser.add_argument('--force', action='store_true', help='Model force otherwise potential')

# Sampling Config
parser.add_argument('--start_state', default='c5', type=str)
parser.add_argument('--end_state', default='c7ax', type=str)
parser.add_argument('--bias_scale', default=20, type=float, help='Scale factor of bias')
parser.add_argument('--num_samples', default=64, type=int, help='Number of paths to sample')
parser.add_argument('--flexible', action='store_true', help='Sample paths with flexible length')
parser.add_argument('--std', default=0.1, type=float, help='Standard deviation of Langevin integrator')
parser.add_argument('--timestep', default=1., type=float, help='Timestep (fs) of the langevin integrator')
parser.add_argument('--temperature', default=300., type=float, help='Temperature (K) of the langevin integrator')
parser.add_argument('--num_steps', default=500, type=int, help='Number of steps in each path i.e. length of trajectory')
parser.add_argument('--friction_coefficient', default=1., type=float, help='Friction_coefficient (ps) of the langevin integrator')
parser.add_argument('--target_std', default=0.05, type=float, help='Standard deviation of gaussian distribution w.r.t. dist matrix of position')


args = parser.parse_args()

if args.wandb:
    wandb.init(project=args.project+"-eval", config=args)

if __name__ == '__main__':
    md = getattr(dynamics, args.molecule.title())(args, args.start_state)

    agent = FlowNetAgent(args, md)
    logger = Logger(args, md)

    mds = MDs(args)
    target_position = torch.tensor(md.position, dtype=torch.float, device=args.device).unsqueeze(0)

    train_log_dir = f"results/{args.molecule}/{args.project}/{args.date}/train/{args.seed}"
    filename = "policy.pt"
    policy_file = f"{train_log_dir}/{filename}"
    if os.path.exists(policy_file):
        agent.policy.load_state_dict(torch.load(policy_file))
    else:
        raise FileNotFoundError("Policy checkpoint not found")
    
    # Sampling and obtain results for evaluation (positions, potentials)

    log = agent.sample(args, mds)

    logger.info(f"Sampling done..!")
    
    logger.info(f"Evaluating results...")

    logger.log(None, agent.policy, 0, **log)
    logger.plot(**log)
    logger.info(f"Evaluation done..!")