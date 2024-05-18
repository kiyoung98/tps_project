import torch
import argparse

from dynamics.mds import MDs
from dynamics import dynamics
from flow import FlowNetAgent
from utils.logging import Logger

parser = argparse.ArgumentParser()

# System Config
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--molecule', default='alanine', type=str)
parser.add_argument('--model_dir', default='model/', type=str)
parser.add_argument('--save_dir', default='results/', type=str)

# Policy Config
parser.add_argument('--force', action='store_true', help='Network predicts force')

# Sampling Config
parser.add_argument('--start_state', default='c5', type=str)
parser.add_argument('--end_state', default='c7ax', type=str)
parser.add_argument('--num_steps', default=1000, type=int, help='Length of paths')
parser.add_argument('--bias_scale', default=1, type=float, help='Scale factor of bias')
parser.add_argument('--timestep', default=1, type=float, help='Timestep of integrator')
parser.add_argument('--num_samples', default=64, type=int, help='Number of paths to sample')
parser.add_argument('--temperature', default=300, type=float, help='Temperature for evaluation')
parser.add_argument('--target_std', default=0.05, type=float, help='Standard deviation of gaussian distribution w.r.t. dist matrix of position')

args = parser.parse_args()

if __name__ == '__main__':
    md = getattr(dynamics, args.molecule.title())(args, args.start_state)
    agent = FlowNetAgent(args, md)
    logger = Logger(args, md)

    mds = MDs(args)

    policy_path = args.save_dir + '/policy.pt'
    agent.policy.load_state_dict(torch.load(args.save_dir))
    
    logger.info(f"Start Evaulation!")
    log = agent.sample(args, mds, args.temperature)
    logger.log(None, agent.policy, 0, **log)
    logger.info(f"Finish Evaluation")