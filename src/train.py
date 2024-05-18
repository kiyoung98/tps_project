import tqdm
import torch
import argparse

from flow import FlowNetAgent
from dynamics.mds import MDs
from dynamics import dynamics
from utils.logging import Logger

parser = argparse.ArgumentParser()

# System Config
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--molecule', default='alanine', type=str)
parser.add_argument('--save_dir', default='results/', type=str)

# Policy Config
parser.add_argument('--force', action='store_true', help='Predict force otherwise potential')

# Sampling Config
parser.add_argument('--start_state', default='c5', type=str)
parser.add_argument('--end_state', default='c7ax', type=str)
parser.add_argument('--num_steps', default=1000, type=int, help='Length of paths')
parser.add_argument('--bias_scale', default=1, type=float, help='Scale factor of bias')
parser.add_argument('--timestep', default=1, type=float, help='Timestep of integrator')
parser.add_argument('--num_samples', default=64, type=int, help='Number of paths to sample')
parser.add_argument('--temperature', default=300, type=float, help='Temperature for evaluation')
parser.add_argument('--target_std', default=0.05, type=float, help='Standard deviation of gaussian distribution w.r.t. dist matrix of position')

# Training Config
parser.add_argument('--train_temperature', default=600, type=float, help='Temperature for training')
parser.add_argument('--max_grad_norm', default=10, type=int, help='Maximum norm of gradient to clip')
parser.add_argument('--num_rollouts', default=5000, type=int, help='Number of rollouts (or sampling)')
parser.add_argument('--log_z_lr', default=1e-2, type=float, help='Learning rate of estimator for log Z')
parser.add_argument('--mlp_lr', default=1e-4, type=float, help='Learning rate of bias potential or force')
parser.add_argument('--buffer_size', default=2048, type=int, help='Size of buffer which stores sampled paths')
parser.add_argument('--trains_per_rollout', default=2000, type=int, help='Number of training per rollout in a rollout')

args = parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(args.seed)

    md = getattr(dynamics, args.molecule.title())(args, args.start_state)
    agent = FlowNetAgent(args, md)
    logger = Logger(args, md)

    logger.info(f"Initialize {args.num_samples} MDs starting at {args.start_state}")
    mds = MDs(args)

    logger.info("Start training")

    for rollout in range(args.num_rollouts):
        print(f'Rollout: {rollout}')

        log = agent.sample(args, mds, args.train_temperature)

        loss = 0
        for _ in tqdm(range(args.trains_per_rollout), desc='Training'):
            loss += agent.train(args)
        loss = loss / args.trains_per_rollout

        logger.log(loss, agent.policy, rollout, **log)

    logger.info("Finish training")
    