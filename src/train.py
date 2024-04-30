import yaml
import wandb
import torch
import argparse
from tqdm import tqdm

from flow import FlowNetAgent
from dynamics.mds import MDs
from dynamics import dynamics
from utils.logging import Logger

parser = argparse.ArgumentParser()

# System Config
parser.add_argument('--molecule', default='alanine', type=str)
parser.add_argument('--project', default='alanine', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--type', default='train', type=str)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--device', default='cuda', type=str)

# Logger Config
parser.add_argument('--logger', default=True, type=bool, help='Use system logger')
parser.add_argument('--date', default="test-run", type=str, help='Date of the training')
parser.add_argument('--save_freq', default=10, type=int, help='Frequency of saving in  rollouts')
parser.add_argument('--server', default="server", type=str, choices=["server", "cluster", "else"], help='Server we are using')
parser.add_argument('--config', default="", type=str, help='Path to config file')

# Policy Config
parser.add_argument('--force', action='store_true', help='Model force otherwise potential')

# Sampling Config
parser.add_argument('--start_state', default='c5', type=str)
parser.add_argument('--end_state', default='c7ax', type=str)
parser.add_argument('--target_std', default=0.05, type=float, help='Standard deviation of gaussian distribution w.r.t. dist matrix of position')
parser.add_argument('--num_steps', default=800, type=int, help='Number of steps in each path i.e. length of trajectory')
parser.add_argument('--num_samples', default=16, type=int, help='Number of paths to sample')
parser.add_argument('--temperature', default=0., type=float, help='In training, must set 0(K) since we use external noise')

# Training Config
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--num_rollouts', default=5000, type=int, help='Number of rollouts (or sampling)')
parser.add_argument('--trains_per_rollout', default=1, type=int, help='Number of training per rollout in a rollout')
parser.add_argument('--buffer_size', default=2048, type=int, help='Size of buffer which stores sampled paths')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--init_buffer', action='store_true')
parser.add_argument('--replay_strategy', default='', type=str)
parser.add_argument('--max_grad_norm', default=10, type=int, help='Maximum norm of gradient to clip')

args = parser.parse_args()

if args.wandb:
    wandb.init(
        project=args.project,
        config=args,
    )

torch.manual_seed(args.seed)

if __name__ == '__main__':
    # NOTE: testing parsing from config file
    # Nothing happends if config file is not provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            for config_class in config.keys():
                for key, value in config[config_class].items():
                    setattr(args, key, value)

    md = getattr(dynamics, args.molecule.title())(args, args.start_state)
    agent = FlowNetAgent(args, md)
    logger = Logger(args, md)

    logger.info(f"Starting MD at {args.start_state}")
    mds = MDs(args)

    if args.init_buffer:
        logger.info(f"Initializing buffer with MD")
        for _ in tqdm(range(args.buffer_size//args.num_samples)):
            agent.sample(args, mds, False)

    logger.info("")
    logger.info(f"Starting training for {args.num_rollouts} rollouts")
    for rollout in range(args.num_rollouts):
        print(f'Rollout: {rollout}')

        print('Sampling...')
        log = agent.sample(args, mds)

        print('Training...')
        loss = 0
        for _ in range(args.trains_per_rollout):
            loss += agent.train(args)
        
        loss = loss / args.trains_per_rollout

        logger.log(loss, agent.policy, rollout, **log)
    
    logger.info("")
    logger.info(f"Training finished for {args.num_rollouts} rollouts..!")
    