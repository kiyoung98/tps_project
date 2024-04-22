import wandb
import torch
import random
import argparse
from tqdm import tqdm

from flow import FlowNetAgent
from dynamics.mds import MDs
from dynamics import dynamics
from utils.logging import Logger

parser = argparse.ArgumentParser()

# System Config
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--molecule', default='alanine', type=str)
parser.add_argument('--project', default='alanine', type=str)
parser.add_argument('--type', default='train', type=str)
parser.add_argument('--logger', default=True, type=bool, help='Use system logger')

# Policy Config
parser.add_argument('--force', action='store_true', help='Model force otherwise potential')
parser.add_argument('--goal_conditioned', action='store_true', help='Receive target position')

# Sampling Config
parser.add_argument('--start_states', default='c5', type=str)
parser.add_argument('--end_states', default='c7ax', type=str)
parser.add_argument('--num_steps', default=500, type=int, help='Number of steps in each path i.e. length of trajectory')
parser.add_argument('--num_samples', default=16, type=int, help='Number of paths to sample')
parser.add_argument('--temperature', default=0., type=float, help='In training, must set 0(K) since we use external noise')

# Training Config
parser.add_argument('--loss', default='tb', type=str)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--std', default=0.1, type=float, help='std of target policy')
parser.add_argument('--start_std', default=0.2, type=float, help='Start std of annealing schedule used in behavior policy')
parser.add_argument('--end_std', default=0.1, type=float, help='End std of annealing schedule used in behavior policy')
parser.add_argument('--hindsight', action='store_true', help='Use hindsight replay proposed by https://arxiv.org/abs/1707.01495')
parser.add_argument('--num_rollouts', default=10000, type=int, help='Number of rollouts (or sampling)')
parser.add_argument('--trains_per_rollout', default=2000, type=int, help='Number of training per rollout in a rollout')
parser.add_argument('--buffer_size', default=200, type=int, help='Size of buffer which stores sampled paths')
parser.add_argument('--terminal_std', default=0.05, type=float, help='Standard deviation of gaussian distribution w.r.t. dist matrix of position')
parser.add_argument('--max_grad_norm', default=10, type=int, help='Maximum norm of gradient to clip')

args = parser.parse_args()

if args.wandb:
    wandb.init(
        project=args.project,
        config=args,
        # entity="postech-ml-tsp"
    )

torch.manual_seed(args.seed)

if __name__ == '__main__':
    start_states = args.start_states.split(',')
    end_states = args.end_states.split(',')

    md = getattr(dynamics, args.molecule.title())(args, start_states[0])
    agent = FlowNetAgent(args, md)
    logger = Logger(args, md)

    mds_dict = {}
    target_position_dict = {}

    logger.info(f"Starting MD at {args.start_states}")
    for state in start_states:
        mds_dict[state] = MDs(args, state)

    logger.info(f"Getting attributes for {args.end_states}")
    for state in end_states:
        target_position = getattr(dynamics, args.molecule.title())(args, state).position
        target_position_dict[state] = torch.tensor(target_position, dtype=torch.float, device=args.device)

    annealing_schedule = torch.linspace(args.start_std, args.end_std, args.num_rollouts, device=args.device)

    logger.info("")
    logger.info(f"Starting training for {args.num_rollouts} rollouts")
    for rollout in range(args.num_rollouts):
        print(f'Rollout: {rollout}')
        std = annealing_schedule[rollout]

        print('Sampling:')
        start_state = random.choice(start_states)
        end_state = random.choice(end_states)

        mds = mds_dict[start_state]         
        target_position = target_position_dict[end_state].unsqueeze(0).unsqueeze(0)
        
        log = agent.sample(args, mds, target_position, std)

        print('Training:')
        loss = 0
        for _ in tqdm(range(args.trains_per_rollout)):
            loss += agent.train(args)
        
        loss = loss / args.trains_per_rollout

        logger.log(loss, agent.policy, start_state, end_state, rollout, **log)
    
    logger.info("")
    logger.info(f"Training finished for {args.num_rollouts} rollouts..!")
    