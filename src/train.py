import wandb
import torch
import random
import argparse
from tqdm import tqdm

from flow import GFlowNet
from dynamics.mds import MDs
from dynamics import dynamics
from utils.logging import Logger

parser = argparse.ArgumentParser()

# System Config
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--molecule', default='alanine', type=str)
parser.add_argument('--project', default='alanine_train', type=str)
parser.add_argument('--type', default='train', type=str)
parser.add_argument('--logger', default=True, type=bool, help='Use system logger')

# Policy Config
parser.add_argument('--bias', action='store_true', help='Use bias in last layer')
parser.add_argument('--force', action='store_true', help='Model force otherwise potential')
parser.add_argument('--goal_conditioned', action='store_true', help='Receive target position')

# Sampling Config
parser.add_argument('--start_states', default='c5', type=str)
parser.add_argument('--end_states', default='c7ax', type=str)
parser.add_argument('--num_steps', default=500, type=int, help='Number of steps in each path i.e. length of trajectory')
parser.add_argument('--num_samples', default=16, type=int, help='Number of paths to sample')
parser.add_argument('--bias_scale', default=1000., type=float, help='Scale of bias which is the output of policy')
parser.add_argument('--timestep', default=1., type=float, help='Timestep (fs) of the langevin integrator')
parser.add_argument('--temperature', default=300., type=float, help='Temperature (K) of the langevin integrator which we want to evaluate')
parser.add_argument('--collision_rate', default=1., type=float, help='Collision Rate (ps) of the langevin integrator')

# Training Config
parser.add_argument('--learning_rate', default=1e-2, type=float)
parser.add_argument('--start_temperature', default=1200., type=float, help='Start of temperature schedule in annealing')
parser.add_argument('--end_temperature', default=300., type=float, help='End of temperature schedule in annealing')
parser.add_argument('--hindsight', action='store_true', help='Use hindsight replay proposed by https://arxiv.org/abs/1707.01495')
parser.add_argument('--num_rollouts', default=5000, type=int, help='Number of rollouts (or sampling)')
parser.add_argument('--trains_per_sample', default=2000, type=int, help='Number of training per sampling in a rollout')
parser.add_argument('--buffer_size', default=100, type=int, help='Size of buffer which stores sampled paths')
parser.add_argument('--terminal_std', default=0.1, type=float, help='Standard deviation of gaussian distribution w.r.t. dist matrix of position')
parser.add_argument('--log_z_scale', default=0.1, type=float, help='Scale of log z to balance learning rate')
parser.add_argument('--max_grad_norm', default=10, type=int, help='Maximum norm of gradient to clip')

args = parser.parse_args()

if args.wandb:
    wandb.init(
        project=args.project,
        config=args,
        entity="postech-ml-tsp"
    )

torch.manual_seed(args.seed)

if __name__ == '__main__':
    start_states = args.start_states.split(',')
    end_states = args.end_states.split(',')

    md_info = getattr(dynamics, args.molecule.title())(args, start_states[0])

    flow = GFlowNet(args, md_info)
    logger = Logger(args, md_info)

    mds_dict = {}
    target_position_dict = {}

    for state in start_states:
        mds_dict[state] = MDs(args, state)

    for state in end_states:
        target_position_dict[state] = getattr(dynamics, args.molecule.title())(args, state).position

    buffer = [] # TODO: Dataset 객체로 바꾸기
    annealing_schedule = torch.linspace(args.start_temperature, args.end_temperature, args.num_rollouts, device=args.device)

    for rollout in range(args.num_rollouts):
        print(f'Rollout: {rollout}')
        temperature = annealing_schedule[rollout]

        print('Sampling:')
        start_state = random.choice(start_states)
        end_state = random.choice(end_states)

        mds = mds_dict[start_state]         
        target_position = target_position_dict[end_state]
        
        data, log = flow.sample(args, mds, target_position, temperature)

        buffer.append(data)
        if len(buffer) > args.buffer_size:
            buffer.pop(0)

        print('Training:')
        loss = 0
        for _ in tqdm(range(args.trains_per_sample)):
            idx = random.randrange(len(buffer))
            data = buffer[idx]
            loss += flow.train(args, data)
        
        loss = loss / args.trains_per_sample

        logger.log(loss, flow.policy, start_state, end_state, rollout, **log)