import time
import wandb
import torch
import argparse

import numpy as np
from tqdm import tqdm
from dynamics.mds import MDs
from dynamics import dynamics
from flow import FlowNetAgent
from utils.logging import Logger

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--type", default="train", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--molecule", default="alanine", type=str)

# Logger Config
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--project", default="alanine", type=str)
parser.add_argument("--save_dir", default="results", type=str)
parser.add_argument("--date", default="date", type=str)
parser.add_argument("--save_freq", default=100, type=int)

# Policy Config
parser.add_argument("--force", action="store_true")
parser.add_argument("--log_z", default="-8", type=float)
parser.add_argument("--dist_feat", action="store_true")

# Sampling Config
parser.add_argument("--start_state", default="c5", type=str)
parser.add_argument("--end_state", default="c7ax", type=str)
parser.add_argument("--num_steps", default=1000, type=int)
parser.add_argument("--bias_scale", default=0.01, type=float)
parser.add_argument("--timestep", default=1, type=float)
parser.add_argument("--sigma", default=0.05, type=float)
parser.add_argument("--num_samples", default=16, type=int)
parser.add_argument("--temperature", default=300, type=float)

# Training Config
parser.add_argument("--start_temperature", default=600, type=float)
parser.add_argument("--end_temperature", default=300, type=float)
parser.add_argument("--max_grad_norm", default=1, type=int)
parser.add_argument("--num_rollouts", default=1000, type=int)
parser.add_argument("--log_z_lr", default=1e-2, type=float)
parser.add_argument(
    "--policy_lr",
    default=1e-4,
    type=float,
)
parser.add_argument(
    "--buffer_size",
    default=2048,
    type=int,
)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument(
    "--trains_per_rollout",
    default=2000,
    type=int,
)

args = parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(args.seed)
    if args.wandb:
        wandb.init(project=args.project, config=args)

    md = getattr(dynamics, args.molecule.title())(args, args.start_state)
    logger = Logger(args, md)

    logger.info(f"Initialize {args.num_samples} MDs starting at {args.start_state}")
    mds = MDs(args)
    agent = FlowNetAgent(args, md, mds)

    temperatures = torch.linspace(
        args.start_temperature, args.end_temperature, args.num_rollouts
    )

    losses = []
    runtimes = []
    rmsds = []
    rmsd_stds = []
    total_start = time.time()

    logger.info("Start training")
    for rollout in range(args.num_rollouts):
        print(f"Rollout: {rollout}")
        start = time.time()

        log = agent.sample(args, mds, temperatures[rollout])
        log = logger.log(agent.policy, rollout, **log)

        loss = 0
        for _ in tqdm(range(args.trains_per_rollout), desc="Training"):
            loss += agent.train(args)
        loss = loss / args.trains_per_rollout
        runtime = time.time() - start

        losses.append(loss)
        runtimes.append(runtime)
        rmsds.append(log["rmsd"])
        rmsd_stds.append(log["std_rmsd"])

        logger.info(f"loss: {loss}")
        logger.info(f"runtime: {runtime}")
        if args.wandb:
            wandb.log({"loss": loss}, step=rollout)
            wandb.log({"runtime": runtime}, step=rollout)
    if args.wandb:
        wandb.log({"total_runtime": time.time() - total_start})
        wandb.log({"runtime_per_rollout": np.mean(runtimes)})

    logger.info(f"total runtime: {time.time() - total_start}")
    logger.info(f"runtime per rollout: {np.mean(runtimes)}")

    np.save(f"{logger.save_dir}/losses.npy", losses)
    np.save(f"{logger.save_dir}/rmsds.npy", rmsds)
    np.save(f"{logger.save_dir}/rmsd_stds.npy", rmsd_stds)

    logger.info("Finish training")
