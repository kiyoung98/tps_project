import os
import sys
import torch
import logging
import datetime

from .plot import plot_paths_alanine
from .metrics import Metric

class Logger():
    def __init__(self, args, md):
        self.metric = Metric(args, md)

        self.best_loss = float('inf')
        self.train = args.train
        self.save_dir = args.save_dir
        self.molecule = args.molecule
        self.num_steps = args.num_steps
        self.num_samples = args.num_samples
            
        if not os.path.exists(f'{self.save_dir}'):
            os.makedirs(f'{self.save_dir}')

        # Logger basic configurations
        self.logger = logging.getLogger("tps")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = args.save_dir + 'train.log' if args.train else args.save_dir + 'eval.log'
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
            
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False
            
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"Date: {date}")
        self.logger.info(f"Log {self.molecule}, Train {self.train}")
        self.logger.info("")

    def info(self, message):
        if self.logger:
            self.logger.info(message)
    
    def log(
            self, 
            loss, 
            policy, 
            rollout, 
            actions,
            last_idx,
            positions, 
            potentials, 
            last_position,
            target_position,
        ):
        # Calculate metrics
        if self.molecule == 'alanine':
            thp, mean_etp, std_etp, mean_efp, std_efp = self.metric.cv_metrics(last_position, target_position, potentials, last_idx)

        mean_ll, std_ll = self.metric.log_likelihood(actions)
        mean_pd, std_pd = self.metric.expected_pairwise_distance(last_position, target_position)
        mean_pcd, std_pcd = self.metric.expected_pairwise_coulomb_distance(last_position, target_position)
        mean_len, std_len = last_idx.float().mean().item(), last_idx.float().std().item()

        # Log
        if self.train:
            self.logger.info("--------------------------------------")
            self.logger.info(f'Rollout: {rollout}')
            self.logger.info(f"loss: {loss}")
            if loss < self.best_loss:
                self.best_loss = loss
                torch.save(policy.state_dict(), f'{self.save_dir}/policy.pt')

        if self.molecule == 'alanine':
            self.logger.info(f"target_hit_percentage (%): {thp}")
            self.logger.info(f"energy_transition_point (kJ/mol): {mean_etp}")
            self.logger.info(f"energy_final_point (kJ/mol): {mean_efp}")
            self.logger.info(f"std_etp: {std_etp}")
            self.logger.info(f"std_etp: {std_efp}")

        self.logger.info(f"log_z: {policy.log_z.item()}")
        self.logger.info(f"log_likelihood: {mean_ll}")
        self.logger.info(f"expected_pairwise_distance (nm): {mean_pd}")
        self.logger.info(f"expected_pairwise_coulomb_distance: {mean_pcd}")
        self.logger.info(f"mean_length: {mean_len}")
        self.logger.info(f"std_ll: {std_ll}")
        self.logger.info(f"std_pd: {std_pd}")
        self.logger.info(f"std_pcd: {std_pcd}")
        self.logger.info(f"std_length: {std_len}")

        if self.molecule == 'alanine' and rollout % 10 == 0:
            plot_paths_alanine(self.save_dir, rollout, positions, target_position, last_idx)