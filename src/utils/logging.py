import sys
import logging
import datetime

from .plot import *
from .metrics import Metric

class Logger():
    def __init__(self, args, md):
        self.metric = Metric(args, md)

        self.seed = args.seed
        self.type = args.type
        self.save_dir = args.save_dir
        self.molecule = args.molecule
        self.num_samples = args.num_samples
        self.save_freq = args.save_freq if args.train else 1
            
        # Logger basic configurations
        self.logger = logging.getLogger("tps")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(args.save_dir, mode="w")
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
        self.logger.info(f"Log {self.type} {self.molecule}, seed {self.seed}")
        self.logger.info("")

    def info(self, message):
        if self.logger:
            self.logger.info(message)
    
    def log(
            self, 
            loss, 
            policy, 
            rollout, 
            last_idx, 
            positions, 
            potentials, 
            last_position,
            log_likelihood, 
            target_position,
            biased_log_likelihood,  
        ):
        # Calculate metrics
        if self.molecule == 'alanine':
            hit, thp, mean_len, std_len, mean_etp, std_etp, mean_efp, std_efp = self.metric.alanine(positions, target_position, potentials)
            true_likelihood = log_likelihood.exp() * hit
            biased_likelihood = biased_log_likelihood.exp()
            ess_ratio = self.metric.effective_sample_size(biased_likelihood, true_likelihood) / self.num_samples

        mean_ll, std_ll = log_likelihood.mean().item(), log_likelihood.std().item()
        mean_pd, std_pd = self.metric.expected_pairwise_distance(last_position, target_position)
        mean_pcd, std_pcd = self.metric.expected_pairwise_coulomb_distance(last_position, target_position)

        # Log
        if self.type == "train":
            self.logger.info(f'Rollout: {rollout}')
            self.logger.info(f"loss: {loss}")
            torch.save(policy.state_dict(), f'{self.save_dir}/policy.pt')

        if self.molecule == 'alanine':
            self.logger.info(f"target_hit_percentage (%): {thp}")
            self.logger.info(f"effective_sample_size_ratio: {ess_ratio}")
            self.logger.info(f"energy_transition_point (kJ/mol): {mean_etp}")
            self.logger.info(f"energy_final_point (kJ/mol): {mean_efp}")
            self.logger.info(f"mean_length: {mean_len}")
            self.logger.info(f"std_length: {std_len}")
            self.logger.info(f"std_etp: {std_etp}")
            self.logger.info(f"std_etp: {std_efp}")

        self.logger.info(f"log_z: {policy.log_z.item()}")
        self.logger.info(f"log_likelihood: {mean_ll}")
        self.logger.info(f"expected_pairwise_distance (pm): {mean_pd}")
        self.logger.info(f"expected_pairwise_coulomb_distance: {mean_pcd}")
        self.logger.info(f"std_ll: {std_ll}")
        self.logger.info(f"std_pd: {std_pd}")
        self.logger.info(f"std_pcd: {std_pcd}")

        if self.molecule == 'alanine' and rollout % 10 == 0:
            plot_paths_alanine(self.save_dir, rollout, positions, target_position, last_idx)