import os
import sys
import pytz
import wandb
import logging
import datetime

from .plot import *
from .metrics import Metric
from tqdm import tqdm

class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""
    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

class Logger():
    def __init__(self, args, md):
        self.type = args.type
        self.wandb = args.wandb
        self.project = args.project
        self.molecule = args.molecule
        self.flexible = args.flexible
        self.start_file = md.start_file
        self.num_samples = args.num_samples
        if self.type == "train":
            self.num_rollouts = args.num_rollouts
            self.save_freq = args.save_freq
        else:
            self.save_freq = 1
            
        self.seed = args.seed
        if not hasattr(args, 'date'):
            raise ValueError("Date is not provided in args")
        self.date = args.date
        kst = pytz.timezone('Asia/Seoul')
        
        self.dir = f'results/{self.molecule}/{self.project}/{self.date}/{self.type}/{args.seed}'

        self.metric = Metric(args, md)
        
        # Set up system logging    
        if args.logger:
            # Check directories
            if not os.path.exists(self.dir):
                os.makedirs(self.dir)
            for dir in ['policy', 'etp', 'efp', 'potential']:
                if not os.path.exists(f'{self.dir}/{dir}'):
                    os.makedirs(f'{self.dir}/{dir}')
            
            # Logger basic configurations
            log_file_name = self.dir + f"/{self.type}.log"
            self.logger = logging.getLogger("tps")
            self.logger.setLevel(logging.INFO)
            
            # File handler
            file_handler = logging.FileHandler(log_file_name, mode="w")
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
            
            # tqdm handler
            # self.logger.addHandler(TqdmLoggingHandler())
            
            log_date = datetime.datetime.now(tz=kst).strftime("%Y-%m-%d %H:%M:%S")
            self.logger.info(f"Date: {log_date}")
            self.logger.info(f"Logging {self.type} {self.molecule}, seed {self.seed}")
            self.logger.info("")

    def info(self, message):
        if self.logger:
            self.logger.info(message)
    
    def log(
            self, 
            policy, 
            loss, 
            rollout, 
            positions, 
            biases, 
            potentials, 
            last_idx, 
            target_position, 
            log_reward, 
            log_md_reward, 
            log_target_reward,
            log_likelihood,  
            plot=False,
        ):
        last_position = positions[torch.arange(self.num_samples), last_idx]

        mean_bias_norm = torch.norm(biases, dim=-1).mean()
        mean_nll, std_nll = -log_md_reward.mean().item(), log_md_reward.std().item()
        ess_ratio = self.metric.effective_sample_size(log_likelihood, log_reward) / self.num_samples
        mean_ppd, std_ppd = self.metric.expected_pairwise_path_distance(positions)
        mean_pd, std_pd = self.metric.expected_pairwise_distance(last_position, target_position)
        mean_psd, std_psd = self.metric.expected_pairwise_scaled_distance(last_position, target_position)
        mean_pcd, std_pcd = self.metric.expected_pairwise_coulomb_distance(last_position, target_position)

        mean_reward, std_reward = log_reward.mean().item(), log_reward.std().item()
        mean_md_reward, std_md_reward = log_md_reward.mean().item(), log_md_reward.std().item()
        mean_target_reward, std_target_reward = log_target_reward.mean().item(), log_target_reward.std().item()

        if self.molecule == 'alanine':
            thp, hit_idxs, mean_len, std_len, mean_etp, std_etp, etps, etp_idxs, mean_efp, std_efp, efps, efp_idxs = self.metric.alanine(positions, target_position, potentials)
        # In case of training logger
        if self.type == "train":
            # Save policy at save_freq and last rollout
            if rollout % self.save_freq == 0:
                torch.save(policy.state_dict(), f'{self.dir}/policy/policy_{rollout}.pt')
                torch.save(policy.state_dict(), f'{self.dir}/policy.pt')
            if rollout == self.num_rollouts - 1 :
                torch.save(policy.state_dict(), f'{self.dir}/policy.pt')

        if rollout % self.save_freq == 0:
            self.logger.info(f"Plotting for {self.num_samples} samples...")
            if self.molecule == 'alanine':
                fig_paths_alanine = plot_paths_alanine(self.dir, positions, target_position, last_idx)
                if thp > 0:
                    fig_etps = plot_etps(self.dir+"/etp", rollout, etps, etp_idxs)
                    fig_efps = plot_efps(self.dir+"/efp", rollout, efps, efp_idxs)
            fig_potentials = plot_potentials(self.dir+"/potential", rollout, potentials, log_reward, last_idx)
            self.logger.info(f"Plotting Done.!!")
 
        # Log to wandb
        if self.wandb:
            log = {
                    'loss': loss,
                    'log_z': policy.log_z.item(),
                    'effective_sample_size_ratio': ess_ratio,
                    'expected_pairwise_path_distance': mean_ppd,
                    'mean_bias_norm': mean_bias_norm,
                    'negative_log_likelihood': mean_nll,
                    'expected_log_reward': mean_reward,
                    'expected_log_md_reward': mean_md_reward,
                    'expected_log_target_reward': mean_target_reward,
                    'expected_pairwise_distance (pm)': mean_pd,
                    'expected_pairwise_scaled_distance': mean_psd,
                    'expected_pairwise_coulomb_distance': mean_pcd,
                }
            wandb.log(log, step=rollout)

            if self.molecule == 'alanine':
                log = {
                        'target_hit_percentage (%)': thp,
                        'energy_transition_point (kJ/mol)': mean_etp,
                        'energy_final_point (kJ/mol)': mean_efp,
                        'mean_length': mean_len,
                        'std_length': std_len,
                    }
                wandb.log(log, step=rollout)

            if self.type == 'eval':
                log = {
                        'std_nll': std_nll,
                        'std_pd': std_pd,
                        'std_psd': std_psd,
                        'std_pcd': std_pcd,
                        'std_ppd': std_ppd,
                    }  
                wandb.log(log, step=rollout)

                if self.molecule == 'alanine':
                    log = {
                            'std_etp': std_etp,
                            'std_efp': std_efp,
                            'std_length': std_len,
                        }
                    wandb.log(log, step=rollout)

            if rollout % self.save_freq==0:
                if self.molecule == 'alanine':
                    wandb.log(
                        {
                            'paths': wandb.Image(fig_paths_alanine)
                        }, 
                        step=rollout
                    )
                
                    if thp > 0:
                        wandb.log(
                            {
                                'etps': wandb.Image(fig_etps),
                                'efps': wandb.Image(fig_efps)
                            }, 
                            step=rollout
                        )

                wandb.log(
                    {
                        'potentials': wandb.Image(fig_potentials),
                    }, 
                    step=rollout
                )

        # Log to system log
        if self.logger:
            if self.type == "train":
                self.logger.info(f'Rollout: {rollout}')
                self.logger.info(f"loss: {loss}")
            self.logger.info("")
            self.logger.info(f"log_z: {policy.log_z.item()}")
            self.logger.info(f"effective_sample_size_ratio: {ess_ratio}")
            self.logger.info(f"expected_pairwise_path_distance: {mean_ppd}")
            self.logger.info(f"negative_log_likelihood: {mean_nll}")
            self.logger.info(f"expected_log_reward: {mean_reward}")
            self.logger.info(f"expected_log_md_reward: {mean_md_reward}")
            self.logger.info(f"expected_log_target_reward: {mean_target_reward}")
            self.logger.info(f"expected_pairwise_distance (pm): {mean_pd}")
            self.logger.info(f"expected_pairwise_scaled_distance: {mean_psd}")
            self.logger.info(f"expected_pairwise_coulomb_distance: {mean_pcd}")
            self.logger.info(f"std_nll: {std_nll}")
            self.logger.info(f"std_pd: {std_pd}")
            self.logger.info(f"std_psd: {std_psd}")
            self.logger.info(f"std_pcd: {std_pcd}")
            self.logger.info(f"std_ppd: {std_ppd}")
            if self.molecule == 'alanine':
                self.logger.info(f"target_hit_percentage (%): {thp}")
                self.logger.info(f"energy_transition_point (kJ/mol): {mean_etp}")
                self.logger.info(f"energy_final_point (kJ/mol): {mean_efp}")
                self.logger.info(f"mean_length: {mean_len}")
                self.logger.info(f"std_length: {std_len}")
                self.logger.info(f"std_etp: {std_etp}")
                self.logger.info(f"std_etp: {std_efp}")

        if plot:            
            if self.molecule == 'alanine':
                plot_paths_alanine(self.dir, positions, target_position, hit_idxs)
                plot_potentials(self.dir, rollout, potentials, log_reward, hit_idxs)
                
                self.logger.info(f"[Plot] Plotting paths")
                plot_path(self.dir, positions, target_position, hit_idxs)

                self.logger.info(f"[Plot] Plotting potentials")
                plot_potential(self.dir, potentials, log_reward, hit_idxs)
                
                self.logger.info(f"[Plot] Plotting 3D view")
                plot_3D_view(self.dir, self.start_file, positions, hit_idxs)
            else:
                self.logger.info(f"[Plot] Plotting potentials")
                plot_potential(self.dir, potentials, log_reward, last_idx)
                
                self.logger.info(f"[Plot] Plotting 3D view")
                plot_3D_view(self.dir, self.start_file, positions, last_idx)