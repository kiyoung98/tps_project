import os
import sys
import pytz
import wandb
import logging
import datetime


from .plot import *
from .metrics import *

class Logger():
    def __init__(self, args, md_info):
        self.wandb = args.wandb
        self.molecule = args.molecule
        self.start_file = md_info.start_file
        
        self.seed = args.seed
        self.type = args.type
        kst = pytz.timezone('Asia/Seoul')
        self.date = datetime.datetime.now(tz=kst).strftime("%Y%m%d-%H%M%S")
        
        self.dir = f'results/{self.molecule}/{self.type}/{self.seed}'
        
        # Set up system logging    
        if args.logger:
            folder_path = self.dir
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            log_file_name = folder_path + f"/{self.type}.log"
            # logging.basicConfig(file_mode="w")
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
            
            log_date = datetime.datetime.now(tz=kst).strftime("%Y-%m-%d %H:%M:%S")
            self.logger.info(f"Date: {log_date}")
            self.logger.info(f"Logging {self.type} {self.molecule}, seed {self.seed}")
            self.logger.info("")


    def info(self, message):
        if self.logger:
            self.logger.info(message)
    
    def log(self, loss, policy, start_state, end_state, rollout, positions, start_position, last_position, target_position, potentials, date=None):
        if self.wandb:
            wandb.log(
                {
                    f'{start_state}_to_{end_state}/expected_pairwise_distance (pm)': expected_pairwise_distance(last_position, target_position),
                    f'{start_state}_to_{end_state}/log_z': policy.get_log_z(start_position, target_position).item(), 
                    'loss': loss,
                },
                step=rollout
            )
            
            if rollout%10==0 and self.molecule == 'alanine':
                wandb.log(
                    {
                        f'{start_state}_to_{end_state}/target_hit_percentage (%)': target_hit_percentage(last_position, target_position),
                        f'{start_state}_to_{end_state}/energy_transition_point (kJ/mol)': energy_transition_point(last_position, target_position, potentials),
                        f'{start_state}_to_{end_state}/paths': wandb.Image(plot_paths_alanine(positions, target_position)),
                    }, 
                    step=rollout
                )

        if self.logger:
            self.logger.info("")
            self.logger.info(f'Rollout: {rollout}')
            self.logger.info(f"{start_state}_to_{end_state}/expected_pairwise_distance (pm): {expected_pairwise_distance(last_position, target_position)}")
            self.logger.info(f"{start_state}_to_{end_state}/log_z: {policy.get_log_z(start_position, target_position).item()}")
            self.logger.info(f"Loss: {loss}")
            
            if rollout % 10 == 0 and self.molecule == 'alanine':
                self.logger.info(f"{start_state}_to_{end_state}/target_hit_percentage (%): {target_hit_percentage(last_position, target_position)}")
                self.logger.info(f"{start_state}_to_{end_state}/energy_transition_point (kJ/mol): {energy_transition_point(last_position, target_position, potentials)}")
        
        torch.save(policy.state_dict(), f'{self.dir}/policy.pt')
    
    def plot(self, positions, target_position, potentials, seed, **kwargs):
        self.logger.info(f"[Plot] Plotting potentials")
        plot_potentials(self.molecule, potentials, self.dir)
        
        self.logger.info(f"[Plot] Plotting 3D view")
        plot_3D_view(self.molecule, self.start_file, positions, self.dir)
        
        if self.molecule == 'alanine':
            self.logger.info(f"[Plot] Plotting paths")
            plot_paths(self.molecule, positions, target_position, self.dir)