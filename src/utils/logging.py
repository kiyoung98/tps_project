import os
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
        
        self.type = args.type
        kst = pytz.timezone('Asia/Seoul')
        self.date = datetime.datetime.now(tz=kst).strftime("%Y%m%d-%H%M%S")
        
        # Set up system logging    
        if args.logger:
            folder_path = f'results/{self.molecule}/{self.date}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            log_file_name = folder_path+f"/{self.type}.log"
            
            self.logger = logging.getLogger("tps")
            self.logger.setLevel(logging.INFO)
            
            handler = logging.FileHandler(log_file_name)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


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
            self.logger.info(f'\nRollout: {rollout}')
            self.logger.info(f"{start_state}_to_{end_state}/expected_pairwise_distance (pm): {expected_pairwise_distance(last_position, target_position)}")
            self.logger.info(f"{start_state}_to_{end_state}/log_z: {policy.get_log_z(start_position, target_position).item()}")
            self.logger.info(f"Loss: {loss}")
            
            if rollout % 10 == 0 and self.molecule == 'alanine':
                self.logger.info(f"{start_state}_to_{end_state}/target_hit_percentage (%): {target_hit_percentage(last_position, target_position)}")
                self.logger.info(f"{start_state}_to_{end_state}/energy_transition_point (kJ/mol): {energy_transition_point(last_position, target_position, potentials)}")
        
        torch.save(policy.state_dict(), f'results/{self.molecule}/{self.date}/policy.pt')
    
    def plot(self, positions, target_position, potentials, date, **kwargs):
        self.logger.info(f"[Plot] potentials")
        plot_potentials(self.molecule, potentials, date)
        
        self.logger.info(f"[Plot] trajectories")
        plot_3D_trajectories(self.molecule, self.start_file, positions, date)
        
        if self.molecule == 'alanine':
            self.logger.info(f"[Plot] paths")
            plot_paths(self.molecule, positions, target_position, date)