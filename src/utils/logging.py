import wandb

from .plot import *
from .metrics import *

class Logger():
    def __init__(self, args, md_info):
        self.wandb = args.wandb
        self.molecule = args.molecule
        self.save_file = args.save_file
        self.start_file = md_info.start_file

    def log(self, loss, policy, start_state, end_state, rollout, positions, start_position, last_position, target_position, potentials):
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

        torch.save(policy.state_dict(), f'{self.save_file}/policy.pt')
    
    def plot(self, positions, target_position, potentials, **kwargs):
        plot_potentials(self.save_file, potentials)
        plot_3D_trajectories(self.save_file, self.start_file, positions)
        if self.molecule == 'alanine':
            plot_paths(self.save_file, positions, target_position)