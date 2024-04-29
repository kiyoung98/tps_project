import torch
from tqdm import tqdm
from dynamics import dynamics

class MDs:
    def __init__(self, args):
        self.args = args
        self.start_state = args.start_state
        self.end_state = args.end_state
        self.device = args.device
        self.molecule = args.molecule
        self.num_samples = args.num_samples

        self.mds = self._init_mds()
        self.target_position = self._init_target_position()

    def _init_mds(self):
        print(f"Initialize dynamics starting at {self.start_state}")

        mds = []
        for _ in tqdm(range(self.num_samples)):
            md = getattr(dynamics, self.molecule.title())(self.args, self.start_state)
            mds.append(md)
        return mds

    def _init_target_position(self):
        print(f"Getting position for {self.end_state}")

        target_position = getattr(dynamics, self.molecule.title())(self.args, self.end_state).position
        target_position = torch.tensor(target_position, dtype=torch.float, device=self.device).unsqueeze(0)
        return target_position

    def step(self, force):
        force = force.detach().cpu().numpy()
        for i in range(self.num_samples):
            self.mds[i].step(force[i])

    def report(self):
        positions, potentials = [], []
        for i in range(self.num_samples):
            position, potential = self.mds[i].report()
            positions.append(position)
            potentials.append(potential)
            
        positions = torch.tensor(positions, dtype=torch.float, device=self.device)
        potentials = torch.tensor(potentials, dtype=torch.float, device=self.device)
        return positions, potentials
    
    def reset(self):
        for i in range(self.num_samples):
            self.mds[i].reset()