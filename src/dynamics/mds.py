import torch
from tqdm import tqdm
from dynamics import dynamics

class MDs:
    def __init__(self, args, state):
        self.args = args
        self.state = state
        self.device = args.device
        self.molecule = args.molecule
        self.num_samples = args.num_samples

        self.mds = self._init_mds()

    def _init_mds(self):
        print(f"Initialize dynamics starting at {self.state}")

        mds = []
        for _ in tqdm(range(self.num_samples)):
            md = getattr(dynamics, self.molecule.title())(self.args, self.state)
            mds.append(md)
        return mds

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