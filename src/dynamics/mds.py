from tqdm import tqdm
from dynamics import dynamics

class MDs:
    def __init__(self, args, state):
        self.args = args
        self.state = state
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

    def step(self, bias):
        bias = bias.detach().cpu().numpy()
        positions, velocities, forces, potentials = [], [], [], []
        for i in range(self.num_samples):
            position, velocity, force, potential = self.mds[i].step(bias[i])
            positions.append(position); velocities.append(velocity); forces.append(force); potentials.append(potential)
        return positions, velocities, forces, potentials
    
    def set(self):
        positions, velocities, forces, potentials = [], [], [], []
        for i in range(self.num_samples):
            position, velocity, force, potential = self.mds[i].set()
            positions.append(position); velocities.append(velocity); forces.append(force); potentials.append(potential)
        return positions, velocities, forces, potentials

    def set_temperature(self, temperature):
        for i in range(self.num_samples):
            self.mds[i].set_temperature(temperature)