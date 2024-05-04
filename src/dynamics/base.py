import numpy as np
import openmm as mm
import openmm.unit as unit
from abc import abstractmethod, ABC

covalent_radii = {
    'H': 31 * unit.picometer / unit.nanometer,
    'C': 76 * unit.picometer / unit.nanometer,
    'N': 71 * unit.picometer / unit.nanometer,
    'O': 66 * unit.picometer / unit.nanometer,
}

nuclear_charge = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
}

class BaseDynamics(ABC):
    def __init__(self, args, state):
        super().__init__()
        self.temperature = args.temperature
        self.start_file = f'./data/{args.molecule}/{state}.pdb'

        self.pdb, self.simulation, self.external_force = self.setup()

        self.simulation.minimizeEnergy()
        self.position = self.report()[0]
        self.reset()
        
        self.num_particles = self.simulation.system.getNumParticles()

        self.covalent_radii_matrix = self.get_covalent_radii_matrix()
        self.charge_matrix = self.get_charge_matrix()
        
    @abstractmethod
    def setup(self):
        pass
    
    def get_covalent_radii_matrix(self):
        covalent_radii_matrix = np.zeros((self.num_particles, self.num_particles))
        topology = self.pdb.getTopology()
        for i, atom_i in enumerate(topology.atoms()):
            for j, atom_j in enumerate(topology.atoms()):
                covalent_radii_matrix[i, j] = covalent_radii[atom_i.element.symbol] + covalent_radii[atom_j.element.symbol]
        return covalent_radii_matrix

    def get_charge_matrix(self):
        charge_matrix = np.zeros((self.num_particles, self.num_particles))
        topology = self.pdb.getTopology()
        for i, atom_i in enumerate(topology.atoms()):
            for j, atom_j in enumerate(topology.atoms()):
                if i == j:
                    charge_matrix[i, j] = 0.5 * nuclear_charge[atom_i.element.symbol]**(2.4)
                else:
                    charge_matrix[i, j] = nuclear_charge[atom_i.element.symbol] * nuclear_charge[atom_j.element.symbol]
        return charge_matrix

    def step(self, forces):
        for i in range(forces.shape[0]):
            self.external_force.setParticleParameters(i, i, forces[i])
        self.external_force.updateParametersInContext(self.simulation.context)
        self.simulation.step(1)

    def report(self):
        state = self.simulation.context.getState(getPositions=True, getEnergy=True)
        positions = state.getPositions().value_in_unit(unit.nanometer)
        potentials = state.getPotentialEnergy().value_in_unit(unit.kilojoules/unit.mole)
        return positions, potentials

    def reset(self):
        for i in range(len(self.position)):
            self.external_force.setParticleParameters(i, i, [0, 0, 0])
        self.external_force.updateParametersInContext(self.simulation.context)
        self.simulation.context.setPositions(self.position)
        self.simulation.context.setVelocitiesToTemperature(0)