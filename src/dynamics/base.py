import openmm.unit as unit
from abc import abstractmethod, ABC


class BaseDynamics(ABC):
    def __init__(self, args, state):
        super().__init__()
        self.device = args.device
        self.temperature = args.temperature
        self.start_file = f'./data/{args.molecule}/{state}.pdb'

        self.simulation, self.external_force = self.setup()

        self.simulation.minimizeEnergy()
        self.num_particles = self.simulation.system.getNumParticles()
        self.position = self.simulation.context.getState(getPositions=True).getPositions().value_in_unit(unit.nanometer)
        self.reset()

        
    @abstractmethod
    def setup(self):
        pass

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
        self.simulation.context.setPositions(self.position)
        self.simulation.context.setVelocitiesToTemperature(0)