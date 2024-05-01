import openmm as mm
from openmm import app
import openmm.unit as unit
from openmmtools.integrators import LangevinIntegrator

from .base import BaseDynamics


class Alanine(BaseDynamics):
    def __init__(self, args, state):
        super().__init__(args, state)

    def setup(self):
        forcefield = app.ForceField('amber99sbildn.xml')
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005
        )
        external_force = mm.CustomExternalForce("fx*x + fy*y + fz*z")

        # creating the parameters
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])

        integrator = mm.LangevinIntegrator(
            self.temperature,  
            self.friction_coefficient,  
            self.timestep,
        ) 

        integrator.setConstraintTolerance(0.00001)

        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        return simulation, external_force