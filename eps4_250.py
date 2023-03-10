#模板参数λ=1，扰动参数=0，模拟温度从300K到270K的退火过程

from openmm import *
from openmm.app import *
from openmm.unit import *
from sys import stdout
import sys
import numpy as np


# Input Files

pdb = PDBFile('wakong.pdb')
forcefield = ForceField('tip4pcarbon.xml','test.xml')

# System Configuration

nonbondedMethod = PME
nonbondedCutoff = 0.9*nanometers
ewaldErrorTolerance = 0.0005
constraints = HBonds
rigidWater = True
constraintTolerance = 0.000001
hydrogenMass = 1.5*amu

# Integration Options

dt = 0.003*picoseconds
temperature = 250*kelvin#300,270两个初始温度
friction = 1.0/picosecond

# Simulation Options

steps = 100 * (60/0.0002)
equilibrationSteps = 1000  #300K初始温度平衡3ns
platform = Platform.getPlatformByName('CUDA')
platformProperties = {'Precision': 'single'}
platformProperties["DeviceIndex"] = "3"
# platform = Platform.getPlatformByName('OpenCL')
# platformProperties = {'Precision': 'double'}
dcdReporter = DCDReporter('eps4_250.dcd', 50000)
dataReporter = StateDataReporter('eps4_250.txt', 2000, totalSteps=steps,
    step=True, speed=True, progress=True,remainingTime=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, separator='\t')
checkpointReporter = CheckpointReporter('eps4_250.chk', 10000)

# Prepare the Simulation

print('Building system...')
topology = pdb.topology
positions = pdb.positions
system = forcefield.createSystem(topology, nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
    constraints=constraints, rigidWater=rigidWater,ewaldErrorTolerance=ewaldErrorTolerance, hydrogenMass=hydrogenMass)
# removing all center of mass motion by CMMotionRemover
system.addForce(CMMotionRemover(500))


integrator = LangevinMiddleIntegrator(temperature, friction, dt)
simulation = Simulation(topology, system, integrator, platform, platformProperties)
# simulation = Simulation(topology, system, integrator, platform)
simulation.context.setPositions(positions)

# Minimize and Equilibrate

print('Performing energy minimization...')
simulation.minimizeEnergy()
# print('Equilibrating...')
simulation.context.setVelocitiesToTemperature(temperature)
#simulation.step(equilibrationSteps)

# Simulate
# steps = 1000 * (120/0.002)
print('Simulating...')
simulation.reporters.append(dcdReporter)
simulation.reporters.append(dataReporter)
simulation.reporters.append(checkpointReporter)
simulation.step(equilibrationSteps)
simulation.currentStep = 0
simulation.step(steps) 
