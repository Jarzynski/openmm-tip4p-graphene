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

# Add a CustomNonbondedForce
# system.addForce(CustomNonbondedForce('4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2); sigma1=sigma[atom1]; sigma2=sigma[atom2]; epsilon1=epsilon[atom1]; epsilon2=epsilon[atom2]'))
energy_function = '-amplitude_all*r*r*amplitude*(0.3*exp0^(-(r-mu1)^2/(2*sigma1^2))+exp0^(-(r-mu2)^2/(2*sigma2^2))+0.65*exp0^(-(r-mu3)^2/(2*sigma3^2))-0.6*exp0^(-(r-mu4)^2/(2*sigma4^2)))/u_max/u_max;'
energy_function += 'amplitude = floor(amplitude_i1 + amplitude_i2);'

o_particles1 = set(range(4536,topology.getNumAtoms(),4))
o_particles2 = set(range(4536,topology.getNumAtoms(),4))
custom_force = openmm.CustomNonbondedForce(energy_function)
custom_force.addGlobalParameter('exp0', 2.718281828459045)
custom_force.addGlobalParameter('amplitude_all',-0.4*kilocalories_per_mole)
custom_force.addPerParticleParameter('amplitude_i')
custom_force.addGlobalParameter('sigma1',0.10*angstrom)
custom_force.addGlobalParameter('sigma2',0.20*angstrom)
custom_force.addGlobalParameter('sigma3',0.30*angstrom)
custom_force.addGlobalParameter('sigma4',0.20*angstrom)

custom_force.addGlobalParameter('mu1',3.1*angstrom)
custom_force.addGlobalParameter('mu2',3.9*angstrom)
custom_force.addGlobalParameter('mu3',3.35*angstrom)
custom_force.addGlobalParameter('mu4',4.5*angstrom)
custom_force.addGlobalParameter('u_max',4.1196*angstrom)
custom_force.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
custom_force.setCutoffDistance(1*nanometer) 

for  i in range(topology.getNumAtoms()):
    if i in o_particles1:
        custom_force.addParticle([0.5])
    else:
        custom_force.addParticle([0])

for i in range(0,4536,1):
    system.setParticleMass(i, 0*amu)

custom_force.addInteractionGroup(o_particles1, o_particles2)
custom_force.setForceGroup(1)
system.addForce(custom_force)


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