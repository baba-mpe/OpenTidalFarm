## Taylor test
from dolfin_adjoint import *
from opentidalfarm import *
from dolfin_adjoint import Control
import Optizelle
from dolfin_adjoint import taylor_test
from numpy import sin, cos, pi

# Create a rectangular domain.
domain = FileDomain("mesh/mesh.xml")

# Specify boundary conditions.
bcs = BoundaryConditionSet()

#bcs.add_bc("u", Constant((2, 0)), facet_id=1) #alternative boundary condition
bcs.add_bc("eta", Constant(0.1), facet_id=1)
bcs.add_bc("eta", Constant(0), facet_id=2)
# The free-slip boundary conditionx`s.
bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="strong_dirichlet") # alternatively: "weak_dirichlet")

# Set the shallow water parameters
prob_params = SteadySWProblem.default_parameters()
prob_params.domain = domain
prob_params.bcs = bcs
prob_params.viscosity = Constant(50.)
prob_params.depth = Constant(50)
prob_params.friction = Constant(0.0025)
turbine = SmearedTurbine()

# Set function space
V = FunctionSpace(domain.mesh, "CG", 3)

# Define reduced functional
def Jhat(friction_function):
    farm = Farm(domain, turbine, function_space=V, site_ids=1)
    farm.friction_function.assign(friction_function)
    prob_params.tidal_farm = farm
    problem = SteadySWProblem(prob_params)
    sol_params = CoupledSWSolver.default_parameters()
    sol_params.dump_period = 1
    solver = CoupledSWSolver(problem, sol_params)
    functional = PowerFunctional(problem)
    control = Control(farm.friction_function) 
    rf = FenicsReducedFunctional(functional, control, solver) 
    return rf.evaluate()

# Define derivative of reduced functional
def dJhat(friction_function):
    farm = Farm(domain, turbine, function_space=V, site_ids=1)
    farm.friction_function.assign(friction_function)
    prob_params.tidal_farm = farm
    problem = SteadySWProblem(prob_params)
    sol_params = CoupledSWSolver.default_parameters()
    sol_params.dump_period = 1
    solver = CoupledSWSolver(problem, sol_params)
    functional = PowerFunctional(problem)
    control = Control(farm.friction_function) 
    rf = FenicsReducedFunctional(functional, control, solver) 
    return rf.derivative()

# Friction function 1
#friction_function = Constant(0.03)
# Friction function 2
#friction_function = interpolate(Expression("1e-08*(1 + x[0]*x[0] + 2*x[1]*x[1])"),V)
# Friction function 3
friction_function = interpolate(Expression("1e-02*(2+ sin(2*pi*x[0]) - cos(2*pi*x[1]))"), V)


# Set control to turbine density
farm = Farm(domain, turbine, function_space=V, site_ids=1)

#import ipdb; ipdb.set_trace() 


farm.friction_function.assign(friction_function)
control = Control(farm.friction_function) 

# Compute reduced funtional and its derivative
J_dens = Jhat(friction_function) 
dJ_dens = dJhat(friction_function)

# Compute convergence rates using imported method from dolfin_adjoint
conv_rate = taylor_test(Jhat, control, J_dens, dJ_dens) 
