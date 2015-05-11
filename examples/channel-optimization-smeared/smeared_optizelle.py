#!/usr/bin/env python

# First try to use optizelle on top of opentidalfarm using the continuous
# drag approach for turbine representation.

#import argparse
from dolfin_adjoint import *
from opentidalfarm import *
from dolfin_adjoint import Control
import Optizelle
from model_turbine import ModelTurbine

# some user arguments
#parser = argparse.ArgumentParser()
#parser.add_argument('--turbines', required=True, type=int, help='number of turbines')
#args = parser.parse_args()

# set up model turbine
model_turbine = ModelTurbine()

# Create domain.
domain = FileDomain("mesh/mesh.xml")

# Specify boundary conditions.
bcs = BoundaryConditionSet()
#bcs.add_bc("u", Constant((2, 0)), facet_id=1)
bcs.add_bc("eta", Constant(0.1), facet_id=1)
bcs.add_bc("eta", Constant(0), facet_id=2)
# The free-slip boundary conditions.
bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="strong_dirichlet") #"weak_dirichlet")

# Set the shallow water parameters
prob_params = SteadySWProblem.default_parameters()
prob_params.domain = domain
prob_params.bcs = bcs
prob_params.viscosity = Constant(50.)
prob_params.depth = Constant(50)
prob_params.friction = Constant(0.0025)

# We here use the smeared turbine approach
turbine = SmearedTurbine()
V = FunctionSpace(domain.mesh, "CG", 1)

# set up farm as shallow water parameter
farm = Farm(domain, turbine, function_space=V, site_ids=1)

init_tf = model_turbine.maximum_smeared_friction/1000. #*args.turbines
farm.friction_function.assign(Constant(init_tf))

prob_params.tidal_farm = farm

# create the shallow water problem
problem = SteadySWProblem(prob_params)

# Next we create a shallow water solver. Here we choose to solve the shallow
# water equations in its fully coupled form. We also set the dump period to 1 in
# order to save the results of each optimisation iteration to disk.

sol_params = CoupledSWSolver.default_parameters()
sol_params.dump_period = 1
solver = CoupledSWSolver(problem, sol_params)

# Next we create a reduced functional, that is the functional considered as a
# pure function of the control by implicitly solving the shallow water PDE. For
# that we need to specify the objective functional (the value that we want to
# optimize), the control (the variables that we want to change), and our shallow
# water solver

functional = PowerFunctional(problem)
control = Control(farm.friction_function) 
rf = FenicsReducedFunctional(functional, control, solver)

# define optimization problem (link to optizelle)
opt_problem = MaximizationProblem(rf, bounds=(0., model_turbine.maximum_smeared_friction))

parameters = {
             "maximum_iterations": 100,
             "optizelle_parameters":
                 {
                 "msg_level" : 10,
                 "algorithm_class" : Optizelle.AlgorithmClass.TrustRegion,
                 "H_type" : Optizelle.Operators.BFGS,
                 "dir" : Optizelle.LineSearchDirection.BFGS,
                 "ipm": Optizelle.InteriorPointMethod.LogBarrier,
                 "sigma": 0.5,
                 "gamma": 0.95,
                 "linesearch_iter_max" : 20,
                 "krylov_iter_max" : 100,
                 "eps_krylov" : 1e-9
                 }
             }

# Alternative parameter set
#parameters = {
#             "maximum_iterations": 100,
#             "optizelle_parameters":
#                 {
#                 "msg_level" : 10,
#                 "algorithm_class" : Optizelle.AlgorithmClass.TrustRegion,
#                 "H_type" : Optizelle.Operators.UserDefined,
#                 "dir" : Optizelle.LineSearchDirection.NewtonCG,
#                 "ipm": Optizelle.InteriorPointMethod.LogBarrier,
#                 "sigma": 0.5,
#                 "gamma": 0.995,
#                 "linesearch_iter_max" : 50,
#                 "krylov_iter_max" : 100,
#                 "eps_krylov" : 1e-9
#                 }
#             }

# set up optimization problem solver
opt_solver = OptizelleSolver(opt_problem, parameters=parameters)

#import ipdb; ipdb.set_trace()

# solve problem and plot result
f_opt = opt_solver.solve()
plot(f_opt, interactive=True)
print "optimal ", assemble(f_opt*dx)
