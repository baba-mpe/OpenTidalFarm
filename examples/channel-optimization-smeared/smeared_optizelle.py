#!/usr/bin/env python

# First try to use optizelle on top of opentidalfarm using the continuous
# drag approach for turbine representation.

from dolfin_adjoint import *
from opentidalfarm import *
from dolfin_adjoint import Control
#from dolfin_adjoint import ReducedFunctional
import Optizelle
from model_turbine import ModelTurbine

model_turbine = ModelTurbine()

# Create a rectangular domain.
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
farm = Farm(domain, turbine, function_space=V, site_ids=1)

prob_params.tidal_farm = farm

# Now we can create the shallow water problem
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
# water solver.

functional = PowerFunctional(problem)
control = Control(farm.friction_function) 
rf = FenicsReducedFunctional(functional, control, solver) #ReducedFunctional(functional, control) # #


#rf_params = ReducedFunctional.default_parameters()
#rf_params.automatic_scaling = False
#rf = ReducedFunctional(functional, control, solver, rf_params)

# As always, we can print all options of the :class:`ReducedFunctional` with:
# print rf_params

# set a maximal turbine friction making this a contrained problem
max_ct = turbine.friction
init_tf = 1e-06 #model_turbine.maximum_smeared_friction/1000.*args.turbines
farm.friction_function.assign(Constant(init_tf))

# define and run optimization problem using optizelle
opt_problem = MaximizationProblem(rf, bounds=(1e-09, model_turbine.maximum_smeared_friction))
parameters = {
             "maximum_iterations": 50,
             "optizelle_parameters":
                 {
                 "msg_level" : 10,
                 "algorithm_class" : Optizelle.AlgorithmClass.TrustRegion,
                 "H_type" : Optizelle.Operators.BFGS,
                 "dir" : Optizelle.LineSearchDirection.BFGS,
                 "ipm": Optizelle.InteriorPointMethod.LogBarrier,
                 "sigma": 0.5,
                 "gamma": 0.5,
                 "linesearch_iter_max" : 20,
                 "krylov_iter_max" : 100,
                 "eps_krylov" : 1e-1
                 }
             }

opt_solver = OptizelleSolver(opt_problem, parameters=parameters)
import ipdb; ipdb.set_trace()
f_opt = opt_solver.solve()
plot(f_opt, interactive=True)
print "optimal ", assemble(f_opt*dx)

