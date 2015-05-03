#!/usr/bin/env python


from opentidalfarm import *

# Create a rectangular domain.
domain = FileDomain("mesh/mesh.xml")

# Specify boundary conditions.
bcs = BoundaryConditionSet()
bcs.add_bc("u", Constant((2, 0)), facet_id=1)
bcs.add_bc("eta", Constant(0), facet_id=2)
# The free-slip boundary conditions.
bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="weak_dirichlet")

# Set the shallow water parameters
prob_params = SteadySWProblem.default_parameters()
prob_params.domain = domain
prob_params.bcs = bcs
prob_params.viscosity = Constant(0.6)
prob_params.depth = Constant(50)
prob_params.friction = Constant(0.0025)

# We here use the smeared tubrine approach
turbine = SmearedTurbine(friction=0.25)

# A farm is defined using the domain and the site area identification.
farm = Farm(domain, turbine=turbine, site_ids=1)

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
control = TurbineFarmControl(farm)
rf_params = ReducedFunctional.default_parameters()
rf_params.automatic_scaling = False
rf = ReducedFunctional(functional, control, solver, rf_params)

# As always, we can print all options of the :class:`ReducedFunctional` with:

#print rf_params

# Now we can define the constraints for the controls and start the
# optimisation.

# Set max value for friction field
max_ct = turbine.friction

f_opt = maximize(rf, bounds=[0., max_ct], method="L-BFGS-B", options={'maxiter': 100, 'ftol':10e-9})




