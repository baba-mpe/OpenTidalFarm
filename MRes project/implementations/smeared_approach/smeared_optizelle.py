#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. _channel_optimization:
#
# .. py:currentmodule:: opentidalfarm
#
# Farm layout optimization
# ========================
#
# This demo optimizes the position of 32 turbines in a tidal farm within a channel.
# The goal of the optimization is to maximise the farm's energy extraction. The
# rectangular channel is 640 m x 320 m large. The farm area is in the channel
# center and 320 m x 160 m large.

# Even though the domain in this demo is quite simple, the concept applies to
# more complex, realistic scenarios.
#
#
# The farm layout optimisation is initialized with a regular layout of 32 turbines:
#
# .. image:: 32turbines_regular.png
#     :scale: 30
#
#
# With this configuration, the flow speed with streamlines is:
#
# .. image:: 32turbines_regular_vel.png
#     :scale: 30
#
# The power extraction by the farm (without taking losses into account) is 46
# MW. This is 1.4 MW per turbine (32 turbines) which is unsatisfactory
# considering that placing a single turbine extracts 2.9 MW.
#
# We can also try a staggered layout:
#
# .. image:: 32turbines_staggered.png
#     :scale: 30
#
# With this configuration, the flow speed with streamlines is:
#
# .. image:: 32turbines_staggered_vel.png
#     :scale: 30
#
# The power extraction by the farm (without taking losses into account) is 64 MW
# or 2.0 MW per turbine. Thats better but still non-optimal.
#
# Applying the layout optimisation in OpenTidalFarm finishes after 92 iterations. The optimised farm layout is:
#
# .. image:: 32turbines_opt.png
#     :scale: 30
#
# The optimization has arranged the turbines to "barrages" perpendicular to the
# flow. Furthermore, it added small north and east barrages of turbines that
# force the water to flow through the prependicular "barrages". The associated
# flow speed with streamlines is:
#
# .. image:: 32turbines_opt_vel.png
#     :scale: 30
#
#
# The power production of the optimised layout is 80 MW, or 2.5 MW per turbine.
# That is the optimisation increased the power production by 74 % compared to
# the initial layout!

# Implementation
# **************
#

# The first part of the program sets up a steady state shallow water problem,
# and is nearly identical to the :ref:`channel_simulation` example:

from dolfin_adjoint import *
from opentidalfarm import *
import Optizelle

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
prob_params.viscosity = Constant(2)
prob_params.depth = Constant(50)
prob_params.friction = Constant(0.0025)

# We here use the smeared turbine approach
turbine = SmearedTurbine(friction=0.25)

# A farm is defined using the domain and the site area identification.
farm = FunctionControlFarm(domain, turbine=turbine, site_ids=1)

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
#control = Control(prob_params.friction)

#rf_params = ReducedFunctional.default_parameters()
#rf_params.automatic_scaling = False
#rf = ReducedFunctional(functional, control, solver, rf_params)

rf = FenicsReducedFunctional(functional, control, solver)

# As always, we can print all options of the :class:`ReducedFunctional` with:

# print rf_params

max_ct = turbine.friction

opt_problem = MaximizationProblem(rf, bounds=(0., max_ct))
parameters = {
             "maximum_iterations": 50,
             "optizelle_parameters":
                 {
                 "msg_level" : 10,
                 "algorithm_class" : Optizelle.AlgorithmClass.TrustRegion,
                 "H_type" : Optizelle.Operators.UserDefined,
                 "dir" : Optizelle.LineSearchDirection.NewtonCG,
                 "ipm": Optizelle.InteriorPointMethod.LogBarrier,
                 "sigma": 0.001,
                 "gamma": 0.995,
                 "linesearch_iter_max" : 50,
                 "krylov_iter_max" : 100,
                 "eps_krylov" : 1e-4
                 }
             }

opt_solver = OptizelleSolver(opt_problem, parameters=parameters)
f_opt = opt_solver.solve()
plot(f_opt, interactive=True)
print "optimal ", assemble(f_opt*dx)



"""# As always, we can print all options of the :class:`ReducedFunctional` with:

print rf_params

# Now we can define the constraints for the controls and start the
# optimisation.

# Set max value for friction field
max_ct = turbine.friction

f_opt = maximize(rf, bounds=[0., max_ct], method="L-BFGS-B", options={'maxiter': 100})

# The example code can be found in ``examples/channel-optimization/`` in the
# ``OpenTidalFarm`` source tree, and executed as follows:

# .. code-block:: bash

#   $ python channel-optimization.py

# You can speed up the calculation by using multiple cores (in this case 4) with:

# .. code-block:: bash

#   $ mpirun -n 4 python channel-optimization.py

# During the optimization run, OpenTidalFarm creates multiple files for
# inspection:
#
# *  turbines.pvd: Stores the position and friction values of the turbines at
#    each optimisation iteration.
# *  iter_*: For each optimisation iteration X, the associated
#    velocity and pressure solutions are stored in a directory named iter_X.
# *  iterate.dat: A testfile that dumps the optimisation progress, e.g. number of
#    iterations, function value, gradient norm, etc
#
# The pvd files can be opened with the open-source software
# `Paraview <http://www.paraview.org>`_."""
