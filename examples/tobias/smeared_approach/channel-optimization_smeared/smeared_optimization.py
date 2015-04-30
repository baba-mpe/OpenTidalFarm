#import argparse
from opentidalfarm import *
from model_turbine import ModelTurbine
#from vorticity_solver import VorticitySolver
#import time

model_turbine = ModelTurbine()
print model_turbine

# Read the command line arguments
#parser = argparse.ArgumentParser()
#parser.add_argument('--turbines', required=True, type=int, help='number of turbines')
#parser.add_argument('--optimize', action='store_true', help='Optimise instead of just simulate')
#parser.add_argument('--withcuts', action='store_true', help='with cut in/out speeds')
#parser.add_argument('--cost', type=float, default=0., help='the cost coefficient')
#args = parser.parse_args()

model_turbine = ModelTurbine()
print model_turbine

# Create a rectangular domain.
domain = FileDomain("mesh/mesh.xml")

# Specify boundary conditions.
bcs = BoundaryConditionSet()
#bcs.add_bc("u", Constant((2, 0)), facet_id=1)
bcs.add_bc("eta", Constant(0.1), facet_id=1)
bcs.add_bc("eta", Constant(0), facet_id=2)
# The free-slip boundary conditions.
bcs.add_bc("u", Constant((0, 0)), facet_id=3, bctype="strong_dirichlet")

# Set the shallow water parameters
prob_params = SteadySWProblem.default_parameters()
prob_params.domain = domain
prob_params.bcs = bcs
prob_params.viscosity = Constant(60.)
prob_params.depth = Constant(50)
prob_params.friction = Constant(0.0025)

# We here use the smeared tubrine approach
turbine = SmearedTurbine()
V = FunctionSpace(domain.mesh, "CG", 1)
farm = Farm(domain, turbine, function_space=V)

# Sub domain for inflow (right)
class FarmDomain(SubDomain):
    def inside(self, x, on_boundary):
        return (1500 <= x[0] <= 2500 and
                1500  <= x[1] <= 2500)

farm_domain = FarmDomain()
domains = MeshFunction("size_t", domain.mesh, domain.mesh.topology().dim())
domains.set_all(0)
farm_domain.mark(domains, 1)
site_dx = Measure("dx")[domains]
farm.site_dx = site_dx(1)
plot(domains, interactive=True)

prob_params.tidal_farm = farm

# Now we can create the shallow water problem

problem = SteadySWProblem(prob_params)

# Next we create a shallow water solver. Here we choose to solve the shallow
# water equations in its fully coupled form. We also set the dump period to 1 in
# order to save the results of each optimisation iteration to disk.

sol_params = CoupledSWSolver.default_parameters()
sol_params.dump_period = 1
solver = CoupledSWSolver(problem, sol_params)

#V = solver.function_space.extract_sub_space([0]).collapse()
#Q = solver.function_space.extract_sub_space([1]).collapse()

#base_u = Function(V, name="base_u")
#base_u_tmp = Function(V, name="base_u_tmp")

# Next we create a reduced functional, that is the functional considered as a
# pure function of the control by implicitly solving the shallow water equations. For
# that we need to specify the objective functional (the value that we want to
# optimize), the control (the variables that we want to change), and our shallow
# water solver.

functional = PowerFunctional(problem)
control = TurbineFarmControl(farm)
rf_params = ReducedFunctional.default_parameters()
rf_params.automatic_scaling = None
rf = ReducedFunctional(functional, control, solver, rf_params)

# As always, we can print all options of the :class:`ReducedFunctional` with:

print rf_params

# Now we can define the constraints for the controls and start the
# optimisation.

init_tf = 0#model_turbine.maximum_smeared_friction/1000.*args.turbines
farm.friction_function.assign(Constant(init_tf))

# Comment this for only forward modelling
#if args.optimize:
maximize(rf, bounds=[0,model_turbine.maximum_smeared_friction],
            method="L-BFGS-B", options={'maxiter': 100})
#maximize(rf, bounds=[0,100.],
#            method="L-BFGS-B", options={'maxiter': 100})


