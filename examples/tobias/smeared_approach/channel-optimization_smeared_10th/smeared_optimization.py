from opentidalfarm import *
from model_turbine import ModelTurbine 
import os.path
import time

# start time for running time computation
tic = time.clock()

set_log_level(INFO)

output_path = "results"

# Create the model turbine
model_turbine = ModelTurbine()
print model_turbine

domain = FileDomain("mesh_smeared/mesh.xml")
domains = MeshFunction("size_t", domain.mesh, "mesh_smeared/mesh_physical_region.xml")
# the mesh has 2 outside and 1 inside the domain, change it to 0 outside and 1 inside
domains.array()[:] = domains.array() % 2

# Set up the domain and site_dx which integrates over the farm only
prob_params = SteadySWProblem.default_parameters()
prob_params.domain = domain
V = FunctionSpace(domain.mesh, "CG", 1)
turbine = SmearedTurbine()
farm = Farm(domain, turbine, function_space=V)
site_dx = Measure("dx")[domains]
farm.site_dx = site_dx(1)
f = File(os.path.join(output_path, "turbine_farms.pvd"))
f << domains

farm.friction_function.assign(Constant(0.0))

prob_params.tidal_farm = farm

# boundary conditions
bcs = BoundaryConditionSet()
bcs.add_bc("u", Constant((2.0,0.0)), facet_id=1, bctype="strong_dirichlet")
bcs.add_bc("eta", Constant(0.0), facet_id=2, bctype="strong_dirichlet")
bcs.add_bc("u", Constant((0.0,0.0)), facet_id=3, bctype="weak_dirichlet")
prob_params.bcs = bcs

# Equation settings
prob_params.viscosity = Constant(0.5)
prob_params.depth = Constant(50.0)
prob_params.friction = Constant(0.0025)
prob_params.initial_condition = Expression(("1e-7", "0", "0"))

print prob_params
problem = SteadySWProblem(prob_params)

# set up the solver
sol_params = CoupledSWSolver.default_parameters()
sol_params.dump_period = 1
sol_params.output_dir = output_path
sol_params.cache_forward_state = False
solver = CoupledSWSolver(problem, sol_params)

# set up functional
power_functional = PowerFunctional(problem)
cost_functional = model_turbine.cost_coefficient * CostFunctional(problem)
functional = power_functional - cost_functional


# set up reduced functional
rf_params = ReducedFunctional.default_parameters()
rf_params.automatic_scaling = None
rf_params.save_checkpoints = True
rf_params.load_checkpoints = True
print rf_params

control = TurbineFarmControl(farm)
rf = ReducedFunctional(functional, control, solver, rf_params)

# Maximize
max_ct = model_turbine.maximum_smeared_friction
m_opt = maximize(rf, bounds=[0, max_ct], scale=1.0,
    method="L-BFGS-B", options={'maxiter': 500})

# stopping time for running time computation
toc = time.clock()

# Get the optimal functional value
j = rf.evaluate(m_opt)

# Save the optimal turbine field
tf = farm.friction_function
optimal_turbine_friction_filename = os.path.join(output_path,"optimal_turbine_friction.xml")
File(optimal_turbine_friction_filename) << tf
print "Wrote optimal turbine friction to %s." % optimal_turbine_friction_filename
print "Run time:", toc - tic

## The default CG1 friction output is ugly at the neightbour 
## elements of the farm boundary (this does not affect the computation 
## as the farm friction is only integrated in the farm area). For
## visualisation we produce a nice output by projecting into a DG1 space.
#V_dg0 = FunctionSpace(config.domain.mesh, 'DG', 0)
#test = TestFunction(V_dg0)
#tfdg = Function(V_dg0)
#form = tfdg*test*config.site_dx - tf*test*config.site_dx(1)
#solve(form == 0, tfdg)
#optimal_turbine_friction_filename = config.params["base_path"]+"/optimal_turbine_friction.pvd"
#File(optimal_turbine_friction_filename) << tfdg

# Compute the total turbine friction
total_friction = assemble(tf*site_dx(1))

# Compute the total cost 
cost = float(model_turbine.cost_coefficient * total_friction)

# Compute the pure power production
# Note: don't use checkpoint here
power_rf_params = rf_params
power_rf_params.save_checkpoints = False
power_rf = ReducedFunctional(PowerFunctional(problem), control, solver, power_rf_params)
power = power_rf.evaluate(m_opt)

# Compute the site area
V_r = FunctionSpace(domain.mesh, 'R', 0)
site_area = assemble(interpolate(Constant(1), V_r)*site_dx(1))

# Print results
print "="*40
print "Site area (m^2): ", site_area
print "Cost coefficient: {}".format(model_turbine.cost_coefficient)
print "Functional (power-cost): %e." % -j
print "Total power: %e." % power
print "Total cost: %e." % cost
print "Maximum smeared turbine friction: %e." % max_ct
print "Total turbine friction: %e." % total_friction 
print "Average smeared turbine friction: %e." % (total_friction / site_area)
print "Total power / total friction: %e." % (power / total_friction)
print "Friction per discrete turbine: {}".format(model_turbine.friction)
print "Estimated number of discrete turbines: {}".format(total_friction/model_turbine.friction)
print "Estimated average power per turbine: {}".format(power / (total_friction/model_turbine.friction))
