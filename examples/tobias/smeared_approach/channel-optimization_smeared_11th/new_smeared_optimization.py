from dolfin_adjoint import *
from opentidalfarm import *
from dolfin_adjoint import Control
import Optizelle
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
#f = File(os.path.join(output_path, "turbine_farms.pvd"))
#f << domains

# Initial turbine friction field
"""Make sure that larger than lower bound of optimization (=zero) as interior point 
algorithm requires so""" 
f = Constant(0.01)
g = Constant(0.001)
h = Expression('(x[0] <= 1250 && x[1] <= 1250 && 750 <= x[0] && 750 <= x[1]) ? f : g', f=f, g=g)
farm.friction_function.assign(h)

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
functional = power_functional  - cost_functional

# Define the control
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
                 "sigma": 0.95, 
                 "gamma": 0.5,
                 "linesearch_iter_max" : 20,
                 "krylov_iter_max" : 100,
                 "eps_krylov" : 1e-9
                 }
             }

opt_solver = OptizelleSolver(opt_problem, parameters=parameters)

# Maximize
f_opt = opt_solver.solve()

# stopping time for running time computation
toc = time.clock()

plot(f_opt, interactive=True)
print "optimal ", assemble(f_opt*dx)
print "Run time:", toc - tic

"""  
# Get the optimal functional value
j = rf.evaluate(m_opt)

# Save the optimal turbine field
tf = farm.friction_function
optimal_turbine_friction_filename = os.path.join(output_path,"optimal_turbine_friction.xml")
File(optimal_turbine_friction_filename) << tf
print "Wrote optimal turbine friction to %s." % optimal_turbine_friction_filename

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
print "Estimated average power per turbine: {}".format(power / (total_friction/model_turbine.friction))"""
