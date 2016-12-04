from dolfin_adjoint import *
from opentidalfarm import *
from dolfin_adjoint import Control
import Optizelle
from model_turbine import ModelTurbine
import os.path
import time
from rectangle_mesh_domain import *

""" Turbine farm layout optimization using continuous turbine density
approach. Underlying PDE described by the shalow water equations in steady state.
Inequality constraints for friction given by lower limit =0 and upper limit 0.589"""

set_log_level(INFO)

output_path = "results"

# Create the model turbine
model_turbine = ModelTurbine()
print model_turbine

# Set up domain
mesh = Mesh('mesh.xml')
domain = RectangularMeshDomain(mesh, 0, 0, 2000, 1000)
domains = domain.cell_ids

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
algorithm otherwise fails""" 
f = Constant(0.01)
g = Constant(0.001)
h = Expression('(x[0] <= 1250 && x[1] <= 650 && 750 <= x[0] && 350 <= x[1]) ? f : g', f=f, g=g, degree=2)
farm.friction_function.assign(h)

prob_params.tidal_farm = farm

# boundary conditions
bcs = BoundaryConditionSet()
bcs.add_bc("u", Constant((2.0,0.0)), facet_id=1, bctype="strong_dirichlet")
bcs.add_bc("eta", Constant(0.0), facet_id=2, bctype="strong_dirichlet")
bcs.add_bc("u", Constant((0.0,0.0)), facet_id=3, bctype="weak_dirichlet")
prob_params.bcs = bcs

# Equation settings
prob_params.viscosity = Constant(5.)
prob_params.depth = Constant(50.0)
prob_params.friction = Constant(0.0025)
prob_params.initial_condition = Expression(("1e-7", "0", "0"), degree=2)

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
rf([farm.friction_function])

# define optimization problem for Optizelle
opt_problem = MaximizationProblem(rf, bounds=(0., model_turbine.maximum_smeared_friction))

parameters = {
             "maximum_iterations": 200,
             "optizelle_parameters":
                 {
                 "msg_level" : 10,
                 "algorithm_class" : Optizelle.AlgorithmClass.TrustRegion,
                 "H_type" : Optizelle.Operators.BFGS,
                 "dir" : Optizelle.LineSearchDirection.BFGS,
                 #"ipm": Optizelle.InteriorPointMethod.LogBarrier,
                 "sigma": 0.95, 
                 "gamma": 0.5,
                 "linesearch_iter_max" : 20,
                 "krylov_iter_max" : 100,
                 "eps_krylov" : 1e-9,
                 "eps_grad" : 1e-200,
                 "eps_dx" : 1e-20
                 }
             }

opt_solver = OptizelleSolver(opt_problem, parameters=parameters)

# Start time for optimisation
tic = time.clock()

# Solve optimisation problem using Optizelle
m_opt = opt_solver.solve()

# stopping time for optimisation
toc = time.clock()

# plot the optimal solution
plot(m_opt, interactive=True)
print "optimal ", assemble(m_opt*dx)

# Get the optimal functional value
j = rf.evaluate(m_opt)

# Save the optimal turbine field
tf = farm.friction_function
optimal_turbine_friction_filename = os.path.join(output_path,"optimal_turbine_friction.xml")
File(optimal_turbine_friction_filename) << tf
print "Wrote optimal turbine friction to %s." % optimal_turbine_friction_filename

# Compute the total turbine friction
total_friction = assemble(tf*site_dx(1))

# Compute the total cost 
cost = float(model_turbine.cost_coefficient * total_friction)

# Compute the pure power production
# Note: don't use checkpoint here
power_rf = FenicsReducedFunctional(power_functional, control, solver)
power = power_rf(m_opt)

# Compute the site area
V_r = FunctionSpace(domain.mesh, 'R', 0)
site_area = assemble(interpolate(Constant(1), V_r)*site_dx(1))

# Print results
print "="*40
print "Site area (m^2): ", site_area
print "Cost coefficient: {}".format(model_turbine.cost_coefficient)
print "Functional (power-cost): %e." % j
print "Total power: %e." % power
print "Total cost: %e." % cost
print "Maximum smeared turbine friction: %e." % model_turbine.maximum_smeared_friction
print "Total turbine friction: %e." % total_friction 
print "Average smeared turbine friction: %e." % (total_friction / site_area)
print "Total power / total friction: %e." % (power / total_friction)
print "Friction per discrete turbine: {}".format(model_turbine.friction)
print "Estimated number of discrete turbines: {}".format(total_friction/model_turbine.friction)
print "Estimated average power per turbine: {}".format(power / (total_friction/model_turbine.friction))
print "Run time:", toc - tic
