from dolfin_adjoint import *
from opentidalfarm import *
from dolfin_adjoint import Control, L2, BaseRieszMap
from model_turbine import ModelTurbine
import os.path
import time

""" Turbine farm layout optimization using continuous turbine density
approach. Underlying PDE described by the shalow water equations in steady state.
Inequality constraints for friction given by lower limit =0 and upper limit 0.589"""

set_log_level(ERROR)

output_path = "results"

# Create the model turbine
model_turbine = ModelTurbine()
print model_turbine

# Set up domain
x_max = 2000
y_max = 1000

N = 20
prob_params = SteadySWProblem.default_parameters()
domain = RectangularDomain(0, 0, x_max, y_max, nx=N, ny=N)
#domain = Domain(Mesh("mesh.xml"), dx, ds)
prob_params.domain = domain

# Define the farm and set site_dx such that we integrate over the farm only
turbine = SmearedTurbine()
W = FunctionSpace(domain.mesh, "DG", 0)
farm = Farm(domain, turbine, function_space=W)

class FarmDomain(SubDomain):
    def inside(self, x, on_boundary):                                                                                                                                                          
        return between(x[0], (0.375*x_max, 0.625*x_max)) and between(x[1], (0.35*y_max, 0.65*y_max))

farm_cf = CellFunction("size_t", domain.mesh)
farm_cf.set_all(0)          
FarmDomain().mark(farm_cf, 1)
#plot(farm_cf, interactive=True)

site_dx = Measure("dx")(subdomain_data=farm_cf)
farm.site_dx = site_dx(1)
prob_params.tidal_farm = farm

# Initial turbine friction field
"""Make sure that larger than lower bound of optimization (=zero) as interior point 
algorithm otherwise fails""" 
#f = Constant(0.01)
#g = Constant(0.001)
#h = Expression('(x[0] <= 1250 && x[1] <= 650 && 750 <= x[0] && 350 <= x[1]) ? f : g', f=f, g=g, degree=2)
#farm.friction_function.assign(h)
#farm.friction_function.vector()[:] = model_turbine.maximum_smeared_friction/2

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
functional *= 1e-6 # convert to MW

# Define the control
control = Control(farm.friction_function) 
rf = FenicsReducedFunctional(-functional, control, solver)
rf([farm.friction_function])

# Define optimization problem 
opt_problem = MaximizationProblem(rf, bounds=(0., model_turbine.maximum_smeared_friction))

parameters = { "monitor": None,
               "type": "blmvm",
               "max_it": 50,
               "subset_type": "matrixfree",
               "fatol": 0.0,
               "frtol": 1e-0,
               "gatol": 0.0,
               "grtol": 0.0,
             }

# Define custom Riesz map
class L2Farm(BaseRieszMap):
    def assemble(self):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        A = inner(u, v)*farm.site_dx
        a = assemble(A, keep_diagonal=True)
        a.ident_zeros()
        return a
                  
# Remove the riesz_map to switch from L2Farm norm to l2 norm    
opt_solver = TAOSolver(opt_problem, parameters, riesz_map=L2Farm(W))

# Start time for optimisation
tic = time.clock()

# Solve optimisation problem using Optizelle
m_opt = opt_solver.solve()
from IPython import embed; embed()

# Alternatively solve with scipy.optimize
#m_opt = minimize(rf, bounds=[0, model_turbine.maximum_smeared_friction],
#                 method="L-BFGS-B", options={'maxiter': 25})

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
power = power_rf([m_opt])

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
