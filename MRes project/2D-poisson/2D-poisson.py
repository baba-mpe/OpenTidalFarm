""" Solves a optimal control problem constrained by the Poisson equation:

    min_(u, m) \int_\Omega 1/2 || u - d ||^2 + 1/2 || f ||^2
    
    subjecct to 

    grad \cdot \grad u = f    in \Omega
    u = 0                     on \partial \Omega

    Using Optizelle (with custom inner product representation) - Newton-CG
    And SciPy - L-BFGS-B
"""
from dolfin import *
from dolfin_adjoint import *
import os.path
import Optizelle
import pylab
import time

output_path = "results"

parameters["adjoint"]["cache_factorizations"] = True

set_log_level(ERROR)

# Create mesh, refined in the center
n = 4
mesh = UnitSquareMesh(n, n)


def randomly_refine(initial_mesh, ratio_to_refine=.35):   
    numpy.random.seed(4)
    cf = CellFunction('bool', initial_mesh)
    for k in xrange(len(cf)):
        if numpy.random.rand() < ratio_to_refine:
            cf[k] = True
    return refine(initial_mesh, cf)


#mesh = refine(mesh)
#mesh = refine(mesh)
#mesh = refine(mesh)
#mesh = refine(mesh)
#mesh = refine(mesh)

#mesh = randomly_refine(mesh)
#mesh = randomly_refine(mesh)
#mesh = randomly_refine(mesh)
#mesh = randomly_refine(mesh)
#mesh = randomly_refine(mesh)
#mesh = randomly_refine(mesh)

cf = CellFunction('bool', mesh)
print len(cf)

  
# Define discrete function spaces and funcions
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

f = interpolate(Expression("0.5"), W, name='Control')
u = Function(V, name='State')
v = TestFunction(V)

# Define and solve the Poisson equation to generate the dolfin-adjoint annotation
F = (inner(grad(u), grad(v)) - f*v)*dx 
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)

# Define functional of interest and the reduced functional
x = SpatialCoordinate(mesh)
d = 1/(2*pi**2)*sin(pi*x[0])*sin(pi*x[1]) # the desired temperature profile
f_analytic = Expression("sin(pi*x[0])*sin(pi*x[1])")


alpha = Constant(1e-6)
J = Functional((0.5*inner(u-d, u-d))*dx + alpha/2*f**2*dx)
control = Control(f)
rf = ReducedFunctional(J, control)

#set minimisation problem for Optizelle
problem = MinimizationProblem(rf)


# Optizelle algorithm parameters 
parameters = {
             "maximum_iterations": 100,
             "optizelle_parameters":
                 {
                 "msg_level" : 10,
                 "algorithm_class" : Optizelle.AlgorithmClass.LineSearch,
                 "H_type" : Optizelle.Operators.UserDefined,
                 "dir" : Optizelle.LineSearchDirection.NewtonCG,
                 "ipm": Optizelle.InteriorPointMethod.PrimalDual,
                 "sigma": 0.9, 
                 "gamma": 0.5,
                 "linesearch_iter_max" : 50,
                 "krylov_iter_max" : 50,
                 "eps_krylov" : 1e-4,
                 "eps_dx" : 1e-3,
                 "eps_grad" : 1e-7
                 }
             }           
             

solver = OptizelleSolver(problem, parameters=parameters)
#import ipdb; ipdb.set_trace()

tic = time.clock()

#SCIPY L-BFGS-B
#f_opt = minimize(rf,  method="L-BFGS-B", options={"maxiter": 150, 'ftol':1e-200, 'gtol': 1e-200})

#OPTIZELLE
f_opt = solver.solve()
  
toc = time.clock()


print "Volume: ", assemble(f_opt*dx)
print "Run Time: ", toc - tic
control_error = errornorm(f_analytic, f_opt)

print "Error: ", control_error

tf = f_opt
filename = os.path.join(output_path,"plot.pvd")
File(filename) << tf




