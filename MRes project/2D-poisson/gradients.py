from dolfin import *
from dolfin_adjoint import *
import numpy as np
import matplotlib.pyplot as plt


output_path = "results"

parameters["adjoint"]["cache_factorizations"] = True

set_log_level(ERROR)

# Create mesh, refined in the center
n = 4
mesh = UnitSquareMesh(n, n)

def randomly_refine(initial_mesh, ratio_to_refine=.35):  
    numpy.random.seed(1)
    cf = CellFunction('bool', initial_mesh)
    for k in xrange(len(cf)):
        if numpy.random.rand() < ratio_to_refine:
            cf[k] = True
    return refine(initial_mesh, cf)

def refine_center(mesh, L=0.2):
    cf = CellFunction("bool", mesh)
    subdomain = CompiledSubDomain('std::abs(x[0]-0.5)<'+str(L)+' && std::abs(x[1]-0.5)<'+str(L))
    subdomain.mark(cf, True)
    return refine(mesh, cf)


#mesh = refine(mesh)
#mesh = refine(mesh)
#mesh = refine(mesh)
#mesh = refine(mesh)

mesh = randomly_refine(mesh)
mesh = randomly_refine(mesh)
mesh = randomly_refine(mesh)
mesh = randomly_refine(mesh)
mesh = randomly_refine(mesh)
mesh = randomly_refine(mesh)

cf = CellFunction('bool', mesh)
print "Element number", len(cf)
  
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


epsilons = np.linspace(-1e-7, 1e-7, num=20)

### Compute steps for little l2 representation

# Compute Riesz gradients
grad_l2 = rf.derivative(forget=False, project=False)[0]
grad_L2 = rf.derivative(forget=False, project=True)[0]

# Compute normalized gradients wrt L2
L2_norm = assemble(inner(grad_L2, grad_L2)*dx)
ngrad_l2 = grad_l2
ngrad_l2.vector()[:] = 1./L2_norm * grad_l2.vector().array()

ngrad_L2 = grad_L2
ngrad_L2.vector()[:] = 1./L2_norm * grad_L2.vector().array()

# Save step sizes in lists
L2_steps = list()
l2_steps = list()

for eps in epsilons:
    
    # Compute perturbated control
    f_new_l2 = interpolate(Expression("0."), W)
    f_new_l2.vector()[:] = f.vector().array() + eps * ngrad_l2.vector().array()

    f_new_L2 = interpolate(Expression("0."), W)
    f_new_L2.vector()[:] = f.vector().array() + eps * ngrad_L2.vector().array()

    # Compare to initial control
    step_size_l2 = (rf(f_new_l2) - rf(f))
    step_size_L2 = (rf(f_new_L2) - rf(f))
    
    l2_steps.append(step_size_l2)
    L2_steps.append(step_size_L2)


# plot options
font = {'size'   : 14}
plt.rc('font', **font)

# plot the results versus the analytic solution
plt.clf()

plt.plot(epsilons, np.array(l2_steps), 'r', label=r"$\nabla \widehat{J} = \mathcal{R}_{\ell^2}(\widehat{J}')$")
plt.plot(epsilons, np.array(L2_steps), 'b', label=r"$\nabla \widehat{J} = \mathcal{R}_{L^2}(\widehat{J}')$")
#plt.title('Title')
plt.legend(loc='best')
plt.xlabel(r"$\varepsilon$")
plt.ylabel(r"$\widehat{J} (f+\varepsilon \nabla \widehat{J}(f)) - \widehat{J} (f)$")
#plt.ylabel('Functional step size')

plt.savefig('gradients.pdf')
plt.show()



