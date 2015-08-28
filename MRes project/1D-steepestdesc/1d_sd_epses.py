from dolfin import *
from dolfin_adjoint import *
import numpy as np
import matplotlib.pyplot as plt

"""Steepest descent in 1D for non-uniformly refined meshes, different convergence
threshold values"""  

# Non-uniform refined meshes
meshes = [Mesh('mesh_1.xml'), Mesh('mesh_2.xml')]#, Mesh('mesh_3.xml'),\
#Mesh('mesh_4.xml'), Mesh('mesh_5.xml'), Mesh('mesh_6.xml'), Mesh('mesh_7.xml'), Mesh('mesh_8.xml'), \
#Mesh('mesh_9.xml'), Mesh('mesh_10.xml'),\
#Mesh('mesh_11.xml'),Mesh('mesh_12.xml'),Mesh('mesh_13.xml'),Mesh('mesh_14.xml')] #,Mesh('mesh_15.xml'),\
#Mesh('mesh_16.xml'), Mesh('mesh_17.xml'), Mesh('mesh_18.xml'), Mesh('mesh_19.xml'),Mesh('mesh_20.xml'),]

# Convergence threshold values
epses = [1e-02,1e-03,1e-04,1e-5,1e-8,1e-14]

iter_list = list()

for eps in epses:
    
    print "Run steepest descent method with convergence threshold =", eps
    print "----------------------------"
    print "----------------------------"

    # Refinement level count
    ref_level = 0

    # Iteration & element numbers
    iters = list()
    elements = list()

    for mesh in meshes:

        # Define discrete function spaces and functions
        V = FunctionSpace(mesh, "CG", 1)
        u = interpolate(Expression("0."), V, name='Control')


        # Define functional of interest
        x = SpatialCoordinate(mesh)
        func = (0.5*inner(u-1., u-1.))*dx


        # Define gradient
        grad = derivative(func, u)


        # Define steepest descent method

        i = 0
        imax = 2500000          # max number of iterations

        x = u.vector().array()              # initial value
        funcVal = assemble(func)            # initial functional value
        d = -assemble(grad).array()         # initial steepest descent direction
        Hessian = assemble(derivative(derivative(func,u),u)).array()    #initial Hessian

        iterates = list()   # save iterates


        while i < imax and funcVal > eps:
            
            # Save iterates as numpy arrays
            iterates.append(x)
            
            # Optimal step size
            alpha = np.inner(d,d)/np.dot(np.dot(Hessian, d),d)
            
            # Updating
            x = x + alpha * d
            u.vector()[:] = x
            
            # Compute new functional, steepest descent direction & Hessian
            func = (0.5*inner(u-1., u-1.))*dx
            funcVal = assemble(func)
            d = -assemble(derivative(func,u)).array()
            Hessian = assemble(derivative(derivative(func,u),u)).array()

            i += 1


        """  
        # Intuitive steepest descent method - choose some intuitively
        # sensible stepsize and if functional decreases keep it (or 
        # even increase it), and otherwise, halve it.

        i = 0
        imax = 100          # max number of iterations
        eps = 0.01          # tolerance
        d = -assemble(grad).array()         # initial steepest descent direction
        alpha = float(1/np.linalg.norm(d))  # initial step size
        x = u.vector().array()              # initial value
        funcVal_old = assemble(func)        # initial functional value

        while i < imax and funcVal_old > eps**2:
            
            x = x + alpha * d
            u.vector()[:] = x

            funcVal_new = assemble((0.5*inner(u-1., u-1.))*dx)
            
            if funcVal_new < funcVal_old:
                
                funcVal_old = funcVal_new
                func = (0.5*inner(u-1., u-1.))*dx
                d = -assemble(derivative(func,u)).array()
                #alpha = 1.5*alpha

            else:
                alpha = 0.5*alpha

            i += 1
        """

        print "Refinement level: ", ref_level 
        ref_level += 1

        print "Iteration number: ", len(iterates)
        print "Element number: ", mesh.num_cells()
        print "----------------------------"
        
        iters.append(len(iterates))
        elements.append(mesh.num_cells())

    # Save iterations in list for plot
    iter_list.append(np.array(iters))
    

# plot options
font = {'size'   : 14}
plt.rc('font', **font)


# compute relations largest/smallest element size per mesh
hmaxmin_list = list()
for i in range(len(meshes)):
    hmaxmin_list.append(meshes[i].hmax()/meshes[i].hmin())



# plot relation largest/smallest element size vs. iteration number for different eps
subplot_list=[321,322,323,324,325,326]

fig1 = plt.figure()
ax = fig1.add_subplot(subplot_list[0])
ax.plot(hmaxmin_list, iter_list[0], marker='o', label=r'$\varepsilon=10^{-2}$' )
plt.xlabel(r'$h_{\max} / h_{\min}$')
plt.ylabel('Iterations')
plt.legend(loc='best')

ax = fig1.add_subplot(subplot_list[1])
ax.plot(hmaxmin_list, iter_list[1], 'g', marker='o', label=r'$\varepsilon=10^{-3}$' )
plt.xlabel(r'$h_{\max} / h_{\min}$')
plt.ylabel('Iterations')
plt.legend(loc='best')

ax = fig1.add_subplot(subplot_list[2])
ax.plot(hmaxmin_list, iter_list[2], 'r', marker='o', label=r'$\varepsilon=10^{-4}$' )
plt.xlabel(r'$h_{\max} / h_{\min}$')
plt.ylabel('Iterations')
plt.legend(loc='best')

ax = fig1.add_subplot(subplot_list[3])
ax.plot(hmaxmin_list, iter_list[3], 'c', marker='o', label=r'$\varepsilon=10^{-5}$' )
plt.xlabel(r'$h_{\max} / h_{\min}$')
plt.ylabel('Iterations')
plt.legend(loc='best')

ax = fig1.add_subplot(subplot_list[4])
ax.plot(hmaxmin_list, iter_list[4], color='purple' , marker='o', label=r'$\varepsilon=10^{-8}$' )
plt.xlabel(r'$h_{\max} / h_{\min}$')
plt.ylabel('Iterations')
plt.legend(loc='best')

ax = fig1.add_subplot(subplot_list[5])
ax.plot(hmaxmin_list, iter_list[5], 'y', marker='o', label=r'$\varepsilon=10^{-12}$' )
plt.xlabel(r'$h_{\max} / h_{\min}$')
plt.ylabel('Iterations')
plt.legend(loc='best')

plt.savefig('hmaxmin_vs_iters_epses.pdf')
plt.show()


# reference convergence rate
eps = 1e-12
x = np.linspace(1,14,100)
y = -3./4*np.log(2*eps)*2**x


# plot refinement level vs. iteration number for different eps
plt.clf()
plt.semilogy(np.arange(1,len(meshes)+1), iter_list[0], marker='o', label=r'$\varepsilon=10^{-2}$' )
plt.semilogy(np.arange(1,len(meshes)+1), iter_list[1], marker='o', label=r'$\varepsilon=10^{-3}$')
plt.semilogy(np.arange(1,len(meshes)+1), iter_list[2], marker='o', label=r'$\varepsilon=10^{-4}$' )
plt.semilogy(np.arange(1,len(meshes)+1), iter_list[3], marker='o', label=r'$\varepsilon=10^{-5}$')
plt.semilogy(np.arange(1,len(meshes)+1), iter_list[4], marker='o', label=r'$\varepsilon=10^{-8}$' )
plt.semilogy(np.arange(1,len(meshes)+1), iter_list[5], marker='o', label=r'$\varepsilon=10^{-14}$' )
plt.semilogy(x,y, linestyle='--', label = r'$\sim 2^r$')
plt.xlabel(r'Refinement level $r$')
plt.ylabel('Iterations')
plt.legend(loc='best')
plt.savefig('reflevel_vs_iters_epses.pdf')
plt.show()



















