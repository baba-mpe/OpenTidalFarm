#!/usr/bin/python

# Outer code for setting up the linear advection problem and calling
# the functions to perform the linear advection
from __future__ import division
import pylab as pl
import types

# all the linear advection schemes, diagnostics and initial conditions
exec(open("./advectionAD_Schemes.py").read())
"""exec(open("./diagnostics.py").read())"""
exec(open("./initialConditions.py").read())

def advectionAD(initialProfile, xmin = 0, xmax = 1, nx =40, \
              nt = 200, c = 0.2, d = 0.01):
    "Advect the given initial conditions function with domain from xmin to xmax"
    "nx points, nt time steps, c Courant number"

    # input checking
    if not isinstance(initialProfile, types.FunctionType):
        raise TypeError("In advection(initialProfile, xmin, xmax, nx, nt, c), \
                         initialProfile must be a function")
    if xmax <= xmin:
        raise ValueError("In advection(initialProfile, xmin, xmax, nx, nt, c), \
                         xmax must be greater than xmin")
    if (nx <= 0) or (nt <= 0):
        raise ValueError("In advection(initialProfile, xmin, xmax, nx, nt, c), \
                         nx and nt must be greater than zero")
    
    # derived variables
    dx = (xmax - xmin)/nx   # spatial resolution
    distanceTravelled = c*dx*nt

    # spatial points for plotting and for defining initial conditions
    x = pl.linspace(xmin, xmax, nx+1)

    # initial conditions
    phiOld = initialProfile(x)

    # Exact solution is the same as the initial conditions but moved around
    # the periodic domain
    phiExact = initialProfile((x - distanceTravelled)%(xmax - xmin))

    # Call function for advecting the profile using FTBS and CTCS for nt timesteps
    phiCTCS_AD = CTCS_AD(phiOld.copy(), c, d, nt)
    phiCTCS = CTCS(phiOld.copy(), c, nt)
    phiCTQUICK = CTQUICK(phiOld.copy(), c, nt)

    """# calculate the error norms, mean and standard deviation of the fields
    FTBSerrors = errorNorms(phiFTBS, phiExact)
    CTCSerrors = errorNorms(phiCTCS, phiExact)

    print("FTBS l1, l2 and linf errors", FTBSerrors)
    print("CTCS l1, l2 and linf errors", CTCSerrors)
    print("Initial mean and standard deviation", pl.mean(phiOld[0:-1]), \
           pl.std(phiOld[0:-2]))
    print("FTBS mean and standard deviation", pl.mean(phiFTBS[0:-1]), \
           pl.std(phiFTBS[0:-2]))
    print("CTCS mean and standard deviation", pl.mean(phiCTCS[0:-1]), \
           pl.std(phiCTCS[0:-2]))"""
 
    # plot options
    font = {'size'   : 14}
    pl.rc('font', **font)

    # plot the results versus the analytic solution
    pl.clf()
    #pl.plot(x, phiOld,      'k--', label='Initial')
    pl.plot(x, phiExact,    'k', label='Exact')
    pl.plot(x, phiCTCS, 'r', label ='CTCS')
    pl.plot(x, phiCTCS_AD,     'b', label='CTCS ArtDif')
    pl.plot(x, phiCTQUICK, 'g', label='CTQUICK')
    pl.legend(loc='best')
    pl.xlabel('x')
    pl.ylabel('$\phi$')

# call the advection function to advect the profile and plot the results
advectionAD(initialProfile=topHat, c = 0.01, nt = 200)
pl.savefig('phi.pdf')
pl.show()

