import pylab as pl
import numpy as np

def CTCS_AD(phiOld, c, d, nt):
    "Linear advection of profile in phiOld using CTCS with artificial diffusion"
    "(FTCS for 1st time step)"

    # input checking
    if pl.ndim(phiOld) != 1:
        raise ValueError("In CTCS_AD(phiOld, c, nt), phiOld must be a one dimensional array")
    if pl.size(phiOld) <= 1:
        raise ValueError("In CTCS_AD(phiOld, c, nt), phiOld must have at least 2 elements")
    if nt < 0:
        raise ValueError("In CTCS_AD(phiOld, c, nt), nt must not be negative")

    # the number of independent points
    nx= len(phiOld)-1

    # add another wrap-around point for periodic boundaries
    phiOld = pl.append(phiOld, [phiOld[1]])
    
    # mid and new time-step arrays for phi
    phi = pl.zeros_like(phiOld)
    phiNew = pl.zeros_like(phiOld)
    
    # FTCS for first time step (as long as at least one time-step is needed)
    if nt >= 1:
    
        for j in np.arange(1,nx+1):
            phi[j] = phiOld[j] - 0.5 * c * ( phiOld[j+1] - phiOld[j-1] )

        phi[0] = phi[nx]
        phi[nx+1] = phi[1]
   
        
    # CTCS for remaining time steps
        for it in np.arange(2,nt+1): 

            for j in np.arange(1,nx+1):
                phiNew[j] = phiOld[j] - c * ( phi[j+1] - phi[j-1] ) + \
                            2 * d * ( phiOld[j+1] - 2 * phiOld[j] + \
                                        phiOld[j-1] )
        
            phiNew[0] = phiNew[nx]
            phiNew[nx+1] = phiNew[1]
        
            phiOld = phi.copy()
            phi = phiNew.copy()
                      
    # return phiNew (without the periodic wrap-around point)
    return phiNew[0:nx+1]

def CTCS(phiOld, c, nt):
    "Linear advection of profile in phiOld using CTCS"
    "(FTCS for 1st time step)"

    # input checking
    if pl.ndim(phiOld) != 1:
        raise ValueError("In CTCS(phiOld, c, nt), phiOld must be a one dimensional array")
    if pl.size(phiOld) <= 1:
        raise ValueError("In CTCS(phiOld, c, nt), phiOld must have at least 2 elements")
    if nt < 0:
        raise ValueError("In CTCS(phiOld, c, nt), nt must not be negative")

    # the number of independent points
    nx= len(phiOld)-1
    # add another wrap-around point for periodic boundaries
    phiOld = pl.append(phiOld, [phiOld[1]])
    
    # mid and new time-step arrays for phi
    phi = pl.zeros_like(phiOld)
    phiNew = pl.zeros_like(phiOld)
    
    # FTCS for first time step (as long as at least one time-step is needed)
    if nt >= 1:
    
        for j in np.arange(1,nx+1):
            phi[j] = phiOld[j] - 0.5 * c * ( phiOld[j+1] - phiOld[j-1] )

        phi[0] = phi[nx]
        phi[nx+1] = phi[1]
   
        
    # CTCS for remaining time steps
        for it in np.arange(2,nt+1): 

            for j in np.arange(1,nx+1):
                phiNew[j] = phiOld[j] - c * ( phi[j+1] - phi[j-1] )
        
            phiNew[0] = phiNew[nx]
            phiNew[nx+1] = phiNew[1]
        
            phiOld = phi.copy()
            phi = phiNew.copy()
                      
    # return phiNew (without the periodic wrap-around point)
    return phiNew[0:nx+1]

def CTQUICK(phiOld, c, nt):
    "Linear advection of profile in phiOld using centred time QUICK"
    "(FTCS for 1st time step)"

    # input checking
    if pl.ndim(phiOld) != 1:
        raise ValueError("In CTQUICK(phiOld, c, nt), phiOld must be a one dimensional array")
    if pl.size(phiOld) <= 1:
        raise ValueError("In CTCQUICK(phiOld, c, nt), phiOld must have at least 2 elements")
    if nt < 0:
        raise ValueError("In CTQUICK(phiOld, c, nt), nt must not be negative")

    # the number of independent points
    nx= len(phiOld)-1
    # add wrap-around points for periodic boundaries
    phiOld = pl.append( phiOld, [phiOld[1]] )
    phiOld = pl.append([phiOld[nx-1]], phiOld)
    
    # mid and new time-step arrays for phi
    phi = pl.zeros_like(phiOld)
    phiNew = pl.zeros_like(phiOld)
    
    # FTCS for first time step (as long as at least one time-step is needed)
    if nt >= 1:
    
        for j in np.arange(2,nx+2):
            phi[j] = phiOld[j] - 0.5 * c * ( phiOld[j+1] - phiOld[j-1] )

        phi[0] = phi[nx]
        phi[1] = phi[nx+1]
        phi[nx+2] = phi[2]
       
        
    # CTQUICK for remaining time steps
        for it in np.arange(2,nt+1): 

            for j in np.arange(2,nx+2):
                phiNew[j] = phiOld[j] - c / 6 * ( 2*phi[j+1] + 3*phi[j] \
                                                - 6*phi[j-1] + phi[j-2])
        
            phiNew[0] = phiNew[nx]
            phiNew[1] = phiNew[nx+1]
            phiNew[nx+2] = phiNew[2]
        
            phiOld = phi.copy()
            phi = phiNew.copy()
                      
    # return phiNew (without the periodic wrap-around point)
    return phiNew[1:nx+2]




