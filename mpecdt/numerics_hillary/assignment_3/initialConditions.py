# Initial conditions for linear advection
import pylab as pl

def cosBell(x):
    "Function defining a cosine bell as a function of position, x"

    # input checking
    if pl.ndim(x) != 1:
        raise ValueError("In cosBell(x), x must be a one dimensional array")

    bell= lambda x: 0.5*(1 - pl.cos(4*pl.pi*x))

    #chooses bell(x) where condition is true, else chooses zeros
    phi = pl.where((x<0.5) | (x>=1.0), bell(x), 0)

    return phi

def cosBell2(x):
    "Function defining a different cosine bell as a function of position, x"

    # input checking
    if pl.ndim(x) != 1:
        raise ValueError("In cosBell2(x), x must be a one dimensional array")
    bell= lambda x: 0.25*(1 - pl.cos(4*pl.pi*x))*(1 - pl.cos(pl.pi*x)**2)

    # chooses bell(x) where condition is true, else chooses zeros
    phi = pl.where((x<0.5) | (x>=1.0), bell(x), 0.)

    return phi

def topHat(x):
    "Function defining a top hat as a function of position, x"

    # input checking
    if pl.ndim(x) != 1:
        raise ValueError("In tophat(x), x must be a one dimensional array")

    # chooses 1 where condition is true, else chooses zeros
    phi = pl.where((x<0.5) | (x>=1.0), 1.0, 0.)

    return phi

def mixed(x):
    "A flat peak in one location and a cosine bell in another"

    # input checking
    if pl.ndim(x) != 1:
        raise ValueError("In mixed(x), x must be a one dimensional array")

    return pl.where((x >= 0.2) & (x <= 0.3), 1, \
           pl.where((x >= 0.4) & (x <= 0.8), 0.5*(1 + pl.cos(5*pl.pi*(x-0.6))),
           pl.where((x > 0.1) & (x < 0.2), 10*(x-0.1), 
           pl.where((x > 0.3) & (x < 0.35), 20*(0.35-x), 0))))

