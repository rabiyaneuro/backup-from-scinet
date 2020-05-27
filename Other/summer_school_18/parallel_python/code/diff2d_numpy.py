#!/usr/bin/env python
#
# diff2d-numpy.py - Simulates two-dimensional diffusion on a square domain
#                   This one uses numpy arrays.
#
# Ramses van Zon
# SciNetHPC, 2016
#
import numpy as np

# import plotdens function
from diff2dplot import plotdens

# driver routine
def main():
    # aweful way to set parameters
    from diff2dparams import D, x1, x2, runtime, dx, outtime, graphics
    # Compute derived parameters.
    nrows     = int((x2-x1)/dx) # number of x points
    ncols     = nrows           # number of y points, same as x in this case.
    npnts     = nrows + 2       # number of x points including boundary points
    dx        = (x2-x1)/nrows   # recompute (previous dx may not fit in [x1,x2])
    dt        = 0.25*dx**2/D    # time step size (edge of stability)
    nsteps    = int(runtime/dt) # number of steps of that size to reach runtime
    nper      = int(outtime/dt) # how many step s between snapshots
    if nper==0: nper = 1
    # Allocate arrays.
    x         = np.linspace(x1-dx,x2+dx,num=npnts) # x values (also y values)
    dens      = np.zeros((npnts,npnts)) # time step t
    densnext  = np.zeros((npnts,npnts)) # time step t+1
    # Initialize.
    simtime=0*dt
    for i in range(1,npnts-1):
        a = 1 - abs(1 - 4*abs((x[i]-(x1+x2)/2)/(x2-x1)))
        for j in range(1,npnts-1):
            b = 1 - abs(1 - 4*abs((x[j]-(x1+x2)/2)/(x2-x1)))
            dens[i][j] = a*b
    # Output initial signal.
    print(simtime)
    if graphics:
        plotdens(dens, x[0], x[-1], first=True)
    # temporary array to hold laplacian
    laplacian = np.zeros((npnts,npnts)) # Laplacian of density inner domain
    for s in range(nsteps):
        # Take one step to produce new density.
        laplacian[1:nrows+1,1:ncols+1] = dens[2:nrows+2,1:ncols+1] + dens[0:nrows+0,1:ncols+1] + dens[1:nrows+1,2:ncols+2] + dens[1:nrows+1,0:ncols+0] - 4*dens[1:nrows+1,1:ncols+1]
        densnext[:,:] = dens + (D/dx**2)*dt*laplacian
        # Swap array pointers so t+1 becomes the new t, and update simulation time.
        dens, densnext = densnext, dens
        simtime += dt
        # Plot and report time.
        if (s+1)%nper == 0:
            print(simtime)
            if graphics:
                plotdens(dens, x[0], x[-1])

if __name__ == '__main__':
    main()

