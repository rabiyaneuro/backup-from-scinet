//
// diff2d.cpp - Simulates two-dimensional diffusion on a square domain
//              C++ version, using rarray (github.com/vanzonr/rarray)
//
// Ramses van Zon
// SciNetHPC, 2016
//
#include <rarray>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "diff2dplot.h"

// driver routine
int main() 
{
    // Include file sets parameters D, x1, x2, runtime, dx, outtime, and graphics:
    enum { False=false, True=true };
    double D;
    double x1;
    double x2;
    double runtime;
    double dx;
    double outtime;
    bool graphics;
    #include "diff2dparams.py"
    // Compute derived parameters.
    int nrows  = (x2-x1)/dx;   // number of x points
    int ncols  = nrows;        // number of y points, same as x in this case.
    int npnts  = nrows + 2;    // number of x points including boundary points
    dx  = (x2-x1)/nrows; // recompute (previous dx may not fit in [x1,x2])
    double dt  = 0.25*dx*dx/D; // time step size
    int nsteps = runtime/dt;   // number of steps of that size to reach runtime
    int nper   = outtime/dt;   // # how many step s between snapshots
    if (nper==0) {
        nper = 1;
    }
    // Allocate arrays.
    rarray<double,1> x(npnts);              // x values (y is the same)
    rarray<double,2> dens(npnts,npnts);     // time step t
    rarray<double,2> densnext(npnts,npnts); // time step t+1
    // Initialize.
    for (int i=0; i<npnts; i++) {
        x[i] = x1+((i-1)*(x2-x1))/nrows;
    }
    dens.fill(0.0);
    densnext.fill(0.0);
    double simtime = 0*dt;
    for (int i=1; i<npnts-1; i++) {
        double a = 1 - fabs(1 - 4*fabs((x[i]-(x1+x2)/2)/(x2-x1)));
        for (int j=1; j<npnts-1; j++) {
            double b = 1- fabs(1 - 4*fabs((x[j]-(x1+x2)/2)/(x2-x1)));
            dens[i][j] = a*b;
        }
    }
    // Output initial signal.
    std::cout << simtime << std::endl;
    if (graphics) {
        std::cerr <<"GRAPHICS!\n";
        plotdens(dens, x[0], x[npnts-1], true);
    }
    // temporary array to hold laplacian
    rarray<double,2> laplacian(npnts,npnts);
    laplacian.fill(0.0);
    for (int s=0; s<nsteps; s++) {
        // Take one step to produce new density.
        #pragma omp parallel default(none) firstprivate(D,dx,dt,ncols,nrows) shared(laplacian,dens,densnext)
        {
        #pragma omp for 
        for (int i=1; i<=nrows; i++) {
            for (int j=1; j<=ncols; j++) {
                laplacian[i][j] = dens[i+1][j]+dens[i-1][j]+dens[i][j+1]+dens[i][j-1]-4*dens[i][j];
	    }
	}
        #pragma omp for 
        for (int i=1; i<=nrows; i++) {
 	    for (int j=1; j<=ncols; j++) {
	        densnext[i][j] = dens[i][j] + (D/(dx*dx))*dt*laplacian[i][j];
	    }
	}
        }
    	// Swap array pointers so t+1 becomes the new t and update simulation time.
	std::swap(dens, densnext);
	simtime += dt;
	// Plot and report time.
	if ((s+1)%nper == 0) {
	    std::cout << simtime << std::endl;
	    if (graphics) {
                plotdens(dens,x[0],x[npnts-1]);
	    }
	}
    }
    if (graphics) {
        endplot();
    }
}

