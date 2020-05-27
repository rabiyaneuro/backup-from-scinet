//
// diff2dplot.cpp - Plots a 2d density to screen (C++ version, uses cpgplot)
//
// Ramses van Zon
// SciNetHPC, 2016
//
#include <cpgplot.h>
#include <algorithm>
#include <iostream>
#include "diff2dplot.h"
void plotdens(const rarray<double,2>& dens, double x1, double x2, bool first)
{    
    static int npnts;
    static rarray<float,1> rdens;
    static float rx1, rx2, vmin, vmax, tr[6];
    static bool xpresent;
    int cpgbeg_result;
    if (first) {
        cpgbeg_result = cpgbeg(0, "/xwindow", 1, 1);
        xpresent = (cpgbeg_result == 1);
        if (xpresent) {
           cpgask(0);
           rx1 = x1;
           rx2 = x2;
           npnts = dens.extent(0);
           cpgenv(rx1, rx2, rx1, rx2, 1, 0);
           //float f[8] = {0.0, .143, .286, .428, .571, .714, .857, 1.0};
           //float r[8] = {0.0, .718, .316, .000, .000, .727, .847, 1.0};
           //float g[8] = {0.0, .000, .316, .559, .592, .727, .057, 1.0}; 
           //float b[8] = {0.0, .718, .991, .559, .000, .000, .057, 1.0};
           // The color scheme defined by the four arrays above is
           // really nice and smooth, but the following matches
           // matplotlib's 'spectral' color map better:
           float f[8] = {0.0, .050, .180, .280, .500, .700, .840, 1.0};
           float r[8] = {0.0, .718, .000, .000, .000, .787, .847, 1.0};
           float g[8] = {0.0, .000, .000, .559, .592, .787, .057, 1.0}; 
           float b[8] = {0.0, .718, .991, .559, .000, .000, .057, 1.0};
           cpgctab(f, r, g, b, 8, -1.0, 0.5);
           vmin = *std::min_element(&dens[0][0], &dens[npnts-1][npnts-1]+1);
           vmin = *std::max_element(&dens[0][0], &dens[npnts-1][npnts-1]+1);
           tr[0] = rx1;
           tr[1] = (rx2-rx1)/npnts;
           tr[2] = 0.0;
           tr[3] = rx1;
           tr[4] = 0.0;
           tr[5] = (rx2-rx1)/npnts;
           rdens = rarray<float,1>(dens.extent(0)*dens.extent(1));
        }
    }
    if (xpresent) {
        for (int i=0;i<npnts*npnts;i++)
            rdens[i] = dens.data()[i];
        cpgimag(rdens.data(), npnts, npnts, 1,npnts, 1,npnts, vmin, vmax, tr);
        if (first)
            cpgwedg("RI", 1.5, 3., vmin, vmax, " ");
    } else {
        std::cerr << "WARNING: No DISPLAY found for graphics" << std::endl;
    }
}
//
void endplot()
{
    cpgend();
}

