//
// diff2dplot.h - Plots a 2d density to screen (C++ version, header file)
//
// Ramses van Zon
// SciNetHPC, 2016
//
#ifndef _DIFF2DPLOTH_
#define _DIFF2DPLOTH_

#include <rarray>

void plotdens(const rarray<double,2>& dens, double x1, double x2, bool first=false);
void endplot();
#endif
