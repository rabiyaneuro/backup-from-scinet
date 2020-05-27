# Makefile for the compiled version for comparison with parallel python codes
# Ramses van Zon
# SciNetHPC, 2017

TIME=time -p
PRINT=./print
MPIRUN=mpirun
MPIRUNOPT=-np 3
RM=\rm -f
RMD=\rm -rf
#CXX=icpc
CXX=g++
OPT=-O3
CXXFLAGS=-std=c++11 $(OPT)
LDFLAGS=$(OPT)
#LDLIBS=-lcpgplot -lpgplot -lX11  -lpng -lifcore -lifport
LDLIBS=-lcpgplot -lpgplot -lX11  -lpng
#F90=ifort
F90=gfortran
FFLAGS=$(OPT)

all: diff2d_cpp.ex diff2d_f90.ex diff2d_cpp_omp.ex diff2d_f90_omp.ex diff2dplot_f90.mod

run: run_diff2d_cpp run_diff2d_f90 run_diff2d_cpp_omp run_diff2d_f90_omp

.PHONY: all clean run_diff2d_cpp run_diff2d_f90 run_diff2d_cpp_omp run_diff2d_f90_omp

run_diff2d_cpp: diff2d_cpp.ex
	$(TIME) ./$< 

run_diff2d_f90: diff2d_f90.ex
	$(TIME) ./$< 

run_diff2d_cpp_omp: diff2d_cpp_omp.ex
	$(TIME) ./$< 

run_diff2d_f90_omp: diff2d_f90_omp.ex
	$(TIME) ./$< 

diff2d_cpp.o: diff2d.cpp diff2dparams.py diff2dplot.h
	$(CXX) -c $(CXXFLAGS) -o $@ $<

diff2d_f90.o: diff2d.f90 diff2dparams.py diff2dplot_f90.mod
	$(F90) -c $(FFLAGS) -o $@ $<

diff2dplot_f90.mod: diff2dplot_f90.o
diff2dplot_f90.o: diff2dplot.f90 pgplot90.mod
	$(F90) -c $(FFLAGS) -o $@ $<

pgplot90.mod: pgplot90.o
pgplot90.o: pgplot90.f90
	$(F90) -c $(FFLAGS) -o $@ $<

diff2dplot_cpp.o: diff2dplot.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

diff2d_cpp_omp.o: diff2d.cpp diff2dparams.py diff2dplot.h
	$(CXX) -c -fopenmp $(CXXFLAGS) -o $@ $<

diff2d_f90_omp.o: diff2d.f90 diff2dparams.py diff2dplot_f90.mod
	$(F90) -c -fopenmp $(FFLAGS) -o $@ $<

diff2d_cpp.ex : diff2d_cpp.o diff2dplot_cpp.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS) -lgfortran

diff2d_f90.ex : diff2d_f90.o diff2dplot_f90.o
	$(F90) $(LDFLAGS) -o $@ $^ $(LDLIBS)

diff2d_cpp_omp.ex : diff2d_cpp_omp.o diff2dplot_cpp.o
	$(CXX) -fopenmp $(LDFLAGS) -o $@ $^ $(LDLIBS) -lgfortran

diff2d_f90_omp.ex : diff2d_f90_omp.o diff2dplot_f90.o
	$(F90) -fopenmp $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	$(RM) *.o *.ex
