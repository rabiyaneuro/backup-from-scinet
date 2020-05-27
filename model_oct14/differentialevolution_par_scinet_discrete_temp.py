"""
differential_evolution: The differential evolution global optimization algorithm
Added by Andrew Nelson 2014

This file used to be called differential_evolution_par.py
This is a modified version of the original diff evol algorithm for use when 
running jobs in parallel
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize.optimize import _status_message
from scipy._lib._util import check_random_state
from scipy._lib.six import xrange
from copy import copy
import time
from memory_profiler import profile

__all__ = ['differential_evolution']

_MACHEPS = np.finfo(np.float64).eps

@profile
def differential_evolution(func, bounds, args=(), strategy='best1bin',
                            strategy1 = None, strategy2 = None,
                           maxiter=1000, popsize=15, tol=0.01,
                           mutation=(0.5, 1), recombination=0.7, seed=None,
                           callback=None, disp=False, polish=True,
                           init='latinhypercube', atol=0, prior = None,
                           mse_thresh = 1, thresh = None, rank = None, 
                           size = None, comm = None, jobid =None):
    
    """Finds the global minimum of a multivariate function.
    Differential Evolution is stochastic in nature (does not use gradient
    methods) to find the minimium, and can search large areas of candidate
    space, but often requires larger numbers of function evaluations than
    conventional gradient based techniques.

    The algorithm is due to Storn and Price [1]_.

    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:

            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'

        The default is 'best1bin'.
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize * len(x)`` individuals.
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        ``U[min, max)``. Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : int or `np.random.RandomState`, optional
        If `seed` is not specified the `np.RandomState` singleton is used.
        If `seed` is an int, a new `np.random.RandomState` instance is used,
        seeded with seed.
        If `seed` is already a `np.random.RandomState instance`, then that
        `np.random.RandomState` instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Display status messages
    callback : callable, `callback(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the population convergence.  When ``val`` is greater than one
        the function halts. If callback returns `True`, then the minimization
        is halted (any polishing is still carried out).
    polish : bool, optional
        If True (default), then `scipy.optimize.minimize` with the `L-BFGS-B`
        method is used to polish the best population member at the end, which
        can improve the minimization slightly.
    init : string, optional
        Specify how the population initialization is performed. Should be
        one of:

            - 'latinhypercube'
            - 'random'

        The default is 'latinhypercube'. Latin Hypercube sampling tries to
        maximize coverage of the available parameter space. 'random' initializes
        the population randomly - this has the drawback that clustering can
        occur, preventing the whole of parameter space being covered.
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.  If `polish`
        was employed, and a lower minimum was obtained by the polishing, then
        OptimizeResult also contains the ``jac`` attribute.

    Notes
    -----
    Differential evolution is a stochastic population based method that is
    useful for global optimization problems. At each pass through the population
    the algorithm mutates each candidate solution by mixing with other candidate
    solutions to create a trial candidate. There are several strategies [2]_ for
    creating trial candidates, which suit some problems more than others. The
    'best1bin' strategy is a good starting point for many systems. In this
    strategy two members of the population are randomly chosen. Their difference
    is used to mutate the best member (the `best` in `best1bin`), :math:`b_0`,
    so far:

    .. math::

        b' = b_0 + mutation * (population[rand0] - population[rand1])

    A trial vector is then constructed. Starting with a randomly chosen 'i'th
    parameter the trial is sequentially filled (in modulo) with parameters from
    `b'` or the original candidate. The choice of whether to use `b'` or the
    original candidate is made with a binomial distribution (the 'bin' in
    'best1bin') - a random number in [0, 1) is generated.  If this number is
    less than the `recombination` constant then the parameter is loaded from
    `b'`, otherwise it is loaded from the original candidate.  The final
    parameter is always loaded from `b'`.  Once the trial candidate is built
    its fitness is assessed. If the trial is better than the original candidate
    then it takes its place. If it is also better than the best overall
    candidate it also replaces that.
    To improve your chances of finding a global minimum use higher `popsize`
    values, with higher `mutation` and (dithering), but lower `recombination`
    values. This has the effect of widening the search radius, but slowing
    convergence.

    .. versionadded:: 0.15.0

    Examples
    --------
    Let us consider the problem of minimizing the Rosenbrock function. This
    function is implemented in `rosen` in `scipy.optimize`.

    >>> from scipy.optimize import rosen, differential_evolution
    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    >>> result = differential_evolution(rosen, bounds)
    >>> result.x, result.fun
    (array([1., 1., 1., 1., 1.]), 1.9216496320061384e-19)

    Next find the minimum of the Ackley function
    (http://en.wikipedia.org/wiki/Test_functions_for_optimization).

    >>> from scipy.optimize import differential_evolution
    >>> import numpy as np
    >>> def ackley(x):
    ...     arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    ...     arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
    ...     return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e
    >>> bounds = [(-5, 5), (-5, 5)]
    >>> result = differential_evolution(ackley, bounds)
    >>> result.x, result.fun
    (array([ 0.,  0.]), 4.4408920985006262e-16)

    References
    ----------
    .. [1] Storn, R and Price, K, Differential Evolution - a Simple and
           Efficient Heuristic for Global Optimization over Continuous Spaces,
           Journal of Global Optimization, 1997, 11, 341 - 359.
    .. [2] http://www1.icsi.berkeley.edu/~storn/code.html
    .. [3] http://en.wikipedia.org/wiki/Differential_evolution
    """

    solver = DifferentialEvolutionSolver(func, bounds, args=args,
                                         strategy=strategy, maxiter=maxiter,
                                         popsize=popsize, tol=tol,
                                         mutation=mutation,
                                         recombination=recombination,
                                         seed=seed, polish=polish,
                                         callback=callback,
                                         disp=disp, init=init, atol=atol, 
                                         prior = prior,
                                         mse_thresh= mse_thresh, 
                                         strategy1 = strategy1,
                                         strategy2 = strategy2, 
                                         thresh = thresh, rank = rank,
                                         size =size, comm = comm, jobid= jobid)
    return solver.solve()


class DifferentialEvolutionSolver(object):

    """This class implements the differential evolution solver

    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:

            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'

        The default is 'best1bin'

    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize * len(x)``
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize * len(x)`` individuals.
    tol : float, optional
        Relative tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        U[min, max). Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : int or `np.random.RandomState`, optional
        If `seed` is not specified the `np.random.RandomState` singleton is
        used.
        If `seed` is an int, a new `np.random.RandomState` instance is used,
        seeded with `seed`.
        If `seed` is already a `np.random.RandomState` instance, then that
        `np.random.RandomState` instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Display status messages
    callback : callable, `callback(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the population convergence.  When ``val`` is greater than one
        the function halts. If callback returns `True`, then the minimization
        is halted (any polishing is still carried out).
    polish : bool, optional
        If True, then `scipy.optimize.minimize` with the `L-BFGS-B` method
        is used to polish the best population member at the end. This requires
        a few more function evaluations.
    maxfun : int, optional
        Set the maximum number of function evaluations. However, it probably
        makes more sense to set `maxiter` instead.
    init : string, optional
        Specify which type of population initialization is performed. Should be
        one of:

            - 'latinhypercube'
            - 'random'
    atol : float, optional
        Absolute tolerance for convergence, the solving stops when
        ``np.std(pop) <= atol + tol * np.abs(np.mean(population_energies))``,
        where and `atol` and `tol` are the absolute and relative tolerance
        respectively.
        
    prior : should be a list of solution vectors (in list format)
    """

    # Dispatch of mutation strategy method (binomial or exponential).
    _binomial = {'best1bin': '_best1',
                 'randtobest1bin': '_randtobest1',
                 'best2bin': '_best2',
                 'rand2bin': '_rand2',
                 'rand1bin': '_rand1'}
    _exponential = {'best1exp': '_best1',
                    'rand1exp': '_rand1',
                    'randtobest1exp': '_randtobest1',
                    'best2exp': '_best2',
                    'rand2exp': '_rand2'}

    def __init__(self, func, bounds, args=(),
                 strategy='best1bin', maxiter=1000, popsize=15,
                 strategy1 = None, strategy2 = None,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                 maxfun=np.inf, callback=None, disp=False, polish=True,
                 init='latinhypercube', atol=0, prior = None, mse_thresh = 1,
                 thresh = None, rank = None, size = None, comm = None,
                 jobid=None):
        
        #MPI params
        self.comm = comm
        self.rank = rank
        self.size = size
        self.jobid = jobid
        
        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy
        self.strategy1 = strategy1
        self.strategy2 = strategy2
        self.thresh = thresh

        self.callback = callback
        self.polish = polish

        # relative and absolute tolerances for convergence
        self.tol, self.atol = tol, atol

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.
        self.scale = mutation
        if (not np.all(np.isfinite(mutation)) or
                np.any(np.array(mutation) >= 2) or
                np.any(np.array(mutation) < 0)):
            raise ValueError('The mutation constant must be a float in '
                             'U[0, 2), or specified as a tuple(min, max)'
                             ' where min < max and min, max are in U[0, 2).')

        self.dither = None
        
        #if mutation const is given as min and max then a random const will be
        #chosen from that range each generation
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination

        self.func = func
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        self.limits = np.array(bounds, dtype='float').T
        if (np.size(self.limits, 0) != 2 or not
                np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence containing '
                             'real valued (min, max) pairs for each value'
                             ' in x')

        if maxiter is None:  # the default used to be None
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:  # the default used to be None
            maxfun = np.inf
        self.maxfun = maxfun

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        
        #np.fabs- Compute the absolute values element-wise.
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.parameter_count = np.size(self.limits, 1)

        self.random_number_generator = check_random_state(seed)

        # default population initialization is a latin hypercube design, but
        # there are other population initializations possible.

        #cast this is as int to allow flaots for popsizes
#        self.num_population_members = int(popsize * self.parameter_count)
        
        #setting popsize to popsize as opposed to whats done in original code:
        #self.num_population_members = popsize * self.parameter_count

        self.num_population_members = popsize

        self.population_shape = (self.num_population_members,
                                 self.parameter_count)

        self._nfev = 0
        if init == 'latinhypercube':
            self.init_population_lhs()
        elif init == 'random':
            self.init_population_random()
        elif init == 'prior':
            self.init_pre_selected(prior)
        else:
            raise ValueError("The population initialization method must be one"
                             "of 'latinhypercube' or 'random'")

        self.disp = disp
        if self.disp:
            print("Initialized solver")
            
        self.mse_thresh = mse_thresh
        # array to hold the values of f(x) over time
        self.cost_over_time = []
        
        # hold the best solution over time
        self.best_over_time = []
        

    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator

        # Each parameter range needs to be sampled uniformly. The scaled
        # parameter range ([0, 1)) needs to be split into
        # `self.num_population_members` segments, each of which has the following
        # size:
        segsize = 1.0 / self.num_population_members

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.
        samples = (segsize * rng.random_sample(self.population_shape)

        # Offset each segment to cover the entire parameter range [0, 1)
                   + np.linspace(0., 1., self.num_population_members,
                                 endpoint=False)[:, np.newaxis])

        # Create an array for population of candidate solutions.
        self.population = np.zeros_like(samples)

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

        # reset population energies
        self.population_energies = (np.ones(self.num_population_members) *
                                    np.inf)

        # reset number of function evaluations counter
        self._nfev = 0
        

    def init_population_random(self):
        """
        Initialises the population at random.  This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.random_sample(self.population_shape)

        # reset population energies
        self.population_energies = (np.ones(self.num_population_members) *
                                    np.inf)

        # reset number of function evaluations counter
        self._nfev = 0
        
        
    def init_population_lhs_mod(self,pop_remaining):
        """
        Fills in population array with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator

        # Each parameter range needs to be sampled uniformly. The scaled
        # parameter range ([0, 1)) needs to be split into
        # `self.num_population_members` segments, each of which has the following
        # size:
        segsize = 1.0 / pop_remaining

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.
        samples = (segsize * rng.random_sample((pop_remaining,
                                                self.parameter_count))

        # Offset each segment to cover the entire parameter range [0, 1)
                   + np.linspace(0., 1., pop_remaining,
                                 endpoint=False)[:, np.newaxis])

        # Create an array for population of candidate solutions.
        pop_remaining_array = np.zeros_like(samples)

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(pop_remaining))
            pop_remaining_array[:, j] = samples[order, j]
            
        #add this to the exisiting population array with our pre-selected 
        #choices
        
        self.population = np.concatenate((self.population, 
                                          pop_remaining_array), axis=0)
        

            
    def init_pre_selected(self, list_of_solutions):
        """
        Initialises the population using the vectors provided.  This is for if
        you already have some a priori knowledge about the solution
        
        list_of_solutions: list of solutions vectors/lists
        """
        
        #turn list of solutions into an array
        self.population = np.array(list_of_solutions)
        
        #filling in the rest of the population matrix with randomized members
        if len(list_of_solutions) < self.num_population_members:
            self.init_population_lhs_mod(self.num_population_members
                                         -len(list_of_solutions))
        
        # reset population energies
        self.population_energies = (np.ones(self.num_population_members) *
                                    np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    @property
    def x(self):
        """
        The best solution from the solver

        Returns
        -------
        x : ndarray
            The best solution from the solver.
        """
        print ("returning best solution")
        return self._scale_parameters(self.population[0])

    @property
    def convergence(self):
        """
        The standard deviation of the population energies divided by their
        mean.
        """
        return (np.std(self.population_energies) /
                np.abs(np.mean(self.population_energies) + _MACHEPS))
    @profile
    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.

        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.  If `polish`
            was employed, and a lower minimum was obtained by the polishing,
            then OptimizeResult also contains the ``jac`` attribute.
        """
        nit, warning_flag = 0, False
        
        # dictionary that holds standard status messages of optimizers
        status_message = _status_message['success']

        # The population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies.
        # Although this is also done in the evolve generator it's possible
        # that someone can set maxiter=0, at which point we still want the
        # initial energies to be calculated (the following loop isn't run).
        
        #np.all checks that there are no 0's in the array 
        if self.maxiter == 0:
            if np.all(np.isinf(self.population_energies)):
                if self.disp:
                    print("Calculating initial energies when maxiter = 0")
                self._calculate_population_energies()
        
#        for i in range(self.num_population_members):
#            print(self.population[i,:])
        # do the optimisation.
        for nit in xrange(1, self.maxiter + 1):
            if self.disp:
                print("iter: ", nit, "rank", self.rank, "time", time.time())
            # evolve the population by a generation
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                status_message = _status_message['maxfev']
                break

            print("differential_evolution step %d: f(x)= %g"
                      % (nit,
                         self.population_energies[0]))
            self.cost_over_time.append(np.average(self.population_energies))
            self.best_over_time.append(np.average(self.population[0]))
            
            if nit ==1 or nit%500 == 0:
                np.save("{}/scratch_/rank{}_nit{}_cost{}.npy".format(
                        self.jobid,self.rank,nit,self.population_energies[0]), 
                self.x)
            
            
            #save populations at each iter and rank to analyze after
#            np.save("before_rank"+str(self.rank)+"iter"+str(nit), self.population)
            
            #migrate
            self.migration()
            
# should the solver terminate?
#            print("Checking if should converge")
#            convergence = self.convergence
#
#            if (self.callback and
#                    self.callback(self._scale_parameters(self.population[0]),
#                                  convergence=self.tol / convergence) is True):
#
#                warning_flag = True
#                status_message = ('callback function requested stop early '
#                                  'by returning True')
#                break
#            print("checking if tolerance level reached")
##            intol = (np.std(self.population_energies) <=
##                     self.atol +
##                     self.tol * np.abs(np.mean(self.population_energies)))
#            
#            intol = self.population_energies[0] <= self.mse_thresh
#            if warning_flag or intol:
#                print("stopping iterations")
#                break
            #print("Starting next iter")


        else:
            status_message = _status_message['maxiter']
            warning_flag = True

        DE_result = OptimizeResult(
            x=self.x,
            fun=self.population_energies[0],
            nfev=self._nfev,
            nit=nit,
            message=status_message,
            success=(warning_flag is not True))
        
        print("done iters")
        if self.polish:
            print("performing final polishing")
            result = minimize(self.func,
                              np.copy(DE_result.x),
                              method='L-BFGS-B',
                              bounds=self.limits.T,
                              args=self.args)

            self._nfev += result.nfev
            DE_result.nfev = self._nfev

            if result.fun < DE_result.fun:
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                # to keep internal state consistent
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)

        return DE_result, self.cost_over_time, self.best_over_time
    @profile
    def migration(self):
        """
        Send the best individual to the next population in the ring
        Receive the best individual from the previous population in the ring
        and replace a random individual in the current pop (not the best one)
        """
        print("RANK in migr func:", self.rank)

        pop_send = np.copy(self.population[0]) #best member from current pop
        energ_send = copy(self.population_energies[0]) #energy of best member
        
        send = [pop_send, energ_send]
        
        right = (self.rank+1)%self.size
        left = (self.rank+self.size-1)%self.size
        
#        data_recv = [np.empty(self.num_population_members, dtype=np.float64), 
#                     np.empty(1, dtype=np.float64)]
        
        data_recv = self.comm.sendrecv(send, right, source=left)
        self.replace_from_mig(data_recv[0], data_recv[1])

#        if rank == 0:
#            print("on rank 0 and sending")
#            #send to ring 1
#            comm.Send(pop_send, dest=1, tag=13)
#            comm.Send(energ_send, dest=1, tag=13)
#            
#            #receive from the last ring
#            pop_recv = np.empty(self.num_population_members, dtype=np.float64)
#            energ_recv = np.empty(1, dtype=np.float64)
#            
#            comm.Recv(energ_recv, source = size-1, tag= 33)
#            comm.Recv(pop_recv, source= size - 1, tag=13)
#            
#            #replace a random individaul with best from previous ring
#            self.replace_from_mig(pop_recv, energ_recv[0])
#            
#        elif rank == size - 1:
#            print("on rank last and sending")
#            comm.Send(pop_send, dest=0, tag=13)
#            comm.Send(energ_send, dest=0, tag=13)
#
#            pop_recv = np.empty(self.num_population_members, dtype=np.float64)
#            energ_recv = np.empty(1, dtype=np.float64)
#            
#            comm.Recv(energ_recv, source = size-2, tag= 33)
#            comm.Recv(pop_recv, source= size - 2, tag=13)
#            
#            self.replace_from_mig(pop_recv, energ_recv[0])
#        else:
#            print("on other ranks and sending")
#            comm.Send(pop_send, dest=rank +1, tag=13)
#            comm.Send(energ_send, dest=rank +1, tag=13)
#            
#            pop_recv = np.empty(self.num_population_members, dtype=np.float64)
#            energ_recv = np.empty(1, dtype=np.float64)
#            
#            comm.Recv(energ_recv, source = rank -1, tag= 33)
#            comm.Recv(pop_recv, source= rank -1, tag=13)
#            
#            self.replace_from_mig(pop_recv, energ_recv[0])
            
    def replace_from_mig(self, pop, energ):
        """
        Obtain a random integer from range(self.num_population_members) not
        including 0. Use this as the index to replace the population member
        with the one you got from migration
        """
        indx = self._select_samples(0,1)
        
        self.population[indx] = pop
        self.population_energies[indx] = energ
    @profile
    def _calculate_population_energies(self):
        """
        Calculate the energies of all the population members at the same time.
        Puts the best member in first place. Useful if the population has just
        been initialised.
        """
        for index, candidate in enumerate(self.population):
            #if reached max number of function evaluations (nfev)
            if self._nfev > self.maxfun:
                break

            parameters = self._scale_parameters(candidate)
            self.population_energies[index] = self.func(parameters,
                                                        *self.args)
            self._nfev += 1

        minval = np.argmin(self.population_energies)

        # put the lowest energy into the best solution position.
        lowest_energy = self.population_energies[minval]
        self.population_energies[minval] = self.population_energies[0]
        self.population_energies[0] = lowest_energy

        self.population[[0, minval], :] = self.population[[minval, 0], :]

    def __iter__(self):
        return self
    @profile
    def __next__(self):
        """
        Evolve the population by a single generation

        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies
        if np.all(np.isinf(self.population_energies)):
            if self.disp:
                print("calculating initial energies")
            self._calculate_population_energies()

        if self.dither is not None: 
            self.scale = (self.random_number_generator.rand()
                          * (self.dither[1] - self.dither[0]) + self.dither[0])

        ##MPI - PARALLELIZE THIS PART AND THEN RETURN THE UPDATED LIST
        ## No need to worry race conditions?
        ## each proc gets own peice of candidates

        for candidate in range(self.num_population_members):
#            if self.disp:
#                print("looping through population members, ", candidate, "of ", 
#                  self.num_population_members)
            
            #if reached max number of function evaluations (nfev)
            if self._nfev > self.maxfun:
                raise StopIteration

            # create a trial solution
#            if self.disp:
#                print("creating trial solution")
            trial = self._mutate(candidate)

            # ensuring that it's in the range [0, 1)
            # self._ensure_constraint(trial)

            # re-initlize
            while not self._ensure_constraint(trial):
                #print("re-intialize")
                trial = self._mutate(candidate)
                

            # scale from [0, 1) to the actual parameter value
            parameters = self._scale_parameters(trial)

            # determine the energy of the objective function
            energy = self.func(parameters, *self.args)
            self._nfev += 1

            # if the energy of the trial candidate is lower than the
            # original population member then replace it
            if energy < self.population_energies[candidate]:
                self.population[candidate] = trial
                self.population_energies[candidate] = energy
                #print("kept the trial")

                # if the trial candidate also has a lower energy than the
                # best solution then replace that as well
                if energy < self.population_energies[0]:
                    self.population_energies[0] = energy
                    self.population[0] = trial
                    #print("NEW BEST")
                    
            #print("didn't keep trial")
            
        return self.x, self.population_energies[0]

    def next(self):
        """
        Evolve the population by a single generation

        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # next() is required for compatibility with Python2.7.
        return self.__next__()

    def _scale_parameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters.
        """
        var = self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2
        var_rounded = np.around([var],decimals = 0)[0].astype(int)

        return var_rounded
    

    def _unscale_parameters(self, parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5
    @profile
    def _ensure_constraint(self, trial):
        """
        make sure the parameters lie between the limits
        """
         # projection
#        for index, param in enumerate(trial):
#            if param > 1:
#                trial[index] = 1
#                print("hit the upper bound")
#            elif param < 0:
#                trial[index] = 0
#                print("hit the lower bound")
                
        # reintialization
#        for index, param in enumerate(trial):
#            if param > 1 or param < 0:
#                return False
#        return True
    
        #Reflection
        for index, param in enumerate(trial):
            # param == trial[index]
            if param > 1:
#                print('param > 1 ', param,
#                      'limit ',self.limits[1][index],
#                      'the param scaled ',
#                      self._scale_parameters(param)[index])
                
                trial[index] = self._unscale_parameters(2*self.limits[1][index]
                - (self._scale_parameters(param)[index]))[index]
                print("new upper est: ", trial[index] )
                if trial[index] > 1:
                    return False
            elif param < 0:
#                print('param <0 ', param,
#                      'limit ',self.limits[1][index],
#                       'the param scaled ',
#                       self._scale_parameters(param)[index])
                
                trial[index] = self._unscale_parameters(2*self.limits[0][index]
                - (self._scale_parameters(param)[index]))[index]
#                print("new lower est: ", trial[index] )
                if trial[index] < 0:
                    return False
                
        return True
    @profile
    def _mutate(self, candidate):
        """
        create a trial vector based on a mutation strategy
        """
        
        delta = np.abs((np.average(self.population_energies)/self.population_energies[0]) -1)
        #print(self.population_energies, "DELTA ", delta)
        
        if delta > self.thresh:
            self.strategy = self.strategy1
#            print("using strategy1: ", self.strategy)
        else:
            self.strategy = self.strategy2
#            print("using strategy2: ", self.strategy)
        
        if self.strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[self.strategy])
        elif self.strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[self.strategy])
            
        #trial is a copy of one of the current solution vector in the population
        trial = np.copy(self.population[candidate])

        rng = self.random_number_generator

        #choosing random place in vector solution vector - fill_point
        fill_point = rng.randint(0, self.parameter_count)

        #not sure what this if statement does
        if (self.strategy == 'randtobest1exp' or
                self.strategy == 'randtobest1bin'):
            bprime = self.mutation_func(candidate,
                                        self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))

        
        """ 
        self._binomial includes:
            _binomial = {'best1bin': '_best1',
                 'randtobest1bin': '_randtobest1',
                 'best2bin': '_best2',
                 'rand2bin': '_rand2',
                 'rand1bin': '_rand1'}
        """
        if self.strategy in self._binomial:
#            if self.disp:
#                print("mutating trial for binomial methods")

            crossovers = rng.rand(self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            # the last one is always from the bprime vector for binomial
            # If you fill in modulo with a loop you have to set the last one to
            # true. If you don't use a loop then you can have any random entry
            # be True.
            crossovers[fill_point] = True
#            if self.disp:
#                print("performin crossovers in trial vector")
            trial = np.where(crossovers, bprime, trial)
            return trial
        
        
        #self._exponential includes: {'best1exp': '_best1', 'rand1exp': '_rand1',
        #               'randtobest1exp': '_randtobest1', 'best2exp': '_best2',
        #                'rand2exp': '_rand2'}
        elif self.strategy in self._exponential:
            i = 0
#            if self.disp:
#                print("mutating trial for exponential methods")
            while (i < self.parameter_count and
                   rng.rand() < self.cross_over_probability):

                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1

            return trial

    def _best1(self, samples):
        """
        best1bin, best1exp
        """
        r0, r1 = samples[:2]
        return (self.population[0] + self.scale *
                (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        """
        rand1bin, rand1exp
        """
        r0, r1, r2 = samples[:3]
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2]))

    def _randtobest1(self, candidate, samples):
        """
        randtobest1bin, randtobest1exp
        """
        r0, r1 = samples[:2]
        bprime = np.copy(self.population[candidate])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r0] -
                                self.population[r1])
        return bprime

    def _best2(self, samples):
        """
        best2bin, best2exp
        """
        r0, r1, r2, r3 = samples[:4]
        bprime = (self.population[0] + self.scale *
                  (self.population[r0] + self.population[r1] -
                   self.population[r2] - self.population[r3]))

        return bprime

    def _rand2(self, samples):
        """
        rand2bin, rand2exp
        """
        r0, r1, r2, r3, r4 = samples
        bprime = (self.population[r0] + self.scale *
                  (self.population[r1] + self.population[r2] -
                   self.population[r3] - self.population[r4]))

        return bprime

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(self.num_population_members),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs

