!
! diff2d.f90 - Simulates two-dimensional diffusion on a square domain
!              Fortran 90 version
!
! Ramses van Zon
! SciNetHPC, 2016
!
program exdiff2d
    use diff2dplot ! provides plotdens and endplot
    implicit none
    ! Include file sets parameters D, x1, x2, runtime, dx, outtime, and graphics:
    logical, parameter :: False=.false., True=.true.
    real*8             :: D, x1, x2, runtime, dx, outtime, dt, simtime, a, b
    logical            :: graphics
    integer            :: nrows, ncols, npnts, nsteps, nper, i, j, s
    real*8, allocatable, dimension(:)             :: x
    real*8, allocatable, dimension(:,:)           :: laplacian
    real*8, pointer, dimension(:,:)               :: dens, densnext, tmp
    real*8, target, allocatable, dimension(:,:,:) :: densbuffer
    include 'diff2dparams.py'
    ! Compute derived parameters.
    nrows     = floor((x2-x1)/dx) ! number of x points
    ncols     = nrows           ! number of y points, same as x in this case.
    npnts     = nrows + 2       ! number of x points including boundary points
    dx        = (x2-x1)/nrows   ! recomputed (previous dx may not fit in [x1,x2])
    dt        = 0.25*dx**2/D    ! time step size (edge of stability)
    nsteps    = int(runtime/dt) ! number of steps of that size to reach runtime
    nper      = int(outtime/dt) ! how many step s between snapshots
    if (nper == 0) then
       nper = 1
    endif
    ! Allocate arrays.
    allocate(x(npnts))
    allocate(densbuffer(npnts,npnts,2))
    allocate(laplacian(npnts,npnts))
    dens => densbuffer(:,:,1)
    densnext => densbuffer(:,:,2)
    ! Initialize.
    do i=1,npnts 
       x(i) = x1+((i-2)*(x2-x1))/nrows ! x values (also y values)
    enddo
    dens=0.0
    densnext=0.0
    simtime=0*dt
    do i=2,npnts-1
        a = 1 - abs(1 - 4*abs((x(i)-(x1+x2)/2)/(x2-x1)))
        do j=2,npnts-1
            b = 1 - abs(1 - 4*abs((x(j)-(x1+x2)/2)/(x2-x1)))
            dens(i,j) = a*b
        end do
    end do
    ! Output initial signal.
    print *, simtime
    if (graphics .eqv. .true.) then
       call plotdens(dens, x(1), x(npnts), .true.)
    endif
    ! Let temporary array hold laplacian.
    laplacian = 0.0 
    do s=1,nsteps
        ! Take one time step to produce new density.
        ! Using explicit iterations over indices instead
        !$omp parallel default(none) private(i,j) firstprivate(D,dx,dt,ncols,nrows) shared(laplacian,dens,densnext)
        !$omp do
        do j=2,ncols+1
            do i=2,nrows+1
                laplacian(i,j) = dens(i+1,j)+dens(i-1,j)+dens(i,j+1)+dens(i,j-1)-4*dens(i,j)
            end do
        end do
        !$omp end do 
        !$omp do
        do j=2,ncols+1
            do i=2,nrows+1
                densnext(i,j) = dens(i,j) + (D/dx**2)*dt*laplacian(i,j)
            end do
        end do
        !$omp end do
        !$omp end parallel
        ! Swap array pointers so t+1 becomes the new t, and update simulation time.
        tmp => dens
        dens => densnext
        densnext => tmp
        simtime = simtime + dt
        ! Plot and report time.
        if (mod(s,nper) == 0) then
            print *, simtime
            if (graphics .eqv. .true.) then
                call plotdens(dens, x(1), x(npnts), .false.)
            endif
        endif
    enddo
    if (graphics .eqv. .true.) then
       call endplot()
    endif
end program exdiff2d

