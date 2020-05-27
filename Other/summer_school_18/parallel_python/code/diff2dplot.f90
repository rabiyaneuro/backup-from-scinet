!
! diff2dplot.f90 - Plots a 2d density to screen (Fortran 90 version, uses pgplot)
!
! Ramses van Zon
! SciNetHPC, 2016
!
module diff2dplot
use pgplot90
contains  
subroutine plotdens(dens, x1, x2, first)
    real*8, dimension(:,:), intent(in) :: dens
    real*8,                 intent(in) :: x1, x2
    logical,                intent(in) :: first
    integer, save :: npnts
    real*4,  save, allocatable :: rdens(:,:)
    real*4,  save :: rx1, rx2, vmin, vmax, tr(6)
    logical, save :: xpresent
    integer*4     :: pgbeg_result
    if (first .eqv. .true.) then
       pgbeg_result = pgbeg(0, "/xwindow", 1, 1)
        if (pgbeg_result == 1) then
            xpresent = .true.
        else
            xpresent = .false.
        endif
        if (xpresent .eqv. .true.) then
            call pgask(.false.)
            rx1 = x1
            rx2 = x2
            npnts = size(dens,1)
            call pgenv(rx1, rx2, rx1, rx2, 1, 0)
            !call pgctab(&
            !     [0.0, .143, .286, .428, .571, .714, .857, 1.0], & 
            !     [0.0, .718, .316, .000, .000, .727, .847, 1.0], &
            !     [0.0, .000, .316, .559, .592, .727, .057, 1.0], &
            !     [0.0, .718, .991, .559, .000, .000, .057, 1.0], &
            !     8, 1.0, 0.5)
            ! The above color scheme is really nice and smooth, but
            ! the following matches matplotlib's 'spectral' color map
            ! better:
            call pgctab(&
                 [0.0, .050, .180, .280, .500, .700, .840, 1.0], & 
                 [0.0, .718, .000, .000, .000, .787, .847, 1.0], &
                 [0.0, .000, .000, .559, .592, .787, .057, 1.0], &
                 [0.0, .718, .991, .559, .000, .000, .057, 1.0], &
                 8, 1.0, 0.5)
            vmin = minval(dens)
            vmax = maxval(dens)
            tr = [rx1,(rx2-rx1)/npnts,0.0,rx1,0.0,(rx2-rx1)/npnts]
            allocate(rdens(npnts,npnts))
        endif
    endif
    if (xpresent .eqv. .true.) then 
        rdens = dens ! copy as a real*4 array
        call pgimag(rdens, npnts, npnts, 1,npnts, 1,npnts, vmin, vmax, tr)
        if (first .eqv. .true.) then
            call pgwedg('RI', 1.5, 3., vmin, vmax, ' ')
         endif
    else
        write (0,*) "WARNING: No DISPLAY found for graphics" 
    endif
end subroutine plotdens
!  
subroutine endplot()
    call pgend()
end subroutine endplot
end module diff2dplot
