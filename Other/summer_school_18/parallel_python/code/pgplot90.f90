!
! pgplot90 - Fortran 90 interface for pgplot library, so return types
!            and arguments are well defined and checked by the compiler.
!
! Note: API of the routines was taken, literally, from
!  https://www.cse.iitb.ac.in/~cs101/2003.2/resources/notesPgplot/subroutines.html
! Only a few of these have tested. Use at your own risk!
!
! Ramses van Zon
! SciNetHPC, 2016
!
module pgplot90
  interface
     !PGARRO -- draw an arrow
     SUBROUTINE PGARRO (X1, Y1, X2, Y2)
       REAL X1, Y1, X2, Y2
     END SUBROUTINE PGARRO
     !PGASK -- control new page prompting
     SUBROUTINE PGASK (FLAG)
       LOGICAL FLAG
     END SUBROUTINE PGASK
     !PGBAND -- read cursor position, with anchor
     INTEGER FUNCTION PGBAND (MODE, POSN, XREF, YREF, X, Y, CH)
       INTEGER MODE, POSN
       REAL XREF, YREF, X, Y
       CHARACTER*(*) CH
     END FUNCTION PGBAND
     !PGBBUF -- begin batch of output (buffer)
     SUBROUTINE PGBBUF()
     END SUBROUTINE PGBBUF
     !PGBEG -- begin PGPLOT, open output device
     INTEGER FUNCTION PGBEG (UNIT, FILE, NXSUB, NYSUB)
       INTEGER       UNIT
       CHARACTER*(*) FILE
       INTEGER       NXSUB, NYSUB
     END FUNCTION PGBEG
     !PGBIN -- histogram of binned data
     SUBROUTINE PGBIN (NBIN, X, DATA, CENTER)
       INTEGER NBIN
       REAL X(*), DATA(*)
       LOGICAL CENTER
     END SUBROUTINE PGBIN
     !PGBOX -- draw labeled frame around viewport
     SUBROUTINE PGBOX (XOPT, XTICK, NXSUB, YOPT, YTICK, NYSUB)
       CHARACTER*(*) XOPT, YOPT
       REAL XTICK, YTICK
       INTEGER NXSUB, NYSUB
     END SUBROUTINE PGBOX
     !PGCIRC -- draw a filled or outline circle
     SUBROUTINE PGCIRC (XCENT, YCENT, RADIUS)
       REAL XCENT, YCENT, RADIUS
     END SUBROUTINE PGCIRC
     !PGCONB -- contour map of a 2D data array, with blanking
     SUBROUTINE PGCONB (A, IDIM, JDIM, I1, I2, J1, J2, C, NC, TR, BLANK)
       INTEGER IDIM, JDIM, I1, I2, J1, J2, NC
       REAL    A(IDIM,JDIM), C(*), TR(6), BLANK
     END SUBROUTINE PGCONB
     !PGCONL -- label contour map of a 2D data array   
     SUBROUTINE PGCONL (A, IDIM, JDIM, I1, I2, J1, J2, C, TR, LABEL, INTVAL, MININT)
       INTEGER IDIM, JDIM, I1, J1, I2, J2, INTVAL, MININT
       REAL A(IDIM,JDIM), C, TR(6)
       CHARACTER*(*) LABEL
     END SUBROUTINE PGCONL
     !PGCONS -- contour map of a 2D data array (fast algorithm)
     SUBROUTINE PGCONS (A, IDIM, JDIM, I1, I2, J1, J2, C, NC, TR)
       INTEGER IDIM, JDIM, I1, I2, J1, J2, NC
       REAL    A(IDIM,JDIM), C(*), TR(6)
     END SUBROUTINE PGCONS
     !PGCONT -- contour map of a 2D data array (contour-following)
     SUBROUTINE PGCONT (A, IDIM, JDIM, I1, I2, J1, J2, C, NC, TR)
       INTEGER IDIM, JDIM, I1, J1, I2, J2, NC
       REAL A(IDIM,JDIM), C(*), TR(6)
     END SUBROUTINE PGCONT
     !PGCONX -- contour map of a 2D data array (non-rectangular)
     SUBROUTINE PGCONX (A, IDIM, JDIM, I1, I2, J1, J2, C, NC, PLOT)
       INTEGER  IDIM, JDIM, I1, J1, I2, J2, NC
       REAL     A(IDIM,JDIM), C(*)
     END SUBROUTINE PGCONX
     !PGCTAB -- install the color table to be used by PGIMAG
     SUBROUTINE PGCTAB(L, R, G, B, NC, CONTRA, BRIGHT)
       INTEGER NC
       REAL    L(NC), R(NC), G(NC), B(NC), CONTRA, BRIGHT
     END SUBROUTINE PGCTAB
     !PGCURS -- read cursor position
     INTEGER FUNCTION PGCURS (X, Y, CH)
       REAL X, Y
       CHARACTER*(*) CH
     END FUNCTION PGCURS
     !PGDRAW -- draw a line from the current pen position to a point
     SUBROUTINE PGDRAW (X, Y)
       REAL X, Y
     END SUBROUTINE PGDRAW
     !PGEBUF -- end batch of output (buffer)
     SUBROUTINE PGEBUF()
     END SUBROUTINE PGEBUF
     !PGEND -- terminate PGPLOT
     SUBROUTINE PGEND()
     END SUBROUTINE PGEND
     !PGENV -- set window and viewport and draw labeled frame
     SUBROUTINE PGENV (XMIN, XMAX, YMIN, YMAX, JUST, AXIS)
       REAL XMIN, XMAX, YMIN, YMAX
       INTEGER JUST, AXIS
     END SUBROUTINE PGENV
     !PGERAS -- erase all graphics from current page
     SUBROUTINE PGERAS()
     END SUBROUTINE PGERAS
     !PGERRB -- horizontal or vertical error bar
     SUBROUTINE PGERRB (DIR, N, X, Y, E, T)
       INTEGER DIR, N
       REAL X(*), Y(*), E(*)
       REAL T
     END SUBROUTINE PGERRB
     !PGERRX -- horizontal error bar
     SUBROUTINE PGERRX (N, X1, X2, Y, T)
       INTEGER N
       REAL X1(*), X2(*), Y(*)
       REAL T
     END SUBROUTINE PGERRX
     !PGERRY -- vertical error bar
     SUBROUTINE PGERRY (N, X, Y1, Y2, T)
       INTEGER N
       REAL X(*), Y1(*), Y2(*)
       REAL T
     END SUBROUTINE PGERRY
     !PGETXT -- erase text from graphics display
     SUBROUTINE PGETXT()
     END SUBROUTINE PGETXT
     !PGFUNT -- function defined by X = F(T), Y = G(T)
     SUBROUTINE PGFUNT (FX, FY, N, TMIN, TMAX, PGFLAG)
       REAL FX, FY
       EXTERNAL FX, FY
       INTEGER N
       REAL TMIN, TMAX
       INTEGER PGFLAG
     END SUBROUTINE PGFUNT
     !PGFUNX -- function defined by Y = F(X)
     SUBROUTINE PGFUNX (FY, N, XMIN, XMAX, PGFLAG)
       REAL FY
       EXTERNAL FY
       INTEGER N
       REAL XMIN, XMAX
       INTEGER PGFLAg
     END SUBROUTINE PGFUNX
     !PGFUNY -- function defined by X = F(Y)
     SUBROUTINE PGFUNY (FX, N, YMIN, YMAX, PGFLAG)
       REAL    FX
       EXTERNAL FX
       INTEGER N
       REAL    YMIN, YMAX
       INTEGER PGFLAG
     END SUBROUTINE PGFUNY
     !PGGRAY -- gray-scale map of a 2D data array
     SUBROUTINE PGGRAY (A, IDIM, JDIM, I1, I2, J1, J2, FG, BG, TR)
       INTEGER IDIM, JDIM, I1, I2, J1, J2
       REAL    A(IDIM,JDIM), FG, BG, TR(6)
     END SUBROUTINE PGGRAY
     !PGHI2D -- cross-sections through a 2D data array
     SUBROUTINE PGHI2D (DATA,NXV,NYV,IX1,IX2,IY1,IY2,X,IOFF,BIAS,CENTER,YLIMS)
       INTEGER NXV, NYV, IX1, IX2, IY1, IY2
       REAL    DATA(NXV,NYV)
       REAL    X(IX2-IX1+1), YLIMS(IX2-IX1+1)
       INTEGER IOFF
       REAL    BIAS
       LOGICAL CENTER
     END SUBROUTINE PGHI2D
     !PGHIST -- histogram of unbinned data
     SUBROUTINE PGHIST(N, DATA, DATMIN, DATMAX, NBIN, PGFLAG)
       INTEGER N
       REAL    DATA(*)
       REAL    DATMIN, DATMAX
       INTEGER NBIN, PGFLAG
     END SUBROUTINE PGHIST
     !PGIDEN -- write username, date, and time at bottom of plot
     SUBROUTINE PGIDEN()
     END SUBROUTINE PGIDEN
     !PGIMAG -- color image from a 2D data array
     SUBROUTINE PGIMAG (A, IDIM, JDIM, I1, I2, J1, J2, A1, A2, TR)
       INTEGER IDIM, JDIM, I1, I2, J1, J2
       REAL    A(IDIM,JDIM), A1, A2, TR(6)
     END SUBROUTINE PGIMAG
     !PGLAB -- write labels for x-axis, y-axis, and top of plot
     SUBROUTINE PGLAB (XLBL, YLBL, TOPLBL)
       CHARACTER*(*) XLBL, YLBL, TOPLBL
     END SUBROUTINE PGLAB
     !PGLCUR -- draw a line using the cursor
     SUBROUTINE PGLCUR (MAXPT, NPT, X, Y)
       INTEGER MAXPT, NPT
       REAL    X(*), Y(*)
     END SUBROUTINE PGLCUR
     !PGLDEV -- list available device types
     SUBROUTINE PGLDEV()
     END SUBROUTINE PGLDEV
     !PGLEN -- find length of a string in a variety of units
     SUBROUTINE PGLEN (UNITS, STRING, XL, YL)
       REAL XL, YL
       INTEGER UNITS
       CHARACTER*(*) STRING
     END SUBROUTINE PGLEN
     !PGLINE -- draw a polyline (curve defined by line-segments)
     SUBROUTINE PGLINE (N, XPTS, YPTS)
       INTEGER  N
       REAL     XPTS(*), YPTS(*)
     END SUBROUTINE PGLINE
     !PGMOVE -- move pen (change current pen position)
     SUBROUTINE PGMOVE (X, Y)
       REAL X, Y
     END SUBROUTINE PGMOVE
     !PGMTXT -- write text at position relative to viewport
     SUBROUTINE PGMTXT (SIDE, DISP, COORD, FJUST, TEXT)
       CHARACTER*(*) SIDE, TEXT
       REAL DISP, COORD, FJUST
     END SUBROUTINE PGMTXT
     !PGNCUR -- mark a set of points using the cursor
     SUBROUTINE PGNCUR (MAXPT, NPT, X, Y, SYMBOL)
       INTEGER MAXPT, NPT
       REAL    X(*), Y(*)
       INTEGER SYMBOL
     END SUBROUTINE PGNCUR
     !PGNUMB -- convert a number into a plottable character string
     SUBROUTINE PGNUMB (MM, PP, FORM, STRING, NC)
       INTEGER MM, PP, FORM
       CHARACTER*(*) STRING
       INTEGER NC
     END SUBROUTINE PGNUMB
     !PGOLIN -- mark a set of points using the cursor
     SUBROUTINE PGOLIN (MAXPT, NPT, X, Y, SYMBOL)
       INTEGER MAXPT, NPT
       REAL    X(*), Y(*)
       INTEGER SYMBOL
     END SUBROUTINE PGOLIN
     !PGPAGE -- advance to new page
     SUBROUTINE PGPAGE()
     END SUBROUTINE PGPAGE
     !PGPANL -- switch to a different panel on the view surface
     SUBROUTINE PGPANL(IX, IY)
       INTEGER IX, IY
     END SUBROUTINE PGPANL
     !PGPAP -- change the size of the view surface
     SUBROUTINE PGPAP (WIDTH, ASPECT)
       REAL WIDTH, ASPECT
     END SUBROUTINE PGPAP
     !PGPIXL -- draw pixels
     SUBROUTINE PGPIXL (IA,IDIM,JDIM,I1,I2,J1,J2,X1,X2,Y1,Y2)
       INTEGER IDIM, JDIM, I1, I2, J1, J2
       INTEGER IA(IDIM,JDIM)
       REAL    X1, X2, Y1, Y2
     END SUBROUTINE PGPIXL
     !PGPNTS -- draw one or more graph markers, not all the same
     SUBROUTINE PGPNTS (N, X, Y, SYMBOL, NS)
       INTEGER N, NS
       REAL X(*), Y(*)
       INTEGER SYMBOL(*)
     END SUBROUTINE PGPNTS
     !PGPOLY -- fill a polygonal area with shading
     SUBROUTINE PGPOLY (N, XPTS, YPTS)
       INTEGER N
       REAL XPTS(*), YPTS(*)
     END SUBROUTINE PGPOLY
     !PGPT -- draw one or more graph markers
     SUBROUTINE PGPT (N, XPTS, YPTS, SYMBOL)
       INTEGER N
       REAL XPTS(*), YPTS(*)
       INTEGER SYMBOL
     END SUBROUTINE PGPT
     !PGPTXT -- write text at arbitrary position and angle
     SUBROUTINE PGPTXT (X, Y, ANGLE, FJUST, TEXT)
       REAL X, Y, ANGLE, FJUST
       CHARACTER*(*) TEXT
     END SUBROUTINE PGPTXT
     !PGQAH -- inquire arrow-head style
     SUBROUTINE PGQAH (FS, ANGLE, VENT)
       INTEGER  FS
       REAL ANGLE, VENT
     END SUBROUTINE PGQAH
     !PGQCF -- inquire character font
     SUBROUTINE PGQCF (FONT)
       INTEGER  FONT
     END SUBROUTINE PGQCF
     !PGQCH -- inquire character height
     SUBROUTINE PGQCH (SIZE)
       REAL SIZE
     END SUBROUTINE PGQCH
     !PGQCI -- inquire color index
     SUBROUTINE PGQCI (CI)
       INTEGER  CI
     END SUBROUTINE PGQCI
     !PGQCIR -- inquire color index range
     SUBROUTINE PGQCIR(ICILO, ICIHI)
       INTEGER   ICILO, ICIHI
     END SUBROUTINE PGQCIR
     !PGQCOL -- inquire color capability
     SUBROUTINE PGQCOL (CI1, CI2)
       INTEGER  CI1, CI2
     END SUBROUTINE PGQCOL
     !PGQCR -- inquire color representation
     SUBROUTINE PGQCR (CI, CR, CG, CB)
       INTEGER CI
       REAL    CR, CG, CB
     END SUBROUTINE PGQCR
     !PGQCS -- inquire character height in a variety of units
     SUBROUTINE PGQCS(UNITS, XCH, YCH)
       INTEGER UNITS
       REAL XCH, YCH
     END SUBROUTINE PGQCS
     !PGQFS -- inquire fill-area style
     SUBROUTINE PGQFS (FS)
       INTEGER  FS
     END SUBROUTINE PGQFS
     !PGQHS -- inquire hatching style
     SUBROUTINE PGQHS (ANGLE, SEPN, PHASE)
       REAL ANGLE, SEPN, PHASE
     END SUBROUTINE PGQHS
     !PGQINF -- inquire PGPLOT general information
     SUBROUTINE PGQINF (ITEM, VALUE, LENGTH)
       CHARACTER*(*) ITEM, VALUE
       INTEGER LENGTH
     END SUBROUTINE PGQINF
     !PGQITF -- inquire image transfer function
     SUBROUTINE PGQITF (ITF)
       INTEGER  ITF
     END SUBROUTINE PGQITF
     !PGQLS -- inquire line style
     SUBROUTINE PGQLS (LS)
       INTEGER  LS
     END SUBROUTINE PGQLS
     !PGQLW -- inquire line width
     SUBROUTINE PGQLW (LW)
       INTEGER  LW
     END SUBROUTINE PGQLW
     !PGQPOS -- inquire current pen position
     SUBROUTINE PGQPOS (X, Y)
       REAL X, Y
     END SUBROUTINE PGQPOS
     !PGQTBG -- inquire text background color index
     SUBROUTINE PGQTBG (TBCI)
       INTEGER  TBCI
     END SUBROUTINE PGQTBG
     !PGQTXT -- find bounding box of text string
     SUBROUTINE PGQTXT (X, Y, ANGLE, FJUST, TEXT, XBOX, YBOX)
       REAL X, Y, ANGLE, FJUST
       CHARACTER*(*) TEXT
       REAL XBOX(4), YBOX(4)
     END SUBROUTINE PGQTXT
     !PGQVP -- inquire viewport size and position
     SUBROUTINE PGQVP (UNITS, X1, X2, Y1, Y2)
       INTEGER UNITS
       REAL    X1, X2, Y1, Y2
     END SUBROUTINE PGQVP
     !PGQVSZ -- find the window defined by the full view surface
     SUBROUTINE PGQVSZ (UNITS, X1, X2, Y1, Y2)
       INTEGER UNITS
       REAL X1, X2, Y1, Y2
     END SUBROUTINE PGQVSZ
     !PGQWIN -- inquire window boundary coordinates
     SUBROUTINE PGQWIN (X1, X2, Y1, Y2)
       REAL X1, X2, Y1, Y2
     END SUBROUTINE PGQWIN
     !PGRECT -- draw a rectangle, using fill-area attributes
     SUBROUTINE PGRECT (X1, X2, Y1, Y2)
       REAL X1, X2, Y1, Y2
     END SUBROUTINE PGRECT
     !PGRND -- find the smallest `round' number greater than x
     REAL FUNCTION PGRND (X, NSUB)
       REAL X
       INTEGER NSUB
     END FUNCTION PGRND
     !PGRNGE -- choose axis limits
     SUBROUTINE PGRNGE (X1, X2, XLO, XHI)
       REAL X1, X2, XLO, XHI
     END SUBROUTINE PGRNGE
     !PGSAH -- set arrow-head style
     SUBROUTINE PGSAH (FS, ANGLE, VENT)
       INTEGER  FS
       REAL ANGLE, VENT
     END SUBROUTINE PGSAH
     !PGSAVE -- save PGPLOT attributes
     SUBROUTINE PGSAVE()
     END SUBROUTINE PGSAVE
     !PGUNSA -- restore PGPLOT attributes
     !ENTRY PGUNSA()?
     !PGSCF -- set character font
     SUBROUTINE PGSCF (FONT)
       INTEGER  FONT
     END SUBROUTINE PGSCF
     !PGSCH -- set character height
     SUBROUTINE PGSCH (SIZE)
       REAL SIZE
     END SUBROUTINE PGSCH
     !PGSCI -- set color index
     SUBROUTINE PGSCI (CI)
       INTEGER  CI
     END SUBROUTINE PGSCI
     !PGSCIR -- set color index range
     SUBROUTINE PGSCIR(ICILO, ICIHI)
       INTEGER   ICILO, ICIHI
     END SUBROUTINE PGSCIR
     !PGSCR -- set color representation
     SUBROUTINE PGSCR (CI, CR, CG, CB)
       INTEGER CI
       REAL    CR, CG, CB
     END SUBROUTINE PGSCR
     !PGSCRN -- set color representation by name
     SUBROUTINE PGSCRN(CI, NAME, IER)
       INTEGER CI
       CHARACTER*(*) NAME
       INTEGER IER
     END SUBROUTINE PGSCRN
     !PGSFS -- set fill-area style
     SUBROUTINE PGSFS (FS)
       INTEGER  FS
     END SUBROUTINE PGSFS
     !PGSHLS -- set color representation using HLS system
     SUBROUTINE PGSHLS (CI, CH, CL, CS)
       INTEGER CI
       REAL    CH, CL, CS
     END SUBROUTINE PGSHLS
     !PGSHS -- set hatching style
     SUBROUTINE PGSHS (ANGLE, SEPN, PHASE)
       REAL ANGLE, SEPN, PHASE
     END SUBROUTINE PGSHS
     !PGSITF -- set image transfer function
     SUBROUTINE PGSITF (ITF)
       INTEGER  ITF
     END SUBROUTINE PGSITF
     !PGSLS -- set line style
     SUBROUTINE PGSLS (LS)
       INTEGER  LS
     END SUBROUTINE PGSLS
     !PGSLW -- set line width
     SUBROUTINE PGSLW (LW)
       INTEGER  LW
     END SUBROUTINE PGSLW
     !PGSTBG -- set text background color index
     SUBROUTINE PGSTBG (TBCI)
       INTEGER  TBCI
     END SUBROUTINE PGSTBG
     !PGSUBP -- subdivide view surface into panels
     SUBROUTINE PGSUBP (NXSUB, NYSUB)
       INTEGER NXSUB, NYSUB
     END SUBROUTINE PGSUBP
     !PGSVP -- set viewport (normalized device coordinates)
     SUBROUTINE PGSVP (XLEFT, XRIGHT, YBOT, YTOP)
       REAL XLEFT, XRIGHT, YBOT, YTOP
     END SUBROUTINE PGSVP
     !PGSWIN -- set window
     SUBROUTINE PGSWIN (X1, X2, Y1, Y2)
       REAL X1, X2, Y1, Y2
     END SUBROUTINE PGSWIN
     !PGTBOX -- draw frame and write (DD) HH MM SS.S labelling
     SUBROUTINE PGTBOX (XOPT, XTICK, NXSUB, YOPT, YTICK, NYSUB)
       REAL XTICK, YTICK
       INTEGER NXSUB, NYSUB
       CHARACTER XOPT*(*), YOPT*(*)
     END SUBROUTINE PGTBOX
     !PGTEXT -- write text (horizontal, left-justified)
     SUBROUTINE PGTEXT (X, Y, TEXT)
       REAL X, Y
       CHARACTER*(*) TEXT
     END SUBROUTINE PGTEXT
     !PGUPDT -- update display
     SUBROUTINE PGUPDT()
     END SUBROUTINE PGUPDT
     !PGVECT -- vector map of a 2D data array, with blanking
     SUBROUTINE PGVECT (A,B,IDIM,JDIM,I1,I2,J1,J2,C,NC,TR,BLANK)
       INTEGER IDIM, JDIM, I1, I2, J1, J2, NC
       REAL    A(IDIM,JDIM), B(IDIM, JDIM), TR(6), BLANK, C
     END SUBROUTINE PGVECT
     !PGVSIZ -- set viewport (inches)
     SUBROUTINE PGVSIZ (XLEFT, XRIGHT, YBOT, YTOP)
       REAL XLEFT, XRIGHT, YBOT, YTOP
     END SUBROUTINE PGVSIZ
     !PGVSTD -- set standard (default) viewport
     SUBROUTINE PGVSTD()
     END SUBROUTINE PGVSTD
     !PGWEDG -- annotate an image plot with a wedge
     SUBROUTINE PGWEDG(SIDE, DISP, WIDTH, FG, BG, LABEL)
       CHARACTER *(*) SIDE,LABEL
       REAL DISP, WIDTH, FG, BG
     END SUBROUTINE PGWEDG
     !PGWNAD -- set window and adjust viewport to same aspect ratio
     SUBROUTINE PGWNAD (X1, X2, Y1, Y2)
       REAL X1, X2, Y1, Y2
     END SUBROUTINE PGWNAD
     !PGADVANCE -- non-standard alias for PGPAGE
     SUBROUTINE PGADVANCE()
     END SUBROUTINE PGADVANCE
     !PGBEGIN -- non-standard alias for PGBEG
     INTEGER FUNCTION PGBEGIN (UNIT, FILE, NXSUB, NYSUB)
       INTEGER       UNIT
       CHARACTER*(*) FILE
       INTEGER       NXSUB, NYSUB
     END FUNCTION PGBEGIN
     !PGCURSE -- non-standard alias for PGCURS
     INTEGER FUNCTION PGCURSE (X, Y, CH)
       REAL X, Y
       CHARACTER*1 CH
     END FUNCTION PGCURSE
     !PGLABEL -- non-standard alias for PGLAB
     SUBROUTINE PGLABEL (XLBL, YLBL, TOPLBL)
       CHARACTER*(*) XLBL, YLBL, TOPLBL
     END SUBROUTINE PGLABEL
     !PGMTEXT -- non-standard alias for PGMTXT
     SUBROUTINE PGMTEXT (SIDE, DISP, COORD, FJUST, TEXT)
       CHARACTER*(*) SIDE, TEXT
       REAL DISP, COORD, FJUST
     END SUBROUTINE PGMTEXT
     !PGNCURSE -- non-standard alias for PGNCUR
     SUBROUTINE PGNCURSE (MAXPT, NPT, X, Y, SYMBOL)
       INTEGER MAXPT, NPT
       REAL    X(*), Y(*)
       INTEGER SYMBOL
     END SUBROUTINE PGNCURSE
     !PGPAPER -- non-standard alias for PGPAP
     SUBROUTINE PGPAPER (WIDTH, ASPECT)
       REAL WIDTH, ASPECT
     END SUBROUTINE PGPAPER
     !PGPOINT -- non-standard alias for PGPT
     SUBROUTINE PGPOINT (N, XPTS, YPTS, SYMBOL)
       INTEGER N
       REAL XPTS(*), YPTS(*)
       INTEGER SYMBOL
     END SUBROUTINE PGPOINT
     !PGPTEXT -- non-standard alias for PGPTXT
     SUBROUTINE PGPTEXT (X, Y, ANGLE, FJUST, TEXT)
       REAL X, Y, ANGLE, FJUST
       CHARACTER*(*) TEXT
     END SUBROUTINE PGPTEXT
     !PGVPORT -- non-standard alias for PGSVP
     SUBROUTINE PGVPORT (XLEFT, XRIGHT, YBOT, YTOP)
       REAL XLEFT, XRIGHT, YBOT, YTOP
     END SUBROUTINE PGVPORT
     !PGVSIZE -- non-standard alias for PGVSIZ
     SUBROUTINE PGVSIZE (XLEFT, XRIGHT, YBOT, YTOP)
       REAL XLEFT, XRIGHT, YBOT, YTOP
     END SUBROUTINE PGVSIZE
     !PGVSTAND -- non-standard alias for PGVSTD
     SUBROUTINE PGVSTAND()
     END SUBROUTINE PGVSTAND
     !PGWINDOW -- non-standard alias for PGSWIN
     SUBROUTINE PGWINDOW (X1, X2, Y1, Y2)
       REAL X1, X2, Y1, Y2
     END SUBROUTINE PGWINDOW
     !
  end interface
end module pgplot90
