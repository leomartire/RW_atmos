c*** locked mode program ***
c input file consists of :
c    n0,iefl,tref   # layers in model(including half space) and earth
c                  flattening control variable ( > 0 applies correction),
c                  reference period for material dispersion correction
c                  (0 for none)
c    d,vp,vs,rho,qbeta,qalpha   model,d is layer thickness. model can
c                  include a one layer ocean (signalled by setting vs = 0
c                  in the top layer). half space can have any thickness
c                  assocoated with it ( 0 is ok)
c    jcom     = 0 quits ;=1 rayleigh waves ; <> 1 love waves
c    output file name
c    c1,c2,nbran1,nbran2  min and max phase velocities and min and max
c                  branch numbers. note that c1 = c2=0 causes program to
c                  choose phase velocity range for itself
c   OLD:  nsrce,npts,dt  # source depths; # points in seismogram and its sample
c                  interval (in secs)
c   NEW:  nsrce,nom,df,fo  # source depths; # frequencies and frequency interval and start frequency in Hz.
c    sdep          source depths
c    rdep          receiver depth
c

	program earthstuf

	implicit real*8 (a-h, o-z)

	include 'sizes.inc'
	include 'units.inc'

	common/m/d(lyrs),ro(lyrs),vp(lyrs),vs(lyrs),fu(lyrs),n,noc,ist,iasc
	common/m0/d0(lyrs),ro0(lyrs),vp0(lyrs),vs0(lyrs),
     &           qb0(lyrs),qa0(lyrs),n0
	common/mq/vps(lyrs),vss(lyrs)
	common/q/qb(lyrs),qa(lyrs)
	common/bran/ce(2),ke(2),de(2),ctry,ceps,um,cm,cmx,cmn
	common/bits/u,nsrce,idep(lsd),nord,tpi,sdep(lsd),ig,idisc,irdep

	character*(256) infil

c -- Arjun: excitations--
        real depth(lyrs)
        integer per_mall, intper, per_mint
        integer nusrper
        real*4, dimension(:), allocatable :: usrper
        real*4, dimension(:), allocatable :: usrom
	character*(256) outfil, outfil_exasc
c -- Arjun: excitations--

	istop = -1
	tpi   = 6.2831853071796d0
c -- Initializing infil as an empty string with 256 characters
	do i = 1,256
	  infil(i:i) = ' '
	enddo
	write(*,'(a$)') ' input file : '
	!read(*,'(a256)') infil
	CALL GETARG(1 , infil)
	open(iinf1,file = infil,status='old')
c	write(*,'(a$)') ' output dispersion file : '
c	read(*,'(a256)') infil
	write(*,'(a$)') 'Write out ascii files ? (0 for no, 1 for yes) : '
	!read(*,'(I1)') iasc
	iasc = 1
        write(*,'(a$)') 'No. of period samples at which to ensure a
     +  solution: '
        !read(*,*) nusrper
        nusrper = 1
        if (nusrper.gt.0) then
          allocate(usrper(nusrper))
          allocate(usrom(nusrper))
          write(*,*) 'Enter period samples (in ascending order) : '
          !read(*,*) (usrper(i), i=1,nusrper)
          usrper(1) = 10000
          usrom=tpi/usrper
          write(*,*) 'usrom is ', usrom
        endif
cccc -- Arjun: ascii files to be written only if iasc=1
        if(iasc.eq.1) then
	  outfil = 'disp.'//trim(infil)
	  open(iouf2,file = outfil,status='replace')
	  outfil = 'disp_vconly.'//trim(infil)
	  open(iouf3,file = outfil,status='replace')
	  outfil = 'tocomputeIO.'//trim(infil) 
          open(150,file=outfil,status='replace')
          outfil = 'eigen.'//trim(infil) 
          open(10,file=outfil,status='replace')
c	  outfil_exasc = 'excitation.'//trim(infil) 
c         open(11,file=outfil_exasc,status='replace')
c -- iouf2 is the stream to the ascii dispersion file
c -- Quentin: iouf3 is the stream to the ascii dispersion file without any derivatives
c -- 10  is the stream to the eigenfunction file
c -- 11 is the stream to the ascii excitaion file
	endif
cccc -- Arjun: ascii files
	read(iinf1,*,end = 777) n0,iefl,tref
	omref = 0.d0
	if (tref.ne.0.d0) omref = tpi/tref

	do i = 1,n0
	  read(iinf1,*) d0(i),vp0(i),vs0(i),ro0(i),qb0(i),qa0(i)
	  if (qb0(i).ne.0.d0) qb0(i) = 1.d0/qb0(i)
	  if (qa0(i).ne.0.d0) qa0(i) = 1.d0/qa0(i)
	enddo
c -- jcom is the number that declares love or rayleigh type, 2 for love, 1 for rayleigh
45	read(iinf1,*)jcom
	if (jcom.eq.0) go to 777
	if (jcom.ne.1) jcom = 2

	do i = 1,256
	  infil(i:i) = ' '
	enddo
	read(iinf1,'(a256)') infil
c -- iouf1 is the stream to the binary file
	open(iouf1,file = infil,status='replace',form='unformatted')
	read(iinf1,*) c1,c2,nbran1,nbran2
	n = n0
	do i = 1,n
	  d(i)   = d0(i)
	  ro(i)  = ro0(i)
	  vps(i) = vp0(i)
	  vss(i) = vs0(i)
	  qa(i)  = qa0(i)
	  qb(i)  = qb0(i)
	enddo
c -- nom is number of frequncies, df is the frequency increment
	read(iinf1,*) nsrce,nom,df,fo
c	read(iinf1,*) nsrce,npts,dt
c	npts = 2*nfac(npts/2)
	read(iinf1,*) (sdep(i),i = 1,nsrce)
	call shell(sdep,nsrce)

	read(iinf1,*) rdep

c  rsplit splits a layer at the receiver depth.
	call rsplit(rdep)
c  split splits a layer at the source depth.the source
	call split(rdep)

c --  Modif. ARJUN: unnecessary information previously written to ascii dispersion file
c	write(iouf2,'(/,a,i3)') ' receiver index = ',irdep
c	write(iouf2,'(/,a)') ' source indices = ' 
c
c	do k = 1,nsrce
c	  write(iouf2,'(2x,i3,1x,f10.4)') idep(k),sdep(k)
c	enddo
c -- End modif. ARJUN
ccc -- Arjun: ascii files ---------------------------------------------------------
      if(iasc.eq.1) then
	write(iouf2,'(i3,1x,a)') 3, ' no. of model parameters for part. der.'
	write(iouf2,'(i4,1x,a)') n, ' no. of layers in model'
	do i = 1,n
	  write(iouf2,'(1X,f7.3,3(1x,f10.6))') d(i),ro(i),vps(i),vss(i)
	enddo

	  write(10,'(a)') 'eigenfunction file from earthsr'
	  write(10,'(i4,1x,a)') n, ' layers in model'
          depth(1) = 0
c -- output depths instead of thicknesses
	  do i=1,n
	    depth(i+1) = depth(i) + d(i)
            !write(10,'(4(1x,f8.3))') depth(i), vss(i), ro(i), vps(i)
            write(10,'(5e15.7,5e15.7,5e15.7,5e15.7)') depth(i), vss(i), ro(i), vps(i)
	  enddo
	endif
ccc -- Arjun: ascii files --------------------------------------------------------

	call flat(jcom,iefl)
      if(iasc.eq.1) then
	write(iouf2,'(/,a)') ' flattened model '

	do i = 1,n
	  write(iouf2,'(1x,f7.3,3(1x,f10.6))') 
     &         d(i),ro(i),vps(i),vss(i)


	enddo
      endif
	noc = 1

	if (vss(1).le.0.d0) noc = 2

	cmin = 0.6d0*vss(noc)
	cmin = dmin1(cmin,vps(1))
	cmn = dmax1(c1,cmin)

	do i = 1,n
	  vss(i) = vss(i)*vss(i)
	  vps(i) = vps(i)*vps(i)
	enddo

c	trec = npts*dt
c	dom  = tpi/trec
c	nh   = npts/2
	dom = tpi*df
	nh  = nom
c -- dom is frequency increment in radians
c -- om is the maximum radial frequency; omax is om + dom.
c   the maximum frequency should be fo + (nh-1)*dom, so
c   omax = fo + nh*dom

c	omax = (nh + 1)*dom
        omax = tpi*fo + nh*dom
	om   = omax - dom
	!WRITE(*,*) ">>>>", omax, fo, df, nh, dom, tpi
	!stop 'TOTO'
	call qcor(om,omref)

	cmax = dsqrt(vs(n)) - 1.d-8
	ctst = c2
	if (ctst.le.0.d0) ctst = cmax
	cmx = dmin1(ctst,cmax)
	cmaxi = cmx
	call detk(cmaxi,om,kei,dei,jcom)
	if (nbran2.lt.0.or.nbran2.ge.kei) nbran2 = kei - 1
	nbran = max0((nbran2 - nbran1 + 1),1)

c -- Arjun: ascii files --
	if(iasc.eq.1) then
	  write(iouf2,'(i3,1x,a)') nbran, 'modes listed in file'
          write(10,'(i3,7x,a)') nbran, 'modes listed in file'
c      write(10,*) "Number of periods included : ",nperiods
	endif
c -- Arjun: ascii files --

c	write(iouf1) nsrce,npts,dt,jcom,nbran
	write(iouf1) nsrce,nom,df,f0,jcom,nbran
c	write(11,'(I2,2X,I3,2X,F6.4,2X,F6.4,2X,I2,2X,I2)') nsrce,nom,df,f0,jcom,nbran
	write(iouf1) (sdep(i),i = 1,nsrce)
c	write(11,'(2X,F6.2)') (sdep(i),i = 1,nsrce)
	nb = nbran1
c  Start with the lower mode, come back here for the next higher mode when this mode is done
30	ctry = 0.5d0*(cmn + cmx)
	ceps = 0.5d0*(cmx - ctry)
	cm = 0.d0
	write(iouf1) nb
c	write(11,'(2X,I2)') nb
c -- Arjun: eigenfunctions
      if(iasc.eq.1) then
        write(10,'(I3,7X,A)') nb, "mode number"
        per_mall = 0
	per_mint = 0
c  Doing the same loop over frequencies as below, except without the call to intrp, which means the eigenfunctions are not calculated here.
c  So this loop is only to determine how many periods per mode will be there in the eigenfunction output file
	do i = 1,nh
	  om = omax - i*dom
	  !WRITE(*,*) ">>>>", i, om/tpi, dom/tpi, omax/tpi
	  call qcor(om,omref)
	  cmax = dsqrt(vs(n)) - 1.d-8
	  ctst = c2
	  if (ctst.le.0.d0) ctst = cmax
	  cmx = dmin1(ctst,cmax)
	  call cex(om,nb,jcom,nev)
          period = tpi/om

	  if (nev.eq.1) then 
		per_mall = per_mall + 1
		if (abs(period-nint(period)).lt.0.00001) then
			per_mint = per_mint + 1
		endif
	  endif
        enddo
        !STOP 'TOTA'
	write(10,'(I3,X,I4,6X,A)') per_mint, per_mall, "period samples this mode (integral, all)"
      endif
c --  End Arjun: eigenfunctions 
        iusr = 1
	do i = 1,nh
	  om = omax - i*dom
          if (nusrper.gt.0 .and. iusr.le.nusrper) then
            if (usrom(iusr).gt.om) then
		!write(*,*) 'Found gap at ', tpi/om, ' for ', tpi/usrom(iusr)
		omuse=usrom(iusr)
                iusr=iusr+1
            else
		omuse=om
            endif
          else
            omuse=om
          endif
          !WRITE(*,*) "omuse", i, usrom(iusr)/tpi, omuse/tpi, om/tpi, omax/tpi, dom/tpi
          
	  call qcor(omuse,omref)
	  cmax = dsqrt(vs(n)) - 1.d-8
	  ctst = c2
	  if (ctst.le.0.d0) ctst = cmax
	  cmx = dmin1(ctst,cmax)
	  call cex(omuse,nb,jcom,nev)
c	  write(*,*), "Period is: ", tpi/omuse, "nev is: ", nev
	  if (nev.eq.0)go to 40
	  call intrp(omuse,dom,jcom)
	enddo
	!stop 'toto'
c -- Arjun: ascii files --
40    if(iasc.eq.1) then
	write(iouf2,905) nb
c       write(10,905) nb
      endif
c -- Arjun: ascii files --

905	format(' ******** mode number: ',i3,' done. ********')
	nb = nb + 1
	write(iouf1) istop
c       write(11,'(2X,I2)') istop
	if (nb.le.nbran2) go to 30
	close(iouf1)
	go to 45

777	close(iinf1)
        if (nusrper.gt.0) then
          deallocate(usrper)
          deallocate(usrom)
        endif
	if(iasc.eq.1) then
        write(*,'(a,a)') ' output s_d vector file is : ', outfil
	 close(iouf2)
	 close(iouf3)
	 close(10)
	 close(150)
c	 close(11)
        endif
	stop
	end



	subroutine split(rdep)
c  subroutine splits a layer at the source depth.the source
c  must be above the half space and not at the surface.
	implicit real*8(a - h,o - z)

	include 'sizes.inc'

	common/m/d(lyrs),ro(lyrs),vp(lyrs),vs(lyrs),fu(lyrs),n,noc,ist
	common/q/qb(lyrs),qa(lyrs)
	common/mq/vps(lyrs),vss(lyrs)
	common/bits/u,nsrce,idep(lsd),nord,tpi,sdep(lsd),ig,idisc,irdep

	do 10 ndp = 1,nsrce
	thk = 0.d0
	iss = 0
	if (sdep(ndp).le.0.d0) go to 10
	do 1 is = 1,n
	iss = iss + 1
	thk = thk + d(is)
    1 if (sdep(ndp).le.thk)go to 2
    2 if (sdep(ndp).eq.thk)go to 10
c  if the source is already at an interface don't do anything
c  split the layer containing the source
	splt = thk - sdep(ndp)
c  shift the model down
	is1 = iss + 1
	nsf = n - iss
	do l = 1,nsf
	  k = n + 1 - l
	  j = k + 1
	  d(j)   = d(k)
	  ro(j)  = ro(k)
	  vps(j) = vps(k)
	  vss(j) = vss(k)
	  qa(j)  = qa(k)
	  qb(j)  = qb(k)
	enddo
c  split the source layer
	d(is1)   = splt
	d(iss)   = d(iss) - splt
	ro(is1)  = ro(iss)
	vps(is1) = vps(iss)
	vss(is1) = vss(iss)
	qa(is1) = qa(iss)
	qb(is1) = qb(iss)
	n = n + 1
	if (sdep(ndp).lt.rdep) irdep = irdep + 1

10	idep(ndp) = iss + 1

	return
	end

	subroutine rsplit(rdep)
c  subroutine splits a layer at the receiver depth.
	implicit real*8(a - h,o - z)
	include 'sizes.inc'

	common/m/d(lyrs),ro(lyrs),vp(lyrs),vs(lyrs),fu(lyrs),n,noc,ist
	common/q/qb(lyrs),qa(lyrs)
	common/mq/vps(lyrs),vss(lyrs)
	common/bits/u,nsrce,idep(lsd),nord,tpi,sdep(lsd),ig,idisc,irdep

	if (rdep.le.0.d0) then
	  irdep = 1
	  return
	endif

	thk = 0.d0
	iss = 0	  
	do 1 is = 1,n
	iss = iss + 1
	thk = thk + d(is)
1	if (rdep.le.thk) go to 2
2	if (rdep.eq.thk) then
c     if the receiver is already at an interface don't do anything
	  irdep = iss + 1
	  return
	endif
	irdep = iss + 1
c  split the layer containing the source
	splt = thk - rdep
c  shift the model down
	is1 = iss + 1
	nsf = n - iss
	do l = 1,nsf
	  k = n + 1 - l
	  j = k + 1
	  d(j)   = d(k)
	  ro(j)  = ro(k)
	  vps(j) = vps(k)
	  vss(j) = vss(k)
	  qa(j)  = qa(k)
	  qb(j)  = qb(k)
	enddo
c  split the receiver layer
	d(is1)   = splt
	d(iss)   = d(iss) - splt
	ro(is1)  = ro(iss)
	vps(is1) = vps(iss)
	vss(is1) = vss(iss)
	qa(is1)  = qa(iss)
	qb(is1)  = qb(iss)

	n = n + 1

	return
	end
	  
	subroutine deriv(cc,w,ls)
c  deriv analytically calculates the rayleigh layer integrals required for the
c  group velocity and for the phase velocity derivatives.it is assumed
c  that uz,ur,tz,tr have been calculated by detray and stored in array x.
c  log derivatives are stored in array der as dc/drho,dc/dalf,dc/dbet,
c  spectra are computed at a distance of 1000km for source size 10.**27
c  cf mendiguren j.g.r. 1977 for excitation functions
	implicit real*8(a - h,o - z)
	include 'sizes.inc'
	include 'units.inc'

	common/x/x(4,lyrs),der(3,lyrs)
	common/q/qb(lyrs),qa(lyrs)
	common/bits/u,nsrce,idep(lsd),nord,tpi,sdep(lsd),ig,idisc,irdep
	common/m/d(lyrs),ro(lyrs),vp2(lyrs),vs2(lyrs),fu(lyrs),n,noc,ist,iasc
c -- new variable added by Arjun
	dimension dep(lyrs)
        integer, save :: iprev = 1
c -- end of new variable declaration

	p = 1.d0/cc
	psq = p*p
	ha = dsqrt(psq - 1.d0/vp2(ls))
	hb = dsqrt(psq - 1.d0/vs2(ls))
	c1 = (hb*x(1,ls) + p*x(2,ls))/(psq - ha*hb)
	c2 = (ha*x(2,ls) + p*x(1,ls))/(psq - ha*hb)
	c3 = ro(ls)*c1*c2*(hb/vp2(ls) + ha/vs2(ls))/(p*(ha + hb))
	t1 = ro(ls)*(ha*c1*c1 + hb*c2*c2 - 2.d0*p*c1*c2)
	t2 = 0.5d0*ro(ls)*c1*c1/(vp2(ls)*ha)
	t3 = 0.5d0*ro(ls)*c2*c2/(vs2(ls)*hb) + t2
c  si1,si2,si3 are the energy integrals
c ***  These are I1, I2, and I3 in eqn 13 and Appendix A in G&M ***
	si1 = t1 + t3
	si2 = 4.d0*vs2(ls)*psq*(t1 + c3) + t3
	si3 = 0.5d0*(si2 + si1)
	der(1,ls) = 0.5d0*(si2 - si1)
	der(2,ls) = t2
	der(3,ls) = si2 - t2
	i = ls
100	i1 = i
	i = i - 1
	r2 = 2.d0*fu(i)*p
	e4 = r2*x(1,i1) - x(4,i1)
	e1 = ro(i)*x(1,i1) - p*e4
	e2 = r2*x(2,i1) - x(3,i1)
	e3 = ro(i)*x(2,i1) - p*e2
	f4 = r2*x(1,i) - x(4,i)
	f1 = ro(i)*x(1,i) - p*f4
	f2 = r2*x(2,i) - x(3,i)
	f3 = ro(i)*x(2,i) - p*f2
	dh = 0.5d0*d(i)*w
	ha = psq - 1.d0/vp2(i)
	hb = psq - 1.d0/vs2(i)
	c1 = dh*(e1*e1 - ha*e2*e2)
	c2 = 0.5d0*(e1*e2 - f1*f2)
	c3 = dh*(e3*e3 - hb*e4*e4)
	c4 = 0.5d0*(e3*e4 - f3*f4)
	c5 = p*(e2*e4 - f2*f4)
	c6 = cc*(e1*e3 - f1*f3) - c5
	t1 = 2.d0*(c5 + c2 + c4)/ro(i)
	t2 = (c2 - c1)/(ro(i)*vp2(i)*ha)
	t3 = (c4 - c3)/(fu(i)*hb) + t2
	sj1 = t1 + t3
	sj2 = 4.d0*vs2(i)*psq*(t1 + c6/ro(i)) + t3
	si1 = si1 + sj1
	si2 = si2 + sj2
	si3 = si3 + 0.5d0*(sj2 + sj1) - (c1 + c3)/ro(i)
	der(1,i) = 0.5d0*(sj2 - sj1)
	der(2,i) = t2
	der(3,i) = sj2 - t2
	if (i.gt.noc) go to 100
	if (noc.eq.1) go to 30
c  energy integrals and partials in ocean layer
	ha = (psq - 1.d0/vp2(1))/ro(1)
	c1 = 0.5d0*d(1)*w*(x(1,noc)*x(1,noc) - ha*x(3,noc)*x(3,noc)/ro(1))
	c2 =  - 0.5d0*x(1,noc)*x(3,noc)/ro(1)
	t2 = (c2 - c1)/(ha*vp2(1))
	t1 = 2.d0*ro(1)*c2 + t2
	si1 = si1 + t1
	si2 = si2 + t2
	si3 = si3 + 0.5d0*(t1 + t2) - ro(1)*c1
	der(1,1) = 0.5d0*(t2 - t1)
	der(2,1) = t2
	der(3,1) = 0.d0
30 	u = cc*si3/si1
c  flan should be almost zero if the eigenvector is accurate
	flan = si1/si2 - 1.d0
	if (dabs(flan).ge.1.e-8) go to 999
c --- Modif. Arjun Sep 2014
c     To overcome the problem of garbage values at depths greater than the maximum depth
c     upto which modal eigenfunctions are calculated. Eigenfunction values at these
c     depths should of course be 0 but before this modification was made, garbage values
c     were written out in their place
      if (irdep>ls) then
	x(1,irdep)=0.0
	x(2,irdep)=0.0
	x(3,irdep)=0.0
	x(4,irdep)=0.0
      endif
      fnorm = 1.d0/si3
      do i = 1,ls
	do j = 1,3
	  der(j,i) = der(j,i)*fnorm
	enddo
      enddo
       per = tpi/w
c -- Arjun: output eigenfunctions --
      if(iasc.eq.1) then
	if (abs(per-nint(per)).lt.0.00001) then
	  intper=1
        else
	  intper=0
        endif
	dep(1)=0
        do ic=1,ls
	  dep(ic+1)=dep(ic)+d(ic)
c	  write(*,*) dep(ic)
	enddo
c       write(10,'(" mode=",I3," period=",F13.7," layers=",I4,6X,I2)') nord,period,ls,intper
        write(10,901) nord,per,cc,u,(w*w/(cc*si3)),ls,intper
        write(10, '(5e15.7)') (dep(ic),(x(ir,ic),ir=1,4), ic=1,ls)
        !write(*,*) 'TOTO', nord,per,ls
        write(150,902) -1, nord,per,cc,u
        
        write(150, '(5e15.7)') (dep(ic),(x(ir,ic),ir=1,4), ic=1,ls)
901     format(I3,3(F13.7,2X),e15.7,4X,I4,4X,I2)
902     format(I7,I7,3(F13.7,2X))
      endif
c -- end Arjun: output eigenfunctions --

	
c    ignore attenuation in the ocean
	q = 0.d0
	do i = noc,ls
	  q = q + der(2,i)*qa(i) + der(3,i)*qb(i)
	enddo
	gam = 0.5d0*w*q/cc
 	gb2 = w*( - x(1,irdep)/cc  +  x(4,irdep)/fu(irdep))
	gb1 = w*( (1.d0 - 2.d0*vs2(irdep)/vp2(irdep))*x(2,irdep)/cc 
     1           +  x(3,irdep)/(ro(irdep)*vp2(irdep)) )
c *** Now output all the quantities for equations 13a and 13b in G&M ***
c *** x(1,irdep) and x(2,irdep) are the same as b1(zr) and b2(zr) in eqns 13a and 13b in G&M ***
	write(iouf1) w,cc,gam,x(1,irdep),x(2,irdep),gb1,gb2
c       write(*,*) "From deriv: ls is ", ls , "& irdep is ", irdep
c	write(11,'(7(2X,F10.6))') w,cc,gam,x(1,irdep),x(2,irdep),gb1,gb2
c -- Arjun: Description of what goes into the (binary) excitations file-
c		w is frequency in radians, cc is phase velocity, 
c               gam is a measure of attenuation (0 for no attenuation)
c		
c -- Arjun: End of description
c  write out the excitations.py1,py2,py3 correspond to a,b,c in mendiguren
c *** py1,py2 and py3 are the terms in the parentheses in eqn 13a in G&M ****
c ***   py1,py2 and py3 each get multiplied by s1,s2, and s3 in equation 13 in G&M ***
c ***  x(1,z),x(2,z),x(3,z),x(4,z) are the same as b1(z),b2(z),b3(z),b4(z) in G&M ***
	fact = p*dsqrt(p*w)/(si3*15.853309d-6)
	do i = 1,nsrce
	  id = idep(i)
	  py1 = x(2,id)*p*fact
c *** py1 is the first term of eqn 13a in G&M ***
	  py2 = x(4,id)*fact/fu(id)
c *** py2 is the third term of eqn 13a in G&M ***
	  sig = ro(id)*vp2(id)
	  py3 =  - (x(3,id)/sig + p*(1.d0 - 2.d0*fu(id)/sig)*x(2,id))*fact
c *** py3 is the second term of eqn 13a in G&M ***
	  if (id.gt.ls) then
	    py1 = 0.d0
	    py2 = 0.d0
	    py3 = 0.d0
	  endif
	  write(iouf1) py1,py2,py3
c	  write(11,'(3(2X,F7.4))') py1,py2,py3
	enddo

	if (q.ne.0.d0)q = 1.d0/q
c ------- Arjun: output to ascii dispersion file --------------------------------
      if(iasc.eq.1) then
c -- Modif. ARJUN - phase velocity partial derivatives now output to the ascii
c -- dispersion file, for select periods
c      idiv=merge(5,10,per.le.50)
c      if (per.le.500) then
c       if (((mod(nint(per),idiv).eq.0) .or. (mod(floor(per),idiv).eq.0))
c     +      .and. ((nint(per).ne.iprev) .and. (floor(per).ne.iprev))) then
c	    write(iouf2,900) nord,per,cc,u,q,flan,ls,1
c            do j = 1,3
c	       write(iouf2,'(7e15.7)') (der(j,i),i=1,ls)
c	    enddo
c            iprev=floor(per)
c        else
c            write(iouf2,900) nord,per,cc,u,q,flan,ls,0 
c        endif
      if ((per.lt.400) .and. (mod(nint(per),5).eq.0) .and.
     +   (abs(per-nint(per)).lt.0.00001)) then
	    write(iouf2,900) nord,per,cc,u,q,flan,ls,1
	    write(iouf3,900) nord,per,cc,u,q,flan
            do j = 1,3
	       write(iouf2,'(7e15.7)') (der(j,i),i=1,ls)
	    enddo
      else
          write(iouf2,900) nord,per,cc,u,q,flan,ls,0 
          write(iouf3,900) nord,per,cc,u,q,flan
      endif
900	format(1x,i5,5g15.7,4X,i3,4X,i1)
       endif
c -- End Arjun: output to the ascii dispersion file -----------------------------
	return

999	write(*,950) flan
950	format(' problem with eigenfunction : flan  = ',g15.7)

	stop
	end

	subroutine detlov(cc,w,de,ifeif)
c  computes the love stress - displacement vector and propagates it
c  upwards. if ifeif = 0 only the determinant at the surface
c  is returned. if ifeif = 1 derivatives,excitations etc. are computed
	implicit real*8(a - h,o - z)
	include 'sizes.inc'
	include 'units.inc'

	common/x/x(2,lyrs),scale(lyrs),der(2,lyrs),dummy(lyrs*2)
	common/m/d(lyrs),ro(lyrs),vp2(lyrs),vs2(lyrs),fu(lyrs),n,noc,ist,iasc
	common/bits/u,nsrce,idep(lsd),nord,tpi,sdep(lsd),ig,idisc,irdep
	common/q/qb(lyrs),qa(lyrs)

c -- new variable added by Arjun
	dimension dep(lyrs)
c -- end of new variable declaration

	p = 1.d0/cc
	psq = p*p
	x(1,ist) = 1.d0
	x(2,ist) =  - fu(ist)*dsqrt(psq - 1.d0/vs2(ist))
	i = ist
	scale(i) = 0.d0
c  propagate the solution up
100	i1 = i
	i = i - 1
	f1 = 1.d0/fu(i)
	hb = psq - 1.d0/vs2(i)
	wd = w*d(i)
	call arg(wd,hb,cb,sb,fac)
	scale(i) = scale(i1) + fac
	fsb = f1*sb
	hsb = hb*sb*fu(i)
	x(1,i) = cb*x(1,i1) + fsb*x(2,i1)
	x(2,i) = hsb*x(1,i1) + cb*x(2,i1)
	if (i.gt.noc) go to 100
	de = x(2,noc)/(dabs(x(2,noc)) + dabs(x(1,noc)))
	if (ifeif.eq.0) return
c  compute derivatives,excitations etc.
	xnorm = 1.d0/x(1,noc)
	knt = noc - 1
	do 25 i = noc,ist
	  knt = knt + 1
	  xfac = 1.d0
	  if (scale(i) - scale(noc).ne.0.d0) xfac = dexp(scale(i) - scale(noc))
	  ls = i
	  x(1,i) = x(1,i)*xnorm*xfac
	  x(2,i) = x(2,i)*xnorm*xfac
	  if (dabs(de).gt.1.d-4) go to 25
	  if (i.lt.noc + 1) go to 25
c  if stress - displacement vector is small enough and solution is no
c  longer oscillatory reduce the model size
	  pbsq = 1.d0/vs2(i)
	  if (xfac.lt.1.d-15.and.psq.ge.pbsq) go to 30
25	continue
30	if (dabs(de).ge.1.d-4)call fixlov(w,p)

	hb = w*dsqrt(psq - 1.d0/vs2(ls))
	fi1 = 0.5d0*x(1,ls)*x(1,ls)/hb
	fi2 = 0.5d0*x(2,ls)*x(2,ls)/(hb*fu(ls))
c  si1,si2,si3 are the energy integrals
c  der(1,i),der(2,i)are the log. derivatives of phase velocity wrt rho and vs
	si1 = ro(ls)*fi1
	si2 = psq*fu(ls)*fi1 + fi2
	si3 = fu(ls)*fi1
	der(1,ls) = 0.5d0*(si2 - si1)
	der(2,ls) = si2
	i = knt
35	i1 = i
	i = i - 1
	c0 = psq*fu(i) - ro(i)
	c1 = 0.5d0*(x(1,i)*x(2,i) - x(1,i1)*x(2,i1))/w
	c2 = 0.5d0*d(i)*(c0*x(1,i1)*x(1,i1) - x(2,i1)*x(2,i1)/fu(i))
	fi1 = (c2 - c1)/c0
	fi2 =  - c2 - c1
	sj1 = ro(i)*fi1
	sj2 = psq*fu(i)*fi1 + fi2
	si1 = si1 + sj1
	si2 = si2 + sj2
	si3 = si3 + fu(i)*fi1
	der(1,i) = 0.5d0*(sj2 - sj1)
	der(2,i) = sj2
	if (i.gt.noc) go to 35
	u = p*si3/si1
       per = tpi/w
c -- Arjun: output eigenfunctions --
      if(iasc.eq.1) then
	if (abs(per-nint(per)).lt.0.00001) then
	  intper=1
        else
	  intper=0
        endif
	dep(1)=0
        do ic=1,ls
	  dep(ic+1)=dep(ic)+d(ic)
c	  write(*,*) dep(ic)
	enddo
c       write(10,'(" mode=",I3," period=",F13.7," layers=",I4,6X,I2)') nord,period,ls,intper
	write(10,'(I3,4(F13.7,2X),I4,6X,I2)') nord,per,cc,u,(si3*cc/w*w),ls,intper
        write(10, '(3e15.7)') (dep(ic),(x(ir,ic),ir=1,2), ic=1,ls)
        !write(15,'(I3,4(F13.7,2X),6X)') nord,per,ls
        
        write(150,9022) -1, nord,per,cc,u
        write(150, '(3e15.7)') (dep(ic),(x(ir,ic),ir=1,2), ic=1,ls)
        
9022    format(I7,I7,3(F13.7,2X))
      endif
c -- end Arjun: output eigenfunctions --

c  flan should be close to zero if the eigenfunction is accurate
	flan = si2/si1 - 1.d0
	if (dabs(flan).ge.1.e-8) go to 999
	fnorm = 1.d0/(psq*si3)
	q = 0.d0
	do 40 i = noc,ls
	der(1,i) = der(1,i)*fnorm
	der(2,i) = der(2,i)*fnorm
40	q = q + der(2,i)*qb(i)
	gam = 0.5d0*w*q/cc
c  compute 'excitation' functions.py1,py2 correspond to a,b in mendiguren
	if (irdep>ls) then
	  x(1,irdep)=0.0
	  x(2,irdep)=0.0
        endif
	gb1 = w*x(2,irdep)/fu(irdep)      
	write(iouf1) w,cc,gam,x(1,irdep),gb1
c	write(*,*) "From detlov: ls is ", ls , "& irdep is ", irdep
c	write(11,'(5(2X,F10.6))') w,cc,gam,x(1,irdep),gb1
	fact = 1.d0/(dsqrt(p*w)*si3*15.853309d-6)
	do 45 i = 1,nsrce
	id = idep(i)
	py1 = x(1,id)*p*fact
	py2 = x(2,id)*fact/fu(id)
	if (id.le.ls) go to 45
	py1 = 0.d0
	py2 = 0.d0
45	write(iouf1) py1,py2
c	write(11,'(2(2X,F7.4))') py1,py2
	per = tpi/w
	if (q.ne.0.d0)q = 1.d0/q
c ------ Arjun: output to ascii dispersion file -------------------------------
        if(iasc.eq.1) then
	 write(iouf2,900) nord,per,cc,u,q,flan,0,0
	 write(iouf3,900) nord,per,cc,u,q,flan,0,0
900	 format(1x,i5,5g15.7,2i3)
        endif
c ------ End Arjun: outout to ascii dispersion file -------------------------
	return

999	write(*,950) flan
950	format(' problem with eigenfunction : flan  = ',g15.7)

	stop
	end

	subroutine intrp(om,dom,jcom)
c  interpolates between bracketing c's to find the root.uses a bisection scheme.
	implicit real*8(a - h,o - z)
	include 'sizes.inc'
	common/m/d(lyrs),ro(lyrs),vp2(lyrs),vs2(lyrs),fu(lyrs),n,noc,ist,iasc
	common/bits/u,nsrce,idep(lsd),nord,tpi,sdep(lsd),ig,idisc,irdep
	common/bran/ce(2),ke(2),de(2),ctry,ceps,um,cm,cmx,cmn
	data tol,dlta/1.d-11,1.d-5/
	fc = de(1)
	fb = de(2)
	if (fc*fb.ge.0.d0) return
	nord = ke(1)
	c = ce(1)
	b = ce(2)
	psq = 1.d0/(b*b)
	call strtdp(psq,om)
	s = c
	fs = fc
c*** bisect ***
5	h = 0.5d0*(b + c)
	t = h*tol
c*** check for convergence ***
	db = dabs(fb)
	dc = dabs(fc)
	if (dabs(h - b).lt.t) go to 35
	if (db.le.dc) go to 10
	y = b
	fy = fb
	gg = b
	fg = fb
	s = c
	fs = fc
	go to 15
10	y = s
	fy = fs
	gg = c
	fg = fc
	s = b
	fs = fb
15	if (fy.eq.fs) go to 20
	b = (s*fy - y*fs)/(fy - fs)
	if (dabs(b - s).lt.t) b = s + dsign(t,gg - s)
	if( (b-h)*(s-b).lt.0.d0) b = h
	go to 25
20	b = h
25	call detray(b,om,fb,0,jcom)
	if (fg*fb.lt.0.d0) go to 30
	c = s
	fc = fs
	go to 5
30	c = gg
	fc = fg
	go to 5
35	if (dc.lt.db) b = c
	call detray(b,om,fb,1,jcom)
	cp = b*(1.d0 - b/u)/om
	ctry = 0.d0
	clin = b - cp*dom
	ctry = 5.d0*cm - 4.d0*b - 2.d0*dom*(um + 2.d0*cp)
	ceps = dmax1(dabs(clin - ctry),ctry*tol)
60	um = cp
	cm = b
	if (ctry.ne.0.d0) return

	ctry = b - cp*dom
	ceps = dmax1(dabs(cp*dom),ctry*tol)

	return
	end

	subroutine detray(cc,w,de,ifeif,jcom)
c  routine computes minor vector in each layer for
c  raleigh waves by progagation upwards. if ifeif = 0
c  only the determinant at the surface is returned.
c  if ifeif = 1 the stress - displacement vector is
c   computed in each layer and deriv is called.
	implicit real*8(a - h,o - z)
        include 'sizes.inc'

	common/x/x(4,lyrs),der(2,lyrs),scale(lyrs)
	common/bits/xxz,nsrce,izz(lsd),nord,tpi,zz(lsd),ig,idisc,irdep
	common/m/d(lyrs),ro(lyrs),vp2(lyrs),vs2(lyrs),fu(lyrs),n,noc,ist
	common/y/y1,y2,y3,y4,y5

	dimension ym(5,lyrs),y(5)
	equivalence (ym,x),(y1,y)

	if (jcom.eq.1) go to 1

	call detlov(cc,w,de,ifeif)
	return
        
c  compute minor vector y at bottom
1	p = 1.d0/cc
	ysav = 0.d0
	psq = p*p
	r2 = 2.d0*fu(ist)*p
	y3 = dsqrt(psq - 1.d0/vp2(ist))
	y4 =  - dsqrt(psq - 1.d0/vs2(ist))
	y1 =  - (y3*y4 + psq)/ro(ist)
	y2 = r2*y1 + p
	y5 = ro(ist) - r2*(p + y2)
	i = ist
c -- Arjun: ist is probably the total number of layers
	scale(i) = 0.d0
	do j = 1,5
	  ym(j,i) = y(j)
	enddo
c**** propagate up layers ****
100	i = i - 1
	wd = w*d(i)
	ha = psq - 1.d0/vp2(i)
	call arg(wd,ha,ca,sa,faca)
	hb = psq - 1.d0/vs2(i)
	call arg(wd,hb,cb,sb,facb)
	scale(i) = scale(i + 1) + faca + facb
	hbs = hb*sb
	has = ha*sa
	r1 = 1.d0/ro(i)
	r2 = 2.d0*fu(i)*p
	b1 = r2*y1 - y2
	g3 = (y5 + r2*(y2 - b1))*r1
	g1 = b1 + p*g3
	g2 = ro(i)*y1 - p*(g1 + b1)
	g1 = g1*dexp( - faca - facb)
	e1 = cb*g2 - hbs*y3
	e2 =  - sb*g2 + cb*y3
	e3 = cb*y4 + hbs*g3
	e4 = sb*y4 + cb*g3
	y3 = ca*e2 - has*e4
	y4 = sa*e1 + ca*e3
	g3 = ca*e4 - sa*e2
	b1 = g1 - p*g3
	y1 = (ca*e1 + has*e3 + p*(g1 + b1))*r1
	y2 = r2*y1 - b1
	y5 = ro(i)*g3 - r2*(y2 - b1)
	do j = 1,5
	  ym(j,i) = y(j)
	enddo
	if (i.gt.noc) go to 100
	if (noc.eq.1) go to 15
c  *** propagate through ocean layer ***
	ha = psq - 1.d0/vp2(1)
	wd = w*d(1)
	call arg(wd,ha,ca,sa,faca)
	y1 = ca*y3 - ha*sa*y5/ro(1)
	y5 = ca*y5 - ro(1)*sa*y3
	y2 = y5

15	de = y5/dsqrt(y1*y1 + y2*y2)
	if (ifeif.eq.0) return

c  compute stress - displacement vector x = n*y
	ynorm = 1.d0/ym(3,noc)
	if (noc.eq.1) go to 20
c  cope with possibility of stoneley mode on ocean floor by integrating
c  from surface
	y1 = ca
	y2 = ro(1)*sa
	ysav = y2/y1
	de1 = de
	de = dmin1(dabs(de1),dabs(ynorm*ym(5,noc)/ysav - 1.d0))
c  specify arbitrary solution y at the surface
20	y1 = 0.d0
	y2 = -ynorm
	y3 = 0.d0
	y4 = 0.d0
	xfac = 1.d0
	sum = 0.d0
	i = noc
c  minor  elements compose matrix n so that x = n*b.(x is the stress - displacement
c  vector) compute x in each layer. this happens to be numerically stable.
c  b is a solution to the equations of motion db/dz = ab
c  Arjun: so b probably contains r1, r2, r3, r4, to which the stresses and displacements 
c  are proportional
25	xx1 = -ym(2,i)*y1 - ym(3,i)*y2 + ym(1,i)*y4
	xx2 = -ym(4,i)*y1 + ym(2,i)*y2 - ym(1,i)*y3
	xx3 = -ym(5,i)*y2 - ym(2,i)*y3 - ym(4,i)*y4
	xx4 =  ym(5,i)*y1 - ym(3,i)*y3 + ym(2,i)*y4
       
        x(1,i) = xx1*xfac
	x(2,i) = xx2*xfac
	x(3,i) = xx3*xfac
	x(4,i) = xx4*xfac

	ls = i
	if (i.eq.ist) go to 30
	if (i.lt.noc + 1) go to 35
	if (dabs(de).gt.1.d-4) go to 35
c  if x becomes  small and the solution is no longer oscillatory
c  reduce the model size
	pbsq = 1.d0/vs2(i)
	if (xfac.lt.1.d-15.and.psq.ge.pbsq) go to 30
35	wd = w*d(i)
	ha = psq - 1.d0/vp2(i)
	call arg(wd,ha,ca,sa,faca)
	hb = psq - 1.d0/vs2(i)
	call arg(wd,hb,cb,sb,facb)
	dfac = dexp(facb - faca)
	cb = dfac*cb
	sb = dfac*sb
	hbs = hb*sb
	has = ha*sa
	r2 = 2.d0*p*fu(i)
	e2 = r2*y2 - y3
	e3 = ro(i)*y2 - p*e2
	e4 = r2*y1 - y4
	e1 = ro(i)*y1 - p*e4
	e6 = ca*e2 - sa*e1
	e8 = cb*e4 - sb*e3
	y1 = (ca*e1 - has*e2 + p*e8)/ro(i)
	y2 = (cb*e3 - hbs*e4 + p*e6)/ro(i)
	y3 = r2*y2 - e6
	y4 = r2*y1 - e8
	i = i + 1
	sum = sum + faca
	xfac = dexp(scale(i) - scale(noc) + sum)
	go to 25

30	if (dabs(de).gt.1.d-4) call fixray(ysav,w,p)
        
c  compute excitations or partial derivatives
50	call deriv(cc,w,ls)

	return
	end

	subroutine qcor(om,omref)
	implicit real*8(a - h,o - z)
	include 'sizes.inc'
	common/mq/vps(lyrs),vss(lyrs)
	common/m/d(lyrs),ro(lyrs),vp(lyrs),vs(lyrs),fu(lyrs),n,noc,ist
	common/q/qb(lyrs),qa(lyrs)

	data pii/.3183098d0/

	omscl = 0.d0
	if (omref.ne.0.d0) omscl = pii*dlog(om/omref)
	do i = 1,n
	  vs(i) = vss(i)*(1.d0 + qb(i)*omscl)**2
	  vp(i) = vps(i)*(1.d0 + qa(i)*omscl)**2
	  fu(i) = ro(i)*vs(i)
	enddo

	return
	end
