# Makefile for surface wave software
# gram: $(obj)
#        gfortran $(obj) -O -o gram -L./grx -lgrx -lX11 \ -ffixed-line-length-none
#        -lm /usr/local/sac2000/sunos/lib/libsac_old.a
#
#

EARTHSR  =  earthsr
SRGRAMF  =  srgramf
DSRGRAMF =  dsrgramf
READ_EARTH = read_earth

EXEC=  $(EARTHSR) $(SRGRAMF) $(READ_EARTH) $(DSRGRAMF)

DEFINC=-Iinclude
SACLIB = /usr/local/sac/lib/libsacio.a
BINDIR = /home/ad605/code/Steve_wtmpi/bin

# Use gcc compiler
CC = gcc
# Fortran
FF      = gfortran
FC      = gfortran
FCOMP   = gfortran
BITTYPE = -m64
CHKBNDS = -fbounds-check
FFLAGS =  -ffixed-line-length-none -fdollar-ok $(BITTYPE) $(CHKBNDS) -O3

ARCH = $(shell uname).m64

all:	$(EXEC)

# Subroutines for EARTHSR
FSRCA=	earthsr.f \
	earthsubs.f \
	peripheral.f \
	char_int.f

# Subroutines for SRGRAMF
FSRCB=	srgramf.f \
	mkhomogsrf.f \
	fftl.f \
	out_sac.f \
	char_int.f \
	udc.f \
	srcget.f \
	nchar.f \
	minmax.f \
	parseprogs_mac.f

# Subroutines for DSRGRAMF
FSRCC=	dsrgramf.f \
	mkhomogsrf.f \
	fftl.f \
	out_sac.f \
	char_int.f \
	udc.f \
	srcget.f \
	nchar.f \
	minmax.f \
	parseprogs_mac.f

FOBJS = $(FSRCS:%.f=$(ARCH)/%.o)

FOBJA = $(FSRCA:%.f=$(ARCH)/%.o)

FOBJG = $(FSRCG:%.f=$(ARCH)/%.o)

FOBJR = $(FSRCR:%.f=$(ARCH)/%.o)

FOBJB = $(FSRCB:%.f=$(ARCH)/%.o)

FOBJC = $(FSRCC:%.f=$(ARCH)/%.o)

$(EARTHSR):: $(FOBJA) 
	$(FF) $(FOBJA) $(FFLAGS) $(DEFINC) -o $(EARTHSR) $(BITTYPE)

$(SRGRAMF):: $(FOBJB)
	$(FF) $(FOBJB) $(FFLAGS) $(SACLIB) $(DEFINC) -o $(SRGRAMF) $(BITTYPE)

$(DSRGRAMF):: $(FOBJC)
	$(FF) $(FOBJC) $(FFLAGS) $(SACLIB) $(DEFINC) -o $(DSRGRAMF) $(BITTYPE)

$(ARCH)/%.o: %.c
	$(CC) $(CFLAGS) $(DEFINC) -c $(@F:.o=.c) -o $@ $(BITTYPE)

$(ARCH)/%.o: %.f
	$(FF) $(FFLAGS) $(DEFINC) -c $(@F:.o=.f) -o $@  $(BITTYPE)

clean:
	@$(RM) $(ARCH)/*.o $(EXEC) *~ *.o

install:
	@if [ ! -d ${BINDIR} ] ; then \
	        mkdir ${BINDIR}; fi
	@for i in ${EXEC}; do \
	        if [ -f $$i ] ; then \
	                echo "Moving $$i to ${BINDIR}"; \
	                mv $$i ${BINDIR}; \
	        fi; \
	done


