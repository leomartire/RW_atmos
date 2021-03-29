Notes on how to run earthsr.f

earthsr.f is a program to compute Rayliegh and Love wave dispersion curves in a one-d medium for sources and receivers at
specified depths.    It is a moderate rewrite of the program earth.f by Steven Roecker (hence the "sr").


I/O:

earthsr requires a single input file and generates two output files (one ascii and one binary).
NB: modified by Arjun, it now generates more than two output files and the format of the output files may have changed.

The contents of the input file look like this:

         201           1   10.000000    
1.9996861103854826    6.2133395821021642        3.5504797271467856        2.7868533043811028        600.00000       300.00000    
1.9990584625211341    6.2278968613472419        3.5587984447904750        2.7936240002272879        600.00000       300.00000    
1.9984310116606139    6.2569002799884554        3.5753717927632938        2.8045927605830929        600.00000       300.00000    

...  several more similar lines discribing the model ...

1.8791763631288632    8.4832766528215799        4.8475866587551888        4.1477210176913539        600.00000       300.00000    
1.8785865400441253    8.4876485389347351        4.8500849433755535        4.1531811552373217        600.00000       300.00000    
0.0000000000000000    8.4900879835442815        4.8514791036365246        4.1560036520216004        600.00000       300.00000    
           1
ray
   0.0000000       0.0000000               0           4
     1    40     0.00100 0.01
8.000   ! source depth
21.00000000   ! receiver depth
           0

What it means


Line 1:		#layers in the model (including the half space)
		Earth flattening control varaiable (0 = no correction; >0 applies correction)
		Reference period for dispersion correction (0 => none)           

		In this example, there are 201 layers (200 + half space), the model is flattened, and the
		reference period is 10 s.  Generally you would just pick a period shorter than anything you
		are going to model.
	
Next 201 Lines:  Specification of the model by layer.  Columns are:
		Thickness (km), Vp (km/s), Vs (km/s), Density (gr/cc), Qb, Qa

		NOTE: The half space thickness (last line) is specified as "0".

1st line after model:   Surface wave type.  1 = Rayleigh; <>1 Love.  In this case we choose the Rayleigh option

2nd line after model:   Filename of binary output of dispersion curves.  In this case it is called "ray"

3rd line after model:   min and max phase velocities and min and max branch (mode) numbers. 
		Note that if we choose the min and max phase velocities to be 0, the program will choose 
		the phase velocity range itself.  In this case case we ask the program to figure out the appropriate 
		range (0.0000000       0.0000000) and solve modes 0 (fundamental) to 4.

4th line after model:  Number of sources, number of frequencies, frequency interval and starting (lowest) frequency.
                In this case we request 1 source, 40 frequencies and a frequency increment of 0.001 Hz starting from 0.01 Hz.  

5th line after model:  Source depths in km. (We have only one source in this case so only one depth at 8 km).

6th line after model:  Receiver depth in km.  (only one receiver allowed)

7th line after model:  This this point the program loops over another set of input lines starting with the surface
		wave type (1st line after model).  If this is set to zero, the program will terminate.


Output:

ASCII file:

	The ascii file repeats information from the input file and a table of dispersion information

	Receiver depth
	Source depths
	Raw Model
	Flattened Model
	Number of Branches (modes)
	For each branch:
		number of sources, number of frequencies, freq interval, wave type, number of branches
		order, period, PhaseV, GroupV, Attenuation term, Accuracy indicator (should be close to zero).	
        #Quentin: In the following lines the firest element on each line is Depth (km) instead of Thickness

Binary file:

	A sequence of reals (and occasional integer) with the same disperion and excitation information as above.

