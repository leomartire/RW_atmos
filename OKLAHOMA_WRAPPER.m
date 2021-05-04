clear all;
close all;
clc;

OUT1 = './OUTPUT_STEP1/';
seismicModel = '/Users/lmartire/Documents/software/rw_atmos_leo/models/Ridgecrest_seismic.txt';

OUT2 = './OUTPUT_STEP2/';
options = [OUT1, 'options_out.pkl']; % must agree with python script for step 1
green = [OUT1, 'Green_RW.pkl']; % must agree with python script for step 1

imech = 0;
OUT3 = [OUT2,'mechanism_',sprintf('%05d', imech),filesep]; % must agree with python script for step 2
field = [OUT3,'RW_field.pkl']; % must agree with python script for step 2
alts = [20]*1e3;
times = [24];
doPlots = 0;
doDumps = 1;

step1 = ['./step1_compute_green.py --output ',OUT1,' --seismicModel ', seismicModel];
step2 = ['./step2_compute_field.py --output ',OUT2,' --options ',options,' --green ', green];
step3 = ['./step3_compute_pressure_section.py --output ',OUT3,' --RWField ',field,' --altitudes ',sprintf('%.3f ', alts),' --times ',sprintf('%.3f ', times),' --doPlots ',num2str(doPlots),' --doDumps ',num2str(doDumps)];

python = ['/usr/local/bin/python3'];

c1 = [python, ' ', step1];
c2 = [python, ' ', step2];
c3 = [python, ' ', step3];