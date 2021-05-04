#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import RW_dispersion
# import mechanisms as mod_mechanisms
# from obspy.core.utcdatetime import UTCDateTime
# import matplotlib.pyplot as plt
# from pyrocko import moment_tensor as mtm
# from obspy.imaging.beachball import beach
from utils import pickleDump, str2bool
import sys
# import numpy as np
import shutil
import argparse

def main():
  parser = argparse.ArgumentParser(description='Computes Green functions with earthsr.')
  
  required = parser.add_argument_group('required arguments')
  required.add_argument('--seismicModel', required=True,
                      help='Path to a SPECFEM-formatted elastic model.')
  required.add_argument('--output', required=True,
                      help='Output folder path.')
  
  parser.add_argument('--outputOverwrite', type=str2bool, choices=[True, False], default=True,
                      help='Overwrite output folder path? Defaults to True.')
  parser.add_argument('--dimension', type=int, choices=[2,3], default=3,
                      help='Atmospheric dimension. Defaults to 3.')
  parser.add_argument('--dimensionSeismic', type=int, choices=[2,3], default=3,
                      help='Seismic dimension. Defaults to 3.')
  
  args = parser.parse_args()
  print(args)

  # Sample path name of the directory created to store data and figures
  output_path               = args.output+'/'
  forceOverwrite            = args.outputOverwrite
  
  # RW-atmos integration options
  options = {}
  options['dimension']         = args.dimension # atmospheric dimension
  options['dimension_seismic'] = args.dimensionSeismic # seismic dimension
  # options['ATTENUATION']    = True # using Graves, Broadband ground-motion simulation using a hybrid approach, 2014
  # options['COMPUTE_MAPS']   = False # Compute and plot x,y,z wavefield. Computationally heavy.
  # options['COMPUTE_XY_PRE'] = 20e3 # Compute xy wavefield above source and at given altitude. Computationally heavy. Set to none to disable.
  options['nb_freq']        = 2**9
  options['nb_kxy']         = 2**8
  options['coef_low_freq']  = 0.001 # minimum frequency (Hz)
  options['coef_high_freq'] = 5. # maximum frequency (Hz)
  options['nb_layers']      = 100 # Number of seismic layers to discretise (earthsr input).
  options['zmax']           = 10.0e3 # Maximum depth (m) for seismic layers (earthsr input).
  options['nb_modes']       = [0, 50] # min, max number of modes
  # options['t_chosen']       = [0., 90.] # time (s) to display wavefield
  options['t_chosen']       = [10, 24, 32]
  options['models'] = {}
  options['models']['specfem'] = args.seismicModel # './models/Ridgecrest_seismic.txt'
  options['type_model']        = 'specfem' # specfem or specfem2d
  options['USE_SPAWN_MPI'] = False
  
  # Check output path is free, make it if necessary.
  if(os.path.isdir(output_path)):
    if(forceOverwrite):
      shutil.rmtree(output_path)
      print('['+sys._getframe().f_code.co_name+'] Output files root folder \''+output_path+'\' existed and has been deleted, as required by script.')
    else:
      sys.exit('['+sys._getframe().f_code.co_name+'] Output files root folder \''+output_path+'\' exists, and script is not set to overwrite. Rename or delete it before running again.')
  os.makedirs(output_path)
  
  # Store inputs.
  shutil.copyfile(options['models']['specfem'], output_path+'seismic_model__specfem.txt')
  options['models']['specfem'] = output_path+'seismic_model__specfem.txt' # Make sure we use the stored one (in case one needs to re-run using the stored "options_inp" file).
  pickleDump(output_path+'options_inp.pkl', options)
  
  # Compute Green functions.
  Green_RW, options_out_rw = RW_dispersion.compute_Green_functions(options)
  options.update(options_out_rw)
  
  # Cleanup run.
  output_path_run = output_path+'earthsrRun/'
  if(not os.path.isdir(output_path_run)):
    os.makedirs(output_path_run)
  for k in ['input_code_earthsr', 'ray', 'tocomputeIO.input_code_earthsr', Green_RW.global_folder]:
    os.rename('./'+k, output_path_run+k)
  Green_RW.global_folder = output_path_run+Green_RW.global_folder # Save the moved/stored folder.
  
  # Store outputs.
  pickleDump(output_path+'Green_RW.pkl', Green_RW)
  pickleDump(output_path+'options_out.pkl', options)

if __name__ == '__main__':
  main()