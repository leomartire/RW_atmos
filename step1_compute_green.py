#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# import mechanisms as mod_mechanisms
# from obspy.core.utcdatetime import UTCDateTime
# import matplotlib.pyplot as plt
# from pyrocko import moment_tensor as mtm
# from obspy.imaging.beachball import beach
import sys
# import numpy as np
import shutil
import argparse

from utils import str2bool
import wrappers

def main():
  parser = argparse.ArgumentParser(description='Computes Green functions with earthsr.')
  
  required = parser.add_argument_group('REQUIRED ARGUMENTS')
  required.add_argument('--seismicModel', required=True,
                      help='Path to a SPECFEM-formatted elastic model.')
  required.add_argument('--output', required=True,
                      help='Output folder path.')
  required.add_argument('--nCPU', type=int, required=True,
                      help='Number of CPUs to use for multithreading the collection of earthsr outputs.')
  
  freqs = parser.add_argument_group('optional arguments - frequency domain')
  def__freqMinMax = [0.001, 5.0];
  freqs.add_argument('--freqMinMax', type=float, nargs=2, default=def__freqMinMax,
                     help=('Frequency bounds in [Hz]. Defaults to [%.3e, %.3e] Hz.' % (def__freqMinMax[0], def__freqMinMax[1])))
  def__nbFreq = 2**8
  freqs.add_argument('--nbFreq', type=int, default=def__nbFreq,
                     help=('Number of frequency points. Defaults to %d.' % (def__nbFreq)))
  # def__nbKXY = 2**6
  # freqs.add_argument('--nbKXY', type=int, default=def__nbKXY,
  #                    help=('Number of wavenumber points. Defaults to %d.' % (def__nbKXY)))
  def__nbModes = [0, 50]
  freqs.add_argument('--nbModes', type=int, nargs=2, default=def__nbModes,
                     help=('Min/max number of modes to compute. Defaults to [%d, %d].' % (def__nbModes[0], def__nbModes[1])))
  
  freqs = parser.add_argument_group('optional arguments - elastic model')
  def__nbLayers = 100
  freqs.add_argument('--nbLayers', type=int, default=def__nbLayers,
                     help=('Number of discretisation layers for the elastic domain. Defaults to %d.' % (def__nbLayers)))
  def__zmax = 10.0e3
  freqs.add_argument('--depthMax', type=float, default=def__zmax,
                     help=('Maximum depth for the elastic domain. Defaults to %.0f m.' % (def__zmax)))
  
  misc = parser.add_argument_group('optional arguments - miscellaneous')
  misc.add_argument('--outputOverwrite', type=str2bool, choices=[True, False], default=True,
                      help='Overwrite output folder path? Defaults to True.')
  misc.add_argument('--dimensionSeismic', type=int, choices=[2,3], default=3,
                      help='Seismic dimension. Defaults to 3.')
  
  args = parser.parse_args()
  print(args)
  print(' ')

  # Sample path name of the directory created to store data and figures
  output_path    = args.output+'/'
  forceOverwrite = args.outputOverwrite
  ncpu           = args.nCPU
  
  # RW-atmos integration options
  options = {}
  # options['dimension']         = args.dimension # atmospheric dimension
  options['dimension_seismic'] = args.dimensionSeismic # seismic dimension
  # options['ATTENUATION']    = True # using Graves, Broadband ground-motion simulation using a hybrid approach, 2014
  # options['COMPUTE_MAPS']   = False # Compute and plot x,y,z wavefield. Computationally heavy.
  # options['COMPUTE_XY_PRE'] = 20e3 # Compute xy wavefield above source and at given altitude. Computationally heavy. Set to none to disable.
  options['nb_freq']        = args.nbFreq
  # options['nb_kxy']         = args.nbKXY
  options['coef_low_freq']  = min(args.freqMinMax) # minimum frequency (Hz)
  options['coef_high_freq'] = max(args.freqMinMax) # maximum frequency (Hz)
  options['nb_layers']      = args.nbLayers # Number of seismic layers to discretise (earthsr input).
  options['zmax']           = args.depthMax # Maximum depth (m) for seismic layers (earthsr input).
  options['nb_modes']       = args.nbModes # min, max number of modes
  # options['t_chosen']       = [0., 90.] # time (s) to display wavefield
  # options['t_chosen']       = [10, 24, 32]
  options['models']            = {}
  options['models']['specfem'] = args.seismicModel # './models/Ridgecrest_seismic.txt'
  options['type_model']        = 'specfem' # specfem or specfem2d
  options['chosen_model']      = 'specfem'
  options['USE_SPAWN_MPI'] = False 
  
  wrappers.computeGreen(output_path, ncpu, options, forceOverwrite=forceOverwrite)

if __name__ == '__main__':
  main()