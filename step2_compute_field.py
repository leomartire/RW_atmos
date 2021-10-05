#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:54:45 2021

@author: lmartire
"""

# from obspy.core.utcdatetime import UTCDateTime
from RWAtmosUtils import pickleLoad
# import shutil
import argparse
import sys

import wrappers
# import multiprocessing as mp
# from multiprocessing.pool import ThreadPool as Pool

# from itertools import repeat
# from utils import str2bool

# def kek(Green_RW_, mech):
#   Green_RW = deepcopy(Green_RW_)
#   Green_RW.set_mechanism(mech)
#   print(Green_RW)

def main():
  parser = argparse.ArgumentParser(description='Computes Rayleigh wave field from Green functions.')
  
  required = parser.add_argument_group('REQUIRED ARGUMENTS')
  required.add_argument('--output', required=True,
                      help='Output folder path.')
  required.add_argument('--options', required=True,
                      help='Path to last set of options Pickle.')
  required.add_argument('--green', required=True,
                      help='Path to Green functions'' object Pickle.')
  
  required.add_argument('--parallel', required=True, type=int, nargs=2,
                      help='Hybrid parallelisation parameters. '+
                           'First integer encodes the MPI parallelisation for sources (parameter sourceIDs). '+
                             '0 = serial; n>0 = MPI over n nodes. '
                           'Second integer encodes the multithreading parallelisation for the RW modes (as specified in the output of step 1). '+
                             '0 = serial (not recommended); n>0 = multithread over n CPUs.')
  
  def__nbKXY = 2**6
  required.add_argument('--atmosDimension', type=int, choices=[2,3], default=3,
                      help='Atmospheric dimension. Defaults to 3.')
  required.add_argument('--nbKXY', type=int, default=def__nbKXY,
                        help=('Number of wavenumber points. Defaults to %d.' % (def__nbKXY)))
  required.add_argument('--sourceIDs', required=True, type=int, nargs='+',
                      help='IDs of the sources to be imported. Must match files under format ''source_#####_in.pkl'' in the chosen output folder path, generated using the script ''makeSource.py''.')
  required.add_argument('--latminlatmaxlonminlonmax', required=True, type=float, nargs=4,
                      help='Min/Max latitude/longitude, regardless of the source mechanism (lat_min, lat_max, lon_min, lon_max).')
  required.add_argument('--f0', required=True, type=float,
                      help='Dominant frequency of the source in [Hz], regardless of the source mechanism.')
  
  misc = parser.add_argument_group('optional arguments - miscellaneous')
  misc.add_argument('--atmosphere', default=[],
                    help='Atmospheric model path? Defaults to the default atmospheric model defined in \'velocity_models.prepare_atmospheric_model\'.')
  
  args = parser.parse_args()
  print(args)
  print(' ')

  # Sample path name of the directory created to store data and figures
  output_path               = args.output+'/'
  # forceOverwrite            = args.outputOverwrite
  
  optionsStep1Path = args.options
  GreenRWPath = args.green
  atmosPath = args.atmosphere
  sourceIDs = args.sourceIDs
  
  parallel = args.parallel
  if(parallel[0]<0 or parallel[1]<0):
    sys.exit('[%s, ERROR] Cannot use a negative value for the parallel parameter.' % (sys._getframe().f_code.co_name))
  if(parallel[0]>len(sourceIDs)):
    sys.exit('[%s, ERROR] Not a good idea to request more nodes (%d) than there are sources (%d), please fix.' % (sys._getframe().f_code.co_name, parallel[0], len(sourceIDs)))
  if(parallel[0]==1 and len(sourceIDs)>1):
    print('[%s, WARNING] Requesting 1 node for multiple (%d) sources. Will run the source loop in serial. This is not optimal but will work.' % (sys._getframe().f_code.co_name, len(sourceIDs)))
  
  options_step2 = {}
  options_step2['dimension'] = args.atmosDimension
  # HERE LOAD NEW OPTIONS FROM ARGS.
  
  # Load previous options, and update with new ones.
  options_step1 = pickleLoad(optionsStep1Path)
  options = options_step1
  options.update(options_step2)
  
  # Source parameters
  options_source = {}
  options_source['stf-data'] = [] # file where stf is located
  options_source['stf']      = 'gaussian' # gaussian or erf
  options_source['f0']       = args.f0 # dominant freuqency (Hz)
  options_source['lat_min']  = min(args.latminlatmaxlonminlonmax[:2])
  options_source['lat_max']  = max(args.latminlatmaxlonminlonmax[:2])
  options_source['lon_min']  = min(args.latminlatmaxlonminlonmax[2:])
  options_source['lon_max']  = max(args.latminlatmaxlonminlonmax[2:])
  
  # Load sources from Earthquake catalog or build custom source
  options_source['DIRECTORY_MECHANISMS'] = []
  options_source['sources'] = []
  for i in sourceIDs:
    sourceFileToImport = (output_path+'source_%05d_in.pkl' % (i))
    source_characteristics = pickleLoad(sourceFileToImport)
    # source_characteristics = { # example of custom source
    #     'id': 0,
    #     'time': UTCDateTime(2019, 8, 9, 0, 9, 57),
    #     'mag': 2.98,
    #     'lat': 35.895,
    #     'lon': -117.679,
    #     'depth': 4.1, #km
    #     'strike': 159,
    #     'dip': 89,
    #     'rake': -156,
    # }
    options_source['sources'].append( source_characteristics )
  list_of_events = sourceIDs # list of event to compute, leave empty if you want all    
  options_source['add_SAC'] = False # Wheter or not add real station locations downloaded from IRIS within the domain
                                    # boundaries defined in options_source['lat_min'], options_source['lat_max'], ...
  
  # Options for ground stations.
  options_IRIS = {}
  # options_IRIS['network'] = 'CI,NN,GS,SN,PB,ZY' # Only if need to download stations.
  # options_IRIS['channel'] = 'HHZ,HNZ,DPZ,BNZ,BHZ,ENZ,EHZ' # Only if need to download stations.
  options_IRIS['stations'] = {}
  # i=0
  # options_IRIS['stations'][i] = mod_mechanisms.create_one_station(x=0., y=-50.0e3, z=0., comp='p', name='station', id=i); i+=1;
  
  # ugly hack: copy options from one dict to another & initialize other options only relevant to Ridgecrest
  options_source['coef_high_freq'] = options['coef_high_freq']
  options_source['nb_kxy']   = args.nbKXY
  # options_source['t_chosen'] = options['t_chosen']
  options_source['activate_LA'] = False # Only relevant for Ridgecrest study
  options_source['rotation'] = False
  
  options_balloon = {}
  
  wrappers.computeRWField(output_path, GreenRWPath, parallel,
                          options, options_source, options_IRIS, options_balloon,
                          list_of_events, atmosPath)

if __name__ == '__main__':
  main()