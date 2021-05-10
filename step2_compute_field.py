#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:54:45 2021

@author: lmartire
"""

# from obspy.core.utcdatetime import UTCDateTime
from utils import pickleLoad, pickleDump
import mechanisms as mod_mechanisms
# import shutil
import argparse
import os
import sys
import RW_atmos
import velocity_models
# import multiprocessing as mp
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool
from copy import deepcopy
import time
import numpy as np
# from itertools import repeat

useMultiProc = False

# def kek(Green_RW_, mech):
#   Green_RW = deepcopy(Green_RW_)
#   Green_RW.set_mechanism(mech)
#   print(Green_RW)

def worker(output_folder, Green_RW_, options, mechanism, i, param_atmos):
  Green_RW = deepcopy(Green_RW_) # copy because there is only one instance of this variable
  # param_atmos = deepcopy(param_atmos_)
  # options = deepcopy(options_)
  
  if((not useMultiProc) or (useMultiProc and i==0)):
    verbose = False
  else:
    verbose = False
  # Compute Rayleigh wave field.
  if(verbose): print('[%s] Update Green functions with chosen focal mechanism (current mechanism ID = %d).' % (sys._getframe().f_code.co_name, i))
  Green_RW.set_mechanism(mechanism)
  if(verbose): print('[%s] Create the Rayleigh wave field.' % (sys._getframe().f_code.co_name))
  RW_field = RW_atmos.create_RW_field(Green_RW, mechanism['domain'], param_atmos, options, verbose=verbose)
  # Store outputs.
  pickleDump(output_folder+'RW_field.pkl', RW_field)
  # Return.
  # return(RW_field)

def main():
  parser = argparse.ArgumentParser(description='Computes Green functions with earthsr.')
  
  required = parser.add_argument_group('REQUIRED ARGUMENTS')
  required.add_argument('--output', required=True,
                      help='Output folder path.')
  required.add_argument('--options', required=True,
                      help='Path to last set of options Pickle.')
  required.add_argument('--green', required=True,
                      help='Path to Green functions'' object Pickle.')
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
  
  options_step2 = {}
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
  options_source['nb_kxy']   = options['nb_kxy']
  # options_source['t_chosen'] = options['t_chosen']
  options_source['activate_LA'] = False # Only relevant for Ridgecrest study
  options_source['rotation'] = False
  
  options_balloon = {}
  
  # Check output path is free, make it if necessary.
  if(not os.path.isdir(output_path)):
    os.makedirs(output_path)
  
  # Prepare source mechanism and domain.
  mechanisms_data = mod_mechanisms.load_source_mechanism_IRIS(options_source, options_IRIS, dimension=options['dimension'], 
                                                              add_SAC = options_source['add_SAC'], add_perturbations = False, 
                                                              specific_events=list_of_events, options_balloon=options_balloon)
  
  # Store inputs.
  pickleDump(output_path+'options_inp.pkl', options)
  pickleDump(output_path+'options_source_inp.pkl', options_source)
  pickleDump(output_path+'options_iris_inp.pkl', options_IRIS)
  if(options_balloon):
    pickleDump(output_path+'options_balloon_inp.pkl', options_IRIS)
  
  output_folders = []
  for i in range(len(mechanisms_data)):
    # Save mechanisms individually (for eventual re-use).
    pickleDump(output_path+'mechanism_data_%05d.pkl' % (i), mechanisms_data.loc[i])
    # Create output folders serially (so as not to chug the parallel process).
    output_folder = (output_path+'mechanism_%05d/' % (i))
    if(not os.path.isdir(output_folder)):
      os.makedirs(output_folder)
      # print('Creating '+output_folder+'.')
    output_folders.append(output_folder)
  
  # Generate atmospheric model (assume it is the same for all mechanisms).
  param_atmos = velocity_models.prepare_atmospheric_model(atmosPath)
  Green_RW = pickleLoad(GreenRWPath)
  velocity_models.plot_atmosphere_and_seismic_fromAtmosFile(output_path, Green_RW.seismic, atmosPath, options['dimension'])
  
  # t1 = time.time()
  if(useMultiProc):
    npool = min([np.floor(len(mechanisms_data)/3), 16]) # min 3 mech per proc, max 16 proc
    print(' ')
    print('[%s, WARNING] Using multiprocessing to parallelise the computation of the fields over every mechanism.' % (sys._getframe().f_code.co_name))
    print('[%s, INFO] Spawning %d processes to take care of the %d mechanisms.' % (sys._getframe().f_code.co_name, npool, len(mechanisms_data)))
    print('[%s, INFO] We only log the output for the first process, so as not to flood the log.' % (sys._getframe().f_code.co_name))
    print(' ')
    pool = Pool(npool)
    
    # Apply method. Theoretically will chug RAM more because there's only one instance of Green_RW, options, and param_atmos.
    for i in range(len(mechanisms_data)):
      # result = pool.apply_async(worker, (output_folders[i], Green_RW, options, mechanisms_data.loc[i], i, param_atmos))
      pool.apply(worker, (output_folders[i], Green_RW, options, mechanisms_data.loc[i], i, param_atmos))
    
    # # Starmap method. Added pre-duplication of variables to try and limit RAM chugging; little to no improvement.
    # Green_RWs = []
    # optionssss = []
    # param_atmossss = []
    # for i in range(len(mechanisms_data)):
    #   Green_RWs.append(deepcopy(Green_RW))
    #   optionssss.append(deepcopy(options))
    #   param_atmossss.append(deepcopy(param_atmos))
    # argzip = zip([output_folders[i] for i in range(len(mechanisms_data))],
    #              # repeat(Green_RW),
    #              [Green_RWs[i] for i in range(len(mechanisms_data))],
    #              # repeat(options),
    #              [optionssss[i] for i in range(len(mechanisms_data))],
    #              [mechanisms_data.loc[i] for i in range(len(mechanisms_data))],
    #              range(len(mechanisms_data)),
    #              # repeat(param_atmos)
    #              [param_atmossss[i] for i in range(len(mechanisms_data))],
    #              )
    # result = pool.starmap_async(worker, argzip)
    
    pool.close()
    pool.join()
    # field = result.get() # Gets returned value for last finished process.
  else:
    for i in range(len(mechanisms_data)):
      # field = worker(output_folders[i], Green_RW, options, mechanisms_data.loc[i], i, param_atmos)
      worker(output_folders[i], Green_RW, options, mechanisms_data.loc[i], i, param_atmos)
  # print(time.time()-t1)

if __name__ == '__main__':
  main()