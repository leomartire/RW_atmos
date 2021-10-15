#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:21:52 2021

@author: lmartire
"""

import os
import shutil
import sys
from datetime import datetime, timedelta
import time
from mpi4py import MPI
import numpy as np
from copy import deepcopy
from pdb import set_trace as bp

import RW_atmos
import mechanisms as mod_mechanisms
import velocity_models
from RWAtmosUtils import pickleLoad, pickleDump
import RW_dispersion

def computeGreen(output_path, ncpu, options, forceOverwrite=False):
  
  # Check output path is free, make it if necessary.
  if(os.path.isdir(output_path)):
    if(forceOverwrite):
      shutil.rmtree(output_path)
      print('['+sys._getframe().f_code.co_name+'] Output files root folder \''+output_path+'\' existed and has been deleted, as required by script.')
    else:
      sys.exit('['+sys._getframe().f_code.co_name+'] Output files root folder \''+output_path+'\' exists, and script is not set to overwrite. Rename or delete it before running again.')
  os.makedirs(output_path)
  
  # Store inputs.
  # shutil.copyfile(options['models']['specfem'], output_path+'seismic_model__specfem.txt')
  os.system('cp %s %s' % (options['models']['specfem'], output_path+'seismic_model__specfem.txt'))
  options['models']['specfem'] = output_path+'seismic_model__specfem.txt' # Make sure we use the stored one (in case one needs to re-run using the stored "options_inp" file).
  pickleDump(output_path+'options_inp.pkl', options)
  
  # Compute Green functions.
  Green_RW, options_out_rw = RW_dispersion.compute_Green_functions(options, ncpu)
  options.update(options_out_rw)
  print(Green_RW)
  
  # Check output.
  if(Green_RW.nb_freqs_actualforMode1<0.5*Green_RW.nb_freqs):
    sys.exit('[%s] > Green functions object contains very few frequencies (%d) compared to what was asked (%d). Something went wrong.'
             % (sys._getframe().f_code.co_name, Green_RW.nb_freqs_actualforMode1, Green_RW.nb_freqs))
  
  # Cleanup run.
  output_path_run = output_path+'earthsrRun/'
  if(not os.path.isdir(output_path_run)):
    os.makedirs(output_path_run)
  print('[%s] Cleaning up earthsr run.' % (sys._getframe().f_code.co_name))
  for k in ['./input_code_earthsr', './ray', './tocomputeIO.input_code_earthsr', Green_RW.global_folder]:
    # print('[%s] > Move \'%s/%s\' to \'%s\' using a Python function.'
    #     % (sys._getframe().f_code.co_name, os.path.abspath(os.getcwd()), k, output_path_run))
    # os.rename(os.path.abspath(os.getcwd())+'/'+k, output_path_run+k)
    print('[%s] > Move \'%s\' to \'%s\' using a system command.'
        % (sys._getframe().f_code.co_name, k, output_path_run))
    os.system('mv '+k+' '+output_path_run)
  Green_RW.global_folder = output_path_run+Green_RW.global_folder # Save the moved/stored folder.
  
  # Store outputs.
  GreenPath = output_path+'Green_RW.pkl'
  optionsOutPath = output_path+'options_out.pkl'
  pickleDump(GreenPath, Green_RW)
  pickleDump(optionsOutPath, options)
  
  # Return for when this module is integrated in something else.
  return(GreenPath, optionsOutPath)

def generateAndSaveRWField(parallel, output_folder, Green_RW_, options, mechanism, i, param_atmos):
  Green_RW = deepcopy(Green_RW_) # copy because there is only one instance of this variable
  # param_atmos = deepcopy(param_atmos_)
  # options = deepcopy(options_)
  
  if(parallel[0]<=1 or (parallel[0]>=2 and i==0)):
    # Print only if we are running serial (parallelSources<=1),
    #         or only the first case if we are running parallel (parallelSources>=2).
    verbose = True
  else:
    verbose = False
  # Compute Rayleigh wave field.
  if(verbose): print('[%s] Update Green functions with chosen focal mechanism (current mechanism ID = %d).' % (sys._getframe().f_code.co_name, i))
  Green_RW.set_mechanism(mechanism)
  if(verbose): print('[%s] Create the Rayleigh wave field.' % (sys._getframe().f_code.co_name))
  RW_field = RW_atmos.create_RW_field(Green_RW, mechanism['domain'], param_atmos, options, ncpus = parallel[1], verbose = verbose)
  # Store outputs.
  pickleDump(output_folder+'RW_field.pkl', RW_field)
  # Return.
  # return(RW_field)

def computeRWField(output_path, GreenRWPath, parallel,
                   options, options_source, options_IRIS, options_balloon,
                   list_of_events, atmosPath):
  print('+------------------------------------------+')
  print("| JOB START TIME = %s   |" % (datetime.now().strftime("%Y/%m/%d @ %H:%M:%S")))
  print('+------------------------------------------+')
  t1 = time.time()
  
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
  
  if(parallel[0]<=1):
    # User set parallel[0] = 0 (explicitely serial) or parallel[0] = 1 (explicitely ask for one node). Simply run in serial, do not ask MPI ressources.
    # Note this consitutes 2 different cases:
    # - Locally, it is best to set parallel[0] = 0 to explicitely state we want a simple serial simulation.
    # - On a cluster, it is best to set parallel[0] = 1 since a request to reserve one node need be made explicit, even if we end up running serial on it.
    print(' ')
    print('[%s] Computation of the fields over every mechanism: remaining serial (no parallelisation).' % (sys._getframe().f_code.co_name))
    print(' ')
    for i in range(len(mechanisms_data)):
      generateAndSaveRWField(parallel, output_folders[i], Green_RW, options, mechanisms_data.loc[i], i, param_atmos)
    
  elif(parallel[0]>=2):
    print(' ')
    print('[%s] Computation of the fields over every mechanism: using MPI parallelisation.' % (sys._getframe().f_code.co_name))
    print('[%s, INFO] Spawning %d MPI processes to take care of the %d mechanisms.' % (sys._getframe().f_code.co_name, parallel[0], len(mechanisms_data)))
    print('[%s, INFO] We only log the output for the first process, so as not to flood the log.' % (sys._getframe().f_code.co_name))
    print(' ')
    # Setup MPI world.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Split the work across the different available MPI processes.
    IDs = [key for key in range(len(mechanisms_data))]
    IDs_perRank = [sublist for sublist in np.array_split(IDs, size) if sublist.size>0]
    # Sync processes just in case.
    comm.Barrier()
    # Let the current MPI process (number "rank") take care of its part of the tasks (IDs_perRank[rank]).
    for i in IDs_perRank[rank]:
      generateAndSaveRWField(parallel, output_folders[i], Green_RW, options, mechanisms_data.loc[i], i, param_atmos)
    # No output, simply barrier and finish.
    comm.Barrier()
    
    # pool = Pool(parallel[0])
    # # Apply method. Theoretically will chug RAM more because there's only one instance of Green_RW, options, and param_atmos.
    # for i in range(len(mechanisms_data)):
    #   # result = pool.apply_async(worker, (output_folders[i], Green_RW, options, mechanisms_data.loc[i], i, param_atmos))
    #   pool.apply(generateAndSaveRWField, (parallel, output_folders[i], Green_RW, options, mechanisms_data.loc[i], i, param_atmos))
    
    # # # Starmap method. Added pre-duplication of variables to try and limit RAM chugging; little to no improvement.
    # # Green_RWs = []
    # # optionssss = []
    # # param_atmossss = []
    # # for i in range(len(mechanisms_data)):
    # #   Green_RWs.append(deepcopy(Green_RW))
    # #   optionssss.append(deepcopy(options))
    # #   param_atmossss.append(deepcopy(param_atmos))
    # # argzip = zip([output_folders[i] for i in range(len(mechanisms_data))],
    # #              # repeat(Green_RW),
    # #              [Green_RWs[i] for i in range(len(mechanisms_data))],
    # #              # repeat(options),
    # #              [optionssss[i] for i in range(len(mechanisms_data))],
    # #              [mechanisms_data.loc[i] for i in range(len(mechanisms_data))],
    # #              range(len(mechanisms_data)),
    # #              # repeat(param_atmos)
    # #              [param_atmossss[i] for i in range(len(mechanisms_data))],
    # #              )
    # # result = pool.starmap_async(worker, argzip)
    # pool.close()
    # pool.join()
    # # field = result.get() # Gets returned value for last finished process.
    
  else:
    sys.exit('[%s, ERROR] We shouldn\'t end up here.' % (sys._getframe().f_code.co_name))
  # print(time.time()-t1)
  
  dur = time.time()-t1
  print('+------------------------------------------+')
  print("| JOB   END TIME = %s   |" % (datetime.now().strftime("%Y/%m/%d @ %H:%M:%S")))
  print('+------------------------------------------+')
  print("| JOB   DURATION = %.5e s (%s) |" % (dur, str(timedelta(seconds=round(dur)))))
  print('+------------------------------------------+')