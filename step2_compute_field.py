#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:54:45 2021

@author: lmartire
"""

from obspy.core.utcdatetime import UTCDateTime
from utils import pickleLoad, pickleDump, str2bool
import mechanisms as mod_mechanisms
import shutil
import argparse
import os
import sys
import RW_atmos
import velocity_models

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
  misc.add_argument('--outputOverwrite', type=str2bool, choices=[True, False], default=True,
                      help='Overwrite output folder path? Defaults to True.')
  
  args = parser.parse_args()
  print(args)

  # Sample path name of the directory created to store data and figures
  output_path               = args.output+'/'
  forceOverwrite            = args.outputOverwrite
  
  optionsStep1Path = args.options
  GreenRWPath = args.green
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
  if(os.path.isdir(output_path)):
    if(forceOverwrite):
      shutil.rmtree(output_path)
      print('['+sys._getframe().f_code.co_name+'] Output files root folder \''+output_path+'\' existed and has been deleted, as required by script.')
    else:
      sys.exit('['+sys._getframe().f_code.co_name+'] Output files root folder \''+output_path+'\' exists, and script is not set to overwrite. Rename or delete it before running again.')
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
    pickleDump(output_path+'options_iris_inp.pkl', options_IRIS)
  for i in range(len(mechanisms_data)):
    pickleDump(output_path+'mechanism_data_%05d.pkl' % (i), mechanisms_data.loc[i])
  
  # Generate atmospheric model (assume it is the same for all mechanisms).
  param_atmos = velocity_models.generate_default_atmos()
  
  # Loop on mechanisms.
  for i in range(len(mechanisms_data)):
    # Select mechanism and transform into dictionnary (more practical for further use).
    mechanism_ = mechanisms_data.loc[i]
    mechanism = {}
    keys_mechanism = ['EVID', 'stf', 'stf-data', 'zsource', 'f0', 'M0', 'M', 'STRIKE', 'DIP', 'RAKE', 'phi', 'station_tab', 'mt', 'domain']
    for key in keys_mechanism:
      mechanism[key] = mechanism_[key]
    
    # Load raw Green functions.
    print('[%s] Load raw Green functions anew (overwrite any previous mechanism).' % (sys._getframe().f_code.co_name))
    Green_RW = pickleLoad(GreenRWPath)
    
    # Set the current working folder.
    options['global_folder'] = (output_path+'mechanism_%05d/' % (i))
    Green_RW.set_global_folder(options['global_folder'])
    if(not os.path.isdir(Green_RW.global_folder)):
      os.makedirs(Green_RW.global_folder)
    
    # Compute Rayleigh wave field.
    print('[%s] Update Green functions with chosen focal mechanism (current mechanism ID = %d).' % (sys._getframe().f_code.co_name, i))
    Green_RW.update_mechanism(mechanism)
    print('[%s] Create the Rayleigh wave field.' % (sys._getframe().f_code.co_name))
    RW_field = RW_atmos.create_RW_field(Green_RW, mechanism['domain'], param_atmos, options)
    
    # Store outputs.
    pickleDump(Green_RW.global_folder+'RW_field.pkl', RW_field)

if __name__ == '__main__':
  main()