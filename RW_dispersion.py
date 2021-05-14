#!/usr/bin/env python3
import numpy as np
import os
# import pandas as pd
import matplotlib
# import matplotlib.pyplot as plt
# from pdb import set_trace as bp
import sys 
from multiprocessing import get_context
from utils import sysErrHdl
import read_earth_io as reo
import velocity_models, utils, RW_atmos
import inspect

## display parameters
font = {'size': 14}
matplotlib.rc('font', **font)

## To make sure that there is no bug when saving and closing the figures
## https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
matplotlib.use('Agg')

## Generate velocity and option files to run earthsr
def generate_model_for_earthsr(side, options):
  print('['+sys._getframe().f_code.co_name+'] Generate velocity and option files to run earthsr.')
  
  format_header = '%d %d %12.12f \n'
  format_string = '%12.12f %12.12f %12.12f %12.12f %12.12f %12.12f \n'
  format_phase  = '%12.12f %12.12f %d %d \n'
  format_freq   = '%d %d %12.12f %12.12f \n'
  
  ## Write input files - LEFT AND RIGHT
  #for nside in range(1,3):
  side['name']   = options['global_folder'] + '/input_code_earthsr'

  ## Generate link for dispersion code
  os.system('rm ' + './input_code_earthsr')
  os.system('ln ' + '-s ' + side['name'])

  ## Open file
  with open(side['name'], 'w') as f:
    f.write(format_header % (options['nb_layers'], options['earth_flattening'], options['ref_period']))
    for l in range(0, options['nb_layers']-1):
            f.write(format_string % (options['h'][l], side['vp'][l], side['vs'][l], side['rho'][l], side['Qa'][l], side['Qb'][l]))
    hend = 0.
    f.write(format_string % (hend, side['vp'][l],side['vs'][l], side['rho'][l], side['Qa'][l], side['Qb'][l])) 
    # Surface wave type.  1 = Rayleigh; <>1 Love.  In this case we choose the Rayleigh option
    f.write('%d\n' % (options['type_wave'] ))
    # Filename of binary output of dispersion curves.  In this case it is called "ray"
    txt_type = 'ray' 
    f.write('%s\n' % (txt_type))
    # min and max phase velocities and min and max branch (mode) numbers.
    f.write(format_phase % (options['min_max_phase'][0], options['min_max_phase'][1], options['nb_modes'][0], options['nb_modes'][1]))
    # Number of sources, number of frequencies, frequency interval and starting (lowest) frequency.
    f.write(format_freq % (options['nb_source'], options['nb_freq'], options['df'], options['freq_range'][0]))
    # Source depths in km.
    f.write('%12.12f \n' % (options['source_depth']))
    # Receiver depths in km.
    f.write('%12.12f \n' % (options['receiver_depth']))
    # This this point the program loops over another set of input lines starting with the surface
    # wave type (1st line after model).
    f.write('%d \n' % (options['Loop']))

def local_collect(title, N, periods):
  return (reo.read_egnfile_allper(title, periods, N), periods)

## Collect eigenfunctions and derivatives from earthsr
def get_eigenfunctions(current_struct, options, ncpu=16):
  print('['+sys._getframe().f_code.co_name+'] Create Green functions object. Collect eigenfunctions and derivatives from earthsr and input them to the object.')

  import multiprocessing as mp
  from functools import partial
  
  ## Construct RW spectrum object 
  Green_RW = RW_atmos.RW_forcing(options)

  periods = 1./np.linspace(options['f_tab'][-1], options['f_tab'][0], len(options['f_tab']))
  
  # UNUSED.
  # uz_tab = []
  # freq_tab  = [[] for ii in range(0,options['nb_modes'][1]+1) ]
  # freqa_tab = [[] for ii in range(0,options['nb_modes'][1]+1) ]
  
  N = ncpu
  list_of_lists = np.array_split(periods, N)
  
  local_collect_partial = partial(local_collect, options['global_folder'] + 'eigen.input_code_earthsr', N)
  
  # ## Setup progress bar
  # toolbar_width = 40
  # total_length  = len(periods) * (options['nb_modes'][1]+1)
  # # sys.stdout.write("Building eigenfunctions: [%s]" % (" " * toolbar_width))
  print('['+sys._getframe().f_code.co_name+'] > Building eigenfunctions from earthsr output.')
  # sys.stdout.write("["+sys._getframe().f_code.co_name+"] Building eigenfunctions: [%s]" % (" " * toolbar_width))
  # sys.stdout.flush()
  # #sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
  
  if N == 1:
    results = [local_collect_partial(periods)]
  else:
    if options['USE_SPAWN_MPI']:
      with get_context("spawn").Pool(processes = N) as p:
        results = p.map(local_collect_partial, list_of_lists)
    else:
      with mp.Pool(processes = N) as p:
        results = p.map(local_collect_partial, list_of_lists)
                          
  # sys.stdout.write("] Done\n")
  
  # ## Setup progress bar
  # toolbar_width = 40
  # total_length  = len(periods) * N
  # sys.stdout.write("["+sys._getframe().f_code.co_name+"] Store eigenfunctions: [%s]" % (" " * toolbar_width))
  print('['+sys._getframe().f_code.co_name+'] > Store eigenfunctions in the local Green functions object.')
  # sys.stdout.flush()
  # id_stat = 0
  # cptbar = 0
  
  offset = 0
  for reoobj_ in results:
    reoobj   = reoobj_[0]
    periods_ = reoobj_[1]
    for iperiod, period in enumerate(periods_):
      iperiod_ = offset + iperiod
      #reoobj=reo.read_egnfile_per(options['global_folder'] + 'eigen.input_code_earthsr', period)
      
      dep     = reoobj.dep
      omega   = 2*np.pi/period
      
      orig_b1 = reoobj.uzmat[iperiod]
      orig_b2 = reoobj.urmat[iperiod]
      orig_b3 = reoobj.tzmat[iperiod]
      orig_b4 = reoobj.trmat[iperiod]
      kmode   = reoobj.wavnum[iperiod].reshape(1,len(reoobj.wavnum[iperiod]))
              
      # origdep = reoobj.dep
      # nmodes  = orig_b1.shape[1]
      mu      = reoobj.mu.reshape(len(reoobj.mu),1)
      lamda   = reoobj.lamda.reshape(len(reoobj.mu),1)
      rho     = reoobj.rho
      kmu    = np.dot(mu,kmode)
      klamda = np.dot(lamda,kmode)
      
      # Eq. (7.28) Aki-Richards
      # r1 = b2 r2 = b1
      # r3 = b4 r4 = b3
      d_b2_dz = (omega*orig_b4-np.multiply(kmu,orig_b1))/mu # numpy.multiply does element wise array multiplication
      d_b1_dz = (np.multiply(klamda,orig_b2)+omega*orig_b3)/(lamda+2*mu)
      # dxz     = np.gradient(orig_b2[:,0])
      # dzz     = np.gradient(orig_b1[:,0])
      
      ## Construct Green's function for a given period 
      Green_RW.add_one_period(period, iperiod_, current_struct, rho, orig_b1, orig_b2, d_b1_dz, d_b2_dz, kmode, dep)
      
      # ## Update progress bar
      # id_stat += 1
      # if(int(toolbar_width*id_stat/total_length) > cptbar):
      #         cptbar = int(toolbar_width*id_stat/total_length)
      #         sys.stdout.write("-")
      #         sys.stdout.flush()
    
    offset += len(periods_)
      
  ## Deallocate
  del results
  
  print('['+sys._getframe().f_code.co_name+'] > Update Green functions object frequency array based on the eigenfunctions that were found.')
  Green_RW.update_frequencies()
  
  # sys.stdout.write("] Done\n")
  print('['+sys._getframe().f_code.co_name+'] > Finished.')
      
  return Green_RW
                
def compute_dispersion_with_earthsr(no, side, options):
  print('['+sys._getframe().f_code.co_name+'] Run earthsr.')
  print('****************************************************************')
  ## Launch dispersion code
  #print(' model: ' + side['name'])
  # os.system('./bin/earthsr ' + 'input_code_earthsr')
  sysErrHdl(sys.path[0]+'/bin/earthsr '+'input_code_earthsr')
  print('****************************************************************')

def move_dispersion_files(no, options):
  print('['+sys._getframe().f_code.co_name+'] Move earthsr files (./disp*, ./eigen*) to \''+options['global_folder']+'\' using a system command.')
  os.system('mv '+'./disp* ' + options['global_folder'])
  os.system('mv '+'./eigen* ' + options['global_folder'])
  if(no > 0):
    os.system('mv ' + 'tocomputeIO* ' + options['global_folder'])

################################################################################################
## Before finishing building coefficients, this routine saves dispersion characteristics to file
def collect_dispersion_from_earthsr_and_save(nside, options):
  print('['+sys._getframe().f_code.co_name+'] Read earthsr output files.')
  
  data_dispersion_file_fund   = utils.load(options['global_folder'] + 'disp_vconly.input_code_earthsr')

  data_dispersion = [{} for i in range(0, options['nb_modes'][1])]
  list_modes_side = [{} for j in range(0, options['nb_modes'][1])]
  for nmode in range(0, options['nb_modes'][1]):
    list_modes_side[nmode]['loc']  = np.where(data_dispersion_file_fund[:,0] == nmode)

    # freq_domain = 0
    if(list_modes_side[nmode]['loc'][0].size > 0):
      data_dispersion[nmode]['period'] = data_dispersion_file_fund[list_modes_side[nmode]['loc'][0],1]
      data_dispersion[nmode]['cphi']   = data_dispersion_file_fund[list_modes_side[nmode]['loc'][0],2]
      data_dispersion[nmode]['cg']     = data_dispersion_file_fund[list_modes_side[nmode]['loc'][0],3]
      data_dispersion[nmode]['QR']     = data_dispersion_file_fund[list_modes_side[nmode]['loc'][0],4]   

    ## Add nan for periods where 1st mode has not been calculated
    if(nmode > 0 and list_modes_side[nmode]['loc'][0].size > 0):
      cpt       = len(data_dispersion[nmode]['period'])-1
      save_cphi = data_dispersion[nmode]['cphi'][-1]*0. + np.inf
      save_cg   = data_dispersion[nmode]['cg'][-1]*0. + np.inf
      save_QR   = data_dispersion[nmode]['QR'][-1]*0. + np.inf
      while data_dispersion[nmode]['period'][-1] < data_dispersion[0]['period'][-1]:
        cpt += 1
        data_dispersion[nmode]['period'] = np.concatenate([data_dispersion[nmode]['period'], [data_dispersion[0]['period'][cpt]]])
        data_dispersion[nmode]['cphi']   = np.concatenate([data_dispersion[nmode]['cphi'], [save_cphi]])
        data_dispersion[nmode]['cg']     = np.concatenate([data_dispersion[nmode]['cg'], [save_cg]])
        data_dispersion[nmode]['QR']     = np.concatenate([data_dispersion[nmode]['QR'], [save_QR]])

  ## Save with name "current_struct" to be consistent with resonance_eigen
  current_struct = data_dispersion
  for nmode in range(0, len(current_struct)):
    if( len(current_struct[nmode]) > 0 ):
      current_struct[nmode]['fks'] = 1./current_struct[nmode]['period']
      
  utils.save_dict(current_struct, options['global_folder'] + 'PARAM_dispersion.mat')
  
  return current_struct

def get_default_options():
  options = {} 
  options['GOOGLE_COLAB'] = False      
  
  ##########
  ## Options
  options['dimension']   = 2
  options['dimension_seismic'] = 2
  options['PLOT_RW_time_series'] = False
  options['COMPUTE_MAPS'] = False
  options['ATTENUATION']  = False
  
  options['nb_modes']    = [0, 5] # min / max
  options['type_wave']   = 1 # Surface wave type.  (1 = Rayleigh; >1 = Love.)
  options['way_forward'] = 1
  options['LOAD_2D_MODEL'] = False
  options['nb_layers']     = 1600#2800
  options['nb_freq']       = 128*4 # Number of frequencies
  options['chosen_header'] = 'coefs_earthsr_sol_'
  options['PLOT']          = 1# 0 = No plot; 1 = plot after computing coef.; 2 = plot without computing coef.
  options['PLOT_folder']   = 'coefs_python_1.2_vs0.5_poisson0.25_h1.0_running_dir_1'
  #options['PLOT_folder']   = 'coefs_python_0.0_17500.0_running_dir_1'
  options['ONLY_purely_1d'] = False

  ## Hetergeneous structure
  options['type_model']    = 'specfem2d'
  options['models'] = {}
  options['models_dimension'] = {}
  #options['models']['specfem'] = '/home/quentin/Documents/DATA/CODES/eclipse_workspace/GIT-DG/current/specfem-dg/EXAMPLES/Ridgecrest_test_38624623_Hare_notopo/Ridgecrest_seismic.txt'
  options['models']['specfem'] = './Ridgecrest_seismic.txt'
  options['models_dimension']['specfem'] = 1
  #options['models']['specfem'] = '/home/quentin/Documents/DATA/Ridgecrest/seismic_models/Ridgecrest_seismic.txt'
  #options['models']['specfem'] = '/home/quentin/Documents/DATA/Ridgecrest/Ridgecrest_SSD/simulations/Ridgecrest_mesh_simu_fine_batch2_3/Ridgecrest_seismic.txt'
  options['chosen_model'] = 'specfem'
  options['zmax'] = 80000.

  ##############
  ## Auxiliaries
  # A1D   = {}
  # A1Dst = {}
  options['dir_earthsr'] = os.path.dirname(os.path.abspath(inspect.getfile(get_default_options)))+'/bin/'
  options['earth_flattening'] = 0 # Earth flattening control variable (0 = no correction; >0 applies correction)
  options['ref_period']  = 10. # Reference period for dispersion correction (0 => none) Generally you would just pick a period shorter than anything you are going to model
  options['output_file'] = 'dispers' # Filename of binary output of dispersion curves.
  options['min_max_phase'] = [0, 0] # min and max phase velocities and min and max branch (mode) numbers. Note that if we choose the min and max phase velocities to be 0, the program will choose the phase velocity range itself.  In this case case we ask the program to figure out the appropriate range (0.0000000       0.0000000) and solve modes 0 (fundamental) to 4.
  options['nb_source'] = 1 # Number of sources
  options['source_depth']   = 6.8 # (km)
  options['receiver_depth'] = 0 # (km)
  options['coef_low_freq']  = 0.001
  options['coef_high_freq'] = 0.5#1.
  options['Loop']           = 0 # This this point the program loops over another set of input lines starting with the surface wave type (1st line after model).  If this is set to zero, the program will terminate.
  
  return(options)

def compute_Green_functions(options_in = {}, ncpu=16):
  print('['+sys._getframe().f_code.co_name+'] Compute Rayleigh waves\' Green functions.')
  print('['+sys._getframe().f_code.co_name+'] > Will run earthsr to obtain the dispersion relations.')
  
  options = get_default_options() # Get default options.
  options.update(options_in) # Update each option (overwrite defaults) based on user input.
  
  # Define frequency domain and store in options.
  f_tab = np.linspace(options['coef_low_freq'], options['coef_high_freq'], options['nb_freq'])
  options['f_tab']   = f_tab
  options['df']      = abs( f_tab[1] - f_tab[0] )
  options['freq_range'] = [f_tab[0], f_tab[-1]]
  
  Green_RW = []
  if(options['PLOT'] < 2):
  
    ###########################################
    ## Build right frequency and spatial ranges
    options_loc = utils.determine_folders(options)
    options.update( options_loc )

    ##############################
    ## Loop over frequency domains
    # freq_domain = 0

    ## Determine adapted model depth for this frequency regime
    #options_loc = get_depth_model(freq_domain, options)
    #options.update( options_loc )
    
    ## Create directory for earthsr eigenfunctions
    #os.makedirs(options['global_folder'])
    
    ## TODO: Creation side vs models
    side = velocity_models.create_velocity_model(options)
    
    ## Create file to use earthsr
    generate_model_for_earthsr(side, options)
    
    no = 0
    compute_dispersion_with_earthsr(no, side, options) # Compute and store dispersion characteristics using earthsr
    
    ## Compute purely1d coefficients
    move_dispersion_files(no, options)
    
    current_struct = collect_dispersion_from_earthsr_and_save(0, options)
    current_struct = [key for key in current_struct if key]
    options['nb_modes'] = [0, len(current_struct)] ## Update modes if necessary
    
    ## Create velocity figures
    velocity_models.create_velocity_figures(current_struct, options)
    
    ## Class containing routine to construct RW/acoustic spectrum at a given location
    Green_RW = get_eigenfunctions(current_struct, options, ncpu=ncpu)
    
    # ## Compute sensitivity maps
    # if(False):
    #   generate_sensitivity_maps(current_struct, Green_RW, options)
      
  print('['+sys._getframe().f_code.co_name+'] Finished computing Rayleigh waves\' Green functions.')
          
  return(Green_RW, options)