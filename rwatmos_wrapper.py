import os
import RW_atmos, RW_dispersion, velocity_models
import mechanisms as mod_mechanisms
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
from pyrocko import moment_tensor as mtm
from obspy.imaging.beachball import beach
from utils import sysErrHdl
import sys
import numpy as np
import shutil

#--------------------------------------------------------------#
# Setup options.                                               #
#--------------------------------------------------------------#

# Sample path name of the directory created to store data and figures
output_root               = './OUTPUTS/'
name_sample               = output_root+'RUN_XXX/'
forceOverwrite            = True

# RW-atmos integration options
options = {}
options['dimension']         = 3 # atmospheric dimension
options['dimension_seismic'] = 3 # seismic dimension
options['ATTENUATION']    = True # using Graves, Broadband ground-motion simulation using a hybrid approach, 2014
options['COMPUTE_MAPS']   = False # Compute and plot x,y,z wavefield. Computationally heavy.
options['COMPUTE_XY_PRE'] = 20.0e3 # Compute xy wavefield above source and at given altitude. Computationally heavy.
options['nb_freq']        = 2**9
options['nb_kxy']         = 2**7
options['coef_low_freq']  = 0.001 # minimum frequency (Hz)
options['coef_high_freq'] = 5. # maximum frequency (Hz)
options['nb_layers']      = 100 # Number of seismic layers for discretization
options['zmax']           = 10.0e3 # maximum depth (m)
options['nb_modes']       = [0, 50] # min, max number of modes
# options['t_chosen']       = [0., 90.] # time (s) to display wavefield
options['t_chosen']       = np.linspace(0,25,4)
options['models'] = {}
options['models']['specfem'] = './models/Ridgecrest_seismic.txt'
options['type_model']        = 'specfem' # specfem or specfem2d

# Source parameters
options_source = {}
options_source['stf-data'] = [] # file where stf is located
options_source['stf']      = 'gaussian' # gaussian or erf
options_source['f0']       = 2. # dominant freuqency (Hz)
options_source['lat_min']  = 35. 
options_source['lat_max']  = 36.
options_source['lon_min']  = -118.
options_source['lon_max']  = -117.

# Load sources from Earthquake catalog or build custom source
options_source['DIRECTORY_MECHANISMS'] = []
options_source['sources'] = []
source_characteristics = { # example of custom source
    'id': 0,
    'time': UTCDateTime(2019, 8, 9, 0, 9, 57),
    'mag': 2.98,
    'lat': 35.895,
    'lon': -117.679,
    'depth': 4.1, #km
    'strike': 159,
    'dip': 89,
    'rake': -156,
}
options_source['sources'].append( source_characteristics )
list_of_events = [0] # list of event to compute, leave empty if you want all    
options_source['add_SAC'] = False # Wheter or not add real station locations downloaded from IRIS within the domain
                                  # boundaries defined in options_source['lat_min'], options_source['lat_max'], ...

# Options for ground stations.
options_IRIS = {}
# options_IRIS['network'] = 'CI,NN,GS,SN,PB,ZY' # Only if need to download stations.
# options_IRIS['channel'] = 'HHZ,HNZ,DPZ,BNZ,BHZ,ENZ,EHZ' # Only if need to download stations.
options_IRIS['stations'] = {}
i=0
options_IRIS['stations'][i] = mod_mechanisms.create_one_station(x=0., y=-50.0e3, z=0., comp='p', name='station', id=i); i+=1;

# Balloon stations.
options_balloon = {}

#--------------------------------------------------------------#
# Start script.                                                #
#--------------------------------------------------------------#

# Check output path is free, make it if necessary.
if(os.path.isdir(output_root)):
  if(forceOverwrite):
    shutil.rmtree(output_root)
    print('['+sys._getframe().f_code.co_name+'] Output files root folder \''+output_root+'\' existed and has been deleted, as required by script.')
  else:
    sys.exit('['+sys._getframe().f_code.co_name+'] Output files root folder \''+output_root+'\' exists, and script is not set to overwrite. Rename or delete it before running again.')
os.makedirs(output_root)

# Plot sources' beachballs.
for source in options_source['sources']:
  mw = source['mag']
  m0 = mtm.magnitude_to_moment(mw)  # convert the mag to moment
  strike, dip, rake = source['strike'], source['dip'], source['rake']
  mt = mtm.MomentTensor(strike=strike, dip=dip, rake=rake, scalar_moment=m0)
  
  fig = plt.figure()
  ax = fig.add_subplot(111)

  aa = (mt.m6_up_south_east()*m0).tolist()
  bball = beach(aa, xy=(0., 0.), width=200, linewidth=1, axes=ax)
  bball.set_zorder(100)
  ax.add_collection(bball)

  plt.title('$M_w$ ' + str(round(mw, 2)) + ' - $f_0 = ' +str(round(options_source['f0'],3))+ '$ Hz'+ ' - depth $ = ' +str(round(source['depth'],3))+ '$ km')
  plt.xlim([-0.5, 0.5])
  plt.ylim([-0.5, 0.5])
  
# ugly hack: copy options from one dict to another & initialize other options only relevant to Ridgecrest
options_source['coef_high_freq'] = options['coef_high_freq']
options_source['nb_kxy']   = options['nb_kxy']
options_source['t_chosen'] = options['t_chosen']
options_source['activate_LA'] = False # Only relevant for Ridgecrest study
options_source['rotation'] = False 
options['USE_SPAWN_MPI'] = False
options['force_dimension'] = False # Only when add_specfem_simu = True
options['force_f0_source'] = False
mechanism, station, domain = {}, {}, {}
keys_mechanism = ['EVID', 'stf', 'stf-data', 'zsource', 'f0', 'M0', 'M', 'phi', 'station_tab', 'mt']

# Load mechanisms/stations data
mechanisms_data = mod_mechanisms.load_source_mechanism_IRIS(options_source, options_IRIS, dimension=options['dimension'], 
                                                            add_SAC = options_source['add_SAC'], add_perturbations = False, 
                                                            specific_events=list_of_events, options_balloon=options_balloon)

Green_RW, options_outrw = RW_dispersion.compute_trans_coefficients(options)
options.update(options_outrw)

tmpFolderForCopy = name_sample.replace('XXX', 'tocopy')
os.makedirs(tmpFolderForCopy)

# Move temporary folder in new folder
sysErrHdl('mv ' + options['global_folder'][:-1]+' '+tmpFolderForCopy)
# os.system('mv ' + options_out['global_folder'] + ' ' + name_sample.replace('XXX', 'tocopy'))

# Save all mechanisms to current folder
mod_mechanisms.save_mt(mechanisms_data, tmpFolderForCopy)
# Loop over each mechanism to generate the atmospheric wavefield
for imecha, mechanism_ in mechanisms_data.iterrows():

    options['global_folder'] = name_sample.replace('XXX', str(imecha+1))
    sysErrHdl('cp -R ' + tmpFolderForCopy[:-1] + ' ' + options['global_folder'])
    # os.system('cp -R ' + name_sample.replace('XXX', 'tocopy')[:-1] + ' ' + options_out['global_folder'])
    Green_RW.set_global_folder(options['global_folder'])

    mechanism = {}
    for key in keys_mechanism:
            mechanism[key] = mechanism_[key]

    # Station distribution
    mod_mechanisms.display_map_stations(mechanism_['EVID'], mechanism_['station_tab'], mechanism_['domain'], options['global_folder'])
    
# # Save all mechanisms to current folder
# mod_mechanisms.save_mt(mechanisms_data, name_sample.replace('XXX', 'tocopy'))
# # Loop over each mechanism to generate the atmospheric wavefield
# for imecha, mechanism_ in mechanisms_data.iterrows():
    
#     options_out['global_folder'] = name_sample.replace('XXX', str(imecha+1))
#     os.system('cp -R ' + name_sample.replace('XXX', 'tocopy')[:-1] + ' ' + options_out['global_folder'])
#     Green_RW.set_global_folder(options_out['global_folder'])

#     mechanism = {}
#     for key in keys_mechanism:
#             mechanism[key] = mechanism_[key]

#     # Station distribution
#     mod_mechanisms.display_map_stations(mechanism_['EVID'], mechanism_['station_tab'], mechanism_['domain'], options_out['global_folder'])
    
    # Generate atmospheric model
    station = mechanism_['station_tab']
    domain  = mechanism_['domain']
    param_atmos = velocity_models.generate_default_atmos()

    # Solve dispersion equations
    RW_atmos.compute_analytical_acoustic(Green_RW, mechanism, param_atmos, station, domain, options)


