#!/usr/bin/env python3
import numpy as np
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib
# import matplotlib.pyplot as plt
# from pdb import set_trace as bp
import pickle
import sys
import lzma
import argparse

## display parameters
font = {'size': 14}
matplotlib.rc('font', **font)

## To make sure that there is no bug when saving and closing the figures
## https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
matplotlib.use('Agg')

def str2bool(v):
    # Useful for argparse booleans.
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def autoAdjustCLim(h):
  (vmin, vmax) = h.get_clim()
  h.set_clim(np.array([-1,1])*np.max(np.abs([vmin, vmax])))

def sysErrHdl(cmd):
  r = os.system(cmd)
  if(r!=0):
    sys.exit('os.system(\''+cmd+'\') failed with error code '+str(r)+'.')

def earthsr_local_folder():
    #return('/staff/quentin/Documents/Codes/RW_atmos')
    return('/Users/lmartire/Documents/software/rw_atmos_leo/bin/')

def pickleDump(fname, var):
  with open(fname, 'wb') as handle:
    pickle.dump(var, handle)
def pickleDumpLZMA(fname, var):
  with lzma.open(fname, 'wb') as handle:
    pickle.dump(var, handle)
def pickleLoad(fname):
  with open(fname, 'rb') as handle:
    var = pickle.load(handle)
  return(var)

def loadAtmosphericModel(path):
  try:
    model = pd.read_csv(path, delim_whitespace=True, header=None)
    model.columns = ['z', 'rho', 'cpa', 'p', 'g', 'kappa', 'mu', 'bulk', 'wx', 'wy', 'cp', 'cv', 'gamma']
  except:
    sys.exit('[%s] Could not read atmospheric model file under the right format.' % (sys._getframe().f_code.co_name))
    # # Eventually try to read another format.
    # model = pd.read_csv(path, delim_whitespace=True, header=None)
    # model.columns = ['z', 'rho', 'dummy1', 'cpa', 'p', 'dummy2', 'g', 'dummy3', 'kappa', 'mu', 'dummy4', 'dummy5', 'dummy6', 'wx', 'cp', 'cv', 'gamma']
    # model['bulk'] = model['mu']
    # model['wy'] = model['wx']
  return(model)

#################################
## Routine to read SPECFEM models
def read_specfem_files(options):
        print('['+sys._getframe().f_code.co_name+'] Read SPECFEM 1D models: '+str(options['models']))
        
        unknown_tab = ['rho', 'vs', 'vp', 'Qp', 'Qs']
        # id_tab      = [1, 3, 2]

        data = {}
        zover0 = []
        for imodel in options['models']:
                data[imodel] = {}
                #for unknown in unknown_tab:
                temp   = pd.read_csv( options['models'][imodel], delim_whitespace=True, header=None )
                temp.columns = ['z', 'rho', 'vp', 'vs', 'Qs', 'Qp']
                
                if(temp['z'].iloc[0] > 0):
                        temp_add = temp.loc[ temp['z'] == temp['z'].min() ].copy()
                        temp_add.loc[0, 'z'] = 0.
                        temp = pd.concat([temp_add, temp]).reset_index()
                
                temp_add = temp.loc[ temp['z'] == temp['z'].max() ].copy()
                temp_add['z'].iloc[0] = 1.e7
                
                temp = pd.concat([temp, temp_add]).reset_index()
                
                zover0 = temp[ 'z' ].values
                cpt_unknown = -1
                for unknown in unknown_tab:
                        cpt_unknown += 1
                        data[imodel][unknown] = temp[ unknown ].values
                        
        
        return zover0, data
        
####################################
## Routine to read SPECFEM 2d models
def read_specfem2d_files(options):
        print('['+sys._getframe().f_code.co_name+'] Read SPECFEM 2D models: '+str(options['models']))
        
        from velocity_models import read_csv_seismic
        unknown_tab = ['rho', 'vs', 'vp', 'Qp', 'Qs']

        data = {}
        zover0 = []
        for imodel in options['models']:
                data[imodel] = {}
                #for unknown in unknown_tab:
                temp = read_csv_seismic(options['models'][imodel], options['models_dimension'][imodel], loc_source = 50000.)
                
                ## TO REMOVE
                #temp.loc[(temp['z'] < 2200) & (temp['z'] > 500), 'vs'] /= 1.5
                
                zover0 = temp[ 'z' ].values
                cpt_unknown = -1
                for unknown in unknown_tab:
                        cpt_unknown += 1
                        data[imodel][unknown] = temp[ unknown ].values
                        if(unknown == 'Qs'):
                                data[imodel][unknown] = 0.05 * data[imodel]['vs']
                        if(unknown == 'Qp'):
                                data[imodel][unknown] = 0.1 * data[imodel]['vs']
        
        return zover0, data
        
################################################################
## Choose the name of the temporary folder to store coefficients
def determine_folders(options):
        # print('['+sys._getframe().f_code.co_name+'] Define working temporary folders for storing coefficients.')
        
        options_loc = {}

        ## Check current folder
        pattern = 'coefs_batch'
        nbdirs = [int(f.split('_')[-1]) for f in os.listdir(options['dir_earthsr']) if pattern in f and not os.path.isfile(os.path.join(options['dir_earthsr'], f))]
        if(nbdirs):
                nbdirs = max(nbdirs)
        else:
                nbdirs = 0
                
        name_simu_folder = './coefs_batch_' + str(nbdirs+1) + '/'
        
        if(os.path.isdir(name_simu_folder)):
          print('['+sys._getframe().f_code.co_name+', WARNING] Folder \''+name_simu_folder+'\' already exists, contents may be overwritten.')
        else:
          if(options['PLOT'] < 2):
            os.makedirs(name_simu_folder)
                
        options_loc['name_simu_subfolder'] = ''
        options_loc['global_folder']       = name_simu_folder + options_loc['name_simu_subfolder']
        
        return options_loc
        
def load(file_name, delimiter = ' '):
    print('['+sys._getframe().f_code.co_name+'] Load \''+file_name+'\'.')
    
    file_r = open(file_name, 'r') 
    data   = file_r.readlines() 
    
    data_array = []
    for line in data:
            data_current = line.strip().split(delimiter)
            data_current = list(filter(None, data_current))
            data_array.append( [float(idata) for idata in data_current] )

    return np.array(data_array)

def save_dict(dict_to_save, filename):

    afile = open(filename, 'wb')
    pickle.dump(dict_to_save, afile)
    afile.close()
        
def generate_trace(t, data, freqmin, freqmax):
                
    import obspy
    tr = obspy.Trace()
    tr.data = data
    tr.stats.delta     = abs( t[1] - t[0] )
    tr.stats.station   = 'station'
    tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
    
    return tr
    
def align_yaxis_np(axes):

    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array(axes)
    extrema = np.array([ax.get_ylim() for ax in axes])

    # reset for divide by zero issues
    for i in range(len(extrema)):
        if np.isclose(extrema[i, 0], 0.0):
            extrema[i, 0] = -1
        if np.isclose(extrema[i, 1], 0.0):
            extrema[i, 1] = 1

    # upper and lower limits
    lowers = extrema[:, 0]
    uppers = extrema[:, 1]

    # if all pos or all neg, don't scale
    all_positive = False
    all_negative = False
    if lowers.min() > 0.0:
        all_positive = True

    if uppers.max() < 0.0:
        all_negative = True

    if all_negative or all_positive:
        # don't scale
        return

    # pick "most centered" axis
    res = abs(uppers+lowers)
    min_index = np.argmin(res)

    # scale positive or negative part
    multiplier1 = abs(uppers[min_index]/lowers[min_index])
    multiplier2 = abs(lowers[min_index]/uppers[min_index])

    for i in range(len(extrema)):
        # scale positive or negative part based on which induces valid
        if i != min_index:
            lower_change = extrema[i, 1] * -1*multiplier2
            upper_change = extrema[i, 0] * -1*multiplier1
            if upper_change < extrema[i, 1]:
                extrema[i, 0] = lower_change
            else:
                extrema[i, 1] = upper_change

        # bump by 10% for a margin
        extrema[i, 0] *= 1.1
        extrema[i, 1] *= 1.1

    # set axes limits
    [axes[i].set_ylim(*extrema[i]) for i in range(len(extrema))]
    
def concat_df_complex(A, B, groupby_lab):
        
    f        = A[groupby_lab].values
    mat_temp = B.drop([groupby_lab], axis=1).values
    
    mat      = A.drop([groupby_lab], axis=1).values
    mat[:mat_temp.shape[0], :] += mat_temp
    A = pd.DataFrame(mat)
    A.columns = np.arange(0, mat.shape[1])
    A[groupby_lab]    = f
    
    return A