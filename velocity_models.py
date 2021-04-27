#!/usr/bin/env python3
import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pdb import set_trace as bp
import sys 
from scipy import interpolate
from pyrocko import moment_tensor as mtm

## Local modules
import utils
try:
    sys.path.append('/staff/quentin/Documents/Projects/Ridgecrest/')
    from extract_velocity_ucvm import get_velocity_ucvm
    UCVM_available = True
except:
    UCVM_available = False

## display parameters
font = {'size': 14}
matplotlib.rc('font', **font)

## To make sure that there is no bug when saving and closing the figures
## https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
matplotlib.use('Agg')

def create_velocity_model_(mt, options, add_Yang_layer, tomo_model='cvmsi'):

        ## Exit message if no UCVM library found
        if not UCVM_available:
            sys.exit('No UCVM library found. Can not build velocity model')

        ## Options
        offset = 0.1
        options_in = {}
        options_in['nb_lats']   = 3
        options_in['nb_depths'] = 50000
        
        options_in['lats'] = [mt['LAT'], mt['LAT']+offset] #Y
        options_in['lons'] = [mt['LON'], mt['LON']+offset] #X
        
        #options_in['lats'] = [mt['balloons']['Tortoise']['balloon']['Lat'], mt['balloons']['Tortoise']['balloon']['Lat']+offset] #Y
        #options_in['lons'] = [mt['balloons']['Tortoise']['balloon']['Lon'], mt['balloons']['Tortoise']['balloon']['Lon']+offset] #X
        
        #options_in['depths'] = [0., 50000.] #Z
        options_in['depths'] = [0., -1*options['zmax']] #Z
        options_in['cvm_model'] = 'cvms5'
        options_in['cvm_model'] = 'cvmsi'
        options_in['cvm_model'] = 'cvmh'
        options_in['cvm_model'] = tomo_model
        #options_in['gtl'] = ',elygtl:ely -z 0,350'
        options_in['gtl'] = ''
        
        options_in['profiles'] = True
        
        ## Auxiliaries
        data_vs = get_velocity_ucvm(options_in)
        data_vs = data_vs.loc[ data_vs['cr_vs'] > 1 ].iloc[::options_in['nb_lats']]
        
        if add_Yang_layer:
                file_yang = '/staff/quentin/Documents/Projects/Ridgecrest/seismic_models/Yang_Vs3km.txt'
                vs_yang   = pd.read_csv(file_yang, header=None, delim_whitespace=True)
                vs_yang   = vs_yang.values
                
                zmax = float(vs_yang.shape[0])
                z_  = np.linspace(0., zmax, vs_yang.shape[0])
                
                vs_ = np.median(vs_yang, axis=1)
                f   = interpolate.interp1d(z_, vs_*1000., kind='linear')
                idx_z = (data_vs['Z']<=zmax).nonzero()[0]
                z_interp  = data_vs['Z'].values[idx_z]
                vs_interp = f(z_interp)
                data_vs.loc[data_vs['Z']<=zmax, 'cr_vs'] = vs_interp
                
                ## Empirical relations (Brocher, 2005a)
                ## eq. (6)
                vp_ = 0.9409 + 2.0947 * vs_ \
                                   - 0.8206 * vs_**2 \
                                   + 0.2683 * vs_**3 \
                                   - 0.0251 * vs_**4
                f   = interpolate.interp1d(z_, vp_*1000., kind='linear')
                vp_interp = f(z_interp)
                data_vs.loc[data_vs['Z']<=zmax, 'cr_vp'] = vp_interp
                
                ## eq. (1)
                rho_  =  1.6612 * vp_ \
                           - 0.4721 * vp_**2 \
                           + 0.0671 * vp_**3 \
                           - 0.0043 * vp_**4 \
                           + 0.000106 * vp_**5
                f   = interpolate.interp1d(z_, rho_*1000., kind='linear')
                rho_interp = f(z_interp)
                data_vs.loc[data_vs['Z']<=zmax, 'cr_rho'] = rho_interp

        ## Save data
        mt['vs'] = data_vs['cr_vs'].values
        mt['vp'] = data_vs['cr_vp'].values
        mt['rho'] = data_vs['cr_rho'].values
        mt['z']   = data_vs['Z'].values
        
        return mt
    
def construct_local_seismic_model(mechanism_data, options):

        name_file = '/staff/quentin/Documents/Projects/Ridgecrest/seismic_models/model_temp.txt'
        data = np.c_[abs(mechanism_data['z']), mechanism_data['rho'], mechanism_data['vp'], mechanism_data['vs'], 0.05*mechanism_data['vs'], 0.1*mechanism_data['vs']]
        np.savetxt(name_file, data)
        
        return (name_file, 'specfem')
        
###############################################################
## Create adapted velocity model on both sides of the interface 
def create_velocity_model(options):
        print('['+sys._getframe().f_code.co_name+'] Create adapted velocity model, with '+str(options['nb_layers'])+' layers.')
        
        ## Definition
        side = {}
        unknown_tab  = ['rho', 'vs', 'vp', 'Qp', 'Qs']
        
        ## Jack's files
        options['z'] = np.arange(0.,100000,2800)
        
        if(options['type_model'] == 'specfem'):
                zover0, data = utils.read_specfem_files(options)
        elif(options['type_model'] == 'specfem2d'):
                zover0, data = utils.read_specfem2d_files(options)
        else:
                sys.exit('Trying to load an external velocity model that is not supported: ' + options['type_model'])     
                   
        options['z'] = zover0.copy()
        z_interp, data_interp = discretize_model_heterogeneous(data, options)

        ## Back to the right format for earthsr and SWRT
        options['z']  = z_interp.copy()/1000.
        options['dz'] = np.diff(options['z'])[0]
        options['h']  = np.diff(options['z'])
        options['nb_layers'] = len(options['z'])
        
        chosen_model = options['chosen_model']
        for iunknown in unknown_tab:
        
                side[iunknown] = data_interp[chosen_model][iunknown].copy()
                
        side['z']  = options['z'].copy()
        ## Add attenuation if needed
        side['Qa'] = side['Qp']*0. + 9999. # P-wave Q
        side['Qb'] = side['Qs']*0. + 9999. # S-wave Q
        if(options['ATTENUATION']):
                side['Qa'] = side['Qp']*1000. # P-wave Q
                side['Qb'] = side['Qs']*1000. # S-wave Q
        
        return side

def create_velocity_figures(current_struct, options):

        nbmodes = np.sum([1 for key in current_struct if key])
        fig, axs = plt.subplots(nrows=nbmodes, ncols=1, sharex=True, sharey=True)
        
        if nbmodes == 1:
                selected_axs = axs
        else:
                selected_axs = axs[-1]
        
        selected_axs.set_ylabel('Velocity (km/s)')
        selected_axs.set_xlabel('Frequency (Hz)')
        fks = current_struct[0]['fks']
        selected_axs.set_xlim([fks.min(), fks.max()])
        for imode in range(0, nbmodes):
                
                if nbmodes > 1:
                        selected_axs = axs[imode]
        
                selected_axs.plot(current_struct[imode]['fks'], current_struct[imode]['cphi'], label='$c_\Phi$')
                selected_axs.plot(current_struct[imode]['fks'], current_struct[imode]['cg'], label='$c_g$', linestyle='--')
                selected_axs.grid()
                selected_axs.text(0.5, 1., 'Mode '+str(imode), horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', edgecolor='black', pad=4.0), transform=selected_axs.transAxes)
                selected_axs.set_xscale('log')
        
        if nbmodes > 1:
                selected_axs = axs[0]
        selected_axs.legend()
        
        if(not options['GOOGLE_COLAB']):
                plt.savefig(options['global_folder'] + 'cphi.png')
                plt.close('all')
                
#######################################
## Discretize continuous model for SWRT
def discretize_model_heterogeneous(data, options): 

        ## Build interpolated depth model
        z_interp = np.linspace(options['z'][0], options['zmax'], options['nb_layers'])
        z_interp_interm = np.linspace(options['z'][0], options['z'][-1], 400)

        ## Loop over models (CVMH/CVMS) and unknowns (rho/vs/vp)
        data_interp = {}
        for imodel in data:

                ## Build layered model
                data_interp[imodel] = {}
                for iunknown in data[imodel]:
                        
                        data_interp[imodel][iunknown] = []

                        temp   = data[imodel][iunknown][:]
                        locnan = np.isnan(temp).nonzero()[0]
                        
                        ## Remove nan values
                        if( locnan.size > 0 ):
                                iz           = locnan[0]
                                temp[iz:]    = temp[iz-1]
                                
                        #f    = interpolate.interp1d(options['z'], temp, kind='previous')
                        #temp_interm = f(z_interp_interm)/1000.
                        #f    = interpolate.interp1d(z_interp_interm, temp_interm, kind='previous')
                        #data_interp[imodel][iunknown] = f(z_interp)
                        
                        f    = interpolate.interp1d(options['z'], temp, kind='next')
                        temp_interm = f(z_interp)/1000.
                        
                        data_interp[imodel][iunknown] = temp_interm
        
        return z_interp, data_interp
        
def generate_default_atmos():
        param_atmos = {}
        param_atmos['isothermal'] = False       
        param_atmos['subsampling'] = True
        param_atmos['subsampling_layers'] = 120
        param_atmos['file'] = './models/default_atmospheric_model.dat'
        param_atmos['cpa']    = 3.2e2 # m/s
        param_atmos['H']      = np.inf # m
        param_atmos['Nsq']    = 1e-10
        param_atmos['wind_x'] = 0.
        param_atmos['wind_y'] = 0.
        
        param_atmos['bulk']   = 1e-3
        param_atmos['shear']  = 1e-3
        param_atmos['kappa']  = 0.
        param_atmos['gamma']  = 1.4
        param_atmos['rho']    = 1.2
        param_atmos['cp']     = 3000.
        
        print('['+sys._getframe().f_code.co_name+'] Generated default atmospheric model from \''+param_atmos['file']+'\'.')
        
        return param_atmos
        
def read_csv_seismic(model, dimension, loc_source = 50000.):
        print('['+sys._getframe().f_code.co_name+'] Read model \''+model+'\'.')
        
        temp   = pd.read_csv( model, delim_whitespace=True, header=None )
        
        print('['+sys._getframe().f_code.co_name+'] Model:')
        print(temp)
        
        if(dimension == 1):
                temp.columns = ['z', 'rho', 'vp', 'vs', 'Qs', 'Qp']
        else:
                temp.columns = ['x', 'z', 'rho', 'vp', 'vs', 'Qs', 'Qp']
                x  = temp['x'].unique()
                ix = np.argmin( abs(x - loc_source) )
                x_chosen = x[ix]
                temp = temp.loc[ temp['x'] == x_chosen, temp.columns != 'x' ].copy()
                
        if(temp['z'].iloc[0] > 0):
                temp_add = temp.loc[ temp['z'] == temp['z'].min() ].copy()
                temp_add.loc[0, 'z'] = 0.
                temp = pd.concat([temp_add, temp]).reset_index()
        
        temp_add = temp.loc[ temp['z'] == temp['z'].max() ].copy()
        temp_add['z'].iloc[0] = 1.e7
        
        temp = pd.concat([temp, temp_add]).reset_index()
        
        return temp
        
def plot_atmosphere_and_seismic(global_folder, seismic, z_atmos, rho, cpa, winds, H, isothermal, dimension, google_colab):
        
  import matplotlib.colors as mcolors
  
  print('['+sys._getframe().f_code.co_name+'] Plot seismic and atmospheric models.')
  
  nb_cols = 3
  fig, axs = plt.subplots(nrows=2, ncols=nb_cols)
  
  colors = [icolor for icolor in mcolors.TABLEAU_COLORS]
  
  iax     = 0
  iax_row = 1
  
  z  = seismic['z'].values/1000.
  zi = np.linspace(z[0], z[-1], 10000)
  
  f = interpolate.interp1d(z, seismic['rho'].values/1000., kind='previous')
  unknown = f(zi)
  axs[iax_row, iax].plot(unknown, zi, color=colors[iax+iax_row*nb_cols])
  axs[iax_row, iax].grid()
  axs[iax_row, iax].set_xlim([unknown.min(), unknown.max()])
  axs[iax_row, iax].set_yscale('log')
  axs[iax_row, iax].set_ylabel('Depth (km)')
  axs[iax_row, iax].set_xlabel('Density (g/cm$^3$)')
  axs[iax_row, iax].text(-0.8, 0.5, 'Seismic', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', edgecolor='black', pad=2.0), transform=axs[iax_row, iax].transAxes, rotation=90)
  axs[iax_row, iax].invert_yaxis()
  
  iax += 1
  f = interpolate.interp1d(z, seismic['vp'].values/1000., kind='previous')
  unknown = f(zi)
  axs[iax_row, iax].plot(unknown, zi, color=colors[iax+iax_row*nb_cols])
  axs[iax_row, iax].grid()
  axs[iax_row, iax].set_xlim([unknown.min(), unknown.max()])
  axs[iax_row, iax].tick_params(axis='both', which='both', labelleft=False)
  axs[iax_row, iax].set_yscale('log')
  axs[iax_row, iax].invert_yaxis()
  axs[iax_row, iax].set_xlabel('$v_p$ (km/s)')
  
  iax += 1
  f = interpolate.interp1d(z, seismic['vs'].values/1000., kind='previous')
  unknown = f(zi)
  axs[iax_row, iax].plot(unknown, zi, color=colors[iax+iax_row*nb_cols])
  axs[iax_row, iax].grid()
  axs[iax_row, iax].set_xlim([unknown.min(), unknown.max()])
  axs[iax_row, iax].tick_params(axis='both', which='both', labelleft=False)
  axs[iax_row, iax].set_yscale('log')
  axs[iax_row, iax].invert_yaxis()
  axs[iax_row, iax].set_xlabel('$v_s$ (km/s)')
  
  axs[iax_row, iax].get_shared_y_axes().join(axs[iax_row, 0], axs[iax_row, 1], axs[iax_row, 2])
  
  ## Create a profile from a few altitude points if isothermal model
  rho = rho
  cpa = cpa
  wx  = winds[0]
  wy  = winds[1]
  z   = z_atmos
  if(isothermal):
          z = np.linspace(0, 50, 100)
          rho = rho[0]*np.exp(-z/(H[0]/1000.))+z*0
          cpa = cpa[0]+z*0
          wx  = winds[0][0]+z*0
          wy  = winds[1][0]+z*0
  else:
          z = z/1000.
          
  iax     = 0
  iax_row = 0
  unknown = rho/1000.
  try:
   axs[iax_row, iax].plot(unknown, z, color=colors[iax+iax_row*nb_cols])
  except:
   bp()
  axs[iax_row, iax].grid()
  if(unknown.min() < 0.5*unknown.max()):
          axs[iax_row, iax].set_xlim([unknown.min(), unknown.max()])
  axs[iax_row, iax].set_ylim([z.min(), z.max()])
  axs[iax_row, iax].set_ylabel('Altitude (km)')
  axs[iax_row, iax].text(-0.8, 0.5, 'Atmosphere', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', edgecolor='black', pad=2.0), transform=axs[iax_row, iax].transAxes, rotation=90)
  axs[iax_row, iax].set_title('Density (g/cm$^3$)')
  axs[iax_row, iax].set_xscale('log')
  
  iax += 1
  unknown = cpa/1000.
  axs[iax_row, iax].plot(unknown, z, color=colors[iax+iax_row*nb_cols])
  axs[iax_row, iax].grid()
  if(not isothermal):
          axs[iax_row, iax].set_xlim([unknown.min(), unknown.max()])
  axs[iax_row, iax].set_ylim([z.min(), z.max()])
  axs[iax_row, iax].tick_params(axis='both', which='both', labelleft=False)
  axs[iax_row, iax].set_title('$c_p$ (km/s)')
  
  
  iax += 1
  unknown = wx/1000.
  axs[iax_row, iax].plot(unknown, z, color=colors[iax+iax_row*nb_cols])
  if(not isothermal):
          axs[iax_row, iax].set_xlim([unknown.min(), unknown.max()])
  axs[iax_row, iax].set_ylim([z.min(), z.max()])
  if(dimension > 2):
          unknown_ = wy/1000.
          axs[iax_row, iax].plot(unknown_, z)
          if(not isothermal):
                  axs[iax_row, iax].set_xlim([min(unknown.min(), unknown_.min()), max(unknown.max(), unknown_.max())])
  axs[iax_row, iax].grid()
  axs[iax_row, iax].tick_params(axis='both', which='both', labelleft=False)
  axs[iax_row, iax].set_title('winds (km/s)')
  
  fig.subplots_adjust(hspace=0.3, right=0.95, left=0.2, top=0.9, bottom=0.15)
  
  if(not google_colab):
    fname = global_folder+'seismic_and_atmos_profiles.pdf'
    print('['+sys._getframe().f_code.co_name+'] Saved seismic and atmospheric models\' plot to \''+fname+'\'.')
    plt.savefig(fname)
    plt.close('all')