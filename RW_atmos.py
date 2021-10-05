#!/usr/bin/env python3
import numpy as np
# import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pdb import set_trace as bp
import sys 
from pyrocko import moment_tensor as mtm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from datetime import datetime, timedelta

from scipy import fftpack
import scipy.integrate as spi
from obspy.signal.tf_misfit import plot_tfr

from scipy import interpolate
# from sympy import symbols, solve
# from sympy.utilities.lambdify import lambdify

from multiprocessing import get_context
import multiprocessing as mp
from functools import partial

# Local modules
# import mechanisms as mod_mechanisms
import RWAtmosUtils, velocity_models

# display parameters
font = {'size': 14}
matplotlib.rc('font', **font)

# To make sure that there is no bug when saving and closing the figures
# https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
matplotlib.use('Agg')

class vertical_velocity():
        
        def __init__(self, period, r2, cphi, cg, I1, kn, QR, directivity):
                
                self.period = period
                self.directivity = directivity
                self.r2   = r2
                self.cg   = cg
                self.cphi = cphi
                self.I1   = I1
                self.kn   = kn
                self.QR   = QR

        def add_attenuation(self, r):
        
                return np.exp( -2.*np.pi*r/(2.*self.cphi*self.QR*self.period) )

        def compute_veloc(self, r, phi, M, depth, unknown = 'd', dimension_seismic = 3):
        
                comp_deriv = -np.pi*2.*1j/self.period if unknown == 'v' else 1.
                comp_deriv = (-np.pi*2.*1j/self.period)*comp_deriv if unknown == 'a' else comp_deriv
                
                # 3d
                if(dimension_seismic == 3):
                        return comp_deriv*(self.r2/(8*self.cphi*self.cg*self.I1))*np.sqrt(2./(np.pi*self.kn*r))*np.exp( 1j*( self.kn*r + np.pi/4. ) )*self.directivity.compute_directivity(phi, M, depth) * self.add_attenuation(r)
                
                # 2d
                elif(dimension_seismic == 2):
                        # 1e3 is okay, because homogeneisation of the equations
                        return 1e3*comp_deriv*(self.r2/(4*self.cphi*self.cg*self.I1))*(1./(self.kn))*np.exp( 1j*( self.kn*r + np.pi/2. ) )*self.directivity.compute_directivity(phi, M, depth) * self.add_attenuation(r)
                else:
                        sys.exit('Seismic dimension not recognized!')
                        
class directivity():

        def __init__(self, dep, dr1dz_source, dr2dz_source, kn, r1_source, r2_source):
        
                self.dep = dep
                self.dr1dz_source = dr1dz_source
                self.dr2dz_source = dr2dz_source
                self.kn = kn
                self.r1_source = r1_source
                self.r2_source = r2_source
        
        def compute_directivity(self, phi, M, depth):
        
                idz = np.argmin( abs(self.dep - depth/1000.) ) # get id of the source depth in the dep vector (which is the depths for the eigenfunctions)
                dr1dz_source = self.dr1dz_source[idz]
                dr2dz_source = self.dr2dz_source[idz]
                r1_source = self.r1_source[idz]
                r2_source = self.r2_source[idz]
                
                phi_rot = phi
                
                return self.kn*r1_source*( M[1]*np.cos(phi_rot)**2 + (-2.*M[5])*np.sin(phi_rot)*np.cos(phi_rot) + M[2]*np.sin(phi_rot)**2 ) \
                                + 1j*dr1dz_source*(M[3]*np.cos(phi_rot) - M[4]*np.sin(phi_rot)) \
                                - 1j*self.kn*r2_source*(M[3]*np.cos(phi_rot) - M[4]*np.sin(phi_rot)) \
                                + dr2dz_source*M[0]

class RW_forcing():
        
        def __str__(self):
          out = ''
          out = out + '+-------------------------------+\n'
          out = out + '|     Rayleigh Wave Forcing     |\n'
          out = out + '|       (Green Functions)       |\n'
          out = out + '+-------------------------------+\n'
          
          out = out + ('| %5d modes                   |\n' % (self.nb_modes))
          
          out = out + ('| %5d frequencies asked       |\n' % (self.nb_freqs))
          out = out + ('| %5d frequencies for mode 1  |\n' % (self.nb_freqs_actualforMode1))
          
          out = out + '+-------------------------------+\n'
          out = out + '| Frequencies asked (mode 1?):  |\n'
          if(self.nb_freqs>11):
            for i in range(5):
              out = out + ('| %.6e %s\n' % (self.f_tab[i], '(v)' if i<self.nb_freqs_actualforMode1 else ''))
            out = out + '|  ...\n'
            for i in range(self.nb_freqs-5, self.nb_freqs):
              out = out + ('| %.6e %s\n' % (self.f_tab[i], '(v)' if i<self.nb_freqs_actualforMode1 else ''))
          else:
            for i in range(self.nb_freqs):
              out = out + ('| %.6e %s\n' % (self.f_tab[i], '(v)' if i<self.nb_freqs_actualforMode1 else ''))
          out = out + '+-------------------------------+\n'
          out = out + ('| uz:          %d*%d array of %s\n' % (len(self.uz), len(self.uz[0]), type(self.uz[0][0])))
          out = out + ('| directivity: %d*%d array of %s\n' % (len(self.directivity), len(self.directivity[0]), type(self.directivity[0][0])))
          out = out + '+-------------------------------+\n'
          out = out + '| Seismic model:                |\n'
          out = out + str(self.seismic)+'\n'
          out = out + '+-------------------------------+\n'
          out = out + '| Chosen storage folder:        |\n'
          out = out + '| '+self.global_folder+'\n'
          out = out + '+-------------------------------+\n'
          
          if(self.has_mechanism):
            out = out + ('| Chosen mechanism:            |\n')
            out = out + ('|   M0:     %.3f\n' % (self.M0))
            out = out + ('|   depth:  %.3f\n' % (self.zsource))
            out = out + ('|   strike: %.0f\n' % (self.strike))
            out = out + ('|   dip:    %.0f\n' % (self.dip))
            out = out + ('|   rake:   %.0f\n' % (self.rake))
            out = out + ('| Source time function:         |\n')
            out = out + ('|   %s\n' % (self.stf))
            out = out + ('|   %d points\n' % (len(self.stf_data)))
            out = out + ('|   f0 = %f\n' % (self.f0))
          else:
            out = out + ('| No associated mechanism yet.  |\n')
          
          out = out + '+-------------------------------+\n'
        
          return(out)
        
        def __repr__(self):
          # Technically wrong, because repr should contain everything needed to build the instance again, but enough for what we want.
          return(self.__str__())
        
        def get_mode_filling(self):
          # Prepare a matrix which is 1 if (period(i), mode(j)) contains a velocity, or 0 if (period(i), mode(j)) is empty.
          mat = np.array([[0 if p==[] else 1 for p in m] for m in self.uz]).T
          return(np.flipud(mat)) # make (freq, mode)
        
        def print_mode_filling(self):
          uz_func_mode_freq = self.get_mode_filling()
          print('         ', end=' ')
          for m in range(self.nb_modes):
            if(m==self.nb_modes-1):
              print('%2d' % (m), end='\n')
            else:
              print('%2d' % (m), end=' ')
          for f in range(self.nb_freqs):
            print('%.3e' % (self.f_tab[f]), end=' ')
            for m in range(self.nb_modes):
              if(m==self.nb_modes-1):
                print('%2d' % (uz_func_mode_freq[f, m]), end='\n')
              else:
                print('%2d' % (uz_func_mode_freq[f, m]), end=' ')
          # dd

        def __init__(self, options):
        
                # Inputs
                self.f_tab = options['f_tab']
                self.nb_freqs = len(self.f_tab)
                self.nb_modes = options['nb_modes'][1]
                
                self.set_global_folder(options['global_folder'])
                
                # Attributes containing seismic/acoustic spectra
                self.directivity = [ [ [] for aa in range(0, self.nb_freqs) ] for bb in range(0, self.nb_modes) ]
                self.uz          = [ [ [] for aa in range(0, self.nb_freqs) ] for bb in range(0, self.nb_modes) ]
                
                # Extract seismic model for later plots
                self.extract_seismic_parameters(options)
                
                # Add source characteristics
                #self.set_mechanism(mechanism)
                self.has_mechanism = False
                
                # MPI parameter
                self.use_spawn = options['USE_SPAWN_MPI']
                
                self.google_colab = options['GOOGLE_COLAB']
                
        def set_global_folder(self, folder):
                self.global_folder = folder # Save folder path from Green's class

        def set_mechanism(self, mechanism):
                self.has_mechanism = True
                self.stf = mechanism['stf']
                self.stf_data = mechanism['stf-data']
                self.zsource = mechanism['zsource'] # m
                if(self.stf == 'gaussian'):
                  self.f0    = mechanism['f0']
                else:
                  self.f0    = mechanism['f0']*1.628
                self.alpha = (np.pi*self.f0)**2
                self.M0    = mechanism['M0']
                self.M     = mechanism['M']*self.M0
                self.phi   = mechanism['phi']
                
                self.strike = mechanism['STRIKE'] # info only because everything is in self.M as a matrix
                self.dip = mechanism['DIP'] # info only because everything is in self.M as a matrix
                self.rake = mechanism['RAKE'] # info only because everything is in self.M as a matrix
                
                self.mt    = []
                if 'mt' in mechanism:
                  self.mt = mechanism['mt']
        
        def clear_mechanism(self):
                self.has_mechanism = False
                delattr(self, 'stf')
                delattr(self, 'stf_data')
                delattr(self, 'zsource')
                delattr(self, 'f0')
                delattr(self, 'alpha')
                delattr(self, 'M0')
                delattr(self, 'M')
                delattr(self, 'phi')
                delattr(self, 'strike')
                delattr(self, 'dip')
                delattr(self, 'rake')
                delattr(self, 'mt')

        def get_mechanism(self):
                mechanism = {}
                mechanism['stf']      = self.stf
                mechanism['stf-data'] = self.stf_data
                mechanism['zsource'] = self.zsource# m
                mechanism['f0']      = self.f0        
                if(self.stf == 'gaussian'):
                  mechanism['f0'] = self.f0
                else:
                  mechanism['f0'] = self.f0/1.628
                mechanism['M0']  = self.M0
                mechanism['M']   = self.M/self.M0
                mechanism['phi'] = self.phi
                mechanism['STRIKE'] = self.strike
                mechanism['DIP'] = self.dip
                mechanism['RAKE'] = self.rake
                mechanism['mt']  = self.mt
                return mechanism

        def source_spectrum(self, period):
        
                if(self.stf == 'gaussian'):
                        return self.M*np.sqrt(np.pi/self.alpha)*np.exp(-((np.pi/period)**2)/self.alpha)*np.exp(2*np.pi*1j*(4./self.f0)/period)
                        #       ^^       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        #      moment                   source fourier transform                                        time shift
                        #       mag
                
                elif(self.stf == 'gaussian_'):
                        return self.M*np.exp(-((np.pi/period)**2)/(self.f0**2))*np.exp(2*np.pi*1j*(4./self.f0)/period)
                
                elif(self.stf == 'erf'):
                        #erf = -1j*(1./self.f0)*np.exp(-(np.pi/(period*self.f0))**2)/(np.pi/(period*self.f0))
                        erf = -1j*np.exp(-(np.pi/(period*self.f0))**2)/(np.pi/period)
                        dirac = 0.
                        return 0.5*(dirac + erf) * self.M * np.exp(2*np.pi*1j*(2.*2./self.f0)/period)
                
                elif(self.stf == 'external'):
                        loc = np.argmin( abs(self.stf_data[0] - 1/period) )
                        return self.M * self.stf_data[1][loc]
                
                else:
                        sys.exit('Source time function "'+self.stf+'" not recognized!')
        
        def add_one_period(self, period, iperiod, current_struct, rho, orig_b1, orig_b2, d_b1_dz, d_b2_dz, kmode, dep):
          # uz    = []
          # freqa = []
          np.seterr(all='raise') # Force raising exceptions when encountering "RuntimeWarning: divide by zero encountered in true_divide".
          for imode in range(0, min(len(current_struct),orig_b1.shape[1])):
            cphi = current_struct[imode]['cphi'][iperiod]
            cg   = current_struct[imode]['cg'][iperiod]
            r2   = orig_b2[:,imode]
            r1   = orig_b1[:,imode]
            kn   = kmode[:,imode]
            d_r2_dz = d_b2_dz[:,imode]
            d_r1_dz = d_b1_dz[:,imode]
            
            try:
              I1 = 0.5*spi.simps(rho[:]*( r1**2 + r2**2 ), dep[:])
            except(FloatingPointError):
              (u, c) = np.unique(dep, return_counts=True)
              if(np.any(c>1)):
                print('[%s] Duplicate depth(s) found during integration:' % (sys._getframe().f_code.co_name))
                print(u[c>1])
                raise(FloatingPointError)
              else:
                sys.exit('what?')
            
            kn = kn[0]
            self.directivity[imode][iperiod] = directivity(dep, d_r1_dz, d_r2_dz, kn, r1, r2)
            
            r2 = r2[0]
            r1 = r1[0]
            
            # Compute quality factor
            #QR  = spi.simps( (2./Qp[:])*(lamda[:] + 2*mu[:])*( kn*r1 + d_r2_dz )**2, dep[:])
            #QR += spi.simps( (2.*mu[:]/Qs[:])*(( kn*r2 + d_r1_dz )**2 - 4*kn*r1*d_r2_dz ), dep[:])
            #QR *= 1./(4.*(kn**2)*cg*cphi*I1)
            QR = current_struct[imode]['QR'][iperiod]
            
            # Store Green's functions for an arbitrary moment tensor
            self.uz[imode][iperiod] = vertical_velocity(period, r2, cphi, cg, I1, kn, QR, self.directivity[imode][iperiod])
                        
        def update_frequencies(self):
          # Compare the set of expected frequencies (self.nb_freq, self.f_tab)
          # to the list of eigenmodes as generated by the calls to
          # self.add_one_period in RW_dispersion.get_eigenfunctions.
          # - In the case where the eigenfrequencies fill the array of
          #   expected frequencies, everything is fine.
          # - In the case where there is less eigenfrequencies than expected
          #   frequencies, the creation of the RW_field will break (see
          #   compute_ifft > response_RW_all_modes > compute_RW_one_mode),
          #   and we need to store the actual frequencies that were found.
          uz_func_mode_freq = self.get_mode_filling()
          nfreqmode1 = np.sum(uz_func_mode_freq[:,0])
          if(nfreqmode1 == self.nb_freqs):
            # Easiest case, first mode contains values for all frequencies.
            print('[%s] > > First mode contains eigenfrequencies for all expected frequencies, everything is fine.'
                  % (sys._getframe().f_code.co_name))
            self.nb_freqs_actualforMode1 = self.nb_freqs
            self.f_tab_actualforMode1 = self.f_tab
          else:
            # Somehow earthsr found less frequencies than asked for.
            print('[%s] > > First mode contains less eigenfrequencies (%d) than expected frequencies (%d).'
                  % (sys._getframe().f_code.co_name, nfreqmode1, self.nb_freqs))
            sys.exit(['[%s, ERROR] This should not happen: fundamental mode should always have all frequencies.'])
          #   # Check for empty frequencies in the middle of the series.
          #   # if mode 1 has [1 1 1 1 1 0 0] then it makes sense (no it does not, cf. error above)
          #   # if mode 1 has [1 1 0 0 1 1 1] then it does not make sense and is probably an error
          #   if(np.sum(uz_func_mode_freq[nfreqmode1:,0])>0):
          #     # If the last indices aren't zero, the zeros are somewhere before, and there is an issue.
          #     print('[%s] > > > On the first mode, some frequencies have no associated eigenfrequency:'
          #           % (sys._getframe().f_code.co_name),
          #           file=sys.stderr)
          #     print(self.f_tab[np.where(uz_func_mode_freq[:,0]==0)], file=sys.stderr)
          #     sys.exit('[%s] > > > This should not happen.' % (sys._getframe().f_code.co_name))
          #   else:
          #     # This is probably fine, store the updated frequency array.
          #     print('[%s] > > > Store frequency span agreeing with the first mode\'s first %d frequencies.' % (sys._getframe().f_code.co_name, nfreqmode1))
          #     self.nb_freqs_actualforMode1 = nfreqmode1
          #     self.f_tab_actualforMode1 = self.f_tab[:self.nb_freqs_actualforMode1]
          # # Check parity.
          # if(self.nb_freqs_actualforMode1%2==1):
          #   print('[%s] > > > Odd number of modes found. This will mess up the FFT routines.' % (sys._getframe().f_code.co_name))
          #   print('[%s] > > > > Dropping to even number below, updating "found frequencies", DELETING THE LAST FREQUENCY in UZ and DIRECTIVITY for ALL MODES.' % (sys._getframe().f_code.co_name))
          #   print('[%s] > > > > This will probably only impact the first mode anyway.' % (sys._getframe().f_code.co_name))
          #   for m in range(self.nb_modes):
          #     self.uz[m][self.nb_freqs_actualforMode1-1]=[]
          #     self.directivity[m][self.nb_freqs_actualforMode1-1]=[]
          #   self.nb_freqs_actualforMode1 -= 1
          #   self.f_tab_actualforMode1 = self.f_tab[:self.nb_freqs_actualforMode1]
          
        def compute_RW_one_mode(self, imode, r, phi, type = 'RW', unknown = 'd', dimension_seismic = 3):
        
                # Source depth
                depth = self.zsource
        
                uz_tab = []
                f_tab  = []
                #print('Compute mode: ', imode)
                # print(imode, len(self.uz[imode]))
                for iuz in self.uz[imode]:
                     
                     if(iuz):
                             M = self.source_spectrum(iuz.period)
                             f  = 1./iuz.period  
                             
                             f_tab.append( f )
                             uz = iuz.compute_veloc(r, phi, M, depth, unknown, dimension_seismic)
                             
                             # If 1d mesh passed we just append
                             if(phi.shape[0] == phi.size):
                                uz_tab.append( uz.reshape(r.size) )

                             # If 2d r/phi mesh passed
                             # Create a 1d array with increments in phi and then r
                             else:   
                                uz_tab.append( uz.reshape(r.shape[1]*r.shape[0],) )
                
                     else:
                        break
                
                response         = pd.DataFrame(np.array(uz_tab)) # Transform list into dataframe
                response.columns = np.arange(0, phi.size)
                response['f']    = np.array(f_tab)
                # print(np.array(uz_tab).shape, response.shape)
                # stop
                
                return response

        def extract_seismic_parameters(self, options):
        
            dimension = 2 if options['type_model'] == 'specfem2d' else 1
            self.seismic = velocity_models.read_csv_seismic(options['models'][options['chosen_model']], dimension)
                
        def local_mode(self, r, phi, type, unknown, dimension_seismic, modes):
                
            for imode_num, imode in enumerate(modes):
            
                    #print('Computing mode', imode)
            
                    response_RW_temp = self.compute_RW_one_mode(imode, r, phi, type, unknown, dimension_seismic)
                    if(imode_num == 0):
                            response_RW = response_RW_temp.copy()
                    else:

                            # Concatenate dataframes with same freq.
                            # we can not use pd.concat since it is too slow for complex numbers
                            response_RW = RWAtmosUtils.concat_df_complex(response_RW, response_RW_temp, 'f')
                    
            return response_RW
        
        def response_RW_all_modes(self, r, phi, type = 'RW', unknown = 'd', mode_max = -1, dimension_seismic = 3, ncpus = 16):
    
            mode_max = len(self.uz) if mode_max == -1 else mode_max
            
            if(ncpus<=1):
              parallel = False
              print('[%s] Get the RW response for all modes. Do the computation in serial.' % (sys._getframe().f_code.co_name))
            else:
              parallel = True
              print('[%s] Get the RW response for all modes. Use multithreading over %d CPUs.' % (sys._getframe().f_code.co_name, ncpus))
            
            if not parallel:
              # SUGGESTION: INSTEAD OF ALL OF WHAT FOLLOWS, SIMPLY CALL "self.local_mode(r, phi, type, unknown, dimension_seismic)".
              for imode in range(0, mode_max):
                #print('Computing mode', imode)
                response_RW_temp = self.compute_RW_one_mode(imode, r, phi, type, unknown, dimension_seismic)
                if(imode == 0):
                  response_RW = response_RW_temp.copy()
                else:
                  # Concatenate dataframes with same freq.
                  # we can not use pd.concat since it is too slow for complex numbers
                  response_RW = RWAtmosUtils.concat_df_complex(response_RW, response_RW_temp, 'f')
            
            else:          
              modes = [key for key in range(0, mode_max)]
              N = ncpus # How many subtasks should be prepared.
              list_of_lists = [sublist for sublist in np.array_split(modes, N) if sublist.size>0] # Split the modes over the requested number of CPUs.
              N = len(list_of_lists) # Make sure the requested N correspond to the number of prepared lists.
              
              local_mode_partial = partial(self.local_mode, r, phi, type, unknown, dimension_seismic)
                          
              if self.use_spawn:
                with get_context("spawn").Pool(processes = N) as p:
                        results = p.map(local_mode_partial, list_of_lists)
              else:
                with mp.Pool(processes = N) as p:
                        results = p.map(local_mode_partial, list_of_lists)
      
              response_RW = results[0]
              for imode, result in enumerate(results): response_RW = RWAtmosUtils.concat_df_complex(response_RW, result, 'f');
            
            if(np.all(np.isnan(response_RW[0]))):
              # Safeguard.
              sys.exit('[%s, ERROR] All values in the Rayleigh wave response are NaNs. Something went terribly wrong.' % (sys._getframe().f_code.co_name))
                    
            return response_RW  
                
        def response_perturbed_solution(self, x, r, phi, type = 'RW', unknown = 'd', mode_max = -1, dimension_seismic = 3, type_opti='min'):
        
            # dip, strike and rake perturbations
            p_strike, p_dip, p_rake = x[0], x[1], x[2]
            
            # Create source from perturbations
            mechanism = self.get_mechanism()
            mt = mechanism['mt']
            strike0, dip0, rake0 = mt.both_strike_dip_rake()[0]
            strike, dip, rake    = strike0 + p_strike, dip0 + p_dip, rake0 + p_rake
            m0 = mt.scalar_moment()
            mt = mtm.MomentTensor(strike=strike, dip=dip, rake=rake, scalar_moment=m0)
            mechanism_save = mechanism.copy()
            
            mechanism['M'] = mt.m6_up_south_east()
            self.set_mechanism(mechanism)
    
            mode_max = len(self.uz) if mode_max == -1 else mode_max
            for imode in range(0, mode_max):
            
                    response_RW_temp = self.compute_RW_one_mode(imode, r, phi, type, unknown, dimension_seismic)
                    if(imode == 0):
                            response_RW = response_RW_temp.copy()
                    else:

                            # Concatenate dataframes with same freq.
                            # we can not use pd.concat since it is too slow for complex numbers
                            response_RW = RWAtmosUtils.concat_df_complex(response_RW, response_RW_temp, 'f')
                            
            self.set_mechanism(mechanism_save)
            
            coef = 1. if type_opti == 'min' else -1.
            return coef * abs(response_RW.loc[:, response_RW.columns != 'f'].values).max(axis=0)[0]                       

        def compute_ifft(self, r_in, phi_in, type, unknown = 'd', mode_max = -1, dimension_seismic = 3, ncpus=16):
        
            # Collect the positive-frequency response of each RW mode
            RW = self.response_RW_all_modes(r_in, phi_in, type, unknown, mode_max, dimension_seismic, ncpus)
            RW = RW.sort_values(by=['f'], ascending=True)
            
            # Positive frequencies
            RW_first = RW.iloc[0:1].copy()
            temp     = pd.DataFrame(RW_first.values*0.)
            temp.columns = RW_first.columns
            RW_first     = temp.copy()
            
            # Negative frequencies
            RW_neg = RW.iloc[:].copy()
            RW_neg.loc[:,'f']  = -RW_neg.loc[:,'f']
            RW = RW_first.append(RW.iloc[:-1])
            RW_neg = RW_neg.sort_values(by=['f'], ascending=True)
            temp   = pd.DataFrame(np.real(RW_neg.iloc[:1].values))
            temp.columns = RW_neg.columns
            RW_neg = temp.append(RW_neg.drop([0]))
            
            temp      = pd.DataFrame(np.real(RW_neg.loc[:, RW_neg.columns != 'f']) + 1j*np.imag(RW_neg.loc[:, RW_neg.columns != 'f']))
            temp['f'] = RW_neg['f']
            RW_neg    = temp.copy()
            temp      = pd.DataFrame(np.real(RW.loc[:, RW.columns != 'f']) - 1j*np.imag(RW.loc[:, RW.columns != 'f']))
            temp['f'] = RW['f'].values
            RW        = temp.copy()
            
            # Concatenate negative and positive frequencies
            RW_tot = pd.concat([RW_neg,RW], ignore_index=True)
            
            # Compute inverse Fourier transform
            ifft_RW = fftpack.ifft(fftpack.fftshift(RW_tot.values[:,:-1], axes=0), axis=0)
            nb_fft  = ifft_RW.shape[0]//2
            ifft_RW = ifft_RW[:nb_fft]
            
            # # Compute corresponding time array
            dt = 1./(2.*abs(RW_neg['f']).max())
            t  = np.arange(0, dt*nb_fft, dt)
            # t = None # If using the updated frequencies in field_RW.__init__, then the time vector is not needed here.
            
            return (t, ifft_RW)

def generate_one_timeseries(t, Mz_t, RW_Mz_t, comp, iz, iy, ix, stat, options):

    # Save waveforms     
    df = pd.DataFrame()
    df['t']  = t
    df['vz'] = np.real(Mz_t)
    name_file = 'waveform_'+comp+'_'+str(stat)+'_'+str(round(ix/1000.,1))+'_'+str(round(iy/1000.,1))+'_'+str(round(iz/1000.,1))+'.csv'
    df.to_csv(options['global_folder'] + name_file, index=False)
    print('['+sys._getframe().f_code.co_name+'] Save waveform to \''+options['global_folder']+name_file+'\'.')
    
    # Deallocate
    df = None
    
    df = pd.DataFrame()
    df['t']  = t
    df['vz'] = np.real(RW_Mz_t)
    name_file = 'RW_waveform_z0_'+comp+'_'+str(stat)+'_'+str(round(ix/1000.,1))+'_'+str(round(iy/1000.,1))+'_'+str(round(iz/1000.,1))+'.csv'
    df.to_csv(options['global_folder'] + name_file, index=False)
    print('['+sys._getframe().f_code.co_name+'] Save waveform to \''+options['global_folder']+name_file+'\'.')
    
    # Deallocate
    df = None
    
    # Create frequency/time plot
    #freq_min, freq_max = Green_RW.f0/10., Green_RW.f0*2.
    freq_min, freq_max = options['coef_low_freq'], options['coef_high_freq']
    tr   = RWAtmosUtils.generate_trace(t, np.real(Mz_t), freq_min, freq_max)
    fig = plot_tfr(tr.data, dt=tr.stats.delta, fmin=freq_min, fmax=freq_max, w0=4., nf=64, fft_zero_pad_fac=4, show=False, t0=0., left=0.16, bottom=0.12, w_2=0.5)
    fig.axes[0].grid()
    fig.axes[0].set_xlabel('Time (s)')
    fig.axes[2].grid()
    fig.axes[2].set_ylabel('Frequency (Hz)')
    fig.axes[1].text(0.1, 1.08, 'E = '+str(round(ix/1000.,1))+' S = '+str(round(iy/1000.,1))+' U = '+str(round(iz/1000.,1)) +' km', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', edgecolor='black', pad=4.0), transform=fig.axes[1].transAxes)
    
    if(not options['GOOGLE_COLAB']):
            name_file = 'freq_time_'+comp+'_'+str(stat)+'_'+str(round(ix/1000.,1))+'_'+str(round(iy/1000.,1))+'_'+str(round(iz/1000.,1))+'.png'
            plt.savefig(options['global_folder'] + name_file)
            plt.close('all')

    tr, fig = None, None

class field_RW():
        
        def __str__(self):
          out = ''
          out = out + '+-------------------------------+\n'
          out = out + ('|    %dD Rayleigh Wave Field     |\n' % (self.dimension))
          out = out + '+-------------------------------+\n'
          
          out = out + '| Seismic model:                |\n'
          out = out + str(self.seismic)+'\n'
          out = out + '+-------------------------------+\n'
          
          out = out + ('| X-Y domain:                   |\n')
          out = out + ('|   x: [%.5e, %.5e] m, %d elements, dx = %.6f\n'
                       % (np.min(self.x), np.max(self.x), self.x.size, np.mean(np.diff(self.x))))
          out = out + ('|   y: [%.5e, %.5e] m, %d elements, dy = %.6f\n'
                       % (np.min(self.y), np.max(self.y), self.y.size, np.mean(np.diff(self.y))))
          
          out = out + ('| Atmospheric model:            |\n')
          if(not self.atmospheric_model_is_generated):
            out = out + ('|   none (atmospheric model is not defined yet).\n')
          else:
            out = out + ('|   z: [%.5e, %.5e] m, %d elements, dz = %.6f\n'
                         % (np.min(self.z), np.max(self.z), self.z.size, np.mean(np.diff(self.z))))
            if(self.isothermal):
              out = out + ('|   Isothermal model.           |\n')
            else:
              out = out + ('|   User-defined model:         |\n')
              out = out + ('|     rho:     %d elements (density)\n' % (self.rho.size))
              out = out + ('|     cpa:     %d elements (sound speed)\n' % (self.rho.size))
              out = out + ('|     winds_x: %d elements\n' % (self.winds[0].size))
              out = out + ('|     winds_y: %d elements\n' % (self.winds[1].size))
              out = out + ('|     bulk:    %d elements\n' % (self.bulk.size))
              out = out + ('|     shear:   %d elements\n' % (self.shear.size))
              out = out + ('|     kappa:   %d elements\n' % (self.kappa.size))
              out = out + ('|     cp:      %d elements (isobaric specific heat capacity)\n' % (self.rho.size))
              out = out + ('|     gamma:   %d elements\n' % (self.gamma.size))
              out = out + ('|     H:       %d elements (scale height)\n' % (self.H.size))
              out = out + ('|     Nsq:     %d elements (Brunt-Väisälä frequency)\n' % (self.rho.size))
          out = out + '+-------------------------------+\n'
          
          out = out + ('| T domain (from IFFT):         |\n')
          out = out + ('|   t: [%.3f, %.6f] s, %d elements, dt = %.6f\n'
                       % (np.min(self.t), np.max(self.t), self.t.size, np.mean(np.diff(self.t))))
          out = out + '+-------------------------------+\n'
          
          out = out + ('| Frequency domain:             |\n')
          if(self.dimension==3):
            out = out + ('|   Omega, KX, KY: %d*%d*%d meshgrid (frequency, x, y)\n'
                         % self.KX.shape)
            out = out + ('| Forcing:                      |\n')
            out = out + ('|   Mo:   %d*%d*%d meshgrid for bottom forcing\n'
                         % self.Mo.shape)
            out = out + ('|   TFMo: %d*%d*%d meshgrid for Fourier transform of bottom forcing\n'
                         % self.Mo.shape)
          else:
            out = out + ('|   Omega, KX, KY: %d*%d meshgrid (frequency, x)\n'
                         % self.KX.shape)
            out = out + ('| Forcing:                      |\n')
            out = out + ('|   Mo:   %d*%d meshgrid for bottom forcing\n'
                         % self.Mo.shape)
            out = out + ('|   TFMo: %d*%d meshgrid for Fourier transform of bottom forcing\n'
                         % self.Mo.shape)
          out = out + '+-------------------------------+\n'
        
          return(out)
        def __repr__(self):
          # Technically wrong, because repr should contain everything needed to build the instance again, but enough for what we want.
          return(self.__str__())
        
        
        default_loc = (30., 0.) # (km, degree)
        def __init__(self, Green_RW, nb_freq, dimension = 2, dx_in = 100., dy_in = 100., xbounds = [100., 100000.], ybounds = [100., 100000.], mode_max = -1, dimension_seismic = 3, ncpus = 16):

                def nextpow2(x):
                        return np.ceil(np.log2(abs(x)))
        
                self.atmospheric_model_is_generated = False
                self.global_folder = Green_RW.global_folder # Save folder path from Green's class
                self.coef_low_freq = [Green_RW.f_tab[0], Green_RW.f_tab[-1]]
                self.type_output = 'a'
        
                #########################
                # Initial call to Green_RW to get the time vector only.
                output = Green_RW.compute_ifft(np.array([field_RW.default_loc[0]]), np.array([field_RW.default_loc[1]]), type='RW', unknown=self.type_output, dimension_seismic = dimension_seismic, ncpus = ncpus)
                t      = output[0]
                # # Probably a good idea to simply compute it analytically using the updated frequencies:
                # dt_anal = 1.0/(2.0*Green_RW.f_tab_actualforMode1.max())
                # t = np.arange(0, Green_RW.nb_freqs_actualforMode1) * dt_anal
                
                # Store seismic model
                self.seismic = Green_RW.seismic
                self.google_colab = Green_RW.google_colab
                
                # Define time/spatial domain boundaries
                # mult_tSpan, mult_xSpan, mult_ySpan = 1, 1, 1
                mult_xSpan, mult_ySpan = 1, 1
                dt_anal, dx_anal, dy_anal = abs(t[1] - t[0]), dx_in, dy_in # Using the dt computed from the updated frequencies above.
                # dx_anal, dy_anal = dx_in, dy_in
                xmin, xmax = xbounds[0], xbounds[1]
                if(dimension > 2):
                        ymin, ymax = ybounds[0], ybounds[1]
                
                # Define frequency/wavenumber boundaries
                NFFT1 = int(2**nextpow2((xmax-xmin)/dx_anal)*mult_xSpan)
                NFFT2 = len(t)
                # NFFT2 = Green_RW.nb_freqs_actualforMode1 # Using the updated frequencies.
                if(dimension > 2):
                        NFFT3 = int(2**nextpow2((ymax-ymin)/dy_anal)*mult_ySpan)
                
                # Define corresponding time and spatial arrays
                x = np.linspace(xmin, xmax, NFFT1)
                # t = dt_anal * np.arange(0,NFFT2) # Already computed at beginning of function (either with Green_RW.compute_ifft or analytically).
                if(dimension > 2):
                        y  = np.linspace(ymin, ymax, NFFT3)
                else:
                        y = np.array([Green_RW.phi])   
                           
                # Define corresponding Frequency Wavenumber arrays
                omega = 2.0*np.pi*(1.0/(dt_anal*NFFT2))*np.concatenate((np.arange(0,NFFT2/2), -np.arange(NFFT2/2,0,-1)))
                kx =    2.0*np.pi*(1.0/(dx_anal*NFFT1))*np.concatenate((np.arange(0,NFFT1/2), -np.arange(NFFT1/2,0,-1)))
                if(dimension > 2):
                        ky = 2.0*np.pi*(1.0/(dy_anal*NFFT3))*np.concatenate((np.arange(0,NFFT3/2), -np.arange(NFFT3/2,0,-1)))
                if(dimension > 2):
                        KX, Omega, KY = np.meshgrid(kx, omega, ky)
                else:
                        KX, Omega = np.meshgrid(kx, omega)
                
                # Initialize bottom RW forcing.
                # Mo  = np.zeros(Omega.shape, dtype=complex) # No need for initialisation since it's completely replaced by a further call.
                
                # Conversion of cartesian coordinates into cylindrical coordinates for 3d
                if(dimension > 2):
                        Y, X = np.meshgrid(y, x)
                        R    = np.sqrt( X**2 + Y**2 )
                        ind_where_yp0 = np.where(Y>0)
                        PHI = X*0.
                        PHI[ind_where_yp0] = np.arccos( X[ind_where_yp0]/R[ind_where_yp0] )
                        ind_where_yp0 = np.where(Y<0)
                        PHI[ind_where_yp0] = -np.arccos( X[ind_where_yp0]/R[ind_where_yp0] )
                
                else:
                        R   = abs(x)
                        PHI = 0. + R*0.
                        PHI[:len(x)//2] = np.pi
                PHI += np.pi/2.
                
                # Compute bottom RW forcing for all modes.
                # Will run through all existing uz for each mode, meaning only the ones with existing frequencies will be added (see compute_ifft > response_RW_all_modes > compute_RW_one_mode).
                # In short, the resulting Mo (or ifft_RW) will have self.nb_freq_filled frequencies, and NOT self.nb_freq.
                temp   = Green_RW.compute_ifft(R/1000., PHI, type='RW', unknown=self.type_output, mode_max = mode_max, dimension_seismic = dimension_seismic, ncpus = ncpus)
                if(dimension > 2):
                        t, Mo  = temp[0], temp[1].reshape( (temp[1].shape[0], PHI.shape[0], PHI.shape[1]) )
                else:
                        t, Mo  = temp[0], temp[1].reshape( (temp[1].shape[0], PHI.size) )
                # # If using the updated frequencies above, then the time vector is not needed.
                # if(dimension > 2):
                #   Mo = temp[1].reshape( (temp[1].shape[0], PHI.shape[0], PHI.shape[1]) )
                # else:
                #   Mo = temp[1].reshape( (temp[1].shape[0], PHI.size) )
                if(np.all(np.isnan(Mo))):
                  # Safeguard.
                  sys.exit('[%s, ERROR] All values in the Rayleigh wave forcing are NaNs. Something went terribly wrong.' % (sys._getframe().f_code.co_name))
                
                # Store forcing parameters
                self.Mo = Mo
                self.TFMo  = fftpack.fftn(self.Mo)
                self.Omega = Omega
                self.KX = -KX
                if(dimension > 2):
                        self.KY = -KY

                # Compute vertical wavenumber
                #self.compute_vertical_wavenumber(TFMo, H, Nsq, winds)
                self.dimension = dimension

                # Store mesh parameters                
                self.x    = x
                self.y    = y
                self.t    = t
                
        def generate_atmospheric_model(self, param_atmos):
        
                self.atmospheric_model_is_generated = True
                
                # Remove errors 
                np.seterr(divide='ignore', invalid='ignore')  
        
                self.isothermal = param_atmos['isothermal']
                if(self.isothermal):
                        self.H = np.array([param_atmos['H']])
                        self.cpa = np.array([param_atmos['cpa']])
                        self.Nsq = np.array([param_atmos['Nsq']])
                        self.winds = []
                        self.winds.append( np.array([param_atmos['wind_x']]) )
                        self.winds.append( np.array([param_atmos['wind_y']]) )
                        self.bulk  = np.array([param_atmos['bulk']])
                        self.shear = np.array([param_atmos['shear']])
                        self.kappa = np.array([param_atmos['kappa']])
                        self.gamma = np.array([param_atmos['gamma']])
                        self.rho   = np.array([param_atmos['rho']])
                        self.cp    = np.array([param_atmos['cp']])
                else:
                        temp = RWAtmosUtils.loadAtmosphericModel(param_atmos['file'])
                        # temp['bulk'] = 2e-4
                        # temp['mu']   = 2e-4
                        
                        if(param_atmos['subsampling']):
                                nb_layers = param_atmos['subsampling_layers']
                                temp_i = pd.DataFrame()
                                zi     = np.linspace(temp['z'].min(), temp['z'].max(), nb_layers)
                                temp_i['z'] = zi
                                for (columnName, columnData) in temp.iteritems():
                                        if('z' in columnName):
                                                continue
                                                
                                        f = interpolate.interp1d(temp['z'].values, columnData.values, kind='cubic')
                                        unknown = f(zi)
                                        temp_i[columnName] = unknown.copy()
                                temp = temp_i.copy()
                        
                        self.z = temp['z'].values
                        zshift = -(np.roll(self.z, 1) - self.z)
                        pshift = np.log( np.roll(temp['p'].values, 1)/temp['p'].values )
                        locbad = np.where(zshift <= 0)
                        if(locbad[0].size > 0):
                                zshift[locbad] = zshift[locbad[0][-1]+1]
                        
                        locbad = np.where(pshift <= 0)
                        if(locbad[0].size > 0):
                                pshift[locbad] = pshift[locbad[0][-1]+1]
                        
                        self.H      = zshift/pshift
                        self.Nsq    = np.sqrt(-(temp['g'].values/temp['rho'].values[0])*np.gradient(temp['rho'].values, self.z, edge_order=2))**2
                        self.Nsq[0] = self.Nsq[1]

                        self.winds = []
                        self.winds.append( temp['wx'].values )
                        self.winds.append( temp['wy'].values )
                        
                        self.cpa = temp['cpa'].values
                        self.rho   = temp['rho'].values
                        self.bulk  = temp['bulk'].values
                        self.shear = temp['mu'].values
                        self.kappa = temp['kappa'].values
                        self.cp    = temp['cp'].values
                        self.gamma = temp['gamma'].values
                                
                                
        def compute_vertical_wavenumber(self, id_layer, correct_wavenumber = True, exact_computation = False):    
        
                # Ignore division/invalid errors in KZ computation
                np.seterr(divide='ignore', invalid='ignore')    
                                
                # Get corresponding atmospheric parameters
                H      = self.H[id_layer]
                Nsq    = self.Nsq[id_layer]
                wind_x = self.winds[0][id_layer]
                wind_y = self.winds[1][id_layer]
                cpa    = self.cpa[id_layer]
                bulk   = self.bulk[id_layer]
                shear  = self.shear[id_layer]
                kappa  = self.kappa[id_layer]
                gamma  = self.gamma[id_layer]
                rho    = self.rho[id_layer]
                cp     = self.cp[id_layer]
                
                # Compute intrinsic frequencies
                Omega_intrinsic = self.Omega - wind_x*self.KX
                if(self.dimension > 2):
                        Omega_intrinsic -= wind_y*self.KY
                     
                # 
                if(not exact_computation):
                
                        if(self.dimension > 2):
                                KZ = np.lib.scimath.sqrt(  -self.KX**2 -self.KY**2 + (self.KX**2 + self.KY**2) * Nsq/(Omega_intrinsic**2) -1./(4.*H**2) \
                                        + (1.+1j*(bulk+(4./3.)*shear+kappa*(gamma-1.)/cp)*Omega_intrinsic/(2.*rho*cpa**2))*(Omega_intrinsic / cpa )**2 )
                             
                        else:
                                KZ = np.lib.scimath.sqrt( -self.KX**2              + (self.KX**2) * Nsq/(Omega_intrinsic**2) -1./(4.*H**2) \
                                        + (1.+1j*(bulk+(4./3.)*shear+kappa*(gamma-1.)/cp)*Omega_intrinsic/(2.*rho*cpa**2))*(Omega_intrinsic / cpa )**2 )
                
                # Exact dispersion equation from Godin, Dissipation of acoustic-gravity waves:An asymptotic approach, 2014.
                # This whole section is not used anywhere, we comment it out.
                # else:
                         
                #         if(self.dimension > 2):
                #                 kx, ky, kz, Omega = symbols('kx, ky, kz, Omega')
                #                 H_, Nsq_, cpa_, rho_, shear_ = symbols('H, gz, c0, rho0, eta0')
                                
                #                 # From Godin, Dissipation of acoustic-gravity waves: An asymptotic approach, 2014
                #                 # eq. (9)
                #                 KZ_exact = solve( \
                #                         (Omega/cpa_)**2 + (kx**2 + ky**2)*Nsq_/Omega**2 \
                #                         + (1j/( Omega*rho_ )) * ( \
                #                           ( ( 7*(Omega**2)/(3*cpa_**2) - kx**2 - ky**2 - kz**2 -1./(4*H_**2) )*( kx**2 + ky**2 + (kz - 1j/(2*H_))**2 ) )*shear_ \
                #                         + ( ((Omega/cpa_)**2) * ( kx**2 + ky**2 + (kz - 1j/(2*H_))**2 ) )*bulk \
                #                         ) - kx**2 - ky**2 - kz**2 -1./(4*H_**2) \
                #                         , kz)
                                
                #                 func = lambdify([kx, ky, Omega, H_, Nsq_, cpa_, rho_, shear_], KZ_exact[1].evalf())
                                
                #                 KZ_ = func(0j+self.KX.reshape(Omega_intrinsic.shape[0]*Omega_intrinsic.shape[1]*Omega_intrinsic.shape[2]), \
                #                    0j+self.KY.reshape(Omega_intrinsic.shape[0]*Omega_intrinsic.shape[1]*Omega_intrinsic.shape[2]), \
                #                    0j+Omega_intrinsic.reshape(Omega_intrinsic.shape[0]*Omega_intrinsic.shape[1]*Omega_intrinsic.shape[2]), \
                #                     H, Nsq, cpa, rho, shear).reshape(\
                #                         Omega_intrinsic.shape[0], Omega_intrinsic.shape[1], Omega_intrinsic.shape[2])
                
                #         else:
                #                 kx, kz, Omega = symbols('kx, kz, Omega')
                                
                #                 # From Godin, Dissipation of acoustic-gravity waves: An asymptotic approach, 2014
                #                 # eq. (9)
                #                 KZ_exact = solve( \
                #                         (Omega/cpa)**2 + (kx**2)*Nsq/Omega**2 \
                #                         + (1j/( Omega*rho )) * ( \
                #                           ( ( 7*(Omega**2)/(3*cpa**2) - kx**2 - kz**2 -1./(4*H**2) )*( kx**2 + (kz - 1j/(2*H))**2 ) )*shear \
                #                         + ( ((Omega/cpa)**2) * ( kx**2 + ky**2 + (kz - 1j/(2*H))**2 ) )*bulk \
                #                         ) - kx**2 - kz**2 -1./(4*H**2) \
                #                         , kz)
                                
                #                 func = lambdify([kx,ky,Omega], KZ_exact[1].evalf())
                                
                #                 KZ_ = func(0j+self.KX.reshape(Omega_intrinsic.shape[0]*Omega_intrinsic.shape[1]*Omega_intrinsic.shape[2]), \
                #                    0j+self.KY.reshape(Omega_intrinsic.shape[0]*Omega_intrinsic.shape[1]*Omega_intrinsic.shape[2]), \
                #                    0j+Omega_intrinsic.reshape(Omega_intrinsic.shape[0]*Omega_intrinsic.shape[1]*Omega_intrinsic.shape[2])).reshape(\
                #                         Omega_intrinsic.shape[0],Omega_intrinsic.shape[1],Omega_intrinsic.shape[2])
                
                # Remove infinite/nan numbers that correspond to zero frequencies
                KZ   = np.nan_to_num(KZ, 0.)
                
                # Correct wavenumbers to remove non-physical solutions
                if(correct_wavenumber):
                        indimag     = np.where(np.imag(KZ)<0)
                        KZ[indimag] = np.conjugate(KZ[indimag])
                        KZ = 0.0 - np.real(KZ)*np.sign(Omega_intrinsic) + 1j*np.imag(KZ)
                
                # Deallocate
                Omega_intrinsic = None
                
                return KZ
        
        # Find all layers for which we have to compute the wavenumbers 
        def _find_id_layers_and_heights(self, z0, z1, zlayer):
        
                id_layers = []
                h_layers  = []
                
                id_first_layer = 0
                if(z0 > zlayer[0]):
                        id_first_layer = np.where(zlayer<z0)[0][-1]
                
                id_last_layer = 0
                if(z1 > zlayer[0]):
                        id_last_layer  = np.where(zlayer<z1)[0][-1]
                        
                zprev = z0
                for current_id in range(id_first_layer, id_last_layer):
                        h = zlayer[current_id+1]-zprev
                        if(h > 0):
                                h_layers.append( h )
                                id_layers.append( current_id )
                        zprev = zlayer[current_id+1]
                
                # Last element    
                if not id_first_layer == id_last_layer:
                        h = z1-zlayer[id_last_layer] 
                else:
                        h = z1-z0
                
                if(h > 0):
                        h_layers.append( h )
                        id_layers.append( id_last_layer )
                
                return h_layers, id_layers
        
        def compute_response_at_given_z(self, z1_in, z0, TFMo_in, comp, KZ_in = [], last_layer_in = -1, return_only_KZ = False, only_TFMo_integration = False):
        
                '''
                If return_only_KZ == True, compute_response_at_given_z returns 1j * sum_i KZ_i * h_i in TFMo
                '''
                
                # Exit messages
                if(only_TFMo_integration and return_only_KZ):
                    sys.exit('In "compute_response_at_given_z": Can not only integrate TFMo and only return KZ simultaneously!')
                
                try:
                    if(only_TFMo_integration and not KZ_in):
                                sys.exit('In "compute_response_at_given_z": Can not only integrate TFMo without KZ provided!')
                except:
                    pass
                        
                # If pressure amplitude decreases with altitude
                coef_amplitude = 1. if comp == 'vz' else -1.
                
                # If we return only KZ, TFMo will contain sum 1j*KZ*z1
                if(not return_only_KZ):
                        TFMo = TFMo_in.copy()
                else:
                        TFMo = np.zeros(self.TFMo.shape, dtype=complex)
                        
                multiple_altitudes_submitted = False
                if(len(z1_in) > 1):
                        multiple_altitudes_submitted = True
                   
                last_layer = last_layer_in
                KZ_tab, field_at_it_tab = [], []   
                for id_z1, z1 in enumerate(z1_in):
                
                        field_at_it = []
                        if(only_TFMo_integration and multiple_altitudes_submitted):
                                KZ = KZ_in[id_z1].copy()   
                        else:
                                KZ = KZ_in.copy()
                                         
                        # We only compute the solution if we are not at the surface or below 
                        if(z1 > 0):
                                
                                # If isothermal model
                                if(self.isothermal):
                                        # Compute the vertical response from the ground forcing
                                        if(not KZ):
                                                KZ = self.compute_vertical_wavenumber(0)       
                                        
                                        if(not return_only_KZ):
                                                field_at_it = np.exp(coef_amplitude*z1/(2*self.H[0])) * fftpack.ifftn( np.exp(1j*(KZ*z1)) * TFMo)
                                          
                                # If layered model
                                else:
                                
                                        #local_inner_loop = partial(self.inner_loop, TFMo, there_is_vz, TFMo_p, there_is_p, x_in, y_in, z_in, comp_in, name_in, t_chosen_in, id_in)
                
                                        #N = 4#mp.cpu_count()
                                        
                                        #with mp.Pool(processes = N) as p:
                                        #        results = p.map(local_inner_loop, combinaisons)
                                
                                        h_layers, id_layers = self._find_id_layers_and_heights(z0, z1, self.z)
                                        for idx, id_layer in enumerate(id_layers):
                                
                                                # Compute the vertical response from the forcing of the layer beneath (idz-1)
                                                if(not last_layer == id_layer):
                                                        KZ = self.compute_vertical_wavenumber(id_layer)       
                                                
                                                if(not return_only_KZ and not only_TFMo_integration):
                                                        TFMo *= np.exp(coef_amplitude*h_layers[idx]/(2*self.H[id_layer])) * np.exp(1j*(KZ*h_layers[idx])) 
                                                        
                                                elif(not return_only_KZ):
                                                        TFMo *= np.exp(coef_amplitude*h_layers[idx]/(2*self.H[id_layer])) #* np.exp(1j*(KZ*h_layers[idx])) 
                                                
                                                else:
                                                        TFMo += 1j*(KZ * h_layers[idx])

                                        if(only_TFMo_integration):
                                                 TFMo *= np.exp(KZ) 
                                        
                                        if(not return_only_KZ):
                                                field_at_it = fftpack.ifftn( TFMo )
                                        
                                        if( len(z1_in) == id_z1+1 ):        
                                                last_layer = id_layer   
                                                
                                        z0 = z1   
                                        
                        elif(not return_only_KZ):
                                field_at_it = fftpack.ifftn(TFMo)
                
                        if(multiple_altitudes_submitted):
                                if(not return_only_KZ):
                                        field_at_it_tab.append( field_at_it.copy() )
                                else:
                                        KZ_tab.append( TFMo.copy() )
                
                if(multiple_altitudes_submitted):
                        return field_at_it_tab, last_layer, KZ, KZ_tab
                else:
                        return field_at_it, last_layer, KZ, TFMo
                
        def convert_TFMo_to_pressure(self):
        
                # Ignore division/invalid errors in P computation
                np.seterr(divide='ignore', invalid='ignore')   
        
                KZ = self.compute_vertical_wavenumber(0, correct_wavenumber = True)
                indnot0 = np.where(abs(self.Omega) > 0)
                P = np.zeros(self.Omega.shape, dtype=complex)
                P[indnot0] = self.rho[0]*(self.cpa[0]**2)*KZ[indnot0]*self.TFMo[indnot0]/(self.Omega[indnot0])
                P = np.nan_to_num(P, 0.)
                
                #bp()
                
                del KZ
                
                return P
                
        def get_index_tabs_time(self, t):
                return(np.argmin(abs(self.t-t)))
        def get_index_tabs(self, t, x, y):
                # Get required time index        
                it = self.get_index_tabs_time(t)
                # Get required location index
                ix = np.argmin( abs(self.x - x) )
                iy = -1
                if(self.dimension > 2):
                        iy = np.argmin( abs(self.y - y) )
                return it, ix, iy
                        
        # Compute wavefield for a given physical domain
        def compute_field_for_xz(self, t, x, y, z, zvect, type_slice, comp):
                print('['+sys._getframe().f_code.co_name+'] Compute a '+comp+' wavefield, along '+type_slice+', at t='+str(t)+', x='+str(x)+', y='+str(y)+', z='+str(z)+'.')
                
                # Build a response matrix based on required slice dimensions
                if(type_slice == 'z'):
                        d1, d2 = len(zvect), len(self.x)
                        Mz_xz = np.zeros((d1, d2), dtype=complex)
                        if(self.dimension > 2):
                                d1, d2 = len(zvect), len(self.y)
                                Mz_yz = np.zeros((d1, d2), dtype=complex)
                elif(type_slice == 'xy'):
                        d1, d2 = len(self.x), len(self.y)
                        Mz_xy = np.zeros((d1, d2), dtype=complex)
                        zvect  = [z]
                else:
                        sys.exit('Slice "'+str(type_slice)+'" not recognized!')
                print('['+sys._getframe().f_code.co_name+'] > Will perform atmopheric computation at '+str(len(zvect))+' altitudes between z='+str(min(zvect))+' and z='+str(max(zvect))+'.')
                        
                # Get required time index        
                # modif 5/1/2020
                #it = np.argmin( abs(self.t - t) )
                it, ix, iy = self.get_index_tabs(t, x, y)
                
                # setup progress bar
                if(len(zvect) > 1):
                        cptbar        = 0
                        toolbar_width = 40
                        total_length  = len(zvect)
                        sys.stdout.write("Building wavefield: [%s]" % (" " * toolbar_width))
                        sys.stdout.flush()
                        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
                
                # Load initial surface forcing
                if(comp == 'vz'):
                        TFMo    = self.TFMo.copy()
                else:
                        TFMo    = self.convert_TFMo_to_pressure().copy()
                        
                # Find location of balloon to extract time series
                zloc = np.argmin( abs(z - np.array(zvect)) )
                        
                iz_prev = 0.
                last_layer_prev = -1
                KZ_prev = []
                # Loop over all layers
                for idz, iz in enumerate(zvect):
                  field_at_it, last_layer, KZ, TFMo = self.compute_response_at_given_z([iz], iz_prev, TFMo, comp, KZ_prev, last_layer_prev)
                  
                  # Store wavenumber if a new wavenumber has been computed
                  if(last_layer > -1):
                    last_layer_prev = last_layer
                    KZ_prev = KZ.copy()
                  iz_prev = iz
                  
                  # 3d
                  if(self.dimension > 2):
                    if(type_slice == 'z'):
                      # modif 5/1/2020
                      #iy = np.argmin( abs(self.y - y) )
                      Mz_xz[idz, :] = field_at_it[it,:,iy]
                      
                      #ix = np.argmin( abs(self.x - x) )
                      Mz_yz[idz, :] = field_at_it[it,ix,:]
                      
                      # Save time series
                      if(idz == zloc):
                        timeseries = field_at_it[:,ix,iy]
                            
                    elif(type_slice == 'xy'):
                      Mz_xy[:, :] = field_at_it[it,:,:]
                      
                      # Save time series
                      if(idz == zloc):
                        # modif 5/1/2020
                        #iy = np.argmin( abs(self.y - y) )
                        #ix = np.argmin( abs(self.x - x) )
                        timeseries = field_at_it[:,ix,iy]
                                          
                  # 2d
                  else:
                    if(type_slice == 'z'):
                      Mz_xz[idz, :] = field_at_it[it,:]
                      # Save time series
                      if(idz == zloc):
                        # modif 5/1/2020
                        #ix = np.argmin( abs(self.x - x) )
                        timeseries = field_at_it[:,ix]
                      
                    elif(type_slice == 'xy'):
                      # An "XY slice" makes little sense in 2D, but we still want to grab the "line" that makes.
                      Mz_xy[:] = np.reshape(field_at_it[it,:], (-1, 1))
                      # Save time series
                      if(idz == zloc):
                        timeseries = field_at_it[:,ix]

                  # update the bar
                  if(len(zvect) > 1):
                    if(int(toolbar_width*idz/total_length) > cptbar):
                      cptbar = int(toolbar_width*idz/total_length)
                      sys.stdout.write("-")
                      sys.stdout.flush()
                
                if(len(zvect) > 1):
                        sys.stdout.write("] Done\n")
                
                if(self.dimension > 2):
                  if(type_slice == 'z'):
                    return Mz_xz, Mz_yz, timeseries
                  elif(type_slice == 'xy'):
                    return Mz_xy, timeseries
                else:
                  if(type_slice == 'z'):
                    return Mz_xz, timeseries
                  elif(type_slice == 'xy'):
                    return Mz_xy, timeseries
                                
        def compute_field_timeseries(self, station_in, merged_computation = False, create_timeseries_here = True):
          # Extract location and component from station dict
          x_in, y_in, z_in = [stat['xs'] for stat in station_in], \
                             [stat['ys'] for stat in station_in], \
                             [stat['zs'] for stat in station_in]
          comp_in = [stat['comp'] for stat in station_in]
          name_in = [stat['name'] for stat in station_in]
          # t_chosen_in = [stat['t_chosen'] for stat in station_in]
          id_in   = [stat['id'] for stat in station_in]
          
          # Setup progress bar
          cptbar        = 0
          toolbar_width = 40
          total_length  = len(x_in)
          sys.stdout.write("Building time series: [%s]" % (" " * toolbar_width))
          sys.stdout.flush()
          sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
          
          if(merged_computation):
            _, _, _, integrated_KZ = self.compute_response_at_given_z(np.unique(z_in).tolist(), 0., [], 'p', return_only_KZ = True) # 'p' is dummy
           
          there_is_vz = False
          if('vz' in comp_in):
            there_is_vz = True
            if(False):
              TFMo = self.TFMo.copy()
          
          there_is_p = False
          if('p' in comp_in):
            there_is_p = True
            if(False):
              TFMo_p = self.convert_TFMo_to_pressure().copy()
           
          station_tab = {}
          Mz, Mo = [], []       
          id_stat = 0
          for comp in np.unique(comp_in):
            if('vz' in comp):
              TFMo_ = self.TFMo.copy()
            if('p' in comp):
              TFMo_ = self.convert_TFMo_to_pressure().copy()
    
            for id_z, z in enumerate(np.unique(z_in)):
              if(not merged_computation):
                if z > 0:
                  field_at_it_, _, _, _ = self.compute_response_at_given_z([z], 0., TFMo_, comp)
              else:
                if(there_is_vz and comp == 'vz'):
                  field_at_it,   _, _, _ = self.compute_response_at_given_z([z], 0., TFMo, comp, KZ_in = integrated_KZ[id_z], only_TFMo_integration = True)
                elif(there_is_p and comp == 'p'):
                  field_at_it_p, _, _, _ = self.compute_response_at_given_z([z], 0., TFMo_p, comp, KZ_in = integrated_KZ[id_z], only_TFMo_integration = True)
                else:
                  sys.exit('Unit ' + comp + ' not recognized!')  
                       
              # Stores fields
              if(not merged_computation and create_timeseries_here):
                field_at_it_loc      = []
                link_field_at_it_loc = []
                for x in np.unique(x_in):
                  ix   = np.argmin( abs(self.x - x) )
                  for y in np.unique(y_in):
                    if(self.dimension==2):
                      # 2D, no need to find y.
                      field_at_it_loc.append( np.real(field_at_it_[:, ix]) )
                    elif(self.dimension==3):
                      iy = np.argmin( abs(self.y - y) )
                      field_at_it_loc.append( np.real(field_at_it_[:, ix, iy]) )
                    else:
                      raise ValueError('[%s] Field dimension is %d, which is impossible.'
                                       % (sys._getframe().f_code.co_name, self.dimension))
                    link_field_at_it_loc.append( {'x': x, 'y': y} )
                del field_at_it_
                       
              # Loop over all required station locations
              # for x, y, z_aux, comp_aux, name, t_chosen, id in zip(x_in, y_in, z_in, comp_in, name_in, t_chosen_in, id_in):
              for x, y, z_aux, comp_aux, name, id in zip(x_in, y_in, z_in, comp_in, name_in, id_in):
                # Skip all stations that do not match z / comp
                if(not z == z_aux or not comp == comp_aux):
                  continue
                
                ix   = np.argmin( abs(self.x - x) )
                
                if(self.dimension==3):
                  iy = np.argmin( abs(self.y - y) )
                  if create_timeseries_here:
                    for id_field, ifield in enumerate(link_field_at_it_loc):
                      if ifield['x'] == x and ifield['y'] == y :
                        id_field_chosen = id_field
                    Mz = field_at_it_loc[id_field_chosen]
                    Mo = self.Mo[:, ix, iy] 
                    del field_at_it_loc
                    
                  else:
                    if z_aux > 0:   
                      Mz.append( field_at_it_[:, ix, iy] )
                    else:   
                      Mz.append( self.Mo[:, ix, iy] )
                    Mo.append( self.Mo[:, ix, iy] )
                    
                elif(self.dimension==2):
                  if create_timeseries_here:
                    for id_field, ifield in enumerate(link_field_at_it_loc):
                      if ifield['x'] == x:
                        id_field_chosen = id_field
                    del field_at_it_loc
                    
                  else:
                    if z_aux > 0:   
                      Mz.append( field_at_it_[:, ix] )
                    else:   
                      Mz.append( self.Mo[:, ix] )
                    Mo.append( self.Mo[:, ix] )
                    
                else:
                  raise ValueError('[%s] Field dimension is %d, which is impossible.'
                                   % (sys._getframe().f_code.co_name, self.dimension))
                
                # Create list of station with the right entries
                station = {}
                station.update( {'id': id} )
                station.update( {'id_field': id_stat} )
                station.update( {'name': name} )
                if(self.dimension > 2):
                  station.update( {'xs': x, 'ys': y, 'zs': z_aux} ) 
                else:
                  station.update( {'xs': x, 'zs': z_aux} ) 
                # station.update( {'t_chosen': t_chosen} )
                station.update( {'type_slice': 'xz'} )
                station.update( {'comp': comp_aux} )
                station_tab[id] = station                            
                        
                # Create waveform within this routine if requested
                if create_timeseries_here:
                  options_in = {}
                  options_in['GOOGLE_COLAB']   = self.google_colab
                  options_in['coef_low_freq']  = self.coef_low_freq[0]
                  options_in['coef_high_freq'] = self.coef_low_freq[-1]
                  options_in['global_folder']  = self.global_folder
                  generate_one_timeseries(self.t, Mz, np.real(self.Mo[:, ix, iy]), comp_aux, z, y, x, name, options_in)  
                  
                  # Delete working arrays to save memory space
                  del Mz, Mo, link_field_at_it_loc, ifield
                        
                # Update progress bar
                id_stat += 1
                if(int(toolbar_width*id_stat/total_length) > cptbar):
                  cptbar = int(toolbar_width*id_stat/total_length)
                  sys.stdout.write("-")
                  sys.stdout.flush()
              
              if(merged_computation):
                del field_at_it, field_at_it_p
              elif(z > 0):
                del field_at_it_        
                    
          sys.stdout.write("] Done\n")
          return(Mz, Mo, station_tab)

def plot_surface_forcing(field, t_station, ix, iy, output_folder, GOOGLE_COLAB=False):
    print('['+sys._getframe().f_code.co_name+'] Plot the Rayleigh wave surface forcing at t='+str(t_station)+'.')
        
    it_, ix_, iy_ = field.get_index_tabs(t_station, ix, iy)
    
    fig, axs = plt.subplots(nrows=1, ncols=1)
    
    plotMo = axs.imshow(np.flipud(np.real(field.Mo[it_, :, :]).T), extent=[field.x[0]/1000., field.x[-1]/1000., field.y[0]/1000., field.y[-1]/1000.], aspect='auto')
    axs.scatter(ix/1000., iy/1000., color='red', zorder=2)
    axs.set_xlabel('West - East [km]')
    axs.set_ylabel('South - North [km]')
    
    axins = inset_axes(axs, width="5%", height="100%", loc='lower left', bbox_to_anchor=(1.02, 0., 1, 1.), bbox_transform=axs.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    
    cbar = plt.colorbar(plotMo, cax=axins)
    RWAtmosUtils.autoAdjustCLim(plotMo)
    
    fig.subplots_adjust(hspace=0.3, right=0.8, left=0.2, top=0.94, bottom=0.15)
    
    if(not GOOGLE_COLAB):
        cbar.ax.set_ylabel('$v_z$ [m/s]', rotation=90)
        plt.savefig(output_folder + 'map_wavefield_forcing_t'+str(round(t_station, 2))+'.pdf')

def create_RW_field(Green_RW, domain, param_atmos, options, ncpus = 16, verbose=True):
    if(verbose): print('['+sys._getframe().f_code.co_name+'] Create Rayleigh wave field from Green functions.')
    
    # Extract parameters.
    dimension         = options['dimension']
    dimension_seismic = options['dimension_seismic']
    nb_freq           = options['nb_freq']
    mode_max          = -1 # -1 will compute all possible modes. Any other value will truncate the computation to the first n modes. Negative values will probably break.
    
    # Define domain.
    if(verbose): print('['+sys._getframe().f_code.co_name+'] > Start by defining domain.')
    # print('[%s] >> Domain is DX * DY * DZ = [%.3f, %.3f] * [%.3f, %.3f] * [%.3f, %.3f] km.'
    #       % (sys._getframe().f_code.co_name, domain['xmin']/1e3, domain['xmax']/1e3, domain['ymin']/1e3, domain['ymax']/1e3, domain['zmin']/1e3, domain['zmax']/1e3))
    # print('['+sys._getframe().f_code.co_name+'] >> Steps are (Dx, Dy, Dz) = (%.0f, %.0f, %.0f) m.'
    #       % (domain['dx'], domain['dy'], domain['dz']))
    xbounds = [domain['xmin'], domain['xmax']]
    ybounds = [domain['ymin'], domain['ymax']]
    # dx, dy, dz = domain['dx'], domain['dy'], domain['dz']
    dx, dy = domain['dx'], domain['dy']
    # z          = np.arange(domain['zmin'], domain['zmax'], dz)
    field = field_RW(Green_RW, nb_freq, dimension, dx, dy, xbounds, ybounds, mode_max, dimension_seismic, ncpus = ncpus)
    
    # Create atmospheric profiles.
    if(verbose): print('['+sys._getframe().f_code.co_name+'] > Update field with atmospheric model.')
    if(not param_atmos):
      param_atmos = velocity_models.generate_default_atmos()
    field.generate_atmospheric_model(param_atmos)
    
    # # Plot profiles.
    # # Using Matplotlib within multithreads breaks everything. Moving this outside of the create_RW_field routine.
    # velocity_models.plot_atmosphere_and_seismic(field.global_folder, field.seismic, field.z, 
    #                                             field.rho, field.cpa, field.winds, field.H, 
    #                                             field.isothermal, field.dimension, field.google_colab)
    if(verbose): print(field)
    return(field)

def compute_analytical_acoustic(Green_RW, mechanism, param_atmos, station, domain, options):
    print('['+sys._getframe().f_code.co_name+'] Generate analytical Rayleigh wave time series and acoustic time series.')
    print('['+sys._getframe().f_code.co_name+'] > Uses the chosen source mechanism and previously computed Green functions.')

    # Exit messages
    if(not mechanism):
      sys.exit('Mechanism has to be provided to build analytical solution')
    if(not domain):
      sys.exit('Domain has to be provided to build analytical solution')
    # if(not station):
    #   sys.exit('Stations have to be provided to build analytical solution')
    
    # Update mechanism.
    print('['+sys._getframe().f_code.co_name+'] Update Green functions with chosen focal mechanism.')
    Green_RW.set_mechanism(mechanism)
    
    # Create the Rayleigh wave field.
    field = create_RW_field(Green_RW, domain, param_atmos, options)
    
    # Plot profiles.
    velocity_models.plot_atmosphere_and_seismic(field.global_folder, field.seismic, field.z, 
                                                field.rho, field.cpa, field.winds, field.H, 
                                                field.isothermal, field.dimension, field.google_colab)
    
    # Compute maps/slices at given instants.
    # Use parameters of first station to build 2d/3d wavefields.
    print('['+sys._getframe().f_code.co_name+'] Compute maps/slices.')
    id_wavefield = 0
    for t_snap in options['t_chosen']:
      
      # Compute atmospheric XY pressure fields.
      if(not options['COMPUTE_XY_PRE']==None):
        print('['+sys._getframe().f_code.co_name+'] Compute atmospheric XY pressure fields.')
        Mxy, Mz_t_tab = field.compute_field_for_xz(t_snap, 0., 0., options['COMPUTE_XY_PRE'], None, 'xy', 'p')
        
        fig = plt.figure()
        hplt = plt.imshow(np.flipud(np.real(Mxy).T), extent=[field.y[0]/1000., field.y[-1]/1000., field.x[0]/1000., field.x[-1]/1000.], aspect='auto')
        plt.xlabel('West-East [km]')
        plt.ylabel('South-North [km]')
        plt.title('Pressure Field at %.1f km' % (options['COMPUTE_XY_PRE']/1e3))
        cbar = plt.colorbar(hplt)
        plt.savefig(options['global_folder']+'map_XY_PRE_t%07.2f.pdf' % (t_snap))
        
        np.real(Mxy).tofile(options['global_folder']+'map_XY_PRE_t%07.2f_%dx%d_z%07.2f.bin'
                            % (t_snap, Mxy.shape[0], Mxy.shape[1], options['COMPUTE_XY_PRE']/1e3))
        if(t_snap == options['t_chosen'][0]):
          np.array([field.x[0], field.x[-1], field.y[0], field.y[-1]]).tofile(options['global_folder']+'map_XY_PRE_XYminmax.bin')
        # Reading is done using A=np.fromfile(filename) in Python, or
        #                       fid=fopen(filename,'r'); A=fread(fid,'real*8'); fclose(fid); in Matlab.
      
      if(station):
        # Somehow stations are needed to compute maps/slices.
        # TODO: parametrise maps/slices separately from stations. Only needs comp and type_slice, since the slice spans the whole domain.
        comp       = station[id_wavefield]['comp']
        iz         = station[id_wavefield]['zs']
        iy         = station[id_wavefield]['ys']
        ix         = station[id_wavefield]['xs']
        # type_slice = station[id_wavefield]['type_slice']
        
        # Compute Rayleigh wave surface forcing.
        plot_surface_forcing(field, t_snap, ix, iy, options['global_folder'], options['GOOGLE_COLAB'])
        
        # Compute maps/slices for a given range of altitudes (m) at a given instant (s)
        if(field.dimension > 2):
          if(options['COMPUTE_MAPS']):
            print('['+sys._getframe().f_code.co_name+'] '+str(field.dimension)+'D. Compute slices along xz, yz, and xy.')
            Mxz, Myz, Mz_t_tab = field.compute_field_for_xz(t_snap, ix, iy, iz, field.z, 'z', comp)
            Mxy, Mz_t_tab      = field.compute_field_for_xz(t_snap, ix, iy, iz, field.z, 'xy', comp)
          nb_cols  = 2
        else:
          if(options['COMPUTE_MAPS']):
            print('['+sys._getframe().f_code.co_name+'] '+str(field.dimension)+'D. Compute slice along xz only.')
            Mxz, Mz_t_tab = field.compute_field_for_xz(t_snap, ix, iy, iz, field.z, 'z', comp) 
          nb_cols = 1
        
        # Display those maps/slices.
        fig, axs = plt.subplots(nrows=2, ncols=nb_cols)
        if(comp == 'p'):
          unknown_label = 'Pressure (Pa)'
        else:
          unknown_label = 'Velocity (m/s)'
        if(options['COMPUTE_MAPS']):
          if(field.dimension > 2):
            vmin, vmax = np.real(Mxy).min(), np.real(Mxy).max()
    
            iax = 0
            iax_col = 0
            axs[iax, iax_col].plot(field.t, np.real(Mz_t_tab), zorder=1)
            axs[iax, iax_col].axvline(t_snap, color='red', zorder=0)
            axs[iax, iax_col].grid(True)
            axs[iax, iax_col].set_xlim([field.t[0], field.t[-1]])
            axs[iax, iax_col].set_xlabel('Time (s)')
            
            axs[iax, iax_col].set_ylabel(unknown_label)
            
            iax_col += 1
            
            # plotMxy = axs[iax, iax_col].imshow(np.flipud(np.real(Mxy).T), extent=[field.y[0]/1000., field.y[-1]/1000., field.x[0]/1000., field.x[-1]/1000.], aspect='auto', vmin=vmin, vmax=vmax)
            axs[iax, iax_col].scatter(iy/1000., ix/1000., color='red', zorder=2)
            axs[iax, iax_col].set_ylabel('West - East (km)')
            axs[iax, iax_col].yaxis.set_label_position("right")
            axs[iax, iax_col].yaxis.tick_right()
            
            iax += 1
            iax_col = 0
            
            plotMxz = axs[iax, iax_col].imshow(np.flipud(np.real(Mxz).T), extent=[field.x[0]/1000., field.x[-1]/1000., field.z[0]/1000., field.z[-1]/1000.], aspect='auto', vmin=vmin, vmax=vmax)
            axs[iax, iax_col].scatter(ix/1000., iz/1000., color='red', zorder=2)
            axs[iax, iax_col].set_xlabel('West - East (km)')
            axs[iax, iax_col].set_ylabel('Altitude (km)')
            
            iax_col += 1
            
            plotMyz = axs[iax, iax_col].imshow(np.flipud(np.real(Myz).T), extent=[field.y[0]/1000., field.y[-1]/1000., field.z[0]/1000., field.z[-1]/1000.], aspect='auto')
            axs[iax, iax_col].scatter(iy/1000., iz/1000., color='red', zorder=2)
            axs[iax, iax_col].set_xlabel('South - North (km)')
            axs[iax, iax_col].text(0.5, 0.1, 't = ' + str(t_snap) + 's', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', edgecolor='black', pad=2.0), transform=axs[iax, iax_col].transAxes)
            axs[iax, iax_col].yaxis.set_label_position("right")
            
            axins = inset_axes(axs[iax, iax_col], width="5%", height="100%", loc='lower left', bbox_to_anchor=(1.02, 0., 1, 1.), bbox_transform=axs[iax, iax_col].transAxes, borderpad=0)
            axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
            
            cbar = plt.colorbar(plotMyz, cax=axins)
               
          else:
            iax = 0
            axs[iax].plot(field.t, np.real(Mz_t_tab), zorder=2)
            
            axs[iax].grid(True)
            axs[iax].set_xlim([field.t[0], field.t[-1]])
            axs[iax].set_xlabel('Time (s)')
            axs[iax].set_ylabel(unknown_label)
            
            if(options['PLOT_RW_time_series']):
              ax2 = axs[iax].twinx()  # instantiate a second axes that shares the same x-axis
              color = 'tab:red'
              ax2.set_ylabel('RW', color=color)  # we already handled the x-label with ax1
              ax2.plot(field.t, np.real(RW_Mz_t_tab[id_wavefield_timeseries]), color=color, zorder=1, linestyle='--')
              ax2.tick_params(axis='y', labelcolor=color)
              
              RWAtmosUtils.align_yaxis_np([axs[iax],ax2])
            
            iax += 1
            plotMxz = axs[iax].imshow(np.flipud(np.real(Mxz).T), extent=[field.x[0]/1000., field.x[-1]/1000., field.z[0]/1000., field.z[-1]/1000.], aspect='auto')
            axs[iax].scatter(ix/1000., iz/1000., color='red', zorder=2)
            axs[iax].set_xlabel('Distance from source (km)')
            axs[iax].set_ylabel('Altitude (km)')
            axs[iax].text(0.15, 0.9, 't = ' + str(t_snap) + 's', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', edgecolor='black', pad=2.0), transform=axs[iax].transAxes)
            
            axins = inset_axes(axs[iax], width="2.5%", height="100%", loc='lower left', bbox_to_anchor=(1.02, 0., 1, 1.), bbox_transform=axs[iax].transAxes, borderpad=0)
            axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
            
            cbar = plt.colorbar(plotMxz, cax=axins)
         
          fig.subplots_adjust(hspace=0.3, right=0.8, left=0.2, top=0.94, bottom=0.15)
        
          if(not options['GOOGLE_COLAB']):
            cbar.ax.set_ylabel(unknown_label, rotation=90) 
            plt.savefig(options['global_folder'] + 'map_wavefield_vz_t%07.2f.pdf' % (t_snap))
            plt.close('all')
    
    if(station):
      # Compute time series for each station.
      print('['+sys._getframe().f_code.co_name+'] Compute time series at each stations.')
      create_timeseries_here = False
      Mz_t_tab, RW_Mz_t_tab, station_updated = field.compute_field_timeseries(station, create_timeseries_here = create_timeseries_here)
      # id_wavefield_timeseries = station_updated[id_wavefield]['id_field']
      if(not create_timeseries_here):
        for id_wavefield in station_updated:
          # Current field
          id_wavefield_timeseries = station_updated[id_wavefield]['id_field']
          Mz_t = Mz_t_tab[id_wavefield_timeseries]
          # Current station parameters
          comp = station[id_wavefield]['comp']
          iz = station[id_wavefield]['zs']
          iy = station[id_wavefield]['ys']
          ix = station[id_wavefield]['xs']
          stat = station[id_wavefield]['name']
          generate_one_timeseries(field.t, Mz_t, RW_Mz_t_tab[id_wavefield_timeseries], comp, iz, iy, ix, stat, options)        
    
    # Deallocate
    del field, Mz_t_tab, RW_Mz_t_tab, station_updated
    
    # Successful exit message.
    print('['+sys._getframe().f_code.co_name+'] Finished generating figures in folder \''+options['global_folder']+'\'.')
    
    #if(not options['GOOGLE_COLAB']):
    #        bp()
   

##############
##############    
# if __name__ == '__main__': 

#         from obspy.core.utcdatetime import UTCDateTime

#         use_individual_SCEC_models = False # Expensive
#         use_individual_SCEC_models_each_iter = True
#         add_perturbations         = False # Very expensive
#         only_perturbations        = True
#         get_normal_reverse_strike = True # Instead of minimizing energy for a focal mecha perturbations, just get extreme cases
#                                          # of strike-slip, reverse and normal faults. Only available for add_perturbations = True
#         add_Yang_layer            = False
#         list_of_events            = [0, 38624623]
        
#         name_sample               = './RIDGECREST_XXX/'
        
#         options_source = {}
#         options_source['stf-data'] = []
#         options_source['stf']      = 'gaussian' # gaussian or erf
#         options_source['f0']      = 5.
#         options_source['rotation'] = False
#         options_source['rotation-towards'] = 'CrazyCat' # name balloon or empty -> will take first balloon
#         #options_source['rotation-towards'] = 'Tortoise' # name balloon or empty -> will take first balloon
#         options_source['lat_min'] = 33.56600971952403 
#         options_source['lat_max'] = 34.40549099227338
#         options_source['lon_min'] = -116.25228005560061
#         options_source['lon_max'] = -115.2953768987713
#         options_source['activate_LA'] = False
        
#         options_IRIS = {}
#         options_IRIS['network'] = 'CI,NN,GS,SN,PB,ZY'
#         options_IRIS['channel'] = 'HHZ,HNZ,DPZ,BNZ,BHZ,ENZ,EHZ'
#         options_IRIS['stations'] = {}
#         x, y, z, comp, name, id = 100., 1000., 0., 'HHZ', 'test', 1
#         options_IRIS['stations'][id] = mod_mechanisms.create_one_station(x, y, z, comp, name, id)
        
#         options = {}
#         options['dimension']   = 3
#         options['dimension_seismic'] = 3
#         options['ATTENUATION']    = True
#         options['COMPUTE_MAPS']   = False
#         options['nb_freq']        = 2**9
#         options['nb_kxy']         = 2**7
#         options['coef_low_freq']  = 0.001
#         options['coef_high_freq'] = 5.
#         options['nb_layers']      = 100
#         options['zmax'] = 60000.
#         options['models'] = {}
#         options['nb_modes']    = [0, 50]
#         options['force_dimension'] = False # Only when add_specfem_simu = True
#         options['force_f0_source'] = False
#         options['t_chosen']        = [40., 90.] # to display wavefield
#         options['USE_SPAWN_MPI'] = False
#         options['models']['specfem'] = '/staff/quentin/Documents/Projects/Ridgecrest/seismic_models/Ridgecrest_seismic.txt'
#         options['type_model']    = 'specfem' # specfem or specfem2d
        
#         if options['USE_SPAWN_MPI']:
#                 set_start_method("spawn") 
        
#         # Load sources from Earthquake catalog
#         options_source['coef_high_freq'] = options['coef_high_freq']
#         options_source['nb_kxy']   = options['nb_kxy']
#         options_source['t_chosen'] = options['t_chosen']
        
#         options_source['sources'] = []
#         source_characteristics = {
#             'id': 0,
#             'time': UTCDateTime(2019, 8, 9, 0, 9, 57),
#             'mag': 2.98,
#             'lat': 35.895,
#             'lon': -117.679,
#             'depth': 4.1, #km
#             'strike': 159,
#             'dip': 89,
#             'rake': -156,
#         }
#         options_source['sources'].append( source_characteristics )
        
#         options_source['DIRECTORY_MECHANISMS'] = []
#         options_source['DIRECTORY_MECHANISMS'].append( '/staff/quentin/Documents/Projects/Ridgecrest/YHS_catalog_August9_new.csv' ) 
#         options_source['DIRECTORY_MECHANISMS'].append( '/staff/quentin/Documents/Projects/Ridgecrest/YHS_catalog_July22_new.csv' )
    
#         options_balloon = {}
#         main_dir_balloon = '/staff/quentin/Documents/Projects/Ridgecrest/data_balloons/Siddharth_balloon/'
#         options_balloon['DIR_BALLOON_GPS'] = []
#         options_balloon['DIR_BALLOON_GPS'].append( [main_dir_balloon+'Hare_GPS.csv', datetime(2019, 7, 22, 0, 0, 0)] )
#         options_balloon['DIR_BALLOON_GPS'].append( [main_dir_balloon+'Tortoise_GPS.csv', datetime(2019, 7, 22, 0, 0, 0)] )
#         options_balloon['DIR_BALLOON_GPS'].append( [main_dir_balloon+'CrazyCat_GPS.csv', datetime(2019, 8, 9, 0, 0, 0)] )
#         options_balloon['DIR_BALLOON_GPS'].append( [main_dir_balloon+'Hare2_GPS.csv', datetime(2019, 8, 9, 0, 0, 0)] )
#         options_balloon = {}
#         mechanisms_data = mod_mechanisms.load_source_mechanism_IRIS(options_source, options_IRIS, dimension=options['dimension'], 
#                                                                     add_SAC = True, add_perturbations = False, specific_events=list_of_events,
#                                                                     options_balloon=options_balloon)
        
#         # Use same seismic model for all simulations        
#         if(not use_individual_SCEC_models):
#             Green_RW, options_out = RW_dispersion.compute_trans_coefficients(options)
        
#         # Add perturbations to mechanisms to account for uncertainties
#         if(add_perturbations): 
#                 if get_normal_reverse_strike:
#                         mechanisms_data = mod_mechanisms.find_extreme_cases(mechanisms_data, get_normal_reverse_strike)
#                 elif not use_individual_SCEC_models:
#                         mechanisms_data = mod_mechanisms.find_extreme_cases(mechanisms_data, get_normal_reverse_strike, Green_RW=Green_RW)
#                 else:
#                         sys.exit('Perturbations in focal mechanism can not be added')
                        
#         # If different seismic models for each simulation, load all seismic models
#         if(use_individual_SCEC_models and not use_individual_SCEC_models_each_iter):
#                 mechanisms_data = mechanisms_data.apply(velocity_models.create_velocity_model_, args=[options, add_Yang_layer], axis=1)
        
#         # Save list of events
#         mechanisms_data.to_pickle("./mechanisms_data.pkl")
        
#         mechanism, station, domain = {}, {}, {}
#         if(not use_individual_SCEC_models):
#                 os.system('mv ' + options_out['global_folder'] + ' ' + name_sample.replace('XXX', 'tocopy'))
        
#         keys_mechanism = ['EVID', 'stf', 'stf-data', 'zsource', 'f0', 'M0', 'M', 'phi', 'station_tab', 'mt']
        
#         # Reset indexes for directory creation
#         mechanisms_data = mechanisms_data.reset_index(drop=True)
        
#         # Only loop over unperturbed focal mechanisms since perturbed ones will have the same seismic profile
#         mechanisms_data_noperturb = mechanisms_data.loc[~mechanisms_data['perturbation'], :].copy()
#         for imecha_, mechanism_data in mechanisms_data_noperturb.iterrows():
                
#                 # If different seismic models for each simulation, create eigenfunctions
#                 if use_individual_SCEC_models:
                
#                         if use_individual_SCEC_models_each_iter:
#                                 mechanisms_data_ = pd.DataFrame([mechanism_data])
#                                 mechanisms_data_ = mechanisms_data_.apply(velocity_models.create_velocity_model_, args=[options, add_Yang_layer], axis=1)
#                                 mechanisms_data_ = mechanisms_data_.iloc[0]
#                                 seismic = velocity_models.construct_local_seismic_model(mechanisms_data_, options)
#                                 del mechanisms_data_
#                         else:
#                                 seismic = velocity_models.construct_local_seismic_model(mechanism_data, options)
                            
#                         options['models']['specfem'] = seismic[0]
#                         options['type_model']        = seismic[1] # specfem or specfem2d
                        
#                         if 'coef_high_freq' in mechanism_data.keys(): 
#                                 options['coef_high_freq'] = mechanism_data['coef_high_freq']
                        
#                         Green_RW, options_out = RW_dispersion.compute_trans_coefficients(options)
                        
#                         os.system('rm -rf ' + name_sample.replace('XXX', 'tocopy'))
#                         os.system('mv ' + options_out['global_folder'] + ' ' + name_sample.replace('XXX', 'tocopy'))
                        
#                         del seismic, options['models']['specfem'], options['type_model'] # Clear variables
                        
#                 mechanisms_data_perturb_loc = mechanisms_data.loc[mechanisms_data['EVID'] == mechanism_data['EVID'], :]
#                 if only_perturbations and add_perturbations:
#                         mechanisms_data_perturb_loc = mechanisms_data_perturb_loc.loc[mechanisms_data_perturb_loc['perturbation'], :]
#                 for imecha, mecha_perturb in mechanisms_data_perturb_loc.iterrows():
                        
#                         options_out['global_folder'] = name_sample.replace('XXX', str(imecha+1))
#                         os.system('cp -R ' + name_sample.replace('XXX', 'tocopy')[:-1] + ' ' + options_out['global_folder'])
#                         Green_RW.set_global_folder(options_out['global_folder'])
                
#                         mechanism = {}
#                         for key in keys_mechanism:
#                                 mechanism[key] = mecha_perturb[key]
                                
#                         # Station distribution
#                         mod_mechanisms.display_map_stations(mecha_perturb['EVID'], mecha_perturb['station_tab'], mecha_perturb['domain'], options_out['global_folder'])
                                
#                         # Save current moment tensor
#                         mod_mechanisms.save_mt(mechanism, options_out['global_folder'])
                        
#                         if 'station_tab' in mecha_perturb.keys(): 
#                                 station = mecha_perturb['station_tab']
                        
#                         if 'domain' in mecha_perturb.keys(): 
#                                 domain = mecha_perturb['domain']
                        
#                         param_atmos = velocity_models.generate_default_atmos()
                        
#                         compute_analytical_acoustic(Green_RW, mechanism, param_atmos, station, domain, options_out)
                        
#                         os.system('rm -f ' + options_out['global_folder'] + 'eigen.input_code_earthsr')
                        
#                 del station, domain, param_atmos, mechanism
#                 if use_individual_SCEC_models:
#                         del Green_RW # Clear variables
                
                