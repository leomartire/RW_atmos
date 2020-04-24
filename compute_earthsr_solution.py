import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pdb import set_trace as bp
import matplotlib
#matplotlib.use('tkAgg')
import matplotlib.patches as patches
import pickle
import json
import sys 
from scipy import fftpack
import scipy.integrate as spi
from scipy import signal

from matplotlib import rc
#rc('font', family='DejaVu Sans', serif='cm10')
#rc('text', usetex=True)
font = {'size': 14}
matplotlib.rc('font', **font)
from scipy import interpolate

def load(file_name, delimiter = ' '):

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

def load_dict(filename):

        afile = open(filename, 'rb')
        output = pickle.load(afile)
        afile.close()
        
        return output

class vertical_velocity():
        
        def __init__(self, period, r2, cphi, cg, I1, kn, directivity, angle_RW):
                
                self.period = period
                self.directivity = directivity
                self.r2   = r2
                self.cg   = cg
                self.cphi = cphi
                self.I1   = I1
                self.kn   = kn
                self.angle_RW = angle_RW

        def compute_veloc(self, r, phi, M, depth, unknown = 'd'):
        
                comp_deriv = np.pi*2.*1j/self.period if unknown == 'v' else 1.
                comp_deriv = (np.pi*2.*1j/self.period)*comp_deriv if unknown == 'a' else comp_deriv
                return comp_deriv*(self.r2/(8*self.cphi*self.cg*self.I1))*np.sqrt(2./(np.pi*self.kn*r))*np.exp( 1j*( self.kn*r + np.pi/4. ) )*self.directivity.compute_directivity(phi, M, depth)

        def compute_acoustic_spectrum(self, r, phi, M, depth, cpa, unknown = 'd'):
        
                perioda = (1./self.compute_veloc(r, phi, M, depth, unknown))*np.sin(self.angle_RW)*self.cphi/cpa
                return 1./perioda

class directivity():

        def __init__(self, dep, dr1dz_source, dr2dz_source, kn, r1_source, r2_source):
        
                self.dep = dep
                self.dr1dz_source = dr1dz_source
                self.dr2dz_source = dr2dz_source
                self.kn = kn
                self.r1_source = r1_source
                self.r2_source = r2_source
        
        def compute_directivity(self, phi, M, depth):
        
                idz = np.argmin( abs(self.dep - depth/1000.) )
                dr1dz_source = self.dr1dz_source[idz]
                dr2dz_source = self.dr2dz_source[idz]
                r1_source = self.r1_source[idz]
                r2_source = self.r2_source[idz]
                
                return self.kn*r1_source*( M[0]*np.cos(phi)**2 + (2.*M[3])*np.sin(phi)*np.cos(phi) + M[1]*np.sin(phi)**2 ) \
                                + 1j*dr1dz_source*(M[4]*np.cos(phi) + M[5]*np.sin(phi)) \
                                - 1j*self.kn*r2_source*(M[4]*np.cos(phi) + M[5]*np.sin(phi)) \
                                + dr2dz_source*M[2]

class RW_forcing():

        def __init__(self, mechanism, options):
        
                ## Inputs
                self.f_tab = options['f_tab']
                self.nb_modes = options['nb_modes'][1]
                
                self.global_folder = options['global_folder'] # Save folder path from Green's class
                
                ## Attributes containing seismic/acoustic spectra
                self.directivity = [ [ [] for aa in range(0, len(self.f_tab)) ] for bb in range(0, options['nb_modes'][1]) ]
                #self.uz_tab      = np.zeros((len(self.f_tab),))
                self.uz          = [ [ [] for aa in range(0, len(self.f_tab)) ] for bb in range(0, options['nb_modes'][1]) ]
                #self.freq_tab    = np.zeros((len(self.f_tab). options['nb_modes'][1]))
                #self.freqa_tab   = np.zeros((len(self.f_tab), options['nb_modes'][1]))
                self.angle_RW    = np.zeros((len(self.f_tab), options['nb_modes'][1]))
                
                ## Extract seismic model for later plots
                self.extract_seismic_parameters(options)
                
                ## Add source characteristics
                self.update_mechanism(mechanism)
                
                ## Medium
                self.update_cpa(0.34)

        def update_mechanism(self, mechanism):
        
                self.zsource = mechanism['zsource'] # m
                self.f0    = mechanism['f0']
                self.alpha = (np.pi*self.f0)**2
                self.M0    = mechanism['M0']
                self.M     = mechanism['M']*self.M0
                self.phi   = mechanism['phi']

        def update_cpa(self, cpa):
        
                self.cpa = cpa

        def source_spectrum(self, period):
        
                return self.M*np.sqrt(np.pi/self.alpha)*np.exp(-((np.pi/period)**2)/self.alpha)*np.exp(-2*np.pi*1j*(1.2/self.f0)/period)
        
        def add_one_period(self, period, iperiod, current_struct, rho, orig_b1, orig_b2, d_b1_dz, d_b2_dz, kmode, dep):
        
                #M = self.source_spectrum(period)
                
                ## Test
                #per_tab = np.linspace(0.01, 1000, 1000)
                #Mtab = [self.source_spectrum(per)[0] for per in per_tab]
                #Mtab = [ np.exp(-((np.pi/per)**2)/self.alpha) for per in per_tab ]
                #plt.figure()
                #plt.plot(per_tab, abs(np.array(Mtab)))
                #plt.show()
                #bp()
                
                uz    = []
                freqa = []
                for imode in range(0, min(len(current_struct),orig_b1.shape[1])):
                        idz_source = np.argmin( abs(dep-self.zsource/1000.) )
                        cphi = current_struct[imode]['cphi'][iperiod]
                        cg   = current_struct[imode]['cg'][iperiod]
                        r2   = orig_b2[:,imode]
                        r1   = orig_b1[:,imode]
                        kn   = kmode[:,imode]
                        d_r2_dz = d_b2_dz[:,imode]
                        d_r1_dz = d_b1_dz[:,imode]
                        
                        I1   = 0.5*spi.simps(rho[:]*( r1**2 + r2**2 ), dep[:])
                        
                        #r2_source = r2[idz_source]
                        #r1_source = r1[idz_source]
                        kn = kn[0]
                        self.directivity[imode][iperiod] = directivity(dep, d_r1_dz, d_r2_dz, kn, r1, r2)
                        
                        r2 = r2[0]
                        r1 = r1[0]
                        
                        #dr2dz_source = d_r2_dz[idz_source]
                        #dr1dz_source = d_r1_dz[idz_source]
                        
                        self.angle_RW[iperiod, imode]  = np.arctan(self.cpa/cphi)
                        self.uz[imode][iperiod]        = vertical_velocity(period, r2, cphi, cg, I1, kn, self.directivity[imode][iperiod], self.angle_RW[iperiod, imode])
                        
                #self.uz_tab[iperiod] = np.sum(uz[iperiod])

        def compute_RW_one_mode(self, imode, r, phi, type = 'RW', unknown = 'd'):
        
                ## Source depth
                depth = self.zsource
        
                uz_tab = []
                f_tab  = []
                #print('Compute mode: ', imode)
                for iuz in self.uz[imode]:
                     
                     if(iuz):
                             
                             M = self.source_spectrum(iuz.period)
                             f  = 1./iuz.period  
                             f_tab.append( f )
                             if(type == 'acoustic'):
                                uz = iuz.compute_acoustic_spectrum(r, phi, M, depth, self.cpa, unknown)
                             else:
                                uz = iuz.compute_veloc(r, phi, M, depth, unknown)
                             
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
                
                return response

        def extract_seismic_parameters(self, options):
        
                self.seismic = pd.read_csv( options['models'][options['chosen_model']], delim_whitespace=True, header=None )
                self.seismic.columns = ['z', 'rho', 'vp', 'vs', 'Qa', 'Qp']
                
        def concat_df_complex(self, A, B, groupby_lab):
        
                f        = A[groupby_lab].values
                mat_temp = B.drop([groupby_lab], axis=1).values
                mat      = A.drop([groupby_lab], axis=1).values
                mat[:mat_temp.shape[0], :] += mat_temp
                A = pd.DataFrame(mat)
                A.columns = np.arange(0, mat.shape[1])
                A[groupby_lab]    = f
                
                return A
                
        def response_RW_all_modes(self, r, phi, type = 'RW', unknown = 'd', mode_max = -1):
        
                mode_max = len(self.uz) if mode_max == -1 else mode_max
                for imode in range(0, mode_max):
                
                        #print('Computing mode', imode)
                
                        response_RW_temp = self.compute_RW_one_mode(imode, r, phi, type, unknown)
                        if(imode == 0):
                                response_RW = response_RW_temp.copy()
                        else:
                                #pd.merge(response_RW, response_RW_temp, on=['f']).set_index(['f']).sum(axis=1)
                                #print('Concatenate mode ', imode)

                                ## Concatenate dataframes with same freq.
                                ## we can not use pd.concat since it is too slow for complex numbers
                                response_RW = self.concat_df_complex(response_RW, response_RW_temp, 'f')
                                
                                #response_RW = pd.concat([response_RW, response_RW_temp], sort=False).groupby(['f']).sum().reset_index()
                                #print('Done concatenate mode ', imode)

                return response_RW                        

        def compute_ifft(self, r_in, phi_in, type, unknown = 'd', mode_max = -1):
        
                #print('Computing positive frequencies')
                RW     = self.response_RW_all_modes(r_in, phi_in, type, unknown, mode_max)
                RW     = RW.sort_values(by=['f'], ascending=True)
                
               #print('Add zero frequency')
                
                RW_first = RW.iloc[0:1].copy()
                temp     = pd.DataFrame(RW_first.values*0.)
                temp.columns = RW_first.columns
                RW_first     = temp.copy()
                #RW     = RW_first.append(RW)
                #RW_first.loc[:,'f'] = 0.*RW_first.loc[:,'f']
                
                #RW.loc[RW.index[0], RW.columns != 'f'] = 0.
                RW_neg = RW.iloc[:].copy()
                RW_neg.loc[:,'f']  = -RW_neg.loc[:,'f']
                
                RW = RW_first.append(RW.iloc[:-1])
                
                # Multiply given value by 2 and returns
                #def change_imag_real(x, ff, sign0, sign1):
                #        imag          = np.imag(x)
                #        indimag       = np.where(imag<0)
                #        imag[indimag] = np.conjugate(imag[indimag])   
                #        bp()  
                #        return sign0*np.sign(ff)*1j*imag + sign1*np.real(x)
                
                #print('Computing negative frequencies')
                
                ## Dataframe indexes
                #phi_idx = np.arange(0, phi.size)
                #RW_neg[phi_idx] = -np.real(RW_neg[phi_idx]) + np.imag(RW_neg[phi_idx])
                
                #RW_neg.loc[:, RW_neg.columns != 'f'] = RW_neg.loc[:, RW_neg.columns != 'f'].apply( lambda x: -np.real(x) + 1j*np.imag(x) )
                
                #RW_neg.loc[:, RW_neg.columns != 'f'] = RW_neg.loc[:, RW_neg.columns != 'f'].apply( change_imag_real, () )
                #KZnew=0.0-real(KZ).*sign(omega_intr)+1i*imag(KZ);
                
                RW_neg = RW_neg.sort_values(by=['f'], ascending=True)
                temp   = pd.DataFrame(np.real(RW_neg.iloc[:1].values))
                temp.columns = RW_neg.columns
                #RW_neg.loc[0, RW_neg.columns != 'f'] = 
                RW_neg = temp.append(RW_neg.drop([0]))
                
                #print('Make sure that pos and neg are conjugate of each other')
                
                temp      = pd.DataFrame(np.real(RW_neg.loc[:, RW_neg.columns != 'f']) + 1j*np.imag(RW_neg.loc[:, RW_neg.columns != 'f']))
                temp['f'] = RW_neg['f']
                RW_neg    = temp.copy()
                #RW_neg.loc[:, RW_neg.columns != 'f'] = np.real(RW_neg.loc[:, RW_neg.columns != 'f']) + np.imag(RW_neg.loc[:, RW_neg.columns != 'f'])
                
                temp      = pd.DataFrame(np.real(RW.loc[:, RW.columns != 'f']) - 1j*np.imag(RW.loc[:, RW.columns != 'f']))
                temp['f'] = RW['f'].values
                RW        = temp.copy()
                
                #print('Concatenate all')
                RW_tot = pd.concat([RW_neg,RW], ignore_index=True)
                
                #RW_tot.loc[:, RW_tot.columns != 'f'] = RW_tot.loc[:, RW_tot.columns != 'f'].apply( change_imag_real, args=(RW_tot['f'].values,1, -1) )
                #RW_tot.loc[:, RW_tot.columns != 'f'] = RW_tot.loc[:, RW_tot.columns != 'f'].apply( change_imag_real, args=(RW_tot['f'].values,1, -1) )
                
                #ifft_RW = fftpack.ifft(fftpack.fftshift(RW_tot.loc[:, RW_neg.columns != 'f'].values), axis=0)
                #ifft_RW = fftpack.ifft(fftpack.fftshift(RW_tot.loc[:, RW_neg.columns != 'f'].values), axis=0)
                #ifft_RW = fftpack.ifft(fftpack.fftshift(RW_tot.iloc[:,:-1].values), axis=0)
                #ifft_RW = fftpack.ifft(fftpack.fftshift(RW_tot.values[:,:-1]), axis=0)
                ifft_RW = fftpack.ifft(fftpack.fftshift(RW_tot.values[:,:-1], axes=0), axis=0)
                nb_fft  = ifft_RW.shape[0]//2
                
                #ifftsave = ifft_RW.copy()
                ifft_RW = ifft_RW[:nb_fft]
                
                df = abs(RW_neg['f'].iloc[1] - RW_neg['f'].iloc[0])
                
                dt = 1./(2.*abs(RW_neg['f']).max())
                t  = np.arange(0, dt*nb_fft, dt)    
                
                #plt.figure(); plt.plot(t, ifft_RW[:nb_fft]); plt.show()
                #plt.figure(); plt.plot(RW_tot['f'].values, np.real(RW_tot[0]), RW_tot['f'].values, np.imag(RW_tot[0])); plt.show()
                #ff  = RW_tot['f'].values
                #ind = np.where(ff < 0)
                #plt.figure(); plt.plot(ff, np.real(RW_tot[0].values)); plt.plot(-ff[ind], np.real(RW_tot[0].values)[ind], ':'); plt.show()
                #plt.figure(); plt.plot(fftpack.fftshift(ff), fftpack.fftshift(RW_tot.loc[:, RW_neg.columns != 'f'].values)); plt.show()
                #plt.figure(); plt.plot(fftpack.fftshift(ff), fftpack.fftshift(RW_tot.loc[:, RW_neg.columns != 'f'].values)); plt.show()
                
                return (t, ifft_RW)

class field_RW():

        default_loc = (30., 0.) # (km, degree)
        def __init__(self, Green_RW, nb_freq, dimension = 2, dx_in = 100., dy_in = 100., xbounds = [100., 100000.], ybounds = [100., 100000.], H = 1e10, Nsq = 1e-10, winds = [0., 0.], mode_max = -1):

                def nextpow2(x):
                        return np.ceil(np.log2(abs(x)))
        
                self.atmospheric_model_is_generated = False
                self.global_folder = Green_RW.global_folder # Save folder path from Green's class
        
                ##################################################
                ## Initial call to Green_RW to get the time vector
                output = Green_RW.compute_ifft(np.array([field_RW.default_loc[0]]), np.array([field_RW.default_loc[0]]), type='RW', unknown='v')
                t      = output[0]
                
                ## Store seismic model
                self.seismic = Green_RW.seismic
                
                ########################################
                ## Define time/spatial domain boundaries
                mult_tSpan, mult_xSpan, mult_ySpan = 1, 1, 1
                dt_anal, dx_anal, dy_anal = abs(t[1] - t[0]), dx_in, dy_in
                xmin, xmax = xbounds[0], xbounds[1]
                if(dimension > 2):
                        ymin, ymax = ybounds[0], ybounds[1]
                
                NFFT1 = int(2**nextpow2((xmax-xmin)/dx_anal)*mult_xSpan)
                NFFT2 = len(t)
                if(dimension > 2):
                        NFFT3 = int(2**nextpow2((ymax-ymin)/dy_anal)*mult_ySpan)
                
                x  = dx_anal * np.arange(0,NFFT1)
                x -= x[-1]/2.

                t = dt_anal * np.arange(0,NFFT2)
                if(dimension > 2):
                        y  = dy_anal * np.arange(0,NFFT3) 
                        y -= y[-1]/2.
                else:
                        y = np.array([0.])   
                   
                ############################        
                ## Frequency/Wavenumber mesh
                omega = 2.0*np.pi*(1.0/(dt_anal*NFFT2))*np.concatenate((np.arange(0,NFFT2/2), -np.arange(NFFT2/2,0,-1)))
                kx =    2.0*np.pi*(1.0/(dx_anal*NFFT1))*np.concatenate((np.arange(0,NFFT1/2), -np.arange(NFFT1/2,0,-1)))
                if(dimension > 2):
                        ky = 2.0*np.pi*(1.0/(dy_anal*NFFT3))*np.concatenate((np.arange(0,NFFT3/2), -np.arange(NFFT3/2,0,-1)))
                
                if(dimension > 2):
                        KX, Omega, KY = np.meshgrid(kx, omega, ky)
                else:
                        KX, Omega = np.meshgrid(kx, omega)
                
                #Nsqtab    = 0.*Nsq + 0.0*Omega;
                #onestab = 0.0*Nsqtab + 1.0;
                
                #####################
                ## Compute RW forcing
                Mo  = np.zeros(Omega.shape, dtype=complex)
                
                ###################################################################
                ## Conversion of cartesian coordinates into cylindrical coordinates
                if(dimension > 2):
                        X, Y   = np.meshgrid(y, x)
                        R      = np.sqrt( X**2 + Y**2 )
                        ind_where_yp0 = np.where(Y>0)
                        PHI = X*0.
                        PHI[ind_where_yp0] = np.arccos( X[ind_where_yp0]/R[ind_where_yp0] )
                        ind_where_yp0 = np.where(Y<0)
                        PHI[ind_where_yp0] = -np.arccos( X[ind_where_yp0]/R[ind_where_yp0] )
                else:
                        #X, Y   = np.meshgrid(y, x)
                        R   = abs(x)
                        #PHI = y[0]+R*0.
                        PHI = y[0]+R*0.
                        PHI[:len(x)//2] = np.pi
                                
                #########################
                ## Compute bottom forcing                
                temp   = Green_RW.compute_ifft(R/1000., PHI, type='RW', unknown='v', mode_max = mode_max)
                if(dimension > 2):
                        t, Mo  = temp[0], temp[1].reshape( (temp[1].shape[0], PHI.shape[0], PHI.shape[1]) )
                else:
                        t, Mo  = temp[0], temp[1].reshape( (temp[1].shape[0], PHI.size) )
                
                ## Taper signal
                if(dimension > 2):
                        self.Mo = Mo*np.tile(signal.tukey(len(t)), len(x)*len(y)).reshape(Mo.shape)
                else:
                        self.Mo = Mo*np.tile(signal.tukey(len(t)), len(x)).reshape(Mo.shape)
                
                self.TFMo  = fftpack.fftn(self.Mo)
                self.Omega = Omega
                self.KX = -KX
                if(dimension > 2):
                        self.KY = -KY

                ##############################
                ## Compute vertical wavenumber
                #self.compute_vertical_wavenumber(TFMo, H, Nsq, winds)

                self.dimension = dimension

                ########################
                ## Store mesh parameters                
                self.x    = x
                self.y    = y
                self.t    = t
           
        def plot_atmosphere_and_seismic(self):
        
                import matplotlib.colors as mcolors
        
                if(not self.atmospheric_model_is_generated):
                        print('Generate atmospheric model first')
                        return
                
                nb_cols = 3
                fig, axs = plt.subplots(nrows=2, ncols=nb_cols)
                
                #seismic = self.seismic

                colors = [icolor for icolor in mcolors.TABLEAU_COLORS]
                
                iax     = 0
                iax_row = 1
                
                z  = self.seismic['z'].values/1000.
                zi = np.linspace(z[0], z[-1], 10000)
                
                f = interpolate.interp1d(z, self.seismic['rho'].values/1000., kind='previous')
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
                f = interpolate.interp1d(z, self.seismic['vp'].values/1000., kind='previous')
                unknown = f(zi)
                axs[iax_row, iax].plot(unknown, zi, color=colors[iax+iax_row*nb_cols])
                axs[iax_row, iax].grid()
                axs[iax_row, iax].set_xlim([unknown.min(), unknown.max()])
                axs[iax_row, iax].tick_params(axis='both', which='both', labelleft=False)
                axs[iax_row, iax].set_yscale('log')
                axs[iax_row, iax].invert_yaxis()
                axs[iax_row, iax].set_xlabel('$v_p$ (km/s)')
                
                iax += 1
                f = interpolate.interp1d(z, self.seismic['vs'].values/1000., kind='previous')
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
                rho = self.rho
                cpa = self.cpa
                wx  = self.winds[0]
                wy  = self.winds[1]
                if(self.isothermal):
                        z = np.linspace(0, 50, 4)
                        rho = self.rho[0]+z*0
                        cpa = self.cpa[0]+z*0
                        wx  = self.winds[0][0]+z*0
                        wy  = self.winds[1][0]+z*0
                else:
                        z = self.z/1000.
                        
                iax     = 0
                iax_row = 0
                unknown = self.rho.copy()/1000.
                axs[iax_row, iax].plot(unknown, z, color=colors[iax+iax_row*nb_cols])
                axs[iax_row, iax].grid()
                axs[iax_row, iax].set_xlim([unknown.min(), unknown.max()])
                axs[iax_row, iax].set_ylim([z.min(), z.max()])
                axs[iax_row, iax].set_ylabel('Altitude (km)')
                axs[iax_row, iax].text(-0.8, 0.5, 'Atmosphere', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', edgecolor='black', pad=2.0), transform=axs[iax_row, iax].transAxes, rotation=90)
                axs[iax_row, iax].set_title('Density (g/cm$^3$)')
                axs[iax_row, iax].set_xscale('log')
                
                iax += 1
                unknown = self.cpa.copy()/1000.
                axs[iax_row, iax].plot(unknown, z, color=colors[iax+iax_row*nb_cols])
                axs[iax_row, iax].grid()
                axs[iax_row, iax].set_xlim([unknown.min(), unknown.max()])
                axs[iax_row, iax].set_ylim([z.min(), z.max()])
                axs[iax_row, iax].tick_params(axis='both', which='both', labelleft=False)
                axs[iax_row, iax].set_title('$c_p$ (km/s)')
                
                
                iax += 1
                unknown = self.winds[0].copy()/1000.
                axs[iax_row, iax].plot(unknown, z, color=colors[iax+iax_row*nb_cols])
                axs[iax_row, iax].set_xlim([unknown.min(), unknown.max()])
                axs[iax_row, iax].set_ylim([z.min(), z.max()])
                if(self.dimension > 2):
                        axs[iax_row, iax].plot(self.winds[1], self.z)
                        axs[iax_row, iax].set_xlim([min(unknown.min(), self.winds[1].min()), max(unknown.max(), self.winds[1].max())])
                axs[iax_row, iax].grid()
                axs[iax_row, iax].tick_params(axis='both', which='both', labelleft=False)
                axs[iax_row, iax].set_title('winds (km/s)')
                
                fig.subplots_adjust(hspace=0.3, right=0.95, left=0.2, top=0.9, bottom=0.15)
                
                if(not options['GOOGLE_COLAB']):
                        plt.savefig(self.global_folder + 'seismic_and_atmos_profiles.png')
                
        def generate_atmospheric_model(self, param_atmos):
        
                self.atmospheric_model_is_generated = True
        
                self.isothermal = param_atmos['isothermal']
                if(self.isothermal):
                        self.H = [param_atmos['H']]
                        self.cpa = [param_atmos['cpa']]
                        self.Nsq = [param_atmos['Nsq']]
                        self.winds = []
                        self.winds.append( [param_atmos['wind_x']] )
                        self.winds.append( [param_atmos['wind_y']] )
                        self.bulk  = [param_atmos['bulk']]
                        self.shear = [param_atmos['shear']]
                        self.kappa = [param_atmos['kappa']]
                        self.gamma = [param_atmos['gamma']]
                        self.rho   = [param_atmos['rho']]
                        self.cp    = [param_atmos['cp']]
                else:
                        #sys.exit('Has to be implemented!')
                        temp   = pd.read_csv( param_atmos['file'], delim_whitespace=True, header=None )
                        temp.columns = ['z', 'rho', 'dummy1', 'cpa', 'p', 'dummy2', 'g', 'dummy3', 'kappa', 'mu', 'dummy4', 'dummy5', 'dummy6', 'wx', 'cp', 'cv', 'gamma']
                        self.z = temp['z'].values
                        self.H     = -(np.roll(self.z, 1) - self.z)/np.log( np.roll(temp['p'].values, 1)/temp['p'].values )
                        self.H[1] = self.H[0]
                        self.H[-1] = self.H[-2]
                        self.Nsq   = np.sqrt(-(temp['g'].values/temp['rho'].values[0])*np.gradient(temp['rho'].values, self.z, edge_order=2))**2
                        self.Nsq[0] = self.Nsq[1]
                        #plt.figure(); plt.plot(self.Nsq, self.z); plt.show()
                        self.winds = []
                        self.winds.append( temp['wx'].values )
                        self.winds.append( temp['wx'].values )
                        self.cpa = temp['cpa'].values
                        self.rho   = temp['rho'].values
                        self.bulk  = temp['mu'].values
                        self.shear = temp['mu'].values
                        self.kappa = temp['kappa'].values
                        self.cp    = temp['cp'].values
                        self.gamma = temp['gamma'].values
                                
                                
        def compute_vertical_wavenumber(self, id_layer):        
                
                ## Get corresponding atmospheric parameters
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
                
                ## Compute intrinsic frequencies
                Omega_intrinsic = self.Omega - wind_x*self.KX
                if(self.dimension > 2):
                        Omega_intrinsic -= wind_y*self.KY
                
                #KZ = np.zeros(Omega.shape, dtype=complex)
                if(self.dimension > 2):
                        KZ = np.lib.scimath.sqrt(  -self.KX**2 -self.KY**2 + (self.KX**2 + self.KY**2) * Nsq/(Omega_intrinsic**2) -1./(4.*H**2) + (1.+1j*(bulk+(4./3.)*shear+kappa*(gamma-1.)/cp)*Omega_intrinsic/(2.*rho*cpa**2))*(Omega_intrinsic / cpa )**2 )
                else:
                        KZ = np.lib.scimath.sqrt( -self.KX**2 + (self.KX**2) * Nsq/(Omega_intrinsic**2) -1./(4.*H**2) + (1.+1j*(bulk+(4./3.)*shear+kappa*(gamma-1.)/cp)*Omega_intrinsic/(2.*rho*cpa**2))*(Omega_intrinsic / cpa )**2 )
                
                ## Remove infinite/nan numbers that correspond to zero frequencies
                KZ   = np.nan_to_num(KZ, 0.)
                
                indimag     = np.where(np.imag(KZ)<0)
                KZ[indimag] = np.conjugate(KZ[indimag])
                # real(KZ) should be positive for positive frequencies and negative for
                # negative frequencies in order to shift signal in positive times
                # restore the sign of KZ depending on Omega-wi*KX
                #     KZnew=real(KZ).*sign((Omega-wind_x*KX)).*sign(KX)+1i*imag(KZ);
                # !!! Why KZ should have a sign opposite to Omega for GW NOT UNDERSTOOD !!!
                # => because vg perpendicular to Vphi ?
                
                KZ = 0.0 - np.real(KZ)*np.sign(Omega_intrinsic) + 1j*np.imag(KZ)
                #KZ = 0.0 + np.real(KZ) + 1j*np.imag(KZ)*np.sign(Omega_intrinsic)
                #KZ = 0.0 + np.real(KZ) + 1j*np.imag(KZ)*np.sign(self.KX)
                #KZ(:,NFFT2/2:-1:2)=0.0-real(KZ(:,NFFT2/2+2:NFFT2))+1i*imag(KZ(:,NFFT2/2+2:NFFT2));
                
                #bp()
                
                
                #ind_m0 = np.where(Omega<0.)
                #ind_p0 = np.where(self.Omega>=0.)
                #KZ[ ind_p0 ] = np.real(KZ[ ind_p0 ]) - 1j*np.imag(KZ[ ind_p0 ])
                
                return KZ
        
        def compute_response_at_given_z(self, z1, z0, TFMo, KZ_in = [], last_layer_in = -1):
        
                ## Find all layers for which we have to compute the wavenumbers 
                def find_id_layers_and_heights(z0, z1, zlayer):
                
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
                        
                        ## Last element    
                        if not id_first_layer == id_last_layer:
                                h = z1-zlayer[id_last_layer] 
                        else:
                                h = z1-z0
                        
                        if(h > 0):
                                h_layers.append( h )
                                id_layers.append( id_last_layer )
                        
                        return h_layers, id_layers  
                
                KZ         = KZ_in
                last_layer = last_layer_in
                ## We only compute the solution if we are not at the surface or below 
                if(z1 > 0):
                        
                        ## If isothermal model
                        if(self.isothermal):
                                ## Compute the vertical response from the ground forcing
                                if(not KZ):
                                        KZ = self.compute_vertical_wavenumber(0)       
                                field_at_it = np.exp(z1/(2*self.H[0])) * fftpack.ifftn( np.exp(1j*(KZ*z1)) * self.TFMo)
                        
                        ## If layered model
                        else:
                        
                                h_layers, id_layers = find_id_layers_and_heights(z0, z1, self.z)
                                for idx, id_layer in enumerate(id_layers):
                        
                                        ## Compute the vertical response from the forcing of the layer beneath (idz-1)
                                        if(not last_layer == id_layer):
                                                KZ = self.compute_vertical_wavenumber(id_layer)       
                                        TFMo *= np.exp(h_layers[idx]/(2*self.H[id_layer])) * np.exp(1j*(KZ*h_layers[idx])) 
                                
                                field_at_it = fftpack.ifftn( TFMo )
                                        
                                last_layer = id_layer        
                                        
                else:
                        field_at_it = fftpack.ifftn(TFMo)
                
                #if(z0 == 0. and z1 > 10000.):
                #        plt.figure(); plt.imshow(np.real(field_at_it), extent=[self.x[0], self.x[-1], self.t[-1], self.t[0]], aspect='auto'); plt.gca().axvline(20000., color='r'); plt.show()
                #        bp()
                
                return field_at_it, last_layer, KZ, TFMo
                
        ## Compute wavefield for a given physical domain
        def compute_field_for_xz(self, t, x, y, z, zvect, type_slice):
                
                ## Build a response matrix based on required slice dimensions
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
                        
                ## Get required time index        
                it = np.argmin( abs(self.t - t) )
                
                ## setup progress bar
                cptbar        = 0
                toolbar_width = 40
                total_length  = len(zvect)
                sys.stdout.write("Building wavefield: [%s]" % (" " * toolbar_width))
                sys.stdout.flush()
                sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
                
                ## Load initial surface forcing
                TFMo    = self.TFMo.copy()
                
                iz_prev = 0.
                last_layer_prev = -1
                KZ_prev = []
                ## Loop over all layers
                for idz, iz in enumerate(zvect):
                
                        field_at_it, last_layer, KZ, TFMo = self.compute_response_at_given_z(iz, iz_prev, TFMo, KZ_prev, last_layer_prev)
                        ## Store wavenumber if a new wavenumber has been computed
                        if(last_layer > -1):
                                last_layer_prev = last_layer
                                KZ_prev = KZ.copy()
                        iz_prev = iz
                        
                        ## 3d
                        if(self.dimension > 2):
                        
                                if(type_slice == 'z'):
                                
                                        iy = np.argmin( abs(self.y - y) )
                                        Mz_xz[idz, :] = field_at_it[it,:,iy]
                                        
                                        ix = np.argmin( abs(self.x - x) )
                                        Mz_yz[idz, :] = field_at_it[it,ix,:]
                                        
                                elif(type_slice == 'xy'):
                                        Mz_xy[:, :] = field_at_it[it,:,:]
                                        
                        ## 2d
                        else:
                                Mz_xz[idz, :] = field_at_it[it,:]

                        ## update the bar
                        if(int(toolbar_width*idz/total_length) > cptbar):
                                cptbar = int(toolbar_width*idz/total_length)
                                sys.stdout.write("-")
                                sys.stdout.flush()
                
                sys.stdout.write("] Done\n")
                
                if(self.dimension > 2 and type_slice == 'z'):
                        return Mz_xz, Mz_yz
                elif(self.dimension > 2 and type_slice == 'xy'):
                        return Mz_xy
                else:
                        return Mz_xz
                        
        def compute_field_timeseries(self, x, y, z):
        
                ix   = np.argmin( abs(self.x - x) )
                #filt = np.exp(1j*(self.KZ*z))
                field_at_it, last_layer, KZ, TFMo = self.compute_response_at_given_z(z, 0., self.TFMo)
                
                # 3d
                if(self.dimension > 2):
                        iy = np.argmin( abs(self.y - y) )
                        #Mz = np.exp(z/(2*self.H)) * fftpack.ifftn(filt*self.TFMo)[:, ix, iy]
                        Mz = field_at_it[:, ix, iy]
                # 2d
                else:
                        #Mz = np.exp(z/(2*self.H)) * fftpack.ifftn(filt*self.TFMo)[:, ix]
                        Mz = field_at_it[:, ix]
                
                return Mz

def generate_default_atmos():
        
        param_atmos = {}
        param_atmos['isothermal'] = False
        param_atmos['file'] = './atmospheric_model.dat'
        param_atmos['cpa']    = 3.4e2 # m/s
        param_atmos['H']      = 7.e3 # m
        param_atmos['Nsq']    = 1e-10
        param_atmos['wind_x'] = 0.
        param_atmos['wind_y'] = 0.
        
        param_atmos['bulk']   = 0.
        param_atmos['shear']  = 0.
        param_atmos['kappa']  = 0.
        param_atmos['gamma']  = 1.4
        param_atmos['rho']    = 1.2
        param_atmos['cp']     = 3000.
        
        return param_atmos

def generate_default_mechanism():

        mechanism = {}
        mechanism['zsource'] = 3400 # m
        mechanism['f0'] = 0.2
        mechanism['M0'] = 1e0
        mechanism['M']  = np.zeros((6,))
        mechanism['M'][0]  = -1.82631379e+13 # Mxx 2.5
        #mechanism['M'][0]  = 453334337148. # Mxx
        #mechanism['M'][0]  = -2.83550741e+16 # Mxx 3.1
        mechanism['M'][1]  = 1.82743497e+13 # Myy 2.5
        mechanism['M'][2]  = -1.12117778e+10 # Mzz 2.5
        #mechanism['M'][2]  = 6.36275923e+16 # Mzz 3.1
        mechanism['M'][3]  = -4.53334337e+11 # Mxy
        mechanism['M'][4]  = 2.73046441e+17 # Mxz 2.5
        #mechanism['M'][4]  = -1.64624203e+16 # Mxz 3.1
        #mechanism['M'][4]  = 0. # Mxz
        mechanism['M'][5]  = 2.21151605e+14 # Myz
        mechanism['M'] /= 1.e15 # Convert N.m = m^2.kg/s^2 to right unit (everything is in km and g/cm^3)
        mechanism['phi']   = 0.
        
        return mechanism

def compute_analytical_acoustic(Green_RW, mechanism, param_atmos, station, domain, options):

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        ## Wavefield dimensions
        dimension = options['dimension']

        ## Update mechanism if needed
        if(not mechanism):
                mechanism = generate_default_mechanism()
        
        Green_RW.update_mechanism(mechanism)

        ## Class to generate field for given x/z t/z combinaison
        nb_freq  = options['nb_freq']
        H     = 7000.
        Nsq   = 1e-4
        winds = [0., 0.]
        mode_max = -1
        if(not domain):
                xbounds    = [-110000., 110000.]
                ybounds    = [-110000., 110000.]
                dx, dy, dz = 600., 2000., 200.
                z         = np.arange(0, 50000., dz)
        else:
                xbounds = [domain['xmin'], domain['xmax']]
                ybounds = [domain['ymin'], domain['ymax']]
                dx, dy, dz  = domain['dx'], domain['dy'], domain['dz']
                z         = np.arange(domain['zmin'], domain['zmax'], dz)
        
        field = field_RW(Green_RW, nb_freq, dimension, dx, dy, xbounds, ybounds, H, Nsq, winds, mode_max)
        
        ## Create atmospheric profiles
        if(not param_atmos):
                param_atmos = generate_default_atmos()
        field.generate_atmospheric_model(param_atmos)
        
        ## Plot profiles
        field.plot_atmosphere_and_seismic()
        
        ## Compute solutions for a given range of altitudes (m) at a given instant (s)
        if(not station):
                iz = 20000.
                iy = 20000.
                ix = 20000.
                t_station = 250.
                type_slice = 'xz'
        else:
                iz = station['zs']
                iy = station['ys']
                ix = station['xs']
                t_station = station['t_chosen']
                type_slice = station['type_slice']
                
        ## Compute solutions for a given range of altitudes (m) at a given instant (s)
        if(dimension > 2):
                Mxz, Myz = field.compute_field_for_xz(t_station, ix, iy, iz, z, 'z')
                Mxy      = field.compute_field_for_xz(t_station, ix, iy, iz, z, 'xy')
                nb_cols  = 2
        else:
                Mxz = field.compute_field_for_xz(t_station, ix, iy, iz, z, 'z') 
                nb_cols = 1
               
        ## COmpute time series at a given location
        Mz_t = field.compute_field_timeseries(ix, iy, iz)
        
        ## Display
        fig, axs = plt.subplots(nrows=2, ncols=nb_cols)
        
        if(dimension > 2):
        
                iax = 0
                iax_col = 0
                axs[iax, iax_col].plot(field.t, np.real(Mz_t))
                axs[iax, iax_col].grid(True)
                axs[iax, iax_col].set_xlim([field.t[0], field.t[-1]])
                axs[iax, iax_col].set_xlabel('Time (s)')
                axs[iax, iax_col].set_ylabel('Velocity (m/s)')
                
                iax_col += 1
                
                plotMyz = axs[iax, iax_col].imshow(np.flip(np.real(Myz), axis=0), extent=[field.y[0]/1000., field.y[-1]/1000., z[0]/1000., z[-1]/1000.], aspect='auto')
                axs[iax, iax_col].scatter(iy/1000., iz/1000., color='red', zorder=2)
                #axs[iax, iax_col].set_xlabel('Distance from source - South (km)')
                axs[iax, iax_col].set_ylabel('Altitude (km)')
                axs[iax, iax_col].text(0.5, 0.1, 't = ' + str(t_station) + 's', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', edgecolor='black', pad=2.0), transform=axs[iax, iax_col].transAxes)
                axs[iax, iax_col].yaxis.set_label_position("right")
                
                axins = inset_axes(axs[iax, iax_col], width="5%", height="100%", loc='lower left', bbox_to_anchor=(1.2, 0., 1, 1.), bbox_transform=axs[iax, iax_col].transAxes, borderpad=0)
                axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
                
                plt.colorbar(plotMyz, cax=axins)
                
                vmin, vmax = np.real(Myz).min(), np.real(Myz).max()
                
                iax += 1
                iax_col = 0
                
                plotMxz = axs[iax, iax_col].imshow(np.flip(np.real(Mxz), axis=0), extent=[field.x[0]/1000., field.x[-1]/1000., z[0]/1000., z[-1]/1000.], aspect='auto', vmin=vmin, vmax=vmax)
                axs[iax, iax_col].scatter(ix/1000., iz/1000., color='red', zorder=2)
                axs[iax, iax_col].set_xlabel('West - East (km)')
                axs[iax, iax_col].set_ylabel('Altitude (km)')
                #axs[iax, iax_col].text(0.15, 0.9, 't = ' + str(t_station) + 's', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', edgecolor='black', pad=2.0), transform=axs[iax, iax_col].transAxes)
                
                iax_col += 1
                
                plotMxy = axs[iax, iax_col].imshow(np.real(Mxy), extent=[field.y[0]/1000., field.y[-1]/1000., field.x[0]/1000., field.x[-1]/1000.], aspect='auto', vmin=vmin, vmax=vmax)
                axs[iax, iax_col].scatter(ix/1000., iy/1000., color='red', zorder=2)
                axs[iax, iax_col].set_xlabel('West - East (km)')
                axs[iax, iax_col].set_ylabel('North - South (km)')
                axs[iax, iax_col].yaxis.set_label_position("right")
                axs[iax, iax_col].yaxis.tick_right()
                #axs[iax, iax_col].text(0.15, 0.9, 't = ' + str(t_station) + 's', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', edgecolor='black', pad=2.0), transform=axs[iax, iax_col].transAxes)
        
        else:
        
                iax = 0
                axs[iax].plot(field.t, np.real(Mz_t))
                axs[iax].grid(True)
                axs[iax].set_xlim([field.t[0], field.t[-1]])
                axs[iax].set_xlabel('Time (s)')
                axs[iax].set_ylabel('Velocity (m/s)')
                
                iax += 1
                plotMxz = axs[iax].imshow(np.flip(np.real(Mxz), axis=0), extent=[field.x[0]/1000., field.x[-1]/1000., z[0]/1000., z[-1]/1000.], aspect='auto')
                axs[iax].scatter(ix/1000., iz/1000., color='red', zorder=2)
                axs[iax].set_xlabel('Distance from source (km)')
                axs[iax].set_ylabel('Altitude (km)')
                axs[iax].text(0.15, 0.9, 't = ' + str(t_station) + 's', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', edgecolor='black', pad=2.0), transform=axs[iax].transAxes)
                
                axins = inset_axes(axs[iax], width="2.5%", height="100%", loc='lower left', bbox_to_anchor=(1.02, 0., 1, 1.), bbox_transform=axs[iax].transAxes, borderpad=0)
                axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
                
                plt.colorbar(plotMxz, cax=axins)
        
        fig.subplots_adjust(hspace=0.3, right=0.8, left=0.2, top=0.94, bottom=0.15)
        
        if(not options['GOOGLE_COLAB']):
                plt.savefig(options['global_folder'] + 'map_wavefield_vz.png')

        ## Save waveform        
        df = pd.DataFrame()
        df['t']  = field.t
        df['vz'] = np.real(Mz_t)
        df.to_csv(options['global_folder'] + 'waveform.csv', index=False)
        print('save waveform to: '+options['global_folder'] + 'waveform.csv')
        
        if(not options['GOOGLE_COLAB']):
                bp()
        
## Generate velocity and option files to run earthsr
def generate_model_for_earthsr(side, options):

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

## Collect eigenfunctions and derivatives from earthsr
def get_eigenfunctions(current_struct, mechanism, options):

        import read_earth_io as reo
        
        ## Define a default mechanism
        if(not mechanism):
                mechanism = generate_default_mechanism()
        
        ## Construct RW spectrum object 
        Green_RW = RW_forcing(mechanism, options)

        #options['f_tab'] = options['f_tab']
        periods = 1./np.linspace(options['f_tab'][-1], options['f_tab'][0], len(options['f_tab']))
        
        # setup toolbar
        cptbar        = 0
        toolbar_width = 40
        total_length  = len(periods)
        sys.stdout.write("Building eigenfunctions: [%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        
        uz_tab = []
        freq_tab  = [[] for ii in range(0,options['nb_modes'][1]+1) ]
        freqa_tab = [[] for ii in range(0,options['nb_modes'][1]+1) ]
        for iperiod, period in enumerate(periods):
        
                reoobj=reo.read_egnfile_per(options['global_folder'] + 'eigen.input_code_earthsr', period)
                
                dep     = reoobj.dep
                omega   = 2*np.pi/period
                orig_b1 = reoobj.uzmat
                orig_b2 = reoobj.urmat
                orig_b3 = reoobj.tzmat
                orig_b4 = reoobj.trmat
                origdep = reoobj.dep
                nmodes  = orig_b1.shape[1]
                matint  = np.zeros((nmodes,nmodes)) # matrix of integrals
                norms   = np.zeros(nmodes) # vector of integrals
                kmode   = reoobj.wavnum.reshape(1,len(reoobj.wavnum))
                mu      = reoobj.mu.reshape(len(reoobj.mu),1)
                lamda   = reoobj.lamda.reshape(len(reoobj.mu),1)
                rho     = reoobj.rho
                kmu    = np.dot(mu,kmode)
                klamda = np.dot(lamda,kmode)
                # Eq. (7.28) Aki-Richards
                # r1 = b2 r2 = b1
                # r3 = b4 r4 = b3
                # Extra omega factor
                d_b2_dz = (omega*orig_b4-np.multiply(kmu,orig_b1))/mu # numpy.multiply does element wise array multiplication
                d_b1_dz = (np.multiply(klamda,orig_b2)+omega*orig_b3)/(lamda+2*mu)
                dxz     = np.gradient(orig_b2[:,0])
                dzz     = np.gradient(orig_b1[:,0])
                
                ## Construct Green's function for a given period 
                Green_RW.add_one_period(period, iperiod, current_struct, rho, orig_b1, orig_b2, d_b1_dz, d_b2_dz, kmode, dep)
                
                #print('Finish reading period ' + str(period))
                # update the bar
                if(int(toolbar_width*iperiod/total_length) > cptbar):
                        cptbar = int(toolbar_width*iperiod/total_length)
                        sys.stdout.write("-")
                        #print(cptbar, iperiod)
                        sys.stdout.flush()
                
        sys.stdout.write("] Done\n")
                
        return Green_RW
                
def compute_dispersion_with_earthsr(no, side, options):

        ## Launch dispersion code
        print(' model: ' + side['name'])
        os.system('./earthsr ' + 'input_code_earthsr')

def move_dispersion_files(no, options):

        os.system('mv ' + 'disp* ' + options['global_folder'])
        os.system('mv ' + 'eigen* ' + options['global_folder'])
        if(no > 0):
                os.system('mv ' + 'tocomputeIO* ' + options['global_folder'])

################################################################################################################
## Move files in running/working directory "running_dir" to the appropriate frequency-domain-dependent subfolder 
def move_files_after_freq_domain(freq_domain, options):

        os.system('mv ' + options['global_folder'] + '/* ' + options['global_folder'] + '_' + str(freq_domain) )
        os.system('rm -rf ' + options['global_folder'])    

############################################
## Save purely 1d coefficients to .mat files  
def save_A1D_coef(AMP1D, AMP1Dst, options):

        save_dict(AMP1D, options['global_folder'] + '/AMP1D.mat')
        if(AMP1Dst):
                save_dict(AMP1Dst, options['global_folder'] + '/AMP1Dst.mat')

#################################################################################
## combine trans./reflec. coefficients from two frequency ranges in the same file
def move_trans_and_reflec(imode1, imode2, subfolder, options):

        str_mode = str(imode1) + str(imode2)
        list_names = ['trans', 'horiz_trans', 'reflec']
        
        no = 1
        for iname in list_names:
        
                file1 = options['global_folder'][2:] + '_' + str(no) + '/' + subfolder + '/data_' + iname + str_mode + '.out'
                file2 = options['global_folder'][2:] + '_' + str(no-1) + '/' + subfolder + '/data_' + iname + str_mode + '.out'
                file_output = options['global_folder'][2:] + '/' + subfolder + '/data_' + iname + str_mode + '.out'
                
                if(os.path.isfile(file1) and os.path.isfile(file2)): 
                        os.system('cat ' + file1 + ' ' + file2 + ' >> ' + file_output)
                elif(os.path.isfile(file1)): 
                        os.system('cat ' + file1 + ' >> ' + file_output)
                elif(os.path.isfile(file2)): 
                        os.system('cat ' + file2 + ' >> ' + file_output)       
                else:
                        continue
                        
###############################################################################
## Move all forward and backward trans./reflec. coefficients to the same folder                        
def save_transmission_coefs(options):

        for way_forward in range(0,2):
    
                subfolder = options['name_simu_subfolder'] + '_way' + str(way_forward+1)
                os.makedirs( options['global_folder'] + '/' + subfolder )

                ## Concatenate files from different frequency domains
                for imode1 in range(0, options['nb_modes'][1]):
                        for imode2 in range(0, options['nb_modes'][1]):
                        
                                ## If files exist, move them accordingly, otherwise do nothing
                                move_trans_and_reflec(imode1, imode2, subfolder, options)

################################################################
## Choose the name of the temporary folder to store coefficients
def determine_folders(options):

        options_loc = {}

        ## Check current folder
        pattern = 'coefs_batch'
        nbdirs = [int(f.split('_')[-1]) for f in os.listdir(options['dir_earthsr']) if pattern in f and not os.path.isfile(os.path.join(options['dir_earthsr'], f))]
        if(nbdirs):
                nbdirs = max(nbdirs)
        else:
                nbdirs = 0
                
        name_simu_folder = './coefs_batch_' + str(nbdirs+1) + '/'
        
        if(options['PLOT'] < 2):
                os.makedirs(name_simu_folder)
                
        options_loc['name_simu_subfolder'] = ''
        options_loc['global_folder']       = name_simu_folder + options_loc['name_simu_subfolder']
        
        return options_loc

################################################################################################
## Before finishing building coefficients, this routine saves dispersion characteristics to file
def collect_dispersion_from_earthsr_and_save(nside, options):

        data_dispersion_file_fund   = load(options['global_folder'] + '/disp_vconly.input_code_earthsr')

        data_dispersion = [{} for i in range(0, options['nb_modes'][1])]
        list_modes_side = [{} for j in range(0, options['nb_modes'][1])]
        for nmode in range(0, options['nb_modes'][1]):
            
                list_modes_side[nmode]['loc']  = np.where(data_dispersion_file_fund[:,0] == nmode)

                freq_domain = 0
                if(list_modes_side[nmode]['loc'][0].size > 0):
                        data_dispersion[nmode]['period'] = data_dispersion_file_fund[list_modes_side[nmode]['loc'][0],1]
                        data_dispersion[nmode]['cphi']   = data_dispersion_file_fund[list_modes_side[nmode]['loc'][0],2]
                        data_dispersion[nmode]['cg']     = data_dispersion_file_fund[list_modes_side[nmode]['loc'][0],3]   

                ## Add nan for periods where 1st mode has not been calculated
                if(nmode > 0 and list_modes_side[nmode]['loc'][0].size > 0):

                        cpt       = len(data_dispersion[nmode]['period'])-1
                        save_cphi = data_dispersion[nmode]['cphi'][-1]
                        save_cg   = data_dispersion[nmode]['cg'][-1]
                        while data_dispersion[nmode]['period'][-1] < data_dispersion[0]['period'][-1]:
                                cpt += 1
                                data_dispersion[nmode]['period'] = np.concatenate([data_dispersion[nmode]['period'], [data_dispersion[0]['period'][cpt]]])
                                data_dispersion[nmode]['cphi']   = np.concatenate([data_dispersion[nmode]['cphi'], [save_cphi]])
                                data_dispersion[nmode]['cg']     = np.concatenate([data_dispersion[nmode]['cg'], [save_cg]])

        ## Save with name "current_struct" to be consistent with resonance_eigen
        current_struct = data_dispersion
        for nmode in range(0, len(current_struct)):
                if( len(current_struct[nmode]) > 0 ):
                        current_struct[nmode]['fks'] = 1./current_struct[nmode]['period']
                
        save_dict(current_struct, options['global_folder'] + '/PARAM_dispersion.mat')
        
        return current_struct

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
                                
                        f    = interpolate.interp1d(options['z'], temp, kind='previous')
                        temp_interm = f(z_interp_interm)/1000.
                        f    = interpolate.interp1d(z_interp_interm, temp_interm, kind='previous')
                        
                        data_interp[imodel][iunknown] = f(z_interp)
                        
        
        return z_interp, data_interp

#################################
## Routine to read SPECFEM models
def read_specfem_files(options):

        unknown_tab = ['rho', 'vs', 'vp']
        id_tab      = [1, 3, 2]

        data = {}
        zover0 = []
        for imodel in options['models']:
                data[imodel] = {}
                #for unknown in unknown_tab:
                temp   = pd.read_csv( options['models'][imodel], delim_whitespace=True, header=None )
                temp.columns = ['z', 'rho', 'vp', 'vs', 'Qa', 'Qp']
                
                if(temp['z'].iloc[0] > 0):
                        temp_add = temp.loc[ temp['z'] == temp['z'].min() ]
                        temp_add['z'].iloc[0] = 0.
                        temp = pd.concat([temp_add, temp]).reset_index()
                
                zover0 = temp[ 'z' ].values
                cpt_unknown = -1
                for unknown in unknown_tab:
                        cpt_unknown += 1
                        data[imodel][unknown] = temp[ unknown ].values
                        
        
        return zover0, data

###############################################################
## Create adapted velocity model on both sides of the interface 
def create_velocity_model(options):

        ## Definition
        side = {}
        unknown_tab  = ['rho', 'vs', 'vp']
        
        ## Jack's files
        options['z'] = np.arange(0.,100000,2800)
        
        if(options['type_model'] == 'specfem'):
                zover0, data = read_specfem_files(options)
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
        ## No attenuation because we only want eigenfunctions        
        side['Qa'] = side['vs']*0 + 9999. # P-wave Q
        side['Qb'] = side['vs']*0 + 9999. # S-wave Q
        
        return side

def create_velocity_figures(current_struct, options):

        nbmodes = len(current_struct)
        fig, axs = plt.subplots(nrows=nbmodes, ncols=1, sharex=True, sharey=True)
        
        axs[-1].set_ylabel('Velocity (km/s)')
        axs[-1].set_xlabel('Frequency (Hz)')
        fks = current_struct[0]['fks']
        axs[-1].set_xlim([fks.min(), fks.max()])
        for imode in range(0, nbmodes):
                axs[imode].plot(current_struct[imode]['fks'], current_struct[imode]['cphi'], label='$c_\Phi$')
                axs[imode].plot(current_struct[imode]['fks'], current_struct[imode]['cg'], label='$c_g$', linestyle='--')
                axs[imode].grid()
                axs[imode].text(0.5, 1., 'Mode '+str(imode), horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', edgecolor='black', pad=4.0), transform=axs[imode].transAxes)
                axs[imode].set_xscale('log')
        
        axs[0].legend()
        
        if(not options['GOOGLE_COLAB']):
                plt.savefig(options['global_folder'] + 'cphi.png')

########################################
########################################
##                                    ##
##           MAIN PROGRAM             ##
##                                    ##  
########################################
########################################
        
def compute_trans_coefficients(options_in = {}):        
        
        options = {} 
        options['GOOGLE_COLAB'] = False      
        
        ##########
        ## Options
        options['dimension']   = 2
        
        options['nb_modes']    = [0, 3] # min / max
        options['type_wave']   = 1 # Surface wave type.  (1 = Rayleigh; >1 = Love.)
        options['way_forward'] = 1
        options['LOAD_2D_MODEL'] = False
        options['type_model']    = 'specfem'
        options['nb_layers']     = 500#2800
        options['nb_freq']       = 128 # Number of frequencies
        options['chosen_header'] = 'coefs_earthsr_sol_'
        options['PLOT']          = 1# 0 = No plot; 1 = plot after computing coef.; 2 = plot without computing coef.
        options['PLOT_folder']   = 'coefs_python_1.2_vs0.5_poisson0.25_h1.0_running_dir_1'
        #options['PLOT_folder']   = 'coefs_python_0.0_17500.0_running_dir_1'
        options['ONLY_purely_1d'] = False

        ## Hetergeneous structure
        options['models'] = {}
        options['models']['specfem'] = '/media/quentin/Samsung_T5/SSD_500GB/Documents/Ridgecrest/simulations/Ridgecrest_mesh_simu_fine_test_Jennifer_1/Ridgecrest_seismic.txt'
        options['models']['specfem'] = '/home/quentin/Desktop/Ridgecrest_seismic.txt'
        options['chosen_model'] = 'specfem'
        options['zmax'] = 50000.

        ##############
        ## Auxiliaries
        A1D   = {}
        A1Dst = {}
        options['dir_earthsr']   = '/home/quentin/Documents/DATA/CODES/earthsr_quentin'
        options['earth_flattening'] = 0 # Earth flattening control variable (0 = no correction; >0 applies correction)
        options['ref_period']  = 10. # Reference period for dispersion correction (0 => none) Generally you would just pick a period shorter than anything you are going to model
        options['output_file'] = 'dispers' # Filename of binary output of dispersion curves.
        options['min_max_phase'] = [0, 0] # min and max phase velocities and min and max branch (mode) numbers. Note that if we choose the min and max phase velocities to be 0, the program will choose the phase velocity range itself.  In this case case we ask the program to figure out the appropriate range (0.0000000       0.0000000) and solve modes 0 (fundamental) to 4.
        options['nb_source'] = 1 # Number of sources
        options['source_depth']   = 6.8 # (km)
        options['receiver_depth'] = 0 # (km)
        options['coef_low_freq']  = 0.001
        options['coef_high_freq'] = 0.3 # 1.85
        options['Loop']           = 0 # This this point the program loops over another set of input lines starting with the surface wave type (1st line after model).  If this is set to zero, the program will terminate.
        
        ## Update each option based on user input
        options.update( options_in )
        
        f_tab = np.linspace(options['coef_low_freq'], options['coef_high_freq'], options['nb_freq'])
        options['f_tab']   = f_tab
        #options['nb_freq'] = len(f_tab)
        options['df']      = abs( f_tab[1] - f_tab[0] )
        options['freq_range'] = [f_tab[0], f_tab[-1]]
        
        Green_RW = []
        if(options['PLOT'] < 2):
        
                ###########################################
                ## Build right frequency and spatial ranges
                options_loc = determine_folders(options)
                options.update( options_loc )
        
                ##############################
                ## Loop over frequency domains
                freq_domain = 0

                ## Determine adapted model depth for this frequency regime
                #options_loc = get_depth_model(freq_domain, options)
                #options.update( options_loc )
                
                ## Create directory for earthsr eigenfunctions
                #os.makedirs(options['global_folder'])
                
                ## TODO: Creation side vs models
                side = create_velocity_model(options)
                
                ## Create file to use earthsr
                generate_model_for_earthsr(side, options)
                
                ## Compute and store dispersion characteristics using earthsr
                no = 0
                compute_dispersion_with_earthsr(no, side, options)
                
                ## Compute purely1d coefficients
                #AMP1D_temp, AMP1Dst_temp, data_I1_temp = store_purely1d_coefficients(side, no, options)
                #AMP1D.update( copy.deepcopy(AMP1D_temp) )
                #AMP1Dst.update( copy.deepcopy(AMP1Dst_temp) )
                #data_I1[no] = copy.deepcopy(data_I1_temp)
                move_dispersion_files(no, options)
                
                current_struct = collect_dispersion_from_earthsr_and_save(0, options)
                
                ## Create velocity figures
                create_velocity_figures(current_struct, options)
                
                ## Class containing routine to construct RW/acoustic spectrum at a given location
                mechanism = {}
                Green_RW = get_eigenfunctions(current_struct, mechanism, options)
                
        return Green_RW, options

if __name__ == '__main__':        

        Green_RW, options_out = compute_trans_coefficients()
        
        mechanism, station, domain = {}, {}, {}
        
        mechanism = generate_default_mechanism()
        
        param_atmos = generate_default_atmos()
        
        compute_analytical_acoustic(Green_RW, mechanism, param_atmos, station, domain, options_out)
