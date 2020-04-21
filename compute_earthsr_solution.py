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

from matplotlib import rc
#rc('font', family='DejaVu Sans', serif='cm10')
#rc('text', usetex=True)
font = {'size': 17}
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
                
                ## Attributes containing seismic/acoustic spectra
                self.directivity = [ [ [] for aa in range(0, len(self.f_tab)) ] for bb in range(0, options['nb_modes'][1]) ]
                #self.uz_tab      = np.zeros((len(self.f_tab),))
                self.uz          = [ [ [] for aa in range(0, len(self.f_tab)) ] for bb in range(0, options['nb_modes'][1]) ]
                #self.freq_tab    = np.zeros((len(self.f_tab). options['nb_modes'][1]))
                #self.freqa_tab   = np.zeros((len(self.f_tab), options['nb_modes'][1]))
                self.angle_RW    = np.zeros((len(self.f_tab), options['nb_modes'][1]))
                
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
        
                import scipy.integrate as spi
        
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

        def compute_RW_one_mode(self, imode, r, type = 'RW', unknown = 'd'):
        
                ## Source depth
                depth = self.zsource
        
                uz_tab = []
                f_tab  = []
                for iuz in self.uz[imode]:
                     
                     if(iuz):
                             
                             M = self.source_spectrum(iuz.period)
                             f  = 1./iuz.period  
                             f_tab.append( f )
                             if(type == 'acoustic'):
                                uz = iuz.compute_acoustic_spectrum(r, self.phi, M, depth, self.cpa, unknown)
                             else:
                                uz = iuz.compute_veloc(r, self.phi, M, depth, unknown)
                             uz_tab.append( uz )
                     
                response = pd.DataFrame()
                response['f']  = np.array(f_tab) 
                response['uz'] = np.array(uz_tab) 
                
                return response
                
        def response_RW_all_modes(self, r, type = 'RW', unknown = 'd', mode_max = -1):
        
                mode_max = len(self.uz) if mode_max == -1 else mode_max
                for imode in range(0, mode_max):
                
                        response_RW_temp = self.compute_RW_one_mode(imode, r, type, unknown)
                        if(imode == 0):
                                response_RW = response_RW_temp.copy()
                        else:
                                #pd.merge(response_RW, response_RW_temp, on=['f']).set_index(['f']).sum(axis=1)
                                response_RW = pd.concat([response_RW, response_RW_temp]).groupby(['f']).sum().reset_index()

                return response_RW                        

        def compute_ifft(self, r, type, unknown = 'd', mode_max = -1):
        
                from scipy import fftpack
        
                rin   = r
        
                RW  = self.response_RW_all_modes(r, type, unknown, mode_max)
                RW_neg = RW.copy()
                RW_neg['f']  = -RW_neg['f']
                RW_neg['uz'] = -np.real(RW_neg['uz']) + np.imag(RW_neg['uz'])
                RW_neg      = RW_neg.sort_values(by=['f'])
                RW_tot      = pd.concat([RW,RW_neg], ignore_index=True)

                ## Invert back to time domain                
                ifft_RW = fftpack.ifft(RW_tot['uz'].values)
                nb_fft  = len(ifft_RW)//2
                
                ifftsave = ifft_RW.copy()
                ifft_RW = ifft_RW[:nb_fft]
                
                df = abs(RW_neg['f'].iloc[1] - RW_neg['f'].iloc[0])
                
                dt = 1./(2.*RW['f'].max())
                t  = np.arange(0, dt*nb_fft, dt)    
                
                #plt.figure(); plt.plot(t, ifft_RW); plt.show()
                #plt.figure(); plt.plot(t, ifftsave[:nb_fft]); plt.show()
                #bp()
                
                return t, ifft_RW

class field_RW():

        def __init__(self, Green_RW, nb_freq, dx_in = 100., dy_in = 100., xbounds = [100., 100000.], H = 1e10, Nsq = 1e-4, winds = [0., 0.], mode_max = -1):

                def nextpow2(x):
                        return np.ceil(np.log2(abs(x)))
        
                ##################################################
                ## Initial call to Green_RW to get the time vector
                t, RW_t = Green_RW.compute_ifft(1., type='RW', unknown='v')

                ########################################
                ## Define time/spatial domain boundaries
                mult_tSpan, mult_xSpan, mult_ySpan = 1, 1, 1
                dt_anal, dx_anal, dy_anal = abs(t[1] - t[0]), dx_in, dy_in
                xmin, xmax = xbounds[0], xbounds[1]
                #ymin, ymax = 100., 50000.
                
                NFFT2 = len(t)
                #NFFT3 = int(2**nextpow2((ymax-ymin)/dy_anal)*mult_ySpan)
                NFFT1 = int(2**nextpow2((xmax-xmin)/dx_anal)*mult_xSpan)
                
                #k = np.zeros((NFFT1,NFFT2,NFFT3))
                k = np.zeros((NFFT1,NFFT2))
                x = dx_anal * np.arange(0,NFFT1) + xmin
                t = dt_anal * np.arange(0,NFFT2)
                
                #y = dy_anal * np.arange(0,NFFT3) + ymin
                #X, Y, T = np.meshgrid(x, y, t)
                
                omega = 2.0*np.pi*(1.0/(dt_anal*NFFT2))*np.concatenate((np.arange(0,NFFT2/2+1), -np.arange(NFFT2/2-1,0,-1)))
                kx =    2.0*np.pi*(1.0/(dx_anal*NFFT1))*np.concatenate((np.arange(0,NFFT1/2+1), -np.arange(NFFT1/2-1,0,-1)))
                #ky =    2.0*np.pi*(1.0/(dy     *NFFT3))*[np.arange(0:1:NFFT3/2] [0.0-[NFFT3/2-1:-1:1]]];
                #KX, KY, Omega = np.meshgrid(kx, ky, omega);
                
                ## Mesh
                X, T      = np.meshgrid(x, t)
                KX, Omega = np.meshgrid(kx, omega);
                Nsqtab    = 0.*Nsq + 0.0*Omega;
                #onestab = 0.0*Nsqtab + 1.0;
                
                #####################
                ## Compute RW forcing
                Mo  = np.zeros(X.shape, dtype=complex)
                
                # setup toolbar
                cptbar        = 0
                toolbar_width = 40
                total_length  = len(x)
                sys.stdout.write("Building wavenumbers: [%s]" % (" " * toolbar_width))
                sys.stdout.flush()
                sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
                for idx, ix in enumerate(x):
                        t, RW_t   = Green_RW.compute_ifft(abs(ix)/1000., type='RW', unknown='v', mode_max = mode_max)
                        Mo[:,idx] = RW_t.copy()
                        
                        # update the bar
                        if(int(toolbar_width*idx/total_length) > cptbar):
                                cptbar = int(toolbar_width*idx/total_length)
                                sys.stdout.write("-")
                                sys.stdout.flush()
                
                sys.stdout.write("] Done\n")
                
                TFMo = fftpack.fftn(Mo)
                
                ## Loop over all layers
                
                
                self.wind_x = winds[0]
                self.wind_y = winds[1]
                Omega_intrinsic = Omega - self.wind_x*KX
                #Omega_intrinsic = Omega - wind_x*KX - wind_y*KY;
                
                KZ = np.sqrt(   (KX**2) * (Nsqtab/(Omega_intrinsic**2) - 1) + (Omega_intrinsic / (Green_RW.cpa*1000.) )**2 )
                #KZ = sqrt(   (KX.^2+KY.^2) .* (Nsqtab./(Omega_intrinsic.^2) - 1) \
                #   + (Omega_intrinsic ./ (Green_RW.cpa*1000.) )**2 );
                
                #indimag = find(imag(KZ)<0);
                #KZ(indimag) = conj(KZ(indimag));
                #ind2=find(isinf(KZ));
                KZ   = np.nan_to_num(KZ, 0.)
                
                indimag = np.where(np.imag(KZ)<0)
                KZ[indimag] = np.conjugate(KZ[indimag])
                # real(KZ) should be positive for positive frequencies and negative for
                # negative frequencies in order to shift signal in positive times
                # restore the sign of KZ depending on Omega-wi*KX
                #     KZnew=real(KZ).*sign((Omega-wind_x*KX)).*sign(KX)+1i*imag(KZ);
                # !!! Why KZ should have a sign opposite to Omega for GW NOT UNDERSTOOD !!!
                # => because vg perpendicular to Vphi ?
                KZ = 0.0 - np.real(KZ)*np.sign(Omega_intrinsic) + 1j*np.imag(KZ)
                
                ## Store wavenumbers
                self.KZ   = KZ
                self.TFMo = TFMo
                self.H    = H
                self.x    = x
                self.t    = t
                
                
        def compute_field_for_xz(self, t, z):
                
                Mz = np.zeros((len(z), len(self.x)), dtype=complex)
                it = np.argmin( abs(self.t - t) )
                
                # setup toolbar
                cptbar        = 0
                toolbar_width = 40
                total_length  = len(z)
                sys.stdout.write("Building wavefield: [%s]" % (" " * toolbar_width))
                sys.stdout.flush()
                sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
                for idz, iz in enumerate(z):
                        filt      = np.exp(1j*(self.KZ*iz))
                        
                        #if(type == 'p'):
                        #        temp = np.exp(iz/(2*self.H)) * filt*self.TFMo
                        #        p    = 1j*( -1*self.wind_x*1j*temp[:,:] )
                        #else:
                        Mz[idz, :] = np.exp(iz/(2*self.H)) * fftpack.ifftn(filt*self.TFMo)[it,:]

                        # update the bar
                        if(int(toolbar_width*idz/total_length) > cptbar):
                                cptbar = int(toolbar_width*idz/total_length)
                                sys.stdout.write("-")
                                sys.stdout.flush()
                
                sys.stdout.write("] Done\n")
                
                return Mz
                #ix = round((x_station-xmin)/dx_anal) + 1 

        def compute_field_timeseries(self, x, z):
        
                ix   = np.argmin( abs(self.x - x) )
                filt = np.exp(1j*(self.KZ*z))
                Mz   = np.exp(z/(2*self.H)) * fftpack.ifftn(filt*self.TFMo)[:,ix]
                
                return Mz

def compute_analytical_acoustic(Green_RW, mechanism, station, domain, options):

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        ## Update mechanism if needed
        if(not mechanism):
                mechanism = {}
                mechanism['zsource'] = 6800 # m
                mechanism['f0'] = 0.2
                mechanism['M0'] = 1e0
                mechanism['M']  = np.zeros((6,))
                #mechanism['M'][0]  = -1826313793918.2844 # Mxx 2.5
                #mechanism['M'][0]  = 453334337148. # Mxx
                mechanism['M'][0]  = -2.83550741e+16 # Mxx 3.1
                mechanism['M'][1]  = 0. # Myy
                #mechanism['M'][2]  = 453334337148. # Mzz 2.5
                mechanism['M'][2]  = 6.36275923e+16 # Mzz 3.1
                mechanism['M'][3]  = 0. # Mxy
                #mechanism['M'][4]  = -11211777000. # Mxz 2.5
                mechanism['M'][4]  = -1.64624203e+16 # Mxz 3.1
                #mechanism['M'][4]  = 0. # Mxz
                mechanism['M'][5]  = 0. # Myz
                mechanism['M'] /= 1.e15 # Convert N.m = m^2.kg/s^2 to right unit (everything is in km and g/cm^3)
                mechanism['phi']   = 0.
        
        Green_RW.update_mechanism(mechanism)

        ## Class to generate field for given x/z t/z combinaison
        nb_freq  = options['nb_freq']
        H     = 7000.
        Nsq   = 1e-4
        winds = [0., 0.]
        mode_max = -1
        if(not domain):
                xbounds    = [-110000., 110000.]
                dx, dy, dz = 600., 600., 200.
                z         = np.arange(0, 35000., dz)
        else:
                xbounds = [domain['xmin'], domain['xmax']]
                dx, dy  = domain['dx'], domain['dy']
                z         = np.arange(domain['zmin'], domain['zmax'], domain['dz'])
        
        field = field_RW(Green_RW, nb_freq, dx, dy, xbounds, H, Nsq, winds, mode_max)
        
        ## Compute solutions for a given range of altitudes (m) at a given instant (s)
        if(not station):
                iz = 25000.
                ix = 100000.
                t_station = 80.
        else:
                iz = station['zs']
                ix = station['xs']
                t_station = station['t_chosen']
                
        ## Compute solutions for a given range of altitudes (m) at a given instant (s)
        Mz   = field.compute_field_for_xz(t_station, z)
                
        ## COmpute time series at a given location
        Mz_t = field.compute_field_timeseries(ix, iz)
        
        ## Display
        fig, axs = plt.subplots(nrows=2, ncols=1)
        
        axs[0].plot(field.t, np.real(Mz_t))
        axs[0].grid(True)
        axs[0].set_xlim([field.t[0], field.t[-1]])
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Velocity (m/s)')
        
        plotMz = axs[1].imshow(np.flip(np.real(Mz), axis=0), extent=[field.x[0]/1000., field.x[-1]/1000., z[0]/1000., z[-1]/1000.], aspect='auto')
        axs[1].scatter(ix/1000., iz/1000., color='red', zorder=2)
        axs[1].set_xlabel('Distance from source (km)')
        axs[1].set_ylabel('Altitude (km)')
        axs[1].text(0.15, 0.9, 't = ' + str(t_station) + 's', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='w', edgecolor='black', pad=2.0), transform=axs[1].transAxes)
        
        axins = inset_axes(axs[1], width="2.5%", height="80%", loc='lower left', bbox_to_anchor=(1.02, 0.1, 1, 1.), bbox_transform=axs[1].transAxes, borderpad=0)
        axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
        
        plt.colorbar(plotMz, cax=axins)
        
        fig.subplots_adjust(hspace=0.28, right=0.8, left=0.2, top=0.98, bottom=0.15)
        
        if(not options['GOOGLE_COLAB']):
                plt.show()

        ## Save waveform        
        df = pd.DataFrame()
        df['t']  = field.t
        df['vz'] = np.real(Mz_t)
        df.to_csv(options['global_folder'] + 'waveform.csv', index=False)
        print('save waveform to: '+options['global_folder'] + 'waveform.csv')
        
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
                mechanism = {}
                mechanism['zsource'] = 6800 # m
                mechanism['f0'] = 0.4
                mechanism['M0'] = 1e0
                mechanism['M']  = np.zeros((6,))
                mechanism['M'][0]  = -1826313793918.2844 # Mxx
                mechanism['M'][1]  = 0. # Myy
                mechanism['M'][2]  = 453334337148. # Mzz
                mechanism['M'][3]  = 0. # Mxy
                mechanism['M'][4]  = -11211777000. # Mxz
                mechanism['M'][5]  = 0. # Myz
                mechanism['M'] /= 1.e15 # Convert N.m = m^2.kg/s^2 to right unit (everything is in km and g/cm^3)
                mechanism['phi']   = 0.
                mechanism['cpa']   = 0.340
        
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
                
        #plt.figure()
        #plt.plot(side['vs'], options['z'])
        #plt.show()
        #bp()
                        
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
        options['coef_high_freq'] = 0.6 # 1.85
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
        
        mechanism = {}
        mechanism['zsource'] = 35800 # m
        mechanism['f0'] = 0.1
        mechanism['M0'] = 1e0
        mechanism['M']  = np.zeros((6,))
        mechanism['M'][0]  = -1826313793918.2844 # Mxx
        mechanism['M'][1]  = 0. # Myy
        mechanism['M'][2]  = 453334337148. # Mzz
        mechanism['M'][3]  = 0. # Mxy
        mechanism['M'][4]  = -11211777000. # Mxz
        mechanism['M'][5]  = 0. # Myz
        mechanism['M'] /= 1.e15 # Convert N.m = m^2.kg/s^2 to right unit (everything is in km and g/cm^3)
        mechanism['phi']   = 0.
        mechanism['cpa']   = 0.340
        
        compute_analytical_acoustic(Green_RW, mechanism, station, domain, options_out)
