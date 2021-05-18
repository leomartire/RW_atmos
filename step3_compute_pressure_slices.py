#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:01:06 2021

@author: lmartire
"""

import sys
import argparse
import matplotlib.pyplot as plt
from utils import str2bool, pickleLoad
import numpy as np
import os
# import shutil
import RW_atmos
import time
from datetime import datetime, timedelta

def main():
  parser = argparse.ArgumentParser(description='Computes Green functions with earthsr.')
  
  required = parser.add_argument_group('REQUIRED ARGUMENTS')
  required.add_argument('--output', required=True,
                      help='Output folder path.')
  required.add_argument('--RWField', required=True,
                      help='Path to RW field''s object Pickle.')
  required.add_argument('--altitudes', required=True,
                      type=float, nargs='+', default=[5.0e3],
                      help='List of altitudes in [m]. Defaults to [5.0e3].')
  required.add_argument('--times', required=True,
                      type=float, nargs='+', default=[5.0],
                      help='List of times in [s]. Defaults to [5.0].')
  
  misc = parser.add_argument_group('optional arguments - miscellaneous')
  misc.add_argument('--doPlots', type=str2bool, choices=[True, False], default=False,
                      help='Do the plots? Defaults to False.')
  misc.add_argument('--doDumps', type=str2bool, choices=[True, False], default=True,
                      help='Do the dumps? Defaults to True.')
  
  print('+------------------------------------------+')
  print("| JOB START TIME = %s   |" % (datetime.now().strftime("%Y/%m/%d @ %H:%M:%S")))
  print('+------------------------------------------+')
  t1 = time.time()
  
  args = parser.parse_args()
  print(args)
  print(' ')
  
  # Sample path name of the directory created to store data and figures
  output_path               = args.output+'/'
  
  RW_field_path = args.RWField
  t_chosen = np.array(args.times)
  altitudes = args.altitudes
  doPlots = args.doPlots
  doDumps = args.doDumps
  
  if((not doPlots) and (not doDumps)):
    sys.exit('[%s, ERROR] Either doPlots or doDumps has to be activated.' % (sys._getframe().f_code.co_name))
  
  # Load Rayleigh wave field.
  RW_field = pickleLoad(RW_field_path)
  
  # Check input.
  if(t_chosen.max() > RW_field.t.max()):
    fm = 0.5/(RW_field.t[1]-RW_field.t[0])
    sys.exit('[%s, ERROR] Asking for a time (%.3f s) that is later than the maximum time of simulation (%.3f s). To get to %.3f s, return to step 1, and either increase --nbFreq to %d or decrease fMax (in --freqMinMax) to %.3e Hz.'
             % (sys._getframe().f_code.co_name, t_chosen.max(), RW_field.t.max(),
                t_chosen.max(), int(np.ceil(t_chosen.max()*2.0*fm+1)), 0.5*(RW_field.t.size-1)/t_chosen.max()))
  
  # Make output path if necessary.
  if(not os.path.isdir(output_path)):
    os.makedirs(output_path)
  
  # Loop timesteps and altitudes.
  for t_snap in t_chosen:
    
    # Rayleigh wave forcing and grid.
    if(doPlots):
      RW_atmos.plot_surface_forcing(RW_field, t_snap, 0, 0, output_path, False)
    if(doDumps):
      if(RW_field.dimension==3):
        M0 = RW_field.Mo[RW_field.get_index_tabs_time(t_snap), :, :]
        tag = 'XY'
        np.real(M0).tofile(output_path+'map_%s_RW_t%07.2f_%dx%d.bin'
                           % (tag, t_snap, M0.shape[0], M0.shape[1]))
      else:
        # Note: this call would work in 3D too, but it is more explicit to separate the two in order to keep in mind the shapes of the involved fields.
        M0 = RW_field.Mo[RW_field.get_index_tabs_time(t_snap), :]
        tag = 'X'
        np.real(M0).tofile(output_path+'map_%s_RW_t%07.2f_%dx1.bin'
                           % (tag, t_snap, M0.shape[0]))
      if(np.all(np.isnan(M0))):
        # Safeguard.
        sys.exit('[%s, ERROR] All values in the Rayleigh wave forcing are NaNs. Something went terribly wrong.' % (sys._getframe().f_code.co_name))
      # Save "grid" only once.
      if(t_snap==t_chosen[0]):
        np.array([RW_field.x[0], RW_field.x[-1], RW_field.y[0], RW_field.y[-1]]).tofile(output_path+'map_%s_PRE_XYminmax.bin' % (tag))
    
    # Horizontal pressure slices.
    for alt in altitudes:
      print('['+sys._getframe().f_code.co_name+'] Compute atmospheric XY pressure slice at altitude = %f.' % (alt))
      Mxy, Mz_t_tab = RW_field.compute_field_for_xz(t_snap, 0., 0., alt, None, 'xy', 'p')
      
      if(doPlots):
        plt.figure()
        hplt = plt.imshow(np.flipud(np.real(Mxy).T), extent=[RW_field.y[0]/1000., RW_field.y[-1]/1000., RW_field.x[0]/1000., RW_field.x[-1]/1000.], aspect='auto')
        plt.xlabel('West-East [km]')
        plt.ylabel('South-North [km]')
        plt.title('Pressure Field at %.1f km' % (alt/1e3))
        cbar = plt.colorbar(hplt)
        cbar.ax.set_ylabel('$p''$ [Pa]', rotation=90) 
        plt.savefig(output_path+'map_%s_PRE_t%07.2f.pdf' % (tag, t_snap))
      if(doDumps):
        np.real(Mxy).tofile(output_path+'map_%s_PRE_t%07.2f_%dx%d_z%07.2f.bin'
                            % (tag, t_snap, Mxy.shape[0], Mxy.shape[1], alt/1e3))
  
  dur = time.time()-t1
  print('+------------------------------------------+')
  print("| JOB   END TIME = %s   |" % (datetime.now().strftime("%Y/%m/%d @ %H:%M:%S")))
  print('+------------------------------------------+')
  print("| JOB   DURATION = %.5e s (%s) |" % (dur, str(timedelta(seconds=round(dur)))))
  print('+------------------------------------------+')
        
if __name__ == '__main__':
  main()