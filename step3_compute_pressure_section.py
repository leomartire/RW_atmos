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
import shutil

def main():
  parser = argparse.ArgumentParser(description='Computes Green functions with earthsr.')
  
  required = parser.add_argument_group('required arguments')
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
  
  parser.add_argument('--outputOverwrite', type=str2bool, choices=[True, False], default=True,
                      help='Overwrite output folder path? Defaults to True.')
  parser.add_argument('--doPlots', type=str2bool, choices=[True, False], default=False,
                      help='Do the plots? Defaults to False.')
  parser.add_argument('--doDumps', type=str2bool, choices=[True, False], default=True,
                      help='Do the dumps? Defaults to True.')
  
  args = parser.parse_args()
  print(args)
  
  # Sample path name of the directory created to store data and figures
  output_path               = args.output+'/'
  forceOverwrite            = args.outputOverwrite
  
  RW_field_path = args.RWField
  t_chosen = args.times
  altitudes = args.altitudes
  doPlots = args.doPlots
  doDumps = args.doDumps
  
  if((not doPlots) and (not doDumps)):
    sys.exit('[%s, ERROR] Either doPlots or doDumps has to be activated.' % (sys._getframe().f_code.co_name))
  
  # Load Rayleigh wave field.
  RW_field = pickleLoad(RW_field_path)
  
  # Check output path is free, make it if necessary.
  if(os.path.isdir(output_path)):
    if(forceOverwrite):
      shutil.rmtree(output_path)
      print('['+sys._getframe().f_code.co_name+'] Output files root folder \''+output_path+'\' existed and has been deleted, as required by script.')
    else:
      sys.exit('['+sys._getframe().f_code.co_name+'] Output files root folder \''+output_path+'\' exists, and script is not set to overwrite. Rename or delete it before running again.')
  os.makedirs(output_path)
  
  # Loop timesteps and altitudes.
  for t_snap in t_chosen:
    for alt in altitudes:
      # Compute atmospheric XY pressure fields.
      print('['+sys._getframe().f_code.co_name+'] Compute atmospheric XY pressure fields.')
      Mxy, Mz_t_tab = RW_field.compute_field_for_xz(t_snap, 0., 0., alt, None, 'xy', 'p')
      
      if(doPlots):
        fig = plt.figure()
        hplt = plt.imshow(np.flipud(np.real(Mxy).T), extent=[RW_field.y[0]/1000., RW_field.y[-1]/1000., RW_field.x[0]/1000., RW_field.x[-1]/1000.], aspect='auto')
        plt.xlabel('West-East [km]')
        plt.ylabel('South-North [km]')
        plt.title('Pressure Field at %.1f km' % (alt/1e3))
        cbar = plt.colorbar(hplt)
        plt.savefig(output_path+'map_XY_PRE_t%07.2f.pdf' % (t_snap))
      
      if(doDumps):
        # Save pressure section.
        np.real(Mxy).tofile(output_path+'map_XY_PRE_z%07.2f_t%07.2f_%dx%d.bin'
                            % (alt/1e3, t_snap, Mxy.shape[0], Mxy.shape[1]))
        
        # Save "grid" only once.
        if(t_snap==t_chosen[0] and alt==altitudes[0]):
          np.array([RW_field.x[0], RW_field.x[-1], RW_field.y[0], RW_field.y[-1]]).tofile(output_path+'map_XY_PRE_XYminmax.bin')

if __name__ == '__main__':
  main()