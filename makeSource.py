#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 08:44:14 2021

@author: lmartire
"""

from obspy.core.utcdatetime import UTCDateTime
from utils import pickleDump
# import mechanisms as mod_mechanisms
# import shutil
import argparse
import os
# import sys
# import RW_atmos
# import velocity_models

def main():
  parser = argparse.ArgumentParser(description='Computes Green functions with earthsr.')
  
  required = parser.add_argument_group('REQUIRED ARGUMENTS')
  required.add_argument('--output', required=True,
                      help='Output file path.')
  required.add_argument('--id', required=True, type=int,
                      help='Event internal ID.')
  required.add_argument('--time', required=True, type=int, nargs=6,
                      help='Event time (YYYY MM DD hh mm ss).')
  required.add_argument('--mag', required=True, type=float,
                      help='Event magnitude.')
  required.add_argument('--latlon', required=True, type=float, nargs=2,
                      help='Event location (LAT LON).')
  required.add_argument('--depth', required=True, type=float,
                      help='Event depth in [km].')
  required.add_argument('--strikeDipRake', required=True, type=float, nargs=3,
                      help='Event strike, dip, and rake (strike dip rake).')
  
  args = parser.parse_args()
  print(args)
  print(' ')
  
  outputPath = args.output
  # Create directory if it does not exist yet.
  if(not os.path.isdir(os.path.dirname(outputPath))):
      os.makedirs(os.path.dirname(outputPath))
  
  source_characteristics = { # example of custom source
      'id': args.id,
      'time': UTCDateTime(args.time[0],args.time[1],args.time[2],args.time[3],args.time[4],args.time[5]),
      'mag': args.mag,
      'lat': args.latlon[0],
      'lon': args.latlon[1],
      'depth': args.depth, #km
      'strike': args.strikeDipRake[0],
      'dip': args.strikeDipRake[1],
      'rake': args.strikeDipRake[2],
  }
    
  # Store output.
  pickleDump(outputPath, source_characteristics)

if __name__ == '__main__':
  main()