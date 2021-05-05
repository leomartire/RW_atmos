clear all;
close all;
clc;

% Step 1 computes and saves the Green functions.
%        INP: one ground model
%        OUT: one Green functions file
% Step 2 computes and saves a Rayleigh wave field.
%        INP: one Green functions file
%             a set of source mechanisms (currently only supports one)
%        OUT: one Rayleigh wave field file
% Step 3 computes pressure slices and Rayleigh wave ground motion.
%        INP: one Rayleigh wave field file
%             a set of altitudes and times
%        OUT: pressure slices at every (altitude, time)
%             Rayleigh wave ground motion at every time

% Parameters.
python = ['/usr/local/bin/python3'];
rwatmosroot = ['/Users/lmartire/Documents/software/rw_atmos_leo/'];
s1p = [rwatmosroot, 'step1_compute_green.py'];
s2p = [rwatmosroot, 'step2_compute_field.py'];
s2p_source = [rwatmosroot, 'makeSource.py'];
s3p = [rwatmosroot, 'step3_compute_pressure_slices.py'];

% Move to this folder.
thisFolder = regexprep(mfilename('fullpath'),mfilename,'');
cd(thisFolder);

% Input.
dryrun = 0; verbose = 0;
seismicModel = '/Users/lmartire/Documents/software/rw_atmos_leo/models/Ridgecrest_seismic.txt';
zmax = 30e3; nlay = zmax/100; fminmax = [0.005, 5]; nfreq = 2^8; nmodes = [0, 20];
f0 = 2;
imech = 1; sources(imech) = struct('id',imech,'time',[2020,7,1,12,0,0],'mag',4,'latlon',[30,-90],'depth',8,'strikeDipRake',[159,89,-156]);
imech = 2; sources(imech) = struct('id',imech,'time',[2020,7,1,14,0,0],'mag',5,'latlon',[30.5,-90.5],'depth',7,'strikeDipRake',[155,45,-6]);
latminlatmaxlonminlonmax = [28, 32, -88, -92]; nkxky = ceil(max(range(latminlatmaxlonminlonmax(1:2)), range(latminlatmaxlonminlonmax(3:4))) * 111/50);
slice_alts = [15]*1e3; slice_tims = [20]; slice_doPlots = 0; slice_doDumps = 1;
rootOutput = ['/Users/lmartire/Documents/software/rw_atmos_leo/ALL_OUTPUTS/'];

% Check input.
dx = max(range(latminlatmaxlonminlonmax(1:2)), range(latminlatmaxlonminlonmax(3:4))) * 111 / nkxky;
if(dx > 1)
  warning(['[',mfilename,', WARNING] Dx will be greater than ',sprintf('%.3f', dx),' km. Consider increasing dkxky to avoid aliasing. 2^',num2str(nextpow2(dx*nkxky/.5)),' will ensure something close to 500 m.']);
end

% Prepare Green functions for each model.
% For now, only one model is needed.
% If multiple models are required, wrap everything from here in a loop.
disp('********************************');
disp('*            STEP 1            *');
disp('********************************');
idModel = 1;
OUT1 = [rootOutput,'/MODEL_',sprintf('%04d', idModel),'/STEP1/'];
step1 = [' --output ',OUT1,' --seismicModel ', seismicModel, ' --nbLayers ',num2str(nlay),' --zmax ',sprintf('%.6e ', zmax),'', ...
         ' --freqMinMax ',sprintf('%.6e ', fminmax),' --nbFreq ',num2str(nfreq),' --nbKXY ',num2str(nkxky),' --nbModes ',sprintf('%d ', nmodes),''];
c1 = [python, ' ', s1p, step1];
options = [OUT1, 'options_out.pkl']; % name must agree with python script for step 1
green = [OUT1, 'Green_RW.pkl']; % name must agree with python script for step 1
if(not(dryrun) && not(exist(green,'file')))
  % Regenerate Green functions.
  [status, result_c1] = system(c1);
  if(status)
    warning(result_c1);
    error(num2str(status));
  else
    if(verbose)
      disp(result_c1);
    else
      quickLog([OUT1,'c1_log.txt'], c1, result_c1);
    end
  end
end

% Create field for each source.
disp('********************************');
disp('*            STEP 2            *');
disp('********************************');
OUT2 = [rootOutput,'/MODEL_',sprintf('%04d', idModel),'/STEP2/'];
c2s = '';
for idSource = 1:numel(sources)
  % Produce a small source file for each wanted source.
  OUT_SOURCE = [OUT2,'source_',sprintf('%05d', idSource-1),'_in.pkl'];
  step2_sources = [' --output ',OUT_SOURCE,' --id ',num2str(sources(idSource).id-1),...
                   ' --time ',sprintf('%d ', sources(idSource).time),' --mag ',sprintf('%.6f', sources(idSource).mag),...
                   ' --latlon ',sprintf('%.6f ', sources(idSource).latlon),' --depth ',sprintf('%.6f', sources(idSource).depth),...
                   ' --strikeDipRake ',sprintf('%.6f ', sources(idSource).strikeDipRake),''];
  c2s = [c2s, python, ' ', s2p_source, step2_sources, ';'];
end
step2 = [' --output ',OUT2,' --options ',options,' --green ', green, ' --sourceIDs ',sprintf('%d ', (1:numel(sources))-1),...
         ' --latminlatmaxlonminlonmax ',sprintf('%.6f ', latminlatmaxlonminlonmax),' --f0 ',sprintf('%.6f', f0),''];
c2 = [python, ' ', s2p, step2];
if(not(dryrun))
  % Generate sources.
  [status, result_c2s] = system(c2s);
  if(status)
    warning(result_c2s);
    error(num2str(status));
  else
    if(verbose)
      disp(result_c2s);
    else
      quickLog([OUT2,'c2s_log.txt'], c2s, result_c2s);
    end
  end
  % Generate fields associated to all mechanisms.
  [status, result_c2] = system(c2);
  if(status)
    warning(result_c2);
    error(num2str(status));
  else
    if(verbose)
      disp(result_c2);
    else
      quickLog([OUT2,'c2_log.txt'], c2, result_c2);
    end
  end
end

% Produce pressure slices for each source.
disp('********************************');
disp('*            STEP 3            *');
disp('********************************');
for idSource = 1:numel(sources)
  OUT3 = [OUT2,'mechanism_',sprintf('%05d', idSource-1),filesep]; % name must agree with python script for step 2
  field = [OUT3,'RW_field.pkl']; % name must agree with python script for step 2
  step3 = [' --output ',OUT3,' --RWField ',field,' --altitudes ',sprintf('%.3f ', slice_alts),' --times ',sprintf('%.3f ', slice_tims),' --doPlots ',num2str(slice_doPlots),' --doDumps ',num2str(slice_doDumps)];
  c3 = [python, ' ', s3p, step3];
  if(not(dryrun))
    % Produce pressure slices.
    [status, result_c3] = system(c3);
    if(status)
      warning(result_c3);
      error(num2str(status));
    else
      if(verbose)
        disp(result_c3);
      else
        quickLog([OUT3,'c3_log.txt'], c3, result_c3);
      end
    end
    % Concatenate pressure slices under Matlab format for current mechanism.
    concatFilePath = concatDumps(OUT3);
  end
end