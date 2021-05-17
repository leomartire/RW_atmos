function [outpath] = concatDumps(root)
  thisFolder = regexprep(mfilename('fullpath'),mfilename,'');
  addpath(genpath([thisFolder,filesep,'..']));
  
  fname_xy = [root,'map_*_PRE_XYminmax.bin'];
  fname_p_filter = [root,'map_*_PRE_t*_*_z*.bin'];
  
  fxy = dir(fname_xy);
  if(numel(fxy)>1)
    error('you should not be able to find more than one xy file');
  elseif(numel(fxy)==0)
    error('no xy file found');
  else
    fname_xy = [fxy(1).folder,filesep,fxy(1).name];
  end

  [xy] = readPreDumps_xy(fname_xy);
  fnames_p = dir(fname_p_filter);
  times = [];
  altitudes = [];
  PRE = [];
  for iif = 1:numel(fnames_p)
    fname_p = fnames_p(iif).name;
    fpath = [fnames_p(iif).folder,filesep,fname_p];

    t = str2num(regexp(regexp(fname_p, 't[^_]+_', 'match', 'once'), '[0-9]+\.[0-9]+', 'match','once'));
    z = str2num(regexp(regexp(fname_p, 'z.+\.bin', 'match', 'once'), '[0-9]+\.[0-9]+', 'match','once'));
    sizeStr = regexp(fname_p,'[0-9]+x[0-9]+','match','once');
    nx = str2num(regexp(regexp(sizeStr,'[0-9]+x','match','once'),'[0-9]+','match','once'));
    ny = str2num(regexp(regexp(sizeStr,'x[0-9]+','match','once'),'[0-9]+','match','once'));
    times = [times, t];
    altitudes = [altitudes, z];

    [p] = readPreDumps(fpath, nx, ny);
    PRE(iif, :, :) = p;

    if(iif==1)
      x = linspace(xy(1), xy(2), nx);
      y = linspace(xy(3), xy(4), ny);
    end
  end
  PREDumpsConcat = struct('z', altitudes, 't', times, 'x', x, 'y', y, 'p', PRE);
  
  outpath = [root,filesep,'PREDumpsConcat.mat'];
  save(outpath, 'PREDumpsConcat');
end