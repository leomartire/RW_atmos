function [xy] = readPreDumps_xy(fname_xy)
  fid = fopen(fname_xy,'r');
  xy = fread(fid,'real*8');
  fclose(fid);
end

