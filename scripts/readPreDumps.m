function [p] = readPreDumps(fname_p, nx, ny)
  fid = fopen(fname_p,'r');
  p = fread(fid,'real*8');
  fclose(fid);
  
  p = reshape(p,nx,ny);
end

