function [] = quickLog(logfile, c2s, result_c2s)

  fid = fopen(logfile,'w');
  
  fprintf(fid, ['Log written on ',datestr(now),'.']);

  fprintf(fid, '\n****************************************************************\n');
  
  fprintf(fid, c2s);

  fprintf(fid, '\n****************************************************************\n\n\n');
  
  fprintf(fid, result_c2s);
  
  fclose(fid);

end