function [t, v] = readWaveformCSV(path)
  fid = fopen(path);
  data = textscan(fid, '%f%f', 'delimiter', ',', 'headerlines', 1);
  t = data{1};
  v = data{2};
  fclose(fid);
end