function [OUTSIDE, INSIDE] = find_annulus(x, y, p, noiseLevel)
  doPlot = 1;
  
  detecc = abs(p);
  detecc(detecc<noiseLevel)=nan;
  detecc(not(isnan(detecc)))=1;
  
  [X, Y] = meshgrid(x, y);
  xydetecc = [X(detecc==1) Y(detecc==1)];
  OUTSIDE = xydetecc(convhull(xydetecc(:, 1), xydetecc(:,2)), :);
  
%   center = mean(xydetecc);
  center = [0,0]; % assume centered
  xydetecc = xydetecc-center;
  rad = hypot(xydetecc(:, 1), xydetecc(:,2));
  rmax = max(rad);
  r = 0.1*rmax;
  while r<rmax
    if(sum(hypot(xydetecc(:, 1), xydetecc(:,2))<r)>0)
      break
    end
    rin = r;
    r = r+0.05*rmax;
  end
  th = linspace(0, 2*pi, 100)';
  INSIDE = center + rin*[cos(th), sin(th)];
  xydetecc = center + xydetecc;

  if(doPlot)
    figure();
    plot(xydetecc(:, 1), xydetecc(:,2), '.'); hold on;
    plot(OUTSIDE(:, 1), OUTSIDE(:,2)); hold on;
    plot(INSIDE(:, 1), INSIDE(:, 2)); hold on;
  end
end