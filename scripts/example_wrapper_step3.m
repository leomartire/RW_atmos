function [] = example_wrapper_step3(c3, OUT3, verbose)
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