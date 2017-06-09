function [names, counts, scores, mean_diffs] = read_SPEID_feature_importance(fileName, to_sort)

  if nargin < 2 % by default, sort results by feature importance
    to_sort = true;
  end

  fileID = fopen(fileName);
  C = textscan(fileID, '%s %f %f %f', 'Delimiter', ',', 'headerLines', 1);
  fclose(fileID);

  % extract columns by name
  names = C{1};
  counts = C{2};
  scores = C{3};
  mean_diffs = C{4};

  if to_sort
    % sort motifs by score (descending)
    [scores, I] = sort(scores, 'descend');
    names = names(I);
    counts = counts(I);
    mean_diffs = mean_diffs(I);
  end

end
