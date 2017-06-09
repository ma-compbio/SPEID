cell_lines = {'GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK'};

root = '/home/sss1/Desktop/projects/DeepInteractions/feature_importances/SPEID/from_HOCOMOCO_motifs/';
suffix = '_feature_importance.csv';

for cell_line_idx = 1:length(cell_lines)

  cell_line = cell_lines{cell_line_idx};

  % Load enhancer results
  file_name = [root cell_line '_enhancers' suffix];
  disp(['Reading file ' file_name]);
  [names, ~, scores] = read_SPEID_feature_importance(file_name, false);
  [names, I] = sort(names);
  scores = scores(I);
  if cell_line_idx == 1 % initialize output matrix, now that we know # features
    importance_enhancers = zeros(length(names), length(cell_lines));
    importance_promoters = zeros(length(names), length(cell_lines));
  end
  importance_enhancers(:, cell_line_idx) = scores;

  % Load promoter results
  file_name = [root cell_line '_promoters' suffix];
  disp(['Reading file ' file_name]);
  [names, ~, scores] = read_SPEID_feature_importance(file_name, false);
  [names, I] = sort(names);
  scores = scores(I);
  importance_promoters(:, cell_line_idx) = scores;

end

save([root 'collected_results.mat'], ...
     'cell_lines', ...
     'names', ...
     'importance_enhancers', ...
     'importance_promoters');
