clear;
close all;
addpath('/home/sss1/Desktop/projects/DeepInteractions/feature_importances/brewermap');

cell_lines = {'GM12878', 'HeLa-S3', 'HUVEC', 'K562', 'IMR90', 'NHEK'};
root = '/home/sss1/Desktop/projects/DeepInteractions/feature_importances/SPEID/from_HOCOMOCO_motifs/';
suffix = '_feature_importance.csv';
fig_dir = '/home/sss1/Desktop/projects/DeepInteractions/feature_importances/figs/';
colormap('winter')

% Load feature importance data for all cell lines
for cell_line_idx = 1:length(cell_lines)

  % enhancers
  cell_line = cell_lines{cell_line_idx};
  file_name = [root cell_line '_enhancers' suffix];
  [names, counts, scores, mean_diffs] = read_SPEID_feature_importance(file_name, false);
  if strcmp(cell_lines{cell_line_idx}, 'K562') || strcmp(cell_lines{cell_line_idx}, 'NHEK')
    mean_diffs = -mean_diffs;
  end
  if cell_line_idx == 1
    importance_enhancers = zeros(length(names), length(cell_lines));
    importance_promoters = zeros(length(names), length(cell_lines));
    count_enhancers = zeros(length(names), length(cell_lines));
    count_promoters = zeros(length(names), length(cell_lines));
    mean_diffs_enhancers = zeros(length(names), length(cell_lines));
    mean_diffs_promoters = zeros(length(names), length(cell_lines));
  end
  % Sort features alphabetically by name
  [names, I] = sort(names);
  importance_enhancers(:, cell_line_idx) = scores(I);
  count_enhancers(:, cell_line_idx) = counts(I);
  mean_diff_enhancers(:, cell_line_idx) = mean_diffs(I);

  % promoters
  file_name = [root cell_line '_promoters' suffix];
  [names, counts, scores, mean_diffs] = read_SPEID_feature_importance(file_name, false);
  if strcmp(cell_lines{cell_line_idx}, 'NHEK')
    mean_diffs = -mean_diffs;
  end
  [names, I] = sort(names);
  importance_promoters(:, cell_line_idx) = scores(I);
  count_promoters(:, cell_line_idx) = counts(I);
  mean_diff_promoters(:, cell_line_idx) = mean_diffs(I);

end

% Plot feature importance histograms for all cell lines
f = figure;
fig_name = [fig_dir 'importance_hist'];
map = brewermap(3, 'Set1');
min_importance = min(min(importance_enhancers(:), min(importance_promoters(:))));
max_importance = max(max(importance_enhancers(:), max(importance_promoters(:))));
for cell_line_idx = 1:length(cell_lines)
  subplot(2, 3, cell_line_idx);
  hold all;
  h_enhancers = histogram(importance_enhancers(:, cell_line_idx), 'facecolor', map(1, :), 'facealpha', 0.5, 'edgecolor', 'none');
  h_promoters = histogram(importance_promoters(:, cell_line_idx), 'facecolor', map(2, :), 'facealpha', 0.5, 'edgecolor', 'none');
  BinWidth = mean([h_enhancers.BinWidth h_promoters.BinWidth]);
  h_promoters.BinWidth = BinWidth;
  h_enhancers.BinWidth = BinWidth;
  xlim([min_importance max_importance]);
  if cell_line_idx == 3
    legend({'Enhancers', 'Promoters'}, 'FontSize', 20);
  end
  title(cell_lines{cell_line_idx}, 'FontSize', 24);
  if cell_line_idx > 3 % Bottom row
    xlabel('Feature Importance', 'FontSize', 20);
  end
  if mod(cell_line_idx, 3) == 1 % Left column
    ylabel('Frequency', 'FontSize', 20);
  end
  set(gca, 'FontSize', 14);
end
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure
saveas(f, [fig_name '.fig']);
saveas(f, [fig_name '.png']);
set(f, 'Units', 'Inches'); pos = get(f, 'Position'); set(f, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)])
print(f, [fig_name '.pdf'], '-dpdf', '-r0')

% Plot feature mean_diff histograms for all cell lines
f = figure;
fig_name = [fig_dir 'mean_diff_hist'];
map = brewermap(3, 'Set1');
min_mean_diff = min(min(mean_diff_enhancers(:), min(mean_diff_promoters(:))));
max_mean_diff = max(max(mean_diff_enhancers(:), max(mean_diff_promoters(:))));
for cell_line_idx = 1:length(cell_lines)
  subplot(2, 3, cell_line_idx);
  hold all;
  h_enhancers = histogram(mean_diff_enhancers(:, cell_line_idx), 'facecolor', map(1, :), 'facealpha', 0.5, 'edgecolor', 'none');
  h_promoters = histogram(mean_diff_promoters(:, cell_line_idx), 'facecolor', map(2, :), 'facealpha', 0.5, 'edgecolor', 'none');
  BinWidth = mean([h_enhancers.BinWidth h_promoters.BinWidth]);
  h_promoters.BinWidth = BinWidth;
  h_enhancers.BinWidth = BinWidth;
  xlim([min_mean_diff max_mean_diff]);
  if cell_line_idx == 3
    legend({'Enhancers', 'Promoters'}, 'FontSize', 20);
  end
  title(cell_lines{cell_line_idx}, 'FontSize', 24);
  if cell_line_idx > 3 % Bottom row
    xlabel('Feature Repression Factor', 'FontSize', 20);
  end
  if mod(cell_line_idx, 3) == 1 % Left column
    ylabel('Frequency', 'FontSize', 20);
  end
  set(gca, 'FontSize', 14);
end
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure
saveas(f, [fig_name '.fig']);
saveas(f, [fig_name '.png']);
set(f, 'Units', 'Inches'); pos = get(f, 'Position'); set(f, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)])
print(f, [fig_name '.pdf'], '-dpdf', '-r0')

% Plot feature importance over feature counts for all cell lines
f = figure;
fig_name = [fig_dir 'importance_over_count'];
for cell_line_idx = 1:length(cell_lines)
  subplot(2, 3, cell_line_idx);
  hold all;
  scatter(count_enhancers(:, cell_line_idx), importance_enhancers(:, cell_line_idx));
  scatter(count_promoters(:, cell_line_idx), importance_promoters(:, cell_line_idx));
  set(gca, 'xscale', 'log');
  if cell_line_idx == 3
    legend({'Enhancers', 'Promoters'}, 'FontSize', 20);
  end
  title(cell_lines{cell_line_idx}, 'FontSize', 24);
  if cell_line_idx > 3 % Bottom row
    xlabel('Motif Count', 'FontSize', 20);
  end
  if mod(cell_line_idx, 3) == 1 % Left column
    ylabel('Motif Importance', 'FontSize', 20);
  end
  set(gca, 'FontSize', 14);
end
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure
saveas(f, [fig_name '.fig']);
saveas(f, [fig_name '.png']);
set(f, 'Units', 'Inches'); pos = get(f, 'Position'); set(f, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)])
print(f, [fig_name '.pdf'], '-dpdf', '-r0')

% Dispay all feature importances as image
f = figure;
fig_name = [fig_dir 'importance_all'];
num_top_features = 639;
subplot(1, 2, 1);
ranked = tiedrank(-mean_diff_enhancers);
[sorted, I] = sort(mean(ranked, 2));
imagesc(ranked(I(1:num_top_features), :));
set(gca, 'XTick', 1:6); set(gca, 'XTickLabel', cell_lines); set(gca,'XTickLabelRotation',45);
title('Enhancers', 'FontSize', 24);
set(gca, 'FontSize', 14);
subplot(1, 2, 2);
ranked = tiedrank(-mean_diff_promoters);
[sorted, I] = sort(mean(ranked, 2));
imagesc(ranked(I(1:num_top_features), :));
set(gca, 'XTick', 1:6); set(gca, 'XTickLabel', cell_lines); set(gca,'XTickLabelRotation',45);
title('Promoters', 'FontSize', 24);
set(gca, 'FontSize', 14);
c = colorbar; set(c, 'YDir', 'reverse' ); ylabel(c, 'Feature Rank', 'FontSize', 20);
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure
saveas(f, [fig_name '.fig']);
saveas(f, [fig_name '.png']);
set(f, 'Units', 'Inches'); pos = get(f, 'Position'); set(f, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)])
print(f, [fig_name '.pdf'], '-dpdf', '-r0')

% Display feature importances of mean-rank-top-20 features for all cell lines as image
f = figure;
fig_name = [fig_dir 'importance_top_20'];
num_top_features = 20;
subplot(1, 2, 1);
ranked = tiedrank(-importance_enhancers);
[sorted, I] = sort(mean(ranked, 2));
imagesc(ranked(I(1:num_top_features), :), [min(ranked(:)) max(ranked(:))]);
set(gca, 'XTick', 1:6); set(gca, 'XTickLabel', cell_lines); set(gca,'XTickLabelRotation',45);
set(gca, 'YTick', 1:num_top_features); set(gca, 'YTickLabel', names(I(1:num_top_features)));
title('Enhancers', 'FontSize', 24);
set(gca, 'FontSize', 14);
subplot(1, 2, 2);
ranked = tiedrank(-importance_promoters);
[sorted, I] = sort(mean(ranked, 2));
imagesc(ranked(I(1:num_top_features), :), [min(ranked(:)) max(ranked(:))]);
set(gca, 'XTick', 1:6); set(gca, 'XTickLabel', cell_lines); set(gca,'XTickLabelRotation',45);
set(gca, 'YTick', 1:num_top_features); set(gca, 'YTickLabel', names(I(1:num_top_features)));
title('Promoters', 'FontSize', 24);
set(gca, 'FontSize', 14);
c = colorbar; set(c, 'YDir', 'reverse' ); ylabel(c, 'Feature Rank', 'FontSize', 20);
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure
saveas(f, [fig_name '.fig']);
saveas(f, [fig_name '.png']);
set(f, 'Units', 'Inches'); pos = get(f, 'Position'); set(f, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)])
print(f, [fig_name '.pdf'], '-dpdf', '-r0')

% Plot feature mean_diff over feature counts for all cell lines
f = figure;
fig_name = [fig_dir 'mean_diff_over_count'];
for cell_line_idx = 1:length(cell_lines)
  subplot(2, 3, cell_line_idx);
  hold all;
  scatter(count_enhancers(:, cell_line_idx), mean_diff_enhancers(:, cell_line_idx));
  scatter(count_promoters(:, cell_line_idx), mean_diff_promoters(:, cell_line_idx));
  set(gca, 'xscale', 'log');
  if cell_line_idx == 3
    legend({'Enhancers', 'Promoters'}, 'FontSize', 20);
  end
  title(cell_lines{cell_line_idx}, 'FontSize', 24);
  if cell_line_idx > 3 % Bottom row
    xlabel('Motif Count', 'FontSize', 20);
  end
  if mod(cell_line_idx, 3) == 1 % Left column
    ylabel('Motif Importance', 'FontSize', 20);
  end
  set(gca, 'FontSize', 14);
end
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure
saveas(f, [fig_name '.fig']);
saveas(f, [fig_name '.png']);
set(f, 'Units', 'Inches'); pos = get(f, 'Position'); set(f, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)])
print(f, [fig_name '.pdf'], '-dpdf', '-r0')

% Display all feature mean_diffs as image
f = figure;
fig_name = [fig_dir 'mean_diff_all'];
num_top_features = 639;
subplot(1, 2, 1);
ranked = tiedrank(-mean_diff_enhancers);
[sorted, I] = sort(mean(ranked, 2));
imagesc(ranked(I(1:num_top_features), :));
set(gca, 'XTick', 1:6); set(gca, 'XTickLabel', cell_lines); set(gca,'XTickLabelRotation',45);
title('Enhancers', 'FontSize', 24);
set(gca, 'FontSize', 14);
subplot(1, 2, 2);
ranked = tiedrank(-mean_diff_promoters);
[sorted, I] = sort(mean(ranked, 2));
imagesc(ranked(I(1:num_top_features), :));
set(gca, 'XTick', 1:6); set(gca, 'XTickLabel', cell_lines); set(gca,'XTickLabelRotation',45);
title('Promoters', 'FontSize', 24);
set(gca, 'FontSize', 14);
c = colorbar; set(c, 'YDir', 'reverse' ); ylabel(c, 'Feature Rank', 'FontSize', 20);
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure
saveas(f, [fig_name '.fig']);
saveas(f, [fig_name '.png']);
set(f, 'Units', 'Inches'); pos = get(f, 'Position'); set(f, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)])
print(f, [fig_name '.pdf'], '-dpdf', '-r0')

% Display feature mean_diffs of mean-rank-top-20 features for all cell lines as image
f = figure;
fig_name = [fig_dir 'mean_diff_top_20'];
num_top_features = 20;
subplot(1, 2, 1);
ranked = tiedrank(-mean_diff_enhancers);
[sorted, I] = sort(mean(ranked, 2));
imagesc(ranked(I(1:num_top_features), :), [min(ranked(:)) max(ranked(:))]);
set(gca, 'XTick', 1:6); set(gca, 'XTickLabel', cell_lines); set(gca,'XTickLabelRotation',45);
set(gca, 'YTick', 1:num_top_features); set(gca, 'YTickLabel', names(I(1:num_top_features)));
title('Enhancers', 'FontSize', 24);
set(gca, 'FontSize', 14);
subplot(1, 2, 2);
ranked = tiedrank(-mean_diff_promoters);
[sorted, I] = sort(mean(ranked, 2));
imagesc(ranked(I(1:num_top_features), :), [min(ranked(:)) max(ranked(:))]);
set(gca, 'XTick', 1:6); set(gca, 'XTickLabel', cell_lines); set(gca,'XTickLabelRotation',45);
set(gca, 'YTick', 1:num_top_features); set(gca, 'YTickLabel', names(I(1:num_top_features)));
title('Promoters', 'FontSize', 24);
set(gca, 'FontSize', 14);
c = colorbar; set(c, 'YDir', 'reverse' ); ylabel(c, 'Feature Rank', 'FontSize', 20);
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure
saveas(f, [fig_name '.fig']);
saveas(f, [fig_name '.png']);
set(f, 'Units', 'Inches'); pos = get(f, 'Position'); set(f, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)])
print(f, [fig_name '.pdf'], '-dpdf', '-r0')
