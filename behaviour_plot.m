%% Load data (run once)
P = load_params();

% Reloaded when data_file changes, so switching session cannot silently reuse
% the last one's data left in the workspace.
if ~exist('data', 'var') || ~exist('loaded_file', 'var') || loaded_file ~= P.data_file
    data = load(fullfile('data', 'raw', P.data_file));
    loaded_file = P.data_file;
end
[~, session] = fileparts(P.data_file);

%% Extract
np_data_reduced = data.np_data_reduced;

excel = np_data_reduced.excel;
trials = np_data_reduced.trials;
units = np_data_reduced.units;
metadata = np_data_reduced.metadata;

%% Figure

% Port colours from parameters.yaml, shared with the UMAP port figures, indexed
% by port number: port_colors(p) is port p's colour. yaml.loadFile hands a list
% back as a cell, so it is made a string array here; hex strings go straight
% into any Color property.
port_colors = string(P.port_colors);

% Saved in the session's own figures folder, as paths.py lays it out, so plots
% of different recordings sit side by side rather than overwriting.
fig_dir = fullfile('figures', session);

% A fixed size in pixels, so the saved PNG comes out the same on every machine,
% and everything placed below can be placed in pixels too -- see Legends.
fig = figure('Name', "behaviour_" + session, 'Units', 'pixels');
fig.Position(3:4) = [1200 700];
theme(fig, 'light');   % R2025a+ follows the desktop's dark theme otherwise, into the saved PNG too

n_trials = height(trials);

% Two axes stacked on one trial axis: the port strip on top, a quarter of the
% height (any less and its eight tick labels crowd), and the rolling
% percentages below it. No gap between them, so the strip reads as a band
% along the top of the plot rather than a separate one. Where exactly they sit
% is set under Legends, once the legends' widths are known.
ax_port = axes(fig, 'Units', 'pixels');
hold(ax_port, 'on');      % or the first scatter below resets the limits, ticks and label set here
ylim(ax_port, [0 8.5]);   % a full step below port 1, so its label clears the 100 beneath
yticks(ax_port, 1:8);
ylabel(ax_port, 'Port');
ax_port.XTickLabel = [];          % the trial numbers are read off the axes below,
ax_port.XAxis.TickLength = [0 0]; % and so are the ticks

ax_rate = axes(fig, 'Units', 'pixels');
hold(ax_rate, 'on');
ylim(ax_rate, [0 100]);
xlabel(ax_rate, 'Trial');
ylabel(ax_rate, '% (rolling)');

% Axis lines and ticks over the block shading rather than under it, or the
% bands hide the line between the two axes and the port strip's ticks.
set([ax_port, ax_rate], 'Layer', 'top');

% One x range for both, set once; linkaxes keeps them aligned under zoom too.
linkaxes([ax_port, ax_rate], 'x');
xlim(ax_rate, [0 n_trials]);

%% Block shading
% A block is a run of trials sharing a CorrectBlock, so a new one starts
% wherever the CorrectBlock value changes -- the same definition
% embedding_and_labels.m uses. CorrectBlock values repeat across the session,
% so blocks are numbered by run rather than by value. CorrectBlock spells out
% the rewarded pair: 56 is ports 5 and 6.
correct_block = trials.CorrectBlock;
block_first = [1; find(diff(correct_block) ~= 0) + 1];
block_last = [block_first(2:end) - 1; n_trials];
n_blocks = numel(block_first);

pair_a = floor(correct_block(block_first) / 10);
pair_b = mod(correct_block(block_first), 10);

% Each block is tinted with its rewarded pair's hue, the darker port's colour
% washed towards white, so the filled circles sit over a background of
% their own colour whenever the mouse is at the right pair. Blocks with the
% same pair therefore share a hue; the legend numbers them apart.
tint = 0.25;   % how much of the port colour survives the wash
block_rgb = 1 - tint * (1 - hex2rgb(port_colors(pair_a)));

% Bands meet halfway between the last trial of one block and the first of the
% next, so each circle sits in its own trial's band, and the outer two run out
% to the ends of the trial axis. They run up through the port strip too, one
% band per axes, full height in each; drawn before anything else, so the
% circles and curves land on top of them. The legend is built from the lower
% axes' bands, one entry per block.
band_edges = [0; block_first(2:end) - 0.5; n_trials];

block_bands = gobjects(n_blocks, 1);
for k = 1:n_blocks
    left = band_edges(k);
    right = band_edges(k + 1);
    patch(ax_port, [left right right left], ax_port.YLim([1 1 2 2]), ...
          block_rgb(k, :), 'EdgeColor', 'none');
    block_bands(k) = patch(ax_rate, [left right right left], ax_rate.YLim([1 1 2 2]), ...
                           block_rgb(k, :), 'EdgeColor', 'none', ...
                           'DisplayName', sprintf('Block %d (Ports %d, %d)', ...
                                                  k, pair_a(k), pair_b(k)));
end
block_key = legend(block_bands, 'AutoUpdate', 'off');   % or the rolling lines drawn later join it

%% Port strip
% One circle per trial, at the port the mouse licked at, in that port's colour:
% filled if the trial was rewarded, open if not. Rewarded is read as recorded
% rather than worked out from the port, because the two part company -- the
% session holds correct trials that went unrewarded and incorrect ones that
% were rewarded.
trial_x = (1:n_trials).';
port = trials.Port;
rewarded = trials.Rewarded == "rewarded";

% A port outside 1..8 has no colour to draw it in, so it is left out and
% counted rather than failing on the index below.
licked = ismember(port, 1:8);
if any(~licked)
    fprintf('%d of %d trials have no port in 1-8 and are not drawn\n', ...
            nnz(~licked), n_trials);
end

port_rgb = zeros(n_trials, 3);
port_rgb(licked, :) = hex2rgb(port_colors(port(licked)));

marker_size = 20;
filled_trials = licked & rewarded;
open_trials = licked & ~rewarded;

scatter(ax_port, trial_x(filled_trials), port(filled_trials), marker_size, ...
        port_rgb(filled_trials, :), 'filled');
scatter(ax_port, trial_x(open_trials), port(open_trials), marker_size, ...
        port_rgb(open_trials, :), 'LineWidth', 1);

% The legend explains fill only, so its two entries are grey stand-ins drawn at
% NaN -- nothing on the axes -- rather than the scatters above, whose colour
% would make a key entry look like it meant one particular port.
legend_grey = [0.5 0.5 0.5];
key_rewarded = scatter(ax_port, NaN, NaN, marker_size, legend_grey, 'filled');
key_unrewarded = scatter(ax_port, NaN, NaN, marker_size, legend_grey, 'LineWidth', 1);
port_key = legend([key_rewarded, key_unrewarded], {'rewarded', 'unrewarded'});

%% Rolling performance
% Two trailing averages over the last rolling_window trials, this one included:
% how many were at one of the block's correct ports, and how many were
% rewarded. Correctness is exactly "at one of the ports CorrectBlock names" --
% checked trial for trial on this session -- so it is read as recorded rather
% than worked out again. The two lines part company where reward and
% correctness do: correct trials that went unrewarded, incorrect ones that
% were rewarded.
rolling_window = 20;
correct = trials.Correctness == "correct";

% Until the window fills, movmean averages over the trials there are so far --
% trial 1 over itself alone, trial 2 over two -- so the lines run from the
% first trial, if noisily at the very start, where one trial moves them most.
pct_correct = 100 * movmean(double(correct), [rolling_window - 1, 0]);
pct_rewarded = 100 * movmean(double(rewarded), [rolling_window - 1, 0]);

correct_line = plot(ax_rate, trial_x, pct_correct, ...
                    'Color', [0.10 0.15 0.50], 'LineWidth', 2, ...
                    'DisplayName', sprintf('%% correct port (last %d trials)', rolling_window));
rewarded_line = plot(ax_rate, trial_x, pct_rewarded, ...
                     'Color', [0.25 0.60 0.85], 'LineWidth', 2, ...
                     'DisplayName', sprintf('%% rewarded (last %d trials)', rolling_window));

% ax_rate's one legend already holds the blocks, so the lines get theirs from
% an invisible axes, through stand-ins drawn at NaN in the same styles. The
% stand-ins are copies, so the key cannot drift from the lines it describes.
ax_line_key = axes(fig, 'Visible', 'off', 'HandleVisibility', 'off');
line_key_entries = copyobj([correct_line, rewarded_line], ax_line_key);
set(line_key_entries, 'XData', NaN, 'YData', NaN);
line_key = legend(ax_line_key, line_key_entries);

%% Legends
% All three in a margin of their own at the right, rather than inside the axes
% where they would sit on the data: the fill key level with the top of the
% port strip, the block key level with the top of the percentages, and the
% line key just below the block key. The margin is sized from the widest
% legend, and both axes end where it begins, so they stay aligned on the trial
% axis.
%
% Everything here is in pixels and placed by hand. A legend left to place
% itself rescales whenever its axes move -- measured at 0.127 of the figure's
% width before a margin was cut and 0.150 after -- so it outgrows any margin
% sized from it; pinned in pixels it keeps the size it was measured at. The
% axes are placed by hand for the same reason: a tiledlayout's axes report
% MATLAB's default position rather than where the layout drew them, so the
% legends could not be lined up against them.
drawnow;   % legends know their size only once drawn
keys = [port_key, block_key, line_key];
for key = keys
    key.Units = 'pixels';
    key.Position = key.Position;   % a manual position, so nothing rescales it again
end

legend_gap = 12;   % pixels between the axes and the legends, and the legends and the edge
legend_margin = max(arrayfun(@(key) key.Position(3), keys)) + 2 * legend_gap;

fig_size = fig.Position(3:4);
axes_left = 70;     % room for the tick labels and the y labels
axes_bottom = 55;   % and for the trial axis's
axes_top = fig_size(2) - 15;
axes_width = fig_size(1) - legend_margin - axes_left;
port_height = (axes_top - axes_bottom) / 4;

ax_rate.Position = [axes_left, axes_bottom, axes_width, axes_top - axes_bottom - port_height];
ax_port.Position = [axes_left, axes_top - port_height, axes_width, port_height];

for pair = {port_key, ax_port; block_key, ax_rate}.'
    [key, ax] = pair{:};
    key.Position(1) = axes_left + axes_width + legend_gap;
    key.Position(2) = sum(ax.Position([2 4])) - key.Position(4);   % top edge level with the axes'
end

line_key.Position(1) = block_key.Position(1);
line_key.Position(2) = block_key.Position(2) - legend_gap - line_key.Position(4);

%% Save
if ~isfolder(fig_dir)
    mkdir(fig_dir);
end

% 'Padding', 'figure' keeps the whole figure rather than cropping to what is
% drawn on it, so the PNG is the figure scaled by one factor and the axes land
% where they were placed above -- which the trial axis below depends on.
png_path = fullfile(fig_dir, 'behaviour.png');
exportgraphics(fig, png_path, 'Padding', 'figure', 'Resolution', 192);

%% Trial axis, for the video
% video.py draws a line on this PNG at the current trial, so it needs the
% pixel column each trial sits at. The scale is read back off the saved file
% rather than assumed from the resolution, and checked to be the same both
% ways. Figure pixels count up from the bottom and image rows down from the
% top, hence the flip. The trial axis counts the plotted trials 1..n rather
% than SessionTrial, and the two part company wherever a trial was dropped, so
% each row pairs a trial's SessionTrial with its column. The top and bottom of
% the axes -- the port strip's top edge, the percentages' bottom -- are where
% the line starts and stops, repeated on every row to keep this one table.
png_info = imfinfo(png_path);
scale = png_info.Width / fig_size(1);
assert(abs(png_info.Height / fig_size(2) - scale) < 0.01, ...
    'behaviour.png is not the whole figure at one scale, so its trial axis cannot be placed');

x_px = scale * (ax_rate.Position(1) + trial_x / n_trials * ax_rate.Position(3));
y_top_px = scale * (fig_size(2) - sum(ax_port.Position([2 4]))) * ones(n_trials, 1);
y_bottom_px = scale * (fig_size(2) - ax_rate.Position(2)) * ones(n_trials, 1);

writetable(table(trials.SessionTrial, x_px, y_top_px, y_bottom_px, ...
                 'VariableNames', {'session_trial', 'x_px', 'y_top_px', 'y_bottom_px'}), ...
           fullfile(fig_dir, 'behaviour_axis.csv'));


