%% Load data (run once)
P = yaml.loadFile('parameters.yaml');

if ~exist('data', 'var')
    data = load(fullfile('data', 'raw', P.data_file));
end

%% Extract
np_data_reduced = data.np_data_reduced;

excel = np_data_reduced.excel;
trials = np_data_reduced.trials;
units = np_data_reduced.units;
metadata = np_data_reduced.metadata;

%% Accumulate spikes

bin_size_ms = P.bin_size_ms;

trial_durations = metadata.SelectedTrialDurations;
session_length_sec = trials.TrialStartGlobalTime(end) + trial_durations(end);

n_bins = ceil(session_length_sec * 1000 / bin_size_ms);

bin_centers_sec = ((1:n_bins) - 0.5) * bin_size_ms / 1000;   % 1 x n_bins

% which trial each bin falls in; NaN for bins in the gaps between trials
assert(numel(trial_durations) == height(trials), 'expected one duration per trial');
trial_starts = trials.TrialStartGlobalTime(:)';
trial_ends = trial_starts + trial_durations(:)';

trial_id_per_bin = discretize(bin_centers_sec, [trial_starts, inf]);

has_trial = ~isnan(trial_id_per_bin);
past_end = false(size(bin_centers_sec));
past_end(has_trial) = bin_centers_sec(has_trial) > trial_ends(trial_id_per_bin(has_trial));
trial_id_per_bin(past_end) = NaN;

% a patch is a run of trials with the same CorrectBlock; block values repeat
% across the session, so patch identity is the run index, not the block value
is_new_patch = [true; diff(trials.CorrectBlock) ~= 0];
patch_id_per_trial = cumsum(is_new_patch);

% recomputed rather than reusing has_trial, which predates the past_end NaNs
in_trial = ~isnan(trial_id_per_bin);
patch_id_per_bin = nan(size(trial_id_per_bin));
patch_id_per_bin(in_trial) = patch_id_per_trial(trial_id_per_bin(in_trial));

% Bins tile the session end to end, but trials do not: consecutive trials are
% separated by the inter-trial interval, and any trial dropped in selection
% leaves a hole the width of that whole trial. Bins landing in those gaps have
% no trial, and so no patch, and are excluded from every mask below.
fprintf('%d of %d bins fall outside any trial (%.1f%%)\n', ...
    sum(~in_trial), n_bins, 100 * mean(~in_trial));

units_filtered = units(ismember(units.group, {'good', 'mua'}), :);
n_units = height(units_filtered);

% Concatenate the spikes into a single big list
global_spike_time = cell(height(trials), 1);
global_spike_identity = cell(height(trials), 1);
for trial_id = 1:height(trials)
    global_spike_time{trial_id} = trials.SpikeTime{trial_id} + trials.TrialStartGlobalTime(trial_id);
    global_spike_identity{trial_id} = trials.SpikeIdentity{trial_id};
end
all_spike_time = vertcat(global_spike_time{:});
all_spike_identity = vertcat(global_spike_identity{:});

% identify which bin each spike belongs to
bin_idx = floor(all_spike_time * 1000 / bin_size_ms) + 1;
assert(all(bin_idx >= 1 & bin_idx <= n_bins), 'A spike fell outside the expected bin range');

% from units filtered out for quality (unit_idx == 0) are expected and
% simply excluded below, not an error.
[~, unit_idx] = ismember(all_spike_identity, units_filtered.cluster_id);
valid = unit_idx > 0;

binned_spikes = accumarray([unit_idx(valid), bin_idx(valid)], 1, [n_units, n_bins]);


%% Smooth spikes

% Gaussian kernel along the time axis, applied per unit, cut off at three
% sigma where it has fallen to about 1% of its peak. See PREPROCESSING.md.
sigma_bins = P.sigma_bins;
kernel_radius = ceil(3 * sigma_bins);
kernel_x = -kernel_radius:kernel_radius;
gaussian_kernel = exp(-kernel_x.^2 / (2 * sigma_bins^2));
gaussian_kernel = gaussian_kernel / sum(gaussian_kernel);

smoothed_spikes = conv2(binned_spikes, gaussian_kernel, 'same');


%% Normalize spikes

% Z-score normalization per unit (see PREPROCESSING.md)
normalized_spikes = (smoothed_spikes - mean(smoothed_spikes, 2)) ./ std(smoothed_spikes, 0, 2);

% A unit that never varies over the session has std 0, so the line above
% divides by zero and its whole row comes out NaN. Nothing downstream can use
% it -- UMAP's correlation metric cannot take a NaN -- so it is dropped here,
% before any labels or masks are written, and every exported file is finite.
% binned_spikes and smoothed_spikes keep the dropped rows; they are
% intermediates and nothing is written from them.
usable_units = all(isfinite(normalized_spikes), 2);
fprintf('removed %d of %d units with no variance over the session\n', ...
    sum(~usable_units), numel(usable_units));

normalized_spikes = normalized_spikes(usable_units, :);
units_filtered = units_filtered(usable_units, :);
n_units = height(units_filtered);



%% Labels based on reward

% Signed seconds to the nearest reward, one value per bin: negative before the
% reward, positive after it, which is the PSTH convention. Computed for every
% bin in the session and against every rewarded trial, so a bin's nearest reward
% may belong to a trial in another patch, or to one that was not kept. That is
% deliberate -- the question is when the animal last saw or next sees a reward,
% not which trial the bin was filed under.
reward_times = trials.TrialStartGlobalTime(trials.Rewarded == "rewarded") + ...
    trials.RewardTimeInTrial(trials.Rewarded == "rewarded");
rt = reward_times(:)';   % 1 x n_rewards, so both results below stay rows
assert(issorted(rt), 'Reward times are not ascending, so discretize cannot bin against them');

% index of the first reward at or after each bin time
next_idx = discretize(bin_centers_sec, [-inf, rt]);
has_next = ~isnan(next_idx);

time_to_reward = nan(size(bin_centers_sec));
time_to_reward(has_next) = rt(next_idx(has_next)) - bin_centers_sec(has_next);

% index of the last reward before each bin time
prev_idx = discretize(bin_centers_sec, [rt, inf]);
has_prev = ~isnan(prev_idx);

time_from_reward = nan(size(bin_centers_sec));
time_from_reward(has_prev) = bin_centers_sec(has_prev) - rt(prev_idx(has_prev));

% whichever reward is closer wins; a NaN on one side leaves the other uncontested
use_next = time_to_reward <= time_from_reward | isnan(time_from_reward);

time_nearest_reward = nan(size(bin_centers_sec));
time_nearest_reward(use_next)  = -time_to_reward(use_next);
time_nearest_reward(~use_next) =  time_from_reward(~use_next);

% Size, in whole milliseconds, of whichever reward time_nearest_reward is
% measured against -- the same use_next split, so a bin pairs with the size of
% the reward it is actually counting down to or up from. RewardDuration is a
% difference of two large timestamps, so nominally equal durations differ in
% the last few bits (0.014999999999999680 alongside 0.015000000000000568); the
% session delivers two sizes, 15 ms and 30 ms, so rounding collapses that noise
% into integers that compare exactly. Left unwindowed on purpose -- how close a
% bin has to be to a reward to display its size is a plotting decision, made
% where REWARD_DISPENSING_S already lives, not a labelling one.
reward_size_ms = round(trials.RewardDuration(trials.Rewarded == "rewarded") * 1000);
rs = reward_size_ms(:)';

reward_size_nearest = nan(size(bin_centers_sec));
reward_size_nearest(use_next & has_next)  = rs(next_idx(use_next & has_next));
reward_size_nearest(~use_next & has_prev) = rs(prev_idx(~use_next & has_prev));


%% Export all bins once, then one mask per patch
label_dir = fullfile('data', 'binned_labels');
if ~isfolder(label_dir)
    mkdir(label_dir);
end

% correct and rewarded are not the same thing here: the session contains both
% correct/unrewarded and incorrect/rewarded trials, so both tests are needed.
% This part does not depend on the patch, so it is computed once.
trial_is_kept = trials.Correctness == "correct" & trials.Rewarded == "rewarded";

keep_trial_bin = false(size(bin_centers_sec));
keep_trial_bin(in_trial) = trial_is_kept(trial_id_per_bin(in_trial));

n_patches = max(patch_id_per_trial);

% Every bin, unfiltered: units down the rows, bins across the columns. The
% per-patch masks below index into these columns. writematrix has no precision
% control and its default runs to ~15 significant digits, which bloats a matrix
% this size, so this one is written with fprintf at 6 significant digits --
% well beyond what smoothed, z-scored rates meaningfully carry.
spikes_path = fullfile(label_dir, 'binned_spikes.csv');
fid = fopen(spikes_path, 'w');
assert(fid > 0, 'could not open %s for writing', spikes_path);
fprintf(fid, [repmat('%.6g,', 1, n_bins - 1), '%.6g\n'], normalized_spikes.');
fclose(fid);

% Bin midpoints in global session seconds, one row per bin, in the same order
% as the columns above. Left at full precision on purpose: these are absolute
% session times, and 6 significant digits would start rounding the midpoint
% away late in a long session.
writematrix(bin_centers_sec.', fullfile(label_dir, 'bin_times.csv'));

% The session trial each bin belongs to, NaN in the gaps between trials, in the
% same order again. SessionTrial rather than the row index discretize hands
% back: the two part company wherever trial selection dropped a trial, and
% SessionTrial is the number the experiment actually ran under.
trial_id_per_bin_session = nan(size(trial_id_per_bin));
trial_id_per_bin_session(in_trial) = trials.SessionTrial(trial_id_per_bin(in_trial));
writematrix(trial_id_per_bin_session(:), fullfile(label_dir, 'trial_ids.csv'));

% Signed seconds to the nearest reward, one row per bin, NaN where there is no
% reward on either side. Written whole rather than per patch: it does not depend
% on the patch, so the plotting side takes whichever subset it wants using the
% same mask it uses for the spikes.
writematrix(time_nearest_reward(:), fullfile(label_dir, 'time_nearest_reward.csv'));

% Size in ms of whichever reward time_nearest_reward pairs with, one row per
% bin, NaN under the same conditions time_nearest_reward is NaN. Also written
% whole rather than per patch, for the same reason.
writematrix(reward_size_nearest(:), fullfile(label_dir, 'reward_size_ms.csv'));

% One mask per patch, one row per bin, covering all n_bins. A patch with no
% correct rewarded trials still gets an all-false mask, so patch numbering
% stays contiguous and nothing downstream has to cope with a missing file.
for selected_patch = 1:n_patches
    keep_bin = keep_trial_bin & patch_id_per_bin == selected_patch;   % NaN fails the test

    fprintf('patch %d: %d bins\n', selected_patch, sum(keep_bin));

    writematrix(uint8(keep_bin).', fullfile(label_dir, sprintf('patch_mask_%d.csv', selected_patch)));
end


%% Head position per bin

% The video is stored per trial, so frames go onto the session clock the same
% way the spikes did: trial start plus the frame's own time within its trial.
% The video runs past the end of the selected trials, so those trailing frames
% have no trial to attach to and are dropped. An unmatched trial from inside
% the selected range would mean the two numberings disagree, which dropping
% frames would paper over rather than fix, so that errors instead.
[is_matched, trial_row] = ismember(excel.trial_number, trials.SessionTrial);

unmatched = unique(excel.trial_number(~is_matched));
inside_range = unmatched(unmatched <= max(trials.SessionTrial));
assert(isempty(inside_range), ...
    'Frames reference unselected trials inside the session: %s', mat2str(inside_range'));

frames = excel(is_matched, :);
frame_global_time = trials.TrialStartGlobalTime(trial_row(is_matched)) + frames.frame_time_in_trial;
assert(issorted(frame_global_time, 'strictascend'), ...
    'Frame times are not strictly increasing, so interp1 cannot index by time');

% DeepLabCut reports how sure it was of the headstage in each frame. A badly
% tracked frame would drag the interpolation towards somewhere the mouse never
% was, so those frames are dropped and the interpolation bridges the hole.
min_likelihood = 0.9;
tracked = frames.headstage_likelihood >= min_likelihood;

% No 'extrap' here on purpose: a bin outside the tracked range comes out NaN
% rather than being silently clamped onto the first or last frame's position.
head_x_per_bin = interp1(frame_global_time(tracked), frames.headstage_x(tracked), bin_centers_sec, 'linear');
head_y_per_bin = interp1(frame_global_time(tracked), frames.headstage_y(tracked), bin_centers_sec, 'linear');

% Bins in the gaps between trials have no frames of their own, so interpolating
% across a gap would draw a straight line through positions the mouse was never
% in. Those bins are NaN, matching the NaNs in trial_id_per_bin.
head_x_per_bin(~in_trial) = NaN;
head_y_per_bin(~in_trial) = NaN;

% One row per bin, [x, y] in DeepLabCut pixels, in the same order as
% bin_times.csv and the columns of binned_spikes.csv.
writematrix([head_x_per_bin(:), head_y_per_bin(:)], fullfile(label_dir, 'head_positions.csv'));


%% Port positions, and which port each bin is at

% Nothing in the tracking says where the ports are, so they are recovered from
% behaviour: at reward the mouse is at the port it chose, so averaging the head
% position at reward over every trial that visited a port gives that port's
% location. Averaged per session rather than hardcoded, because the tracking is
% in pixels of each session's own video and the ports land somewhere different
% in each one.
%
% Trials are not restricted to rewarded ones. A wrong choice still puts the
% mouse at the port it visited.
n_ports = 8;

% Interpolated through the tracked frames, the same way head_x_per_bin and
% head_y_per_bin are above, rather than snapping to the single nearest raw
% frame and dropping the trial if that one frame is under min_likelihood. Ports
% 5-8 sit in a part of the frame where most single reward-time frames land
% under threshold, which used to leave some ports (port 8 entirely) with no
% recovered position at all even though nearby tracked frames were fine.
reward_time_global = trials.TrialStartGlobalTime + trials.RewardTimeInTrial;

reward_x = interp1(frame_global_time(tracked), frames.headstage_x(tracked), reward_time_global, 'linear');
reward_y = interp1(frame_global_time(tracked), frames.headstage_y(tracked), reward_time_global, 'linear');
reward_usable = ~isnan(reward_x) & ~isnan(reward_y);

port_positions = nan(n_ports, 2);
for port = 1:n_ports
    at_port = reward_usable & trials.Port == port;
    port_positions(port, :) = [mean(reward_x(at_port)), mean(reward_y(at_port))];
    fprintf('port %d: %3d trials, centre (%.0f, %.0f)\n', port, sum(at_port), ...
        port_positions(port, 1), port_positions(port, 2));
end

% Nearest port within this many pixels of the bin's tracked position, or 0 for
% none. Read off the position rather than from the tracking export's "In zone"
% columns, which were found unreliable. A port with no computed position is all
% NaN and so never wins; a bin with no position gets 0 the same way.
%
% Port centres sit 229-1072 px apart (closest pair: ports 1 and 2), so there is
% no risk of this threshold confusing two ports with each other -- it only
% trades off how much real dwell-time around a port counts as "there" against
% how many genuine visits get missed by a few pixels. 5 px lost 63 of 410
% trials from the per-bin labels entirely; 10 px recovers most of those
% (16 lost) while adding only a modest number of bins (4142 -> 10440).
% Diminishing returns set in above 12 px.
port_threshold_px = 10;

dx = head_x_per_bin(:) - port_positions(:, 1)';
dy = head_y_per_bin(:) - port_positions(:, 2)';
[nearest_px, port_per_bin] = min(sqrt(dx.^2 + dy.^2), [], 2);
port_per_bin(isnan(nearest_px) | nearest_px >= port_threshold_px) = 0;

fprintf('%d of %d bins within %g px of a port\n', ...
    sum(port_per_bin > 0), numel(port_per_bin), port_threshold_px);

% A trial is lost if none of its own bins land within port_threshold_px of the
% port it was actually at -- its choice happened, but nothing in port_ids.csv
% reflects it, so nothing coloured by port on the UMAP will ever show it.
port_per_bin_row = port_per_bin(:)';

matched_bin = false(size(bin_centers_sec));
matched_bin(in_trial) = port_per_bin_row(in_trial) == trials.Port(trial_id_per_bin(in_trial))';

trial_has_match = false(height(trials), 1);
trial_has_match(trial_id_per_bin(in_trial & matched_bin)) = true;

fprintf('\ntrials whose port never appears in the per-bin labels (lost from the UMAP):\n');
for port = 1:n_ports
    is_port_trial = trials.Port == port;
    n_lost = sum(is_port_trial & ~trial_has_match);
    fprintf('port %d: %d of %d trials lost\n', port, n_lost, sum(is_port_trial));
end

writematrix(port_per_bin, fullfile(label_dir, 'port_ids.csv'));
