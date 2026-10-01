"""Sweep the session, building a panel video from every window along the way.

Reads parameters.yaml and hands it to video.py, which holds the work. Run from
the project root, after run_umap.py has written the embeddings and
behaviour_plot.m the behaviour figure:

    python main.py
    python main.py --end 600   # only clips starting before 600 s, for a quick run

The session and the cameras come from parameters.yaml. The window does not: the
sweep below sets it per clip, so the start_time_s and duration_s written in the
file are only what a single hand-run would have used, and are overwritten here.

Every window is rendered, including the stretches where the mouse was getting
the task wrong -- it is the incorrect trajectories that are worth watching.
Every block is on screen in every clip. While the mouse is somewhere a column's
embedding was not fitted on, that column's trail carries on in magenta, at
where run_umap.py placed those bins with UMAP.transform.

Clips are rendered one after another, each frame after the last. Running
several clips at once in separate processes was tried and measured no faster:
one clip's x264 encoder already keeps every core busy.

Everything lands under figures/<session>/ -- see paths.py. The clips go into
clips/, in one folder named for the sweep, so a second sweep at another length
sits beside the first rather than mixing into it. The interactive plots go into
umap/, written once before the first clip is rendered. Each clip also saves the
frame at every trial end it passes through into trial_ends/, so a trial is only
saved if its last bin falls inside a rendered clip.
"""

import argparse
import time

import numpy as np

import video
from paths import load_params, session_paths

parser = argparse.ArgumentParser(
    description='Sweep the session and build clips. With --end, only clips '
                'starting before that session time (s) are rendered, for a '
                'quick partial run.')
parser.add_argument('--end', type=float, default=None,
                    help='session time in seconds; clips starting at or '
                         'after this are skipped (default: whole session)')
args = parser.parse_args()

params = load_params()
PATHS = session_paths(params)

# The session's own length, read off the last bin rather than written down, so
# a different recording needs no edit here.
BIN_TIMES = np.loadtxt(PATHS.label_dir / 'bin_times.csv')

# Clip length, named once: the sweep steps by it and the output folder is named
# after it, so the name cannot drift from what is in it. The session is already
# in the path, as the folder the clips folder sits in.
CLIP_S = 60

# All the clips of one sweep together, under the session they came from.
CLIP_DIR = PATHS.clip_dir / f'full_session_{CLIP_S}'


# One session, loaded once and reused for every clip. Nothing load_session
# builds depends on the window, so rebuilding it per clip would re-render every
# plotly figure to no effect. session.cfg is params['vid'] itself, the same dict
# object, so moving the window is a matter of writing to it between calls.
#
# The interactive plots are written before the first clip, so they are on disk
# within minutes even when the sweep is stopped early. --end does not trim
# them, since it limits clips, not plots.
session = video.load_session(params)
video.write_interactive_plots(session)

# Back-to-back clips of CLIP_S seconds, tiling the whole session. The last one
# runs past the end of the session; its frames there hold on the final bin.
starts = [start for start in range(0, round(BIN_TIMES[-1]), CLIP_S)
          if args.end is None or start < args.end]

CLIP_DIR.mkdir(parents=True, exist_ok=True)
print(f'writing {len(starts)} clips to {CLIP_DIR}')

sweep_started = time.perf_counter()
for done, start_time in enumerate(starts, start=1):
    session.cfg['start_time_s'] = start_time
    session.cfg['duration_s'] = CLIP_S

    started = time.perf_counter()
    video.write_video(session, CLIP_DIR / f'{start_time}s.mp4')
    elapsed = time.perf_counter() - started

    frames = round(CLIP_S * video.frame_rate(session.bins))
    running = time.perf_counter() - sweep_started
    average = running / done
    print(f'  [{done}/{len(starts)}] {start_time}s took {elapsed:.1f} s for '
          f'{frames} frames ({frames / elapsed:.0f} a second) | '
          f'{running / 60:.1f} min so far, {average:.1f} s per video on average, '
          f'~{average * (len(starts) - done) / 60:.0f} min left', flush=True)

print(f'sweep took {(time.perf_counter() - sweep_started) / 60:.1f} min')
