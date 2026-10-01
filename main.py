"""Sweep the session, building a panel video from every window along the way.

Reads parameters.yaml and hands it to video.py, which holds the work. Run from
the project root, after run_umap.py has written the embeddings and
behaviour_plot.m the behaviour figure:

    python main.py
    python main.py --end 600   # only clips starting before 600 s, for a quick run

The session, the cameras and the frame rate come from parameters.yaml. The
window does not: the sweep below sets it per clip, so the start_time_s and
duration_s written in the file are only what a single hand-run would have used,
and are overwritten here.

Every window is rendered, including the stretches where the mouse was getting
the task wrong -- it is the incorrect trajectories that are worth watching.
Every block is on screen in every clip, so a column's trail simply drops out
while the mouse is somewhere that column's embedding was not fitted on.

Everything lands under figures/<session>/ -- see paths.py. The clips go into
clips/, in one folder named for the sweep, so a second sweep at other lengths
sits beside the first rather than mixing into it. The interactive plots go into
umap/, written once before the first clip is rendered.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

import video
from paths import session_paths

parser = argparse.ArgumentParser(
    description='Sweep the session and build clips. With --end, only clips '
                'starting before that session time (s) are rendered, for a '
                'quick partial run.')
parser.add_argument('--end', type=float, default=None,
                    help='session time in seconds; clips starting at or '
                         'after this are skipped (default: whole session)')
args = parser.parse_args()

params = yaml.safe_load(Path('parameters.yaml').read_text())
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

CLIP_DIR.mkdir(parents=True, exist_ok=True)
print(f'writing clips to {CLIP_DIR}')

# Back-to-back clips of CLIP_S seconds, tiling the whole session. The last one
# runs past the end of the session; its frames there hold on the final bin.
for start_time in range(0, round(BIN_TIMES[-1]), CLIP_S):
    if args.end is not None and start_time >= args.end:
        continue

    session.cfg['start_time_s'] = start_time
    session.cfg['duration_s'] = CLIP_S

    started = time.perf_counter()
    video.write_video(session, CLIP_DIR / f'{start_time}s.mp4')
    elapsed = time.perf_counter() - started

    frames = round(CLIP_S * session.cfg['fps'])
    print(f'  took {elapsed:.1f} s for {frames} frames '
          f'({frames / elapsed:.0f} a second)')
