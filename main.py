"""Sweep the session, building a panel video from every window along the way.

Reads parameters.yaml and hands it to video.py, which holds the work. Run from
the project root, after run_umap.py has written the embeddings:

    python main.py
    python main.py --end 600   # only clips starting before 600 s, for a quick run

The session, the cameras and the frame rate come from parameters.yaml. The
window and the patch do not: the sweep below sets them per clip, so the
start_time_s, duration_s and patch written in the file are only what a single
hand-run would have used, and are overwritten here.

Every window is rendered, including the stretches where the mouse was getting
the task wrong. Those hold no bins in any patch mask, so the patch panels draw
their cloud as usual but carry no trail across it, and the information panel
reads "excluded" the whole way through, while the maze and the whole-session
embedding are trailed as normal -- which is the point: it is the incorrect
trajectories that are worth watching.

Each clip is otherwise rendered against the patch that owns most of its bins --
see get_patch -- which is not always the patch its start time falls in.

The clips land together in one folder named for the sweep and the session, so a
second sweep at other lengths sits beside the first rather than mixing into it.
The interactive plots stay in figures/ alongside, one set per patch rather than
one per clip.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

import video

parser = argparse.ArgumentParser(
    description='Sweep the session and build clips. With --end, only clips '
                'starting before that session time (s) are rendered, for a '
                'quick partial run.')
parser.add_argument('--end', type=float, default=None,
                    help='session time in seconds; clips starting at or '
                         'after this are skipped (default: whole session)')
args = parser.parse_args()

params = yaml.safe_load(Path('parameters.yaml').read_text())

# The session's own length, read off the last bin rather than written down, so
# a different recording needs no edit here.
BIN_TIMES = np.loadtxt(video.DATA_DIR / 'bin_times.csv')

# Clip lengths, named once: the sweep zips them against its start times and the
# output folder is named after them, so the name cannot drift from what is in it.
DURATIONS = (30, 60, 120)

# All the clips of one sweep together, under the session they came from. The
# session is the data file, which is what the information panel calls it too.
CLIP_DIR = video.PLOT_DIR / ('full_session_'
                             + '-'.join(str(d) for d in DURATIONS)
                             + '_' + Path(params['data_file']).stem)

# Every patch's mask, read once. A patch whose trials were all dropped in
# selection has an all-false mask and can never win a window, but it is kept
# here so the numbering matches the files on disk.
PATCH_MASKS = {int(path.stem.rsplit('_', 1)[1]): np.loadtxt(path).astype(bool)
               for path in video.DATA_DIR.glob('patch_mask_*.csv')}

# When each patch runs, for the windows that hold no kept bins to fall back on.
PATCH_SPANS = video.patch_spans()


def nearest_patch(time_s):
    """The patch running at this moment, or the closest one if none is.

    Distance is zero anywhere inside a patch's span, so this only has to choose
    between patches for a moment that falls between two -- or before the first,
    which the start of the session does: patch 1's block opens on trial 1 at
    0 s, but its span is measured from its first correct rewarded bin at 80 s,
    because the mouse got the first five trials wrong.
    """
    def distance(span):
        _, first, last = span
        return max(first - time_s, 0.0, time_s - last)

    patch, _, _ = min(PATCH_SPANS, key=distance)
    return patch


def get_patch(start_time_s, duration_s):
    """Whichever patch owns the most bins in this window.

    A window is not guaranteed to sit inside one patch: patches are runs of
    trials, so a long enough clip runs off the end of one and into the next, and
    a clip starting in the inter-trial gap between two may hold bins from both.
    Counting the bins and taking the majority picks the patch the clip mostly
    shows, which is the one whose embedding is worth rendering beside it.

    The masks hold correct rewarded bins only, so a window lying entirely in
    incorrect or unrewarded trials counts nothing anywhere. That is not a reason
    to skip it -- those are the trials worth watching the mouse during -- so it
    falls back to whichever patch is nearest in time, and which patch that is
    barely matters: none of the window's bins are in any of them.

    Such a clip loses nothing but the patch panels' trail. Their cloud is a
    still of the whole patch and renders as it always does; it is the moving
    dots that are missing, because patch_row is -1 for a bin the patch does not
    contain and embedding_panel skips those. The maze and the whole-session
    embedding are trailed as normal, since their rows cover every bin, and the
    information panel reads "excluded" throughout.
    """
    window = (BIN_TIMES >= start_time_s) & (BIN_TIMES < start_time_s + duration_s)

    counts = {patch: int((mask & window).sum()) for patch, mask in PATCH_MASKS.items()}
    best = max(counts, key=counts.get)

    return best if counts[best] else nearest_patch(start_time_s + duration_s / 2)


# Three clips off every sweep point: a short one at the point itself, then two
# longer ones starting later, so the same stretch is seen at three lengths.
#
# One session per patch, kept and reused. Nothing load_session builds depends on
# the window -- only on the patch -- so rebuilding it per clip would re-render
# ten plotly figures to no effect. session.cfg is params['vid'] itself, the same
# dict object, so moving the window is a matter of writing to it between calls.
sessions = {}

CLIP_DIR.mkdir(parents=True, exist_ok=True)
print(f'writing clips to {CLIP_DIR}')

for t in range(0, round(BIN_TIMES[-1]), 210):
    for start_time, movie_duration in zip((t, t + 30, t + 90), DURATIONS):
        if args.end is not None and start_time >= args.end:
            continue

        patch = get_patch(start_time, movie_duration)

        if patch not in sessions:
            params['vid']['patch'] = patch
            sessions[patch] = video.load_session(params)
            video.write_interactive_plots(sessions[patch])
        session = sessions[patch]

        session.cfg['start_time_s'] = start_time
        session.cfg['duration_s'] = movie_duration

        started = time.perf_counter()
        video.write_video(session, CLIP_DIR /
                          f'patch{patch}_{start_time}s_{movie_duration}s.mp4')
        elapsed = time.perf_counter() - started

        frames = round(movie_duration * session.cfg['fps'])
        print(f'  took {elapsed:.1f} s for {frames} frames '
              f'({frames / elapsed:.0f} a second)')


    




