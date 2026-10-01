"""Where one session's inputs and outputs live, named once for every script.

A session is its data file, and everything derived from it sits under a folder
named for that file's stem, so switching data_file in parameters.yaml switches
every input and output together and a second session never overwrites the first:

    data/raw/                              shared inputs: .mat files, maze images
    data/<session>/binned_labels/          embedding_and_labels.m
    data/<session>/embeddings/             run_umap.py
    figures/<session>/behaviour.png        behaviour_plot.m
    figures/<session>/umap/                video.write_interactive_plots
    figures/<session>/clips/<sweep>/       main.py
    figures/<session>/trial_ends/          main.py, one frame per trial's last bin
    figures/<session>/session_video.mp4    video.write_video, when run by hand

The MATLAB scripts build the same paths themselves; keep them in step with this.

Which session that is comes from load_params: the one the SESSION_ENV
environment variable names, which is how pipeline.py runs each session in turn,
or the first one listed in parameters.yaml for a script run by hand.
load_params.m does the same on the MATLAB side.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

RAW_DIR = Path('data') / 'raw'
SESSION_ENV = 'PIPELINE_SESSION'                 # the session pipeline.py is running


@dataclass(frozen=True)
class SessionPaths:
    session: str                                 # the data file's stem
    maze_png: Path                               # its blackout frame
    label_dir: Path                              # per-bin csvs
    emb_dir: Path                                # umap_*.npy
    fig_dir: Path                                # everything plotted from it
    umap_dir: Path                               # the interactive html plots
    clip_dir: Path                               # one folder per sweep
    trial_end_dir: Path                          # the frame at each trial's end


def load_params(path='parameters.yaml'):
    """parameters.yaml, with the chosen session's data_file and maze_png lifted
    to the top level, where everything downstream reads them.

    The session is the one SESSION_ENV names, which pipeline.py sets for each
    session in turn. Without it -- a script run by hand -- it is the first one
    listed under `sessions`. The two files always come from the same entry, so
    a session's data can never be paired with another session's maze.
    """
    params = yaml.safe_load(Path(path).read_text())
    sessions = params.get('sessions') or {}
    if not sessions:
        raise SystemExit(f'no sessions listed in {path} -- are they all commented out?')

    name = os.environ.get(SESSION_ENV)
    if name is None:
        name = next(iter(sessions))
        print(f'{SESSION_ENV} not set; running {name}, the first session in {path}')
    elif name not in sessions:
        raise SystemExit(f'no session {name!r} in {path}; it has {", ".join(sessions)}')

    params['session'] = name
    params.update(params['sessions'][name])
    return params


def session_paths(params):
    """Every path for the session parameters.yaml names."""
    session = Path(params['data_file']).stem
    fig_dir = Path('figures') / session
    return SessionPaths(
        session=session,
        maze_png=RAW_DIR / params['maze_png'],
        label_dir=Path('data') / session / 'binned_labels',
        emb_dir=Path('data') / session / 'embeddings',
        fig_dir=fig_dir,
        umap_dir=fig_dir / 'umap',
        clip_dir=fig_dir / 'clips',
        trial_end_dir=fig_dir / 'trial_ends')
