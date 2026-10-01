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
    figures/<session>/session_video.mp4    video.write_video, when run by hand

The MATLAB scripts build the same paths themselves; keep them in step with this.
"""

from dataclasses import dataclass
from pathlib import Path

RAW_DIR = Path('data') / 'raw'


@dataclass(frozen=True)
class SessionPaths:
    session: str                                 # the data file's stem
    maze_png: Path                               # its blackout frame
    label_dir: Path                              # per-bin csvs
    emb_dir: Path                                # umap_*.npy
    fig_dir: Path                                # everything plotted from it
    umap_dir: Path                               # the interactive html plots
    clip_dir: Path                               # one folder per sweep


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
        clip_dir=fig_dir / 'clips')
