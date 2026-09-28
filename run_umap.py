"""UMAP embeddings of the binned session — the whole session, its correct
rewarded bins, and each block both filtered that way and whole.

Reads the CSVs written by embedding_and_labels.m and writes one .npy per embedding into
data/<session>/embeddings/, for the session parameters.yaml names -- see
paths.py. No plotting: this only produces the embeddings.

    umap_full.npy              (n_bins, n_components)   every bin, in file order
    umap_correct_rewarded.npy  (n_kept, n_components)   correct_rewarded
    umap_block_<N>_cr.npy      (n_kept, n_components)   correct_rewarded & block_id == N
    umap_block_<N>_all.npy     (n_kept, n_components)   block_id == N

Every selection is made from three per-bin columns, correct.csv and rewarded.csv
(0/1, combined here into correct_rewarded) and block_id.csv (the block each bin falls in, the gaps between trials
included) -- see selections(). A masked embedding's row i is the i-th True
entry of its mask, so the matching times and positions are bin_times.csv[mask]
and head_positions.csv[mask].

Fits are cached by content, so re-running does not refit anything whose inputs
and parameters are unchanged. UMAP is not seeded here, so the cache is also what
keeps an embedding stable from run to run — delete .cache/ to force fresh fits.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import yaml

from paths import session_paths

PATHS = session_paths(yaml.safe_load(Path('parameters.yaml').read_text()))
LABEL_DIR = PATHS.label_dir
OUT_DIR = PATHS.emb_dir
CACHE_DIR = Path('.cache')

UMAP_PARAMS = dict(
    n_neighbors=40,
    n_components=3,
    min_dist=0.1,
    n_jobs=-1,
    metric='correlation',
)

# Selections smaller than this are skipped. Below n_neighbors, UMAP silently
# truncates the neighbourhood to the sample count and every point becomes every
# other point's neighbour, so the layout stops reflecting the data. A block with
# no correct rewarded trials selects nothing for its _cr embedding and lands
# here too.
MIN_BINS = 5 * UMAP_PARAMS['n_neighbors']


def file_digest(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_spikes():
    """(n_bins, n_units) from the units-by-bins CSV, plus that file's digest.

    embedding_and_labels.m drops zero-variance units before writing, so the file is already
    finite and needs no filtering here. The check below is a guard, not a fix:
    UMAP's correlation metric cannot take a NaN, and failing loudly beats an
    embedding quietly built on garbage.
    """
    path = LABEL_DIR / 'binned_spikes.csv'
    if not path.exists():
        raise SystemExit(f'{path} not found — run embedding_and_labels.m first')

    spikes = pd.read_csv(path, header=None).to_numpy().T   # (n_bins, n_units)

    bad = ~np.isfinite(spikes).all(axis=0)
    if bad.any():
        raise SystemExit(
            f'{path}: {int(bad.sum())} of {bad.size} units are non-finite. '
            'embedding_and_labels.m is meant to drop these — re-run it to refresh the export'
        )

    return spikes, file_digest(path)


def load_column(name, n_bins):
    """One per-bin CSV as a flat array, checked against the spikes' bin count."""
    path = LABEL_DIR / name
    if not path.exists():
        raise SystemExit(f'{path} not found — run embedding_and_labels.m first')

    column = pd.read_csv(path, header=None).to_numpy().ravel()

    if column.shape[0] != n_bins:
        raise SystemExit(
            f'{path.name}: {column.shape[0]} rows but binned_spikes.csv has {n_bins} '
            'bins — the two CSVs are from different runs of embedding_and_labels.m'
        )

    return column


def mask_digest(mask):
    """A digest of exactly which bins a mask selects.

    Taken over the mask written out one 0 or 1 per line with Windows line
    endings -- byte for byte what the per-block mask files embedding_and_labels.m
    used to write held. A block's correct rewarded selection therefore keys the
    cache exactly as its old mask file did, and the fits made from those files
    are found again rather than refitted. That matters because UMAP is not
    seeded: a refit would give the block a new layout, and the camera tuned for
    it in parameters.yaml would no longer suit.
    """
    return hashlib.sha256(b''.join(np.where(mask, b'1\r\n', b'0\r\n'))).hexdigest()


def selections(n_bins):
    """(output name, row mask) for every masked embedding, from the label columns.

    Block N's correct rewarded bins are correct_rewarded & block_id == N, and
    the whole of block N is block_id == N. Every bin has a block -- the gaps
    between trials take the block of the trial before them -- but only bins
    inside a correct rewarded trial are correct_rewarded, so the _cr selections
    hold no gap bins and the _all ones do.
    """
    correct_rewarded = (load_column('correct.csv', n_bins).astype(bool)
                        & load_column('rewarded.csv', n_bins).astype(bool))
    block_id = load_column('block_id.csv', n_bins)

    found = [('correct_rewarded', correct_rewarded)]
    for block in np.unique(block_id[np.isfinite(block_id)]).astype(int):
        in_block = block_id == block
        found.append((f'block_{block}_cr', correct_rewarded & in_block))
        found.append((f'block_{block}_all', in_block))
    return found


def embedding_for(rows, key_source):
    """Fit UMAP on these rows, or reuse a cached fit of the same inputs.

    The key covers the spikes file's contents, which rows were selected (the
    mask's digest, or the word 'full'), and every UMAP parameter. A re-export,
    a different selection, or an edited parameter therefore all miss the cache
    rather than quietly returning a stale or unrelated embedding. Keys are
    content-addressed, so selections never collide with each other or with any
    other script sharing .cache/.
    """
    key = hashlib.sha256(
        (key_source + repr(sorted(UMAP_PARAMS.items()))).encode()
    ).hexdigest()[:16]
    cache_path = CACHE_DIR / f'umap_{key}.npy'

    if cache_path.exists():
        print(f'  reusing cached embedding: {cache_path}')
        return np.load(cache_path)

    print(f'  no cached embedding for these inputs — fitting UMAP on {rows.shape[0]} bins...')
    embedding = umap.UMAP(**UMAP_PARAMS).fit_transform(rows)

    CACHE_DIR.mkdir(exist_ok=True)
    np.save(cache_path, embedding)
    print(f'  cached embedding: {cache_path}')
    return embedding


spikes, spikes_digest = load_spikes()
n_bins, n_units = spikes.shape
print(f'{n_bins} bins x {n_units} units')

# (output name, row selector, the part of the cache key that identifies the rows)
targets = [('full', np.ones(n_bins, dtype=bool), 'full')]
targets += [(name, mask, mask_digest(mask)) for name, mask in selections(n_bins)]

OUT_DIR.mkdir(parents=True, exist_ok=True)

for name, mask, selector in targets:
    print(f'{name}:')
    n_selected = int(mask.sum())

    if n_selected < MIN_BINS:
        print(f'  only {n_selected} bins — skipping, need at least {MIN_BINS}')
        continue

    embedding = embedding_for(spikes[mask], spikes_digest + selector)

    out_path = OUT_DIR / f'umap_{name}.npy'
    np.save(out_path, embedding)
    print(f'  wrote {out_path}  {embedding.shape}')
