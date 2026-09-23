"""UMAP embeddings of the binned session — the whole session and each patch.

Reads the CSVs written by embedding_and_labels.m and writes one .npy per embedding into
data/embeddings/. No plotting: this only produces the embeddings.

    umap_full.npy       (n_bins, n_components)      every bin, in file order
    umap_patch_<N>.npy  (n_kept, n_components)      the bins patch_mask_<N> selects

A patch embedding's row i is the i-th True entry of that patch's mask, so the
matching times and positions are bin_times.csv[mask] and head_positions.csv[mask].

Fits are cached by content, so re-running does not refit anything whose inputs
and parameters are unchanged. UMAP is not seeded here, so the cache is also what
keeps an embedding stable from run to run — delete .cache/ to force fresh fits.
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import umap

LABEL_DIR = Path('data') / 'binned_labels'
OUT_DIR = Path('data') / 'embeddings'
CACHE_DIR = Path('.cache')

UMAP_PARAMS = dict(
    n_neighbors=40,
    n_components=3,
    min_dist=0.1,
    n_jobs=-1,
    metric='correlation',
)

# Patches smaller than this are skipped. Below n_neighbors, UMAP silently
# truncates the neighbourhood to the sample count and every point becomes every
# other point's neighbour, so the layout stops reflecting the data. Patches with
# no correct rewarded trials get an all-false mask and land here too.
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


def available_patches():
    """Patch numbers with a mask file present, read off the filenames."""
    found = []
    for path in sorted(LABEL_DIR.glob('patch_mask_*.csv')):
        match = re.fullmatch(r'patch_mask_(\d+)\.csv', path.name)
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def load_mask(patch, n_bins):
    """One patch's 0/1 column as a boolean over all bins, with the file's digest."""
    path = LABEL_DIR / f'patch_mask_{patch}.csv'
    mask = pd.read_csv(path, header=None).to_numpy().ravel()

    if mask.shape[0] != n_bins:
        raise SystemExit(
            f'{path.name}: {mask.shape[0]} rows but binned_spikes.csv has {n_bins} '
            'bins — the two CSVs are from different runs of embedding_and_labels.m'
        )

    return mask.astype(bool), file_digest(path)


def embedding_for(rows, key_source):
    """Fit UMAP on these rows, or reuse a cached fit of the same inputs.

    The key covers the spikes file's contents, whatever selected the rows (a
    mask file's contents, or the word 'full'), and every UMAP parameter. A
    re-export, a different patch, or an edited parameter therefore all miss the
    cache rather than quietly returning a stale or unrelated embedding. Keys are
    content-addressed, so patches never collide with each other or with any
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

patches = available_patches()
if not patches:
    print(f'no patch_mask_*.csv in {LABEL_DIR} — embedding the full session only')

# (output name, row selector, the part of the cache key that identifies the rows)
targets = [('full', np.ones(n_bins, dtype=bool), 'full')]
for patch in patches:
    mask, mask_digest = load_mask(patch, n_bins)
    targets.append((f'patch_{patch}', mask, mask_digest))

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
