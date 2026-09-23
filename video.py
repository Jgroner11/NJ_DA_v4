"""Twelve-panel video: a 4x3 grid, every panel live.

Twelve panels, four across and three down, numbered row-major from 1 in the top
left. Odd columns hold the whole session and even columns the selected patch, so
each pair sits side by side and the two are read together:

     1 full 3D     2 patch 3D     3 full xy     4 patch xy      plain, with trail
     5 full 3D     6 patch 3D     7 full xy     8 patch xy      by reward time
     9 full 3D    10 patch 3D    11 maze       12 info          by port

Only the plain row carries the moving trail. The coloured panels are static,
because their colorbars and legends shift the plot area, so the fitted pixel
maps describe the plain panels' layout rather than theirs.

Every panel is the same square, PANEL pixels a side, which means the maze is now
scaled down to fit one. Its tracking coordinates are therefore no longer pixel
positions in the panel and have to be mapped through the same resize -- see
maze_geometry.

The trailed panels each draw the current bin plus the previous TRAIL_S seconds
of bins, so the same moment is picked out in the maze and in every embedding at
once. Age shows twice over: the dot cools along a colour scale and fades in
opacity at the same time.

The window comes from start_time_s / duration_s / fps in parameters.yaml.

Every panel is a cached background with dots painted on in numpy, so no frame
re-renders anything through plotly. Re-rendering the embeddings per frame was
measured at about 11 s a frame, nearly all of it kaleido's fixed per-call cost,
which worked out at five hours for a minute of video.

Painting on a cached background needs to know where a 3D embedding point lands
in the rendered image. umap_plots.fit_projection works that out by rendering one
extra figure of known points and solving for the camera matrix plotly used; see
it for why this is measured rather than derived.

Nothing here reads parameters.yaml or runs on import. main.py loads the file and
calls the three entry points at the foot of this module, in order:

    load_session             read the per-bin data and render every background
    write_interactive_plots  save each embedding as an orbitable html plot
    write_video              paint the trail onto those backgrounds, frame by frame

Needs kaleido (plotly's static image export) and imageio-ffmpeg (the encoder):

    pip install kaleido imageio-ffmpeg
"""

import os
import re
import threading
from dataclasses import dataclass, fields
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import plotly.colors as pc
from PIL import Image, ImageDraw, ImageFont, ImageOps

import umap_plots as plots
from umap_plots import BACKGROUND

DATA_DIR = Path('data') / 'alternation'
EMB_DIR = DATA_DIR / 'embeddings'
MAZE_PNG = Path('NPC4_2026_05_17_17_10_50_blackout.png')
PLOT_DIR = Path('figures')
OUT_PATH = PLOT_DIR / 'patch_video.mp4'

COLUMNS = 4                                      # panels across
ROWS = 3                                         # and down
PANEL = 480                                      # side of one panel, in pixels

PLACEHOLDER_BG = '#ffffff'                       # an empty panel
PLACEHOLDER_INK = '#9a9892'                      # and the number written on it
PLACEHOLDER_EDGE = '#e1e0d9'                     # a hairline so the grid is visible
INFO_INK = '#52514e'                             # values on the information panel
INFO_LABEL = '#9a9892'                           # and the labels beside them
INFO_X = PANEL // 12                             # its left margin
INFO_VALUE_X = PANEL // 2                        # where the values line up
INFO_TOP = PANEL // 3                            # the first labelled row
INFO_STEP = PANEL // 8                           # and the gap to the next
REWARD_DISPENSING_S = 2.0                        # how long the info panel shows a reward's size for
TRAIL_SCALE = 'Turbo'                            # newest dot hot, oldest cold
TRAIL_HOT = 0.85                                 # where on the scale the newest dot sits
TRAIL_COLD = 0.20                                # and the oldest
DOT_RADIUS = 3                                   # trail dot radius on the maze, in pixels
UMAP_DOT_RADIUS = 4                             # and on the embedding panels
TRAIL_S = 6.0                                    # seconds of trail drawn behind the mouse
MIN_ALPHA = 0.1                                  # opacity of its oldest dot
CAMERA_ZOOM = 0.7                                # below 1 pulls the viewer in
DEFAULT_EYE = dict(x=1.25, y=1.25, z=1.25)       # plotly's own default 3D eye

FULL_TITLE = 'Full session'
XY = (0, 1)                                      # the dimensions a flat view keeps


# --------------------------------------------------------------------------
# geometry and images
# --------------------------------------------------------------------------

def read_camera(cfg, key):
    """One panel's fixed camera position, as plotly wants it.

    CAMERA_ZOOM scales the eye without turning it. Plotly's eye is a position
    rather than a direction, so shortening it walks the viewer towards the
    cloud and the cloud fills more of the panel, while the angle chosen in the
    interactive plot survives untouched.

    Falls back to plotly's own default eye if parameters.yaml has no entry
    under `key` yet -- a new patch has nothing hand-tuned for it the first
    time it is rendered, and this is what lets that first render happen at
    all, so its own umap_patch_N_uncolored.html can be orbited afterward to
    find a real angle and add it under `key`.
    """
    if key not in cfg:
        print(f'no {key} in parameters.yaml; using the default view')
    eye = cfg.get(key, DEFAULT_EYE)
    return dict(eye=dict(x=eye['x'] * CAMERA_ZOOM,
                         y=eye['y'] * CAMERA_ZOOM,
                         z=eye['z'] * CAMERA_ZOOM))


def disc_offsets(radius):
    """Row and column offsets of a filled circle, so a dot stamps in one go."""
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    inside = dy**2 + dx**2 <= radius**2
    return dy[inside], dx[inside]


def maze_geometry():
    """How the blackout frame is laid into a panel: size on screen, and offset.

    Mirrors what ImageOps.pad does -- scale to fit preserving the aspect, then
    centre -- because maze_pixels has to reproduce it exactly to put dots in the
    right place. Written out rather than assumed so the two cannot drift: the
    rounding here is PIL's own, and // 2 in place of round() would sit a pixel
    off for some panel sizes.
    """
    width, height = Image.open(MAZE_PNG).size

    if width > height:                        # the panel is square, so this is the test
        new_width, new_height = PANEL, round(height / width * PANEL)
    else:
        new_width, new_height = round(width / height * PANEL), PANEL

    return (width, height, new_width, new_height,
            round((PANEL - new_width) * 0.5), round((PANEL - new_height) * 0.5))


def maze_pixels(positions):
    """Tracking coordinates mapped into the scaled-down maze panel.

    head_positions.csv is in pixels of the full-size blackout frame, and the
    panel holds a copy scaled to fit and centred, so the coordinates need the
    same treatment. The half-pixel terms are the gap between a pixel's corner
    and its centre: a resize maps the centre of source pixel x to
    (x + 0.5) * scale - 0.5. Checked against PIL's own output, this lands within
    0.02 px, where dropping those terms is out by up to half a pixel.
    """
    width, height, new_width, new_height, offset_x, offset_y = maze_geometry()

    x = (positions[:, 0] + 0.5) * (new_width / width) - 0.5 + offset_x
    y = (positions[:, 1] + 0.5) * (new_height / height) - 0.5 + offset_y
    return np.column_stack([x, y])


def maze_image():
    """The blackout frame, scaled to fit a panel and centred on it."""
    image = Image.open(MAZE_PNG).convert('RGB')
    return np.asarray(ImageOps.pad(image, (PANEL, PANEL), color=BACKGROUND))


def placeholder_font(size):
    """A font for the placeholder numbers, whatever this machine happens to have."""
    for name in ('arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fitted_font(text, max_width, size):
    """The largest font at or below `size` that keeps `text` inside max_width.

    The session filename is the one string here whose length is not known in
    advance, and a longer one would otherwise run off the panel.
    """
    while size > 8:
        font = placeholder_font(size)
        if font.getlength(text) <= max_width:
            return font
        size -= 1
    return placeholder_font(8)


def blank_panel(number):
    """An empty panel: white, its number in the middle, a hairline round the edge.

    The border is not decoration -- without it thirteen white squares merge into
    one white area and the grid it is meant to show cannot be seen.
    """
    image = Image.new('RGB', (PANEL, PANEL), PLACEHOLDER_BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, PANEL - 1, PANEL - 1], outline=PLACEHOLDER_EDGE)

    text = f'panel {number}'
    font = placeholder_font(PANEL // 12)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(((PANEL - (right - left)) / 2 - left,
               (PANEL - (bottom - top)) / 2 - top),
              text, fill=PLACEHOLDER_INK, font=font)

    return np.asarray(image)


def placeholder(number):
    """A panel function for a slot with no plot in it yet.

    The image is built once and handed back unchanged every frame, so an empty
    panel costs nothing per frame beyond being copied into the grid.
    """
    image = blank_panel(number)
    return lambda time_s: image


def static_panel(image):
    """A panel function for a rendering that does not move between frames.

    The coloured and port views are properties of the bins rather than of the
    frame, so the rendered image goes back unchanged, the way a placeholder's
    does.

    None of them carries a trail. Their colorbars and legends shrink the plot
    area relative to the plain panels, so the two do not share a projection:
    painting with a plain panel's pixel map here would put the dots in visibly
    the wrong place. Giving one a trail would mean fitting a projection against
    that figure's own layout, which fit_projection does not currently model.
    """
    return lambda time_s: image


def trail_ramp(n_bins):
    """One colour per bin of the trail, hottest first, as rows of r, g, b.

    Trimmed to TRAIL_HOT..TRAIL_COLD rather than run end to end, because Turbo is
    nearly black at both ends: untrimmed, the newest dot and the oldest would
    both disappear into the maze photo. The middle of the scale stays saturated
    throughout, which is what lets one ramp work on the dark maze and the pale
    embedding panels at once.
    """
    fractions = np.linspace(TRAIL_HOT, TRAIL_COLD, n_bins)
    sampled = pc.sample_colorscale(pc.get_colorscale(TRAIL_SCALE), fractions,
                                   colortype='rgb')
    return np.array([[float(v) for v in colour[4:-1].split(',')] for colour in sampled])


# --------------------------------------------------------------------------
# per-bin data
# --------------------------------------------------------------------------

@dataclass
class Bins:
    """Everything recorded per time bin, and what the trail is drawn from.

    full_row and patch_row give the row of each embedding a bin corresponds to.
    For the full session that is the bin index itself; for a patch it is the
    bin's rank among the mask's set bins, and -1 for bins the patch does not
    contain.
    """
    times: np.ndarray
    trial_ids: np.ndarray                        # NaN between trials
    time_nearest_reward: np.ndarray
    reward_size_ms: np.ndarray                   # size of whichever reward that is, NaN with it
    port_ids: np.ndarray                         # 0 where not at a port
    head_xy: np.ndarray
    patch_mask: np.ndarray
    full_row: np.ndarray
    patch_row: np.ndarray
    width: float                                 # seconds in one bin
    trail_bins: int                              # how many of them the trail spans
    trail_rgb: np.ndarray                        # and its colour, newest first


def patch_spans():
    """When each patch runs, as (patch, first, last) in session seconds.

    A patch is a run of trials sharing a CorrectBlock, so the patches tile the
    session in order and never overlap -- which is what makes a span enough to
    say which one a given moment belongs to. A patch whose trials were all
    dropped in selection has an all-false mask and no span, so it is left out
    here rather than handed back empty.

    The span is measured off the mask, and the mask holds only correct rewarded
    bins, so the true patch starts a little before `first` and ends a little
    after `last` -- by however much of its first and last trials were dropped.
    Close enough to choose which embedding to render; not close enough to call
    a boundary. The exact per-bin answer is patch_id_per_bin in
    embedding_and_labels.m, which is computed there but not currently exported.
    """
    times = np.loadtxt(DATA_DIR / 'bin_times.csv')

    spans = []
    for path in sorted(DATA_DIR.glob('patch_mask_*.csv'),
                       key=lambda p: int(p.stem.rsplit('_', 1)[1])):
        patch = int(path.stem.rsplit('_', 1)[1])
        bins = np.flatnonzero(np.loadtxt(path).astype(bool))
        if bins.size:
            spans.append((patch, times[bins[0]], times[bins[-1]]))

    return spans


def load_bins(patch):
    """Read the per-bin csvs this patch's video needs, and size the trail."""
    times = np.loadtxt(DATA_DIR / 'bin_times.csv')
    patch_mask = np.loadtxt(DATA_DIR / f'patch_mask_{patch}.csv').astype(bool)

    width = times[1] - times[0]
    trail_bins = round(TRAIL_S / width)

    return Bins(
        times=times,
        trial_ids=np.loadtxt(DATA_DIR / 'trial_ids.csv'),
        time_nearest_reward=np.loadtxt(DATA_DIR / 'time_nearest_reward.csv'),
        reward_size_ms=np.loadtxt(DATA_DIR / 'reward_size_ms.csv'),
        port_ids=np.loadtxt(DATA_DIR / 'port_ids.csv'),
        head_xy=maze_pixels(np.loadtxt(DATA_DIR / 'head_positions.csv', delimiter=',')),
        patch_mask=patch_mask,
        full_row=np.arange(len(times)),
        patch_row=np.where(patch_mask, np.cumsum(patch_mask) - 1, -1),
        width=width,
        trail_bins=trail_bins,
        trail_rgb=trail_ramp(trail_bins))


def bin_at(bins, time_s):
    """Index of the bin covering this frame time.

    A binary search against the bin centres rather than arithmetic on the frame
    time: at 30 fps with 100 ms bins every third frame lands exactly on a bin
    boundary, and there time_s / bins.width picks a side on floating-point error
    alone, so neighbouring frames can jump back and forth by a bin.
    """
    return min(int(np.searchsorted(bins.times, time_s)), len(bins.times) - 1)


def trail(bins, time_s):
    """Bins within TRAIL_S of this frame, oldest first, with opacity and colour.

    Shared by every trailed panel so they always highlight the same bins. Near
    the start of the session there are fewer than bins.trail_bins bins behind
    the current one, so the trail grows in rather than starting full.
    """
    newest = bin_at(bins, time_s)
    indices = np.arange(max(newest - bins.trail_bins + 1, 0), newest + 1)

    ages = newest - indices
    alphas = 1.0 - (1.0 - MIN_ALPHA) * ages / bins.trail_bins
    return indices, alphas, bins.trail_rgb[ages]


# --------------------------------------------------------------------------
# rendered embeddings
# --------------------------------------------------------------------------

@dataclass
class Layer:
    """One embedding as this video uses it.

    `figure` is written out as an interactive plot; `background` is that figure
    rasterised once, to be copied per frame. `pixels` says where each point of
    the cloud landed in that rasterisation, and is None for the views that carry
    no trail -- see static_panel for why those cannot borrow a plain view's map.
    `camera_key` is the parameters.yaml entry a 3D view's readout reports, and
    None for a flat projection, which has no camera to report.
    """
    figure: object
    background: np.ndarray
    pixels: np.ndarray = None
    name: str = ''                               # stem of its html file
    camera_key: str = None


@dataclass
class Layers:
    """Every rendered view, in the order the panels read them."""
    full: Layer
    patch: Layer
    full_coloured: Layer
    patch_coloured: Layer
    full_xy: Layer
    patch_xy: Layer
    full_xy_coloured: Layer
    patch_xy_coloured: Layer
    full_ports: Layer
    patch_ports: Layer

    def all(self):
        """Each layer once, in declaration order."""
        return [getattr(self, f.name) for f in fields(self)]


def report_labels(labels, points, what):
    """Check the reward labels pair with the cloud, and say how many are coloured."""
    assert len(labels) == len(points), \
        f'{len(labels)} labels but {len(points)} embedded bins'
    print(f'{int((np.abs(labels) <= plots.TIME_RADIUS).sum())} of {len(labels)} '
          f'{what} bins within {plots.TIME_RADIUS:g} s of a reward')


def render_layers(bins, cfg):
    """Every embedding rendered once, with the pixel map its trail needs.

    These are the only kaleido renders in the whole run: a background is built
    here and every frame then paints onto a copy of it.

    The coloured views pair each bin with how far it sits from a reward. Those
    labels were written for every bin in the session, so a patch's own come out
    under the same mask as its spikes, while the full session's pair with the
    full embedding directly.
    """
    patch = cfg['patch']
    patch_title = f'Patch {patch}'
    patch_camera_key = f'patch_{patch}_camera'
    full_camera = read_camera(cfg, 'full_camera')
    patch_camera = read_camera(cfg, patch_camera_key)

    full_points = np.load(EMB_DIR / 'umap_full.npy')
    patch_points = np.load(EMB_DIR / f'umap_patch_{patch}.npy')

    full_figure, full_background, full_pixels = plots.panel_and_pixels(
        full_points, FULL_TITLE, full_camera, PANEL)
    patch_figure, patch_background, patch_pixels = plots.panel_and_pixels(
        patch_points, patch_title, patch_camera, PANEL)

    full_labels = bins.time_nearest_reward
    patch_labels = bins.time_nearest_reward[bins.patch_mask]
    report_labels(patch_labels, patch_points, 'patch')
    report_labels(full_labels, full_points, 'session')

    # Same cameras and the same hidden axes as the plain views, so the two are
    # comparable at a glance and only the colour differs. Expect a far paler
    # picture from the session than from the patch -- a patch is correct
    # rewarded trials, where almost every bin sits near a reward, while most of
    # a session does not.
    full_coloured = plots.coloured_figure(full_points, full_labels, FULL_TITLE,
                                          full_camera, PANEL)
    patch_coloured = plots.coloured_figure(patch_points, patch_labels, patch_title,
                                           patch_camera, PANEL)

    # Flat views of the same two clouds, down the same pair of dimensions, so
    # the patch and the session are read the same way. The plain ones also get
    # the pixel map their trail needs: a flat view has no camera, but it still
    # has to be told where a point landed -- see umap_plots.fit_flat_projection.
    full_xy_title = f'{FULL_TITLE} - xy projection'
    patch_xy_title = f'{patch_title} - xy projection'

    full_xy, full_xy_background, full_xy_pixels = plots.flat_panel_and_pixels(
        full_points, XY, full_xy_title, PANEL)
    patch_xy, patch_xy_background, patch_xy_pixels = plots.flat_panel_and_pixels(
        patch_points, XY, patch_xy_title, PANEL)

    full_xy_coloured = plots.coloured_projection(full_points, full_labels, XY,
                                                 full_xy_title, PANEL)
    patch_xy_coloured = plots.coloured_projection(patch_points, patch_labels, XY,
                                                  patch_xy_title, PANEL)

    # Both clouds again, coloured by which port the mouse was at. The labels are
    # per bin and spatial -- nearest port centre within a few pixels of the
    # tracked position -- so unlike the per-trial reward labels they mark only
    # the moments actually spent at a port.
    patch_ports = bins.port_ids[bins.patch_mask]
    full_port_figure = plots.port_figure(full_points, bins.port_ids, FULL_TITLE,
                                         full_camera, PANEL)
    patch_port_figure = plots.port_figure(patch_points, patch_ports, patch_title,
                                          patch_camera, PANEL)

    print(f'{int((patch_ports > 0).sum())} of {len(patch_ports)} patch bins at a port; '
          f'{int((bins.port_ids > 0).sum())} of {len(bins.port_ids)} session bins')

    # The coloured and port figures are rendered as well as written out, because
    # they are panels now. One more kaleido render each on a cold cache, nothing
    # on a warm one.
    return Layers(
        full=Layer(full_figure, full_background, full_pixels,
                   'umap_full', 'full_camera'),
        patch=Layer(patch_figure, patch_background, patch_pixels,
                    f'umap_patch_{patch}_uncolored', patch_camera_key),
        full_coloured=Layer(full_coloured,
                            plots.figure_image(full_coloured, PANEL),
                            name='umap_full_colored', camera_key='full_camera'),
        patch_coloured=Layer(patch_coloured,
                             plots.figure_image(patch_coloured, PANEL),
                             name=f'umap_patch_{patch}_colored',
                             camera_key=patch_camera_key),
        full_xy=Layer(full_xy, full_xy_background, full_xy_pixels,
                      'umap_full_xy_uncolored'),
        patch_xy=Layer(patch_xy, patch_xy_background, patch_xy_pixels,
                       f'umap_patch_{patch}_xy_uncolored'),
        full_xy_coloured=Layer(full_xy_coloured,
                               plots.figure_image(full_xy_coloured, PANEL),
                               name='umap_full_xy_colored'),
        patch_xy_coloured=Layer(patch_xy_coloured,
                                plots.figure_image(patch_xy_coloured, PANEL),
                                name=f'umap_patch_{patch}_xy_colored'),
        full_ports=Layer(full_port_figure,
                         plots.figure_image(full_port_figure, PANEL),
                         name='umap_full_ports'),
        patch_ports=Layer(patch_port_figure,
                          plots.figure_image(patch_port_figure, PANEL),
                          name=f'umap_patch_{patch}_ports'))


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------
#
# Each builder below does its one-off work when it is called and hands back a
# function of the frame time, so the per-frame path holds nothing but the
# painting itself.

def maze_panel(bins):
    """Panel 11: the maze, with the mouse's position and its recent trail.

    Drawn on a copy, so the trail lasts one frame instead of accumulating, and
    oldest first so the current position sits on top where dots overlap. Bins
    with no tracked position are skipped, leaving a gap in the trail.
    """
    maze = maze_image()
    dot_dy, dot_dx = disc_offsets(DOT_RADIUS)

    def draw(time_s):
        frame = maze.copy()

        for index, alpha, colour in zip(*trail(bins, time_s)):
            x, y = bins.head_xy[index]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            rows = np.clip(round(y) + dot_dy, 0, PANEL - 1)
            cols = np.clip(round(x) + dot_dx, 0, PANEL - 1)
            # rounded, not truncated: assigning the float blend straight into a
            # uint8 frame would round every dot down and darken the trail
            frame[rows, cols] = np.round(frame[rows, cols] * (1 - alpha)
                                         + colour * alpha)

        return frame

    return draw


def embedding_panel(layer, bins, bin_to_row):
    """One embedding panel: the cached rendering with this frame's trail on it.

    The same blend the maze panel uses, at the pixels the embedding points were
    projected to. Painting on top means a trail dot geometrically behind the
    cloud still shows, where plotly would have hidden it -- which is what you
    want from a marker you are trying to follow.
    """
    dot_dy, dot_dx = disc_offsets(UMAP_DOT_RADIUS)

    def draw(time_s):
        frame = layer.background.copy()

        indices, alphas, colours = trail(bins, time_s)
        rows = bin_to_row[indices]
        present = rows >= 0          # a trail bin outside this patch has no point

        for row, alpha, colour in zip(rows[present], alphas[present], colours[present]):
            x, y = layer.pixels[row]
            dots = np.clip(round(y) + dot_dy, 0, PANEL - 1)
            cols = np.clip(round(x) + dot_dx, 0, PANEL - 1)
            frame[dots, cols] = np.round(frame[dots, cols] * (1 - alpha)
                                         + colour * alpha)

        return frame

    return draw


def info_panel(bins, data_file, patch):
    """Panel 12: in words, what the other panels are showing at this instant.

    The session name and the label beside each row are baked into the background
    once, so a frame only has to add the three values -- which is what keeps
    this panel to about a millisecond, the whole reason it is PIL text rather
    than a plotly figure.

    "included" is read straight off the patch mask, so it answers exactly the
    question the patch panel poses -- is this bin one of the ones plotted there
    -- rather than the looser question of whether the trial belongs to the patch
    at all. A bin between trials has no trial number.

    "reward size" reads bins.time_nearest_reward, which is signed (negative
    before a reward, positive after -- see embedding_and_labels.m), so 0 marks
    reward onset and the window checked is [0, REWARD_DISPENSING_S]. Outside
    that window, or where time_nearest_reward is NaN (no reward on either side
    of this bin), it reads None rather than bins.reward_size_ms -- that field
    is paired with time_nearest_reward but not itself windowed, so reading it
    unconditionally would show a reward's size long before or after it was
    actually dispensed.
    """
    labels = ['time', 'trial', f'patch {patch}', 'reward size']
    font = placeholder_font(PANEL // 18)

    background = Image.new('RGB', (PANEL, PANEL), PLACEHOLDER_BG)
    fixed = ImageDraw.Draw(background)
    fixed.rectangle([0, 0, PANEL - 1, PANEL - 1], outline=PLACEHOLDER_EDGE)

    heading = placeholder_font(PANEL // 26)
    fixed.text((INFO_X, INFO_X), 'session', font=heading, fill=INFO_LABEL)

    width = PANEL - 2 * INFO_X
    fixed.text((INFO_X, INFO_X + PANEL // 20), data_file,
               font=fitted_font(data_file, width, PANEL // 22), fill=INFO_INK)

    for row, label in enumerate(labels):
        fixed.text((INFO_X, INFO_TOP + row * INFO_STEP), label,
                   font=font, fill=INFO_LABEL)

    def draw(time_s):
        image = background.copy()
        pen = ImageDraw.Draw(image)

        index = bin_at(bins, time_s)
        trial = bins.trial_ids[index]

        reward_offset = bins.time_nearest_reward[index]
        dispensing = 0 <= reward_offset <= REWARD_DISPENSING_S   # NaN compares False

        values = [f'{bins.times[index]:.2f} s',
                  '-' if np.isnan(trial) else f'{int(trial)}',
                  'included' if bins.patch_mask[index] else 'excluded',
                  f'{int(bins.reward_size_ms[index])} ms' if dispensing else 'None']

        for row, value in enumerate(values):
            pen.text((INFO_VALUE_X, INFO_TOP + row * INFO_STEP), value,
                     font=font, fill=INFO_INK)

        return np.asarray(image)

    return draw


def build_panels(bins, layers, data_file, patch):
    """The grid, row-major from 1 in the top left, as functions of the frame time.

    The placeholder fallback goes unused while all twelve are filled; it is what
    a thirteenth panel would fall back to if ROWS grew.
    """
    live = {
        1: embedding_panel(layers.full, bins, bins.full_row),
        2: embedding_panel(layers.patch, bins, bins.patch_row),
        3: embedding_panel(layers.full_xy, bins, bins.full_row),
        4: embedding_panel(layers.patch_xy, bins, bins.patch_row),
        5: static_panel(layers.full_coloured.background),
        6: static_panel(layers.patch_coloured.background),
        7: static_panel(layers.full_xy_coloured.background),
        8: static_panel(layers.patch_xy_coloured.background),
        9: static_panel(layers.full_ports.background),
        10: static_panel(layers.patch_ports.background),
        11: maze_panel(bins),
        12: info_panel(bins, data_file, patch),
    }
    return [live[n] if n in live else placeholder(n)
            for n in range(1, COLUMNS * ROWS + 1)]


def compose(panels, time_s):
    """One video frame: every panel drawn, then tiled into the grid."""
    drawn = [panel(time_s) for panel in panels]
    rows = [np.hstack(drawn[row * COLUMNS:(row + 1) * COLUMNS])
            for row in range(ROWS)]
    return np.vstack(rows)


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

_BENIGN_FFMPEG_WARNING = re.compile(rb'Truncating packet of size \d+ to \d+')


class _SuppressBenignFfmpegWarning:
    """Hides one specific, harmless ffmpeg warning from the console.

    Every run logs "Truncating packet of size ... to ...": ffmpeg's rawvideo
    demuxer probes the first frame before our pipe has delivered all of it --
    a pipe can't be rewound the way a file can, which is why this never
    happens with a file input -- and warns about the short read. Checked with
    a frame carrying a row-by-row marker pattern that the probe's packet is
    still queued and decoded correctly: the encoded video is byte-for-byte
    what it would be without the warning, so it is filtered here rather than
    investigated as a real bug.

    imageio hands ffmpeg our actual stderr file descriptor with no hook to
    intercept it, so this swaps fd 2 for a pipe for the duration of the
    block and relays everything else straight through unfiltered.
    """

    def __enter__(self):
        self._saved_fd = os.dup(2)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 2)
        os.close(write_fd)
        self._reader = os.fdopen(read_fd, 'rb')
        self._thread = threading.Thread(target=self._relay, daemon=True)
        self._thread.start()
        return self

    def _relay(self):
        passthrough = os.fdopen(os.dup(self._saved_fd), 'wb')
        for line in self._reader:
            if not _BENIGN_FFMPEG_WARNING.search(line):
                passthrough.write(line)
                passthrough.flush()
        passthrough.close()

    def __exit__(self, *exc_info):
        os.dup2(self._saved_fd, 2)
        os.close(self._saved_fd)
        self._reader.close()
        self._thread.join(timeout=5)


@dataclass
class Session:
    """One run's worth of loaded data and rendered backgrounds."""
    cfg: dict
    bins: Bins
    layers: Layers
    panels: list


def frame_size():
    """The video's pixel dimensions, checked against what the encoder accepts."""
    width, height = COLUMNS * PANEL, ROWS * PANEL
    assert width % 2 == 0 and height % 2 == 0, \
        f'frame is {width}x{height}; libx264 needs even dimensions'
    return width, height


def load_session(params):
    """Everything a frame needs: the per-bin csvs read, every background rendered.

    The expensive half of a run, and the only half that touches plotly. What it
    hands back is enough for write_video to work in numpy alone.
    """
    cfg = params['vid']

    width, height = frame_size()
    print(f'frame {width}x{height}: {COLUMNS}x{ROWS} panels of {PANEL}x{PANEL}')

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    bins = load_bins(cfg['patch'])
    layers = render_layers(bins, cfg)
    panels = build_panels(bins, layers, params['data_file'], cfg['patch'])

    return Session(cfg=cfg, bins=bins, layers=layers, panels=panels)


def write_interactive_plots(session):
    """Every rendered embedding saved as an html plot beside the video.

    Orbit a 3D one by hand and the readout in its corner names the camera
    position to paste into parameters.yaml.
    """
    for layer in session.layers.all():
        path = PLOT_DIR / f'{layer.name}.html'
        print(f'wrote {plots.write_html(layer.figure, path, layer.camera_key, CAMERA_ZOOM)}')


def write_video(session, out_path=OUT_PATH):
    """Paint the trail onto the cached backgrounds, frame by frame, and encode."""
    cfg, bins = session.cfg, session.bins
    start, duration, fps = cfg['start_time_s'], cfg['duration_s'], cfg['fps']

    in_window = (bins.times >= start) & (bins.times < start + duration)
    print(f'{in_window.sum()} bins in the {duration:g} s window from {start:g} s')

    n_frames = round(duration * fps)

    # macro_block_size=1 keeps the frame at exactly the grid's size; the default
    # of 16 silently rescales to the next multiple of 16.
    with _SuppressBenignFfmpegWarning():
        with imageio.get_writer(out_path, fps=fps, codec='libx264', quality=8,
                                macro_block_size=1) as writer:
            for i in range(n_frames):
                writer.append_data(compose(session.panels, start + i / fps))

    print(f'wrote {out_path}  ({n_frames} frames at {fps} fps, {n_frames / fps:.1f} s)')
    return out_path
