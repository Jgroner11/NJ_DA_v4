"""Twelve-panel video: a 4x3 grid, every panel live.

Twelve panels, four across and three down, numbered row-major from 1 in the top
left. Odd columns hold the whole session and even columns the selected block, so
each pair sits side by side and the two are read together:

     1 full 3D     2 block 3D     3 full xy     4 block xy      plain, with trail
     5 full 3D     6 block 3D     7 full xy     8 block xy      by reward time
     9 full 3D    10 block 3D    11 maze       12 info          by port

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

import imageio.v2 as imageio
import numpy as np
import plotly.colors as pc
from PIL import Image, ImageDraw, ImageFont, ImageOps

import umap_plots as plots
from paths import session_paths
from umap_plots import BACKGROUND

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
    under `key` yet -- a new block has nothing hand-tuned for it the first
    time it is rendered, and this is what lets that first render happen at
    all, so its own umap_block_N_cr_uncolored.html can be orbited afterward to
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


def maze_geometry(maze_png):
    """How the blackout frame is laid into a panel: size on screen, and offset.

    Mirrors what ImageOps.pad does -- scale to fit preserving the aspect, then
    centre -- because maze_pixels has to reproduce it exactly to put dots in the
    right place. Written out rather than assumed so the two cannot drift: the
    rounding here is PIL's own, and // 2 in place of round() would sit a pixel
    off for some panel sizes.
    """
    width, height = Image.open(maze_png).size

    if width > height:                        # the panel is square, so this is the test
        new_width, new_height = PANEL, round(height / width * PANEL)
    else:
        new_width, new_height = round(width / height * PANEL), PANEL

    return (width, height, new_width, new_height,
            round((PANEL - new_width) * 0.5), round((PANEL - new_height) * 0.5))


def maze_pixels(positions, maze_png):
    """Tracking coordinates mapped into the scaled-down maze panel.

    head_positions.csv is in pixels of the full-size blackout frame, and the
    panel holds a copy scaled to fit and centred, so the coordinates need the
    same treatment. The half-pixel terms are the gap between a pixel's corner
    and its centre: a resize maps the centre of source pixel x to
    (x + 0.5) * scale - 0.5. Checked against PIL's own output, this lands within
    0.02 px, where dropping those terms is out by up to half a pixel.
    """
    width, height, new_width, new_height, offset_x, offset_y = maze_geometry(maze_png)

    x = (positions[:, 0] + 0.5) * (new_width / width) - 0.5 + offset_x
    y = (positions[:, 1] + 0.5) * (new_height / height) - 0.5 + offset_y
    return np.column_stack([x, y])


def maze_image(maze_png):
    """The blackout frame, scaled to fit a panel and centred on it."""
    image = Image.open(maze_png).convert('RGB')
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

    full_row and block_row give the row of each embedding a bin corresponds to.
    For the full session that is the bin index itself; for a block it is the
    bin's rank among the mask's set bins, and -1 for bins the block does not
    contain.

    block_mask is the block's correct rewarded bins -- the ones its _cr
    embedding holds, and the only block embedding the video shows -- and
    block_all_mask is every bin of the block, gaps between trials included.
    """
    times: np.ndarray
    trial_ids: np.ndarray                        # NaN between trials
    time_nearest_reward: np.ndarray
    reward_size_ms: np.ndarray                   # size of whichever reward that is, NaN with it
    port_ids: np.ndarray                         # 0 where not at a port
    switch_stay: np.ndarray                      # 1 switch, 2 stay, 0 neither
    at_trial_port: np.ndarray                    # at its own trial's port
    patch_identified: np.ndarray                 # after its block's patch was identified
    correct: np.ndarray                          # in a correct trial
    rewarded: np.ndarray                         # in a rewarded trial
    head_xy: np.ndarray
    block_mask: np.ndarray
    block_all_mask: np.ndarray
    full_row: np.ndarray
    block_row: np.ndarray
    width: float                                 # seconds in one bin
    trail_bins: int                              # how many of them the trail spans
    trail_rgb: np.ndarray                        # and its colour, newest first


def load_correct_rewarded(paths):
    """Whether each bin is in a correct rewarded trial, from correct.csv and rewarded.csv."""
    return (np.loadtxt(paths.label_dir / 'correct.csv').astype(bool)
            & np.loadtxt(paths.label_dir / 'rewarded.csv').astype(bool))


def switch_stay_groups(bins):
    """Each bin's code in umap_plots.SWITCH_STAY_GROUPS, 0 for the pale underlay.

    Only bins where the mouse is at its own rewarded trial's port are coloured:
    switch and stay as MATLAB labelled them, then the remaining correct rewarded
    trials, split by whether their block's patch had been identified yet, then
    the incorrect rewarded ones. Unrewarded trials, bins away from the port and
    the gaps between trials are all 0.
    """
    at_reward = bins.at_trial_port & bins.rewarded
    groups = np.zeros(len(bins.times), dtype=int)
    groups[at_reward & bins.correct] = 3
    groups[at_reward & bins.correct & bins.patch_identified] = 5
    groups[at_reward & ~bins.correct] = 4
    groups[at_reward & (bins.switch_stay == 1)] = 1
    groups[at_reward & (bins.switch_stay == 2)] = 2
    return groups


def load_block_id(paths):
    """block_id.csv: the block each bin falls in, the gaps between trials included."""
    return np.loadtxt(paths.label_dir / 'block_id.csv')


def block_spans(paths):
    """When each block runs, as (block, first, last) in session seconds.

    A block is a run of trials sharing a CorrectBlock, and block_id gives every
    bin one -- the gaps between trials take the block of the trial before them
    -- so the spans tile the session in order and never overlap, which is what
    makes a span enough to say which block a given moment belongs to. They are
    exact: first and last are the block's own first and last bins.

    A block with no correct rewarded bins is left out. Its _cr embedding cannot
    be fitted, so there is nothing to render for it, and choosing it would only
    fail on loading an embedding that does not exist.
    """
    times = np.loadtxt(paths.label_dir / 'bin_times.csv')
    correct_rewarded = load_correct_rewarded(paths)
    block_id = load_block_id(paths)

    spans = []
    for block in np.unique(block_id[np.isfinite(block_id)]).astype(int):
        in_block = block_id == block
        if (in_block & correct_rewarded).any():
            bins = np.flatnonzero(in_block)
            spans.append((int(block), times[bins[0]], times[bins[-1]]))

    return spans


def load_bins(block, paths):
    """Read the per-bin csvs this block's video needs, and size the trail."""
    labels = paths.label_dir
    times = np.loadtxt(labels / 'bin_times.csv')
    block_all_mask = load_block_id(paths) == block
    block_mask = load_correct_rewarded(paths) & block_all_mask

    width = times[1] - times[0]
    trail_bins = round(TRAIL_S / width)

    return Bins(
        times=times,
        trial_ids=np.loadtxt(labels / 'trial_ids.csv'),
        time_nearest_reward=np.loadtxt(labels / 'time_nearest_reward.csv'),
        reward_size_ms=np.loadtxt(labels / 'reward_size_ms.csv'),
        port_ids=np.loadtxt(labels / 'port_ids.csv'),
        switch_stay=np.loadtxt(labels / 'switch_stay.csv'),
        at_trial_port=np.loadtxt(labels / 'at_trial_port.csv').astype(bool),
        patch_identified=np.loadtxt(labels / 'patch_identified.csv').astype(bool),
        correct=np.loadtxt(labels / 'correct.csv').astype(bool),
        rewarded=np.loadtxt(labels / 'rewarded.csv').astype(bool),
        head_xy=maze_pixels(np.loadtxt(labels / 'head_positions.csv', delimiter=','),
                            paths.maze_png),
        block_mask=block_mask,
        block_all_mask=block_all_mask,
        full_row=np.arange(len(times)),
        block_row=np.where(block_mask, np.cumsum(block_mask) - 1, -1),
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
    None for a flat projection, which has no camera to report. `background` is
    None for a view no panel shows, which is written out but never rasterised.
    """
    figure: object
    background: np.ndarray
    pixels: np.ndarray = None
    name: str = ''                               # stem of its html file
    camera_key: str = None


@dataclass
class Layers:
    """Every view: the rendered ones in the order the panels read them, then the
    ones only written out."""
    full: Layer
    block: Layer
    full_coloured: Layer
    block_coloured: Layer
    full_xy: Layer
    block_xy: Layer
    full_xy_coloured: Layer
    block_xy_coloured: Layer
    full_ports: Layer
    block_ports: Layer

    # Written out as interactive plots only; no panel reads these yet.
    correct_rewarded: Layer
    correct_rewarded_coloured: Layer
    correct_rewarded_ports: Layer
    block_all: Layer
    block_all_coloured: Layer
    block_all_ports: Layer
    full_switch_stay: Layer
    block_all_switch_stay: Layer

    def all(self):
        """Each layer once, in declaration order."""
        return [getattr(self, f.name) for f in fields(self)]


def report_labels(labels, points, what):
    """Check the reward labels pair with the cloud, and say how many are coloured."""
    assert len(labels) == len(points), \
        f'{len(labels)} labels but {len(points)} embedded bins'
    print(f'{int((np.abs(labels) <= plots.TIME_RADIUS).sum())} of {len(labels)} '
          f'{what} bins within {plots.TIME_RADIUS:g} s of a reward')


def render_layers(bins, cfg, port_colours, paths):
    """Every embedding rendered once, with the pixel map its trail needs.

    These are the only kaleido renders in the whole run: a background is built
    here and every frame then paints onto a copy of it.

    The coloured views pair each bin with how far it sits from a reward. Those
    labels were written for every bin in the session, so a block's own come out
    under the same mask as its spikes, while the full session's pair with the
    full embedding directly.
    """
    block = cfg['block']
    block_title = f'Block {block}'
    block_camera_key = f'block_{block}_cr_camera'
    full_camera = read_camera(cfg, 'full_camera')
    block_camera = read_camera(cfg, block_camera_key)

    full_points = np.load(paths.emb_dir / 'umap_full.npy')
    block_points = np.load(paths.emb_dir / f'umap_block_{block}_cr.npy')

    full_figure, full_background, full_pixels = plots.panel_and_pixels(
        full_points, FULL_TITLE, full_camera, PANEL)
    block_figure, block_background, block_pixels = plots.panel_and_pixels(
        block_points, block_title, block_camera, PANEL)

    full_labels = bins.time_nearest_reward
    block_labels = bins.time_nearest_reward[bins.block_mask]
    report_labels(block_labels, block_points, 'block')
    report_labels(full_labels, full_points, 'session')

    # Same cameras and the same hidden axes as the plain views, so the two are
    # comparable at a glance and only the colour differs. Expect a far paler
    # picture from the session than from the block -- a block is correct
    # rewarded trials, where almost every bin sits near a reward, while most of
    # a session does not.
    full_coloured = plots.coloured_figure(full_points, full_labels, FULL_TITLE,
                                          full_camera, PANEL)
    block_coloured = plots.coloured_figure(block_points, block_labels, block_title,
                                           block_camera, PANEL)

    # Flat views of the same two clouds, down the same pair of dimensions, so
    # the block and the session are read the same way. The plain ones also get
    # the pixel map their trail needs: a flat view has no camera, but it still
    # has to be told where a point landed -- see umap_plots.fit_flat_projection.
    full_xy_title = f'{FULL_TITLE} - xy projection'
    block_xy_title = f'{block_title} - xy projection'

    full_xy, full_xy_background, full_xy_pixels = plots.flat_panel_and_pixels(
        full_points, XY, full_xy_title, PANEL)
    block_xy, block_xy_background, block_xy_pixels = plots.flat_panel_and_pixels(
        block_points, XY, block_xy_title, PANEL)

    full_xy_coloured = plots.coloured_projection(full_points, full_labels, XY,
                                                 full_xy_title, PANEL)
    block_xy_coloured = plots.coloured_projection(block_points, block_labels, XY,
                                                  block_xy_title, PANEL)

    # Both clouds again, coloured by which port the mouse was at. The labels are
    # per bin and spatial -- nearest port centre within a few pixels of the
    # tracked position -- so unlike the per-trial reward labels they mark only
    # the moments actually spent at a port.
    block_ports = bins.port_ids[bins.block_mask]
    full_port_figure = plots.port_figure(full_points, bins.port_ids, port_colours,
                                         FULL_TITLE, full_camera, PANEL)
    block_port_figure = plots.port_figure(block_points, block_ports, port_colours,
                                          block_title, block_camera, PANEL)

    print(f'{int((block_ports > 0).sum())} of {len(block_ports)} block bins at a port; '
          f'{int((bins.port_ids > 0).sum())} of {len(bins.port_ids)} session bins')

    # The correct rewarded bins of every block in one embedding, drawn three
    # ways: plain, by reward time, and by port. Nothing here depends on the
    # block, so each block's session builds the same three. No panel shows them
    # yet, so they are built as figures and written out as interactive plots,
    # but never rasterised: no kaleido render is spent on them. Their labels come
    # out of the per-bin files under the same mask that selected the
    # embedding's rows.
    cr_mask = load_correct_rewarded(paths)
    cr_points = np.load(paths.emb_dir / 'umap_correct_rewarded.npy')
    cr_title = 'Correct rewarded, all blocks'
    cr_camera_key = 'correct_rewarded_camera'
    cr_camera = read_camera(cfg, cr_camera_key)

    cr_labels = bins.time_nearest_reward[cr_mask]
    cr_ports = bins.port_ids[cr_mask]
    report_labels(cr_labels, cr_points, 'correct rewarded')

    # The whole of this block -- every bin, gaps between trials included, not
    # just its correct rewarded ones -- drawn the same three ways, and likewise
    # written out only. Its own camera key, since its layout is its own fit and
    # shares nothing with the _cr embedding's.
    all_points = np.load(paths.emb_dir / f'umap_block_{block}_all.npy')
    all_title = f'Block {block}, all bins'
    all_camera_key = f'block_{block}_all_camera'
    all_camera = read_camera(cfg, all_camera_key)

    all_labels = bins.time_nearest_reward[bins.block_all_mask]
    all_ports = bins.port_ids[bins.block_all_mask]
    report_labels(all_labels, all_points, 'whole block')

    groups = switch_stay_groups(bins)

    # The coloured and port figures are rendered as well as written out, because
    # they are panels now. One more kaleido render each on a cold cache, nothing
    # on a warm one.
    return Layers(
        full=Layer(full_figure, full_background, full_pixels,
                   'umap_full', 'full_camera'),
        block=Layer(block_figure, block_background, block_pixels,
                    f'umap_block_{block}_cr_uncolored', block_camera_key),
        full_coloured=Layer(full_coloured,
                            plots.figure_image(full_coloured, PANEL),
                            name='umap_full_colored', camera_key='full_camera'),
        block_coloured=Layer(block_coloured,
                             plots.figure_image(block_coloured, PANEL),
                             name=f'umap_block_{block}_cr_colored',
                             camera_key=block_camera_key),
        full_xy=Layer(full_xy, full_xy_background, full_xy_pixels,
                      'umap_full_xy_uncolored'),
        block_xy=Layer(block_xy, block_xy_background, block_xy_pixels,
                       f'umap_block_{block}_cr_xy_uncolored'),
        full_xy_coloured=Layer(full_xy_coloured,
                               plots.figure_image(full_xy_coloured, PANEL),
                               name='umap_full_xy_colored'),
        block_xy_coloured=Layer(block_xy_coloured,
                                plots.figure_image(block_xy_coloured, PANEL),
                                name=f'umap_block_{block}_cr_xy_colored'),
        full_ports=Layer(full_port_figure,
                         plots.figure_image(full_port_figure, PANEL),
                         name='umap_full_ports'),
        block_ports=Layer(block_port_figure,
                          plots.figure_image(block_port_figure, PANEL),
                          name=f'umap_block_{block}_cr_ports'),
        correct_rewarded=Layer(
            plots.plain_figure(cr_points, cr_title, cr_camera, PANEL), None,
            name='umap_correct_rewarded_uncolored', camera_key=cr_camera_key),
        correct_rewarded_coloured=Layer(
            plots.coloured_figure(cr_points, cr_labels, cr_title, cr_camera, PANEL), None,
            name='umap_correct_rewarded_colored', camera_key=cr_camera_key),
        correct_rewarded_ports=Layer(
            plots.port_figure(cr_points, cr_ports, port_colours, cr_title, cr_camera, PANEL),
            None, name='umap_correct_rewarded_ports'),
        block_all=Layer(
            plots.plain_figure(all_points, all_title, all_camera, PANEL), None,
            name=f'umap_block_{block}_all_uncolored', camera_key=all_camera_key),
        block_all_coloured=Layer(
            plots.coloured_figure(all_points, all_labels, all_title, all_camera, PANEL), None,
            name=f'umap_block_{block}_all_colored', camera_key=all_camera_key),
        block_all_ports=Layer(
            plots.port_figure(all_points, all_ports, port_colours, all_title, all_camera, PANEL),
            None, name=f'umap_block_{block}_all_ports'),
        full_switch_stay=Layer(
            plots.switch_stay_figure(full_points, groups, FULL_TITLE, full_camera, PANEL),
            None, name='umap_full_switch_stay', camera_key='full_camera'),
        block_all_switch_stay=Layer(
            plots.switch_stay_figure(all_points, groups[bins.block_all_mask],
                                     all_title, all_camera, PANEL),
            None, name=f'umap_block_{block}_all_switch_stay', camera_key=all_camera_key))


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------
#
# Each builder below does its one-off work when it is called and hands back a
# function of the frame time, so the per-frame path holds nothing but the
# painting itself.

def maze_panel(bins, maze_png):
    """Panel 11: the maze, with the mouse's position and its recent trail.

    Drawn on a copy, so the trail lasts one frame instead of accumulating, and
    oldest first so the current position sits on top where dots overlap. Bins
    with no tracked position are skipped, leaving a gap in the trail.
    """
    maze = maze_image(maze_png)
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
        present = rows >= 0          # a trail bin outside this block has no point

        for row, alpha, colour in zip(rows[present], alphas[present], colours[present]):
            x, y = layer.pixels[row]
            dots = np.clip(round(y) + dot_dy, 0, PANEL - 1)
            cols = np.clip(round(x) + dot_dx, 0, PANEL - 1)
            frame[dots, cols] = np.round(frame[dots, cols] * (1 - alpha)
                                         + colour * alpha)

        return frame

    return draw


def info_panel(bins, data_file, block):
    """Panel 12: in words, what the other panels are showing at this instant.

    The session name and the label beside each row are baked into the background
    once, so a frame only has to add the three values -- which is what keeps
    this panel to about a millisecond, the whole reason it is PIL text rather
    than a plotly figure.

    "included" is read straight off the block mask, so it answers exactly the
    question the block panel poses -- is this bin one of the ones plotted there
    -- rather than the looser question of whether the trial belongs to the block
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
    labels = ['time', 'trial', f'block {block}', 'reward size']
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
                  'included' if bins.block_mask[index] else 'excluded',
                  f'{int(bins.reward_size_ms[index])} ms' if dispensing else 'None']

        for row, value in enumerate(values):
            pen.text((INFO_VALUE_X, INFO_TOP + row * INFO_STEP), value,
                     font=font, fill=INFO_INK)

        return np.asarray(image)

    return draw


def build_panels(bins, layers, data_file, block, maze_png):
    """The grid, row-major from 1 in the top left, as functions of the frame time.

    The placeholder fallback goes unused while all twelve are filled; it is what
    a thirteenth panel would fall back to if ROWS grew.
    """
    live = {
        1: embedding_panel(layers.full, bins, bins.full_row),
        2: embedding_panel(layers.block, bins, bins.block_row),
        3: embedding_panel(layers.full_xy, bins, bins.full_row),
        4: embedding_panel(layers.block_xy, bins, bins.block_row),
        5: static_panel(layers.full_coloured.background),
        6: static_panel(layers.block_coloured.background),
        7: static_panel(layers.full_xy_coloured.background),
        8: static_panel(layers.block_xy_coloured.background),
        9: static_panel(layers.full_ports.background),
        10: static_panel(layers.block_ports.background),
        11: maze_panel(bins, maze_png),
        12: info_panel(bins, data_file, block),
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
    with statement and relays everything else straight through unfiltered.
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
    paths: object                                # paths.SessionPaths
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
    paths = session_paths(params)

    width, height = frame_size()
    print(f'frame {width}x{height}: {COLUMNS}x{ROWS} panels of {PANEL}x{PANEL}')

    bins = load_bins(cfg['block'], paths)
    layers = render_layers(bins, cfg, params['port_colors'], paths)
    panels = build_panels(bins, layers, params['data_file'], cfg['block'],
                          paths.maze_png)

    return Session(cfg=cfg, paths=paths, bins=bins, layers=layers, panels=panels)


def write_interactive_plots(session):
    """Every rendered embedding saved as an html plot, in the session's umap folder.

    Orbit a 3D one by hand and the readout in its corner names the camera
    position to paste into parameters.yaml.
    """
    session.paths.umap_dir.mkdir(parents=True, exist_ok=True)
    for layer in session.layers.all():
        path = session.paths.umap_dir / f'{layer.name}.html'
        print(f'wrote {plots.write_html(layer.figure, path, layer.camera_key, CAMERA_ZOOM)}')


def write_video(session, out_path=None):
    """Paint the trail onto the cached backgrounds, frame by frame, and encode.

    Without an out_path, the video lands as block_video.mp4 in the session's
    figures folder.
    """
    cfg, bins = session.cfg, session.bins
    if out_path is None:
        session.paths.fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = session.paths.fig_dir / 'block_video.mp4'
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
