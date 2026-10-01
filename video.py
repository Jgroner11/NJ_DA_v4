"""Panel video: an 8x5 grid, one embedding per column, one colouring per row.

Eight columns, one embedding each, and four rows, one colouring each, then a
fifth row for what is not an embedding:

              Full   CR   B1   B2   B3   B4   B5   Region
    plain      .     .    .    .    .    .    .    .        with trail
    reward     .     .    .    .    .    .    .    .        by reward time
    ports      .     .    .    .    .    .    .    .        by port
    switch     .     .    .    .    .    .    .    .        by switch / stay group
    row 5    info  maze  behaviour (3 wide) ----   -    -    -

CR is the correct rewarded bins of every block in one fit, B1..B5 the whole of
each block, gaps between trials included, and Region the bins between the two
lines of the Maze lines section in embedding_and_labels.m. A block beyond the
fifth is fitted by run_umap.py but never shown; a column whose embedding does
not exist -- a session with fewer blocks, or a selection too small to fit --
is left as blank panels.

Only the plain row carries the moving trail, and only for bins the column's
embedding holds: a column's trail drops out while the mouse is somewhere that
embedding was not fitted on. The coloured panels are static, because their
colorbars and legends shift the plot area, so the fitted pixel maps describe the
plain panels' layout rather than theirs.

The behaviour panel is behaviour_plot.m's figure, with a line at the current
trial, placed from the axis table that script writes beside it.

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
from dataclasses import dataclass

import imageio.v2 as imageio
import numpy as np
import plotly.colors as pc
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

import umap_plots as plots
from paths import session_paths
from umap_plots import BACKGROUND

COLUMNS = 8                                      # panels across
ROWS = 5                                         # and down
PANEL = 480                                      # side of one panel, in pixels
BLOCKS_SHOWN = 5                                 # block columns, B1 to this
BEHAVIOUR_SPAN = 3                               # panels the behaviour figure spans
BEHAVIOUR_LINE = '#52514e'                       # the current-trial line on it
BEHAVIOUR_LINE_WIDTH = 2                         # in pixels

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

XY = (0, 1)                                      # the dimensions a flat view keeps

# One per row of the grid, top to bottom: the suffix of each view's html file.
VIEWS = ['uncolored', 'colored', 'ports', 'switch_stay']


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
    under `key` yet -- a new embedding has nothing hand-tuned for it the first
    time it is rendered, and this is what lets that first render happen at
    all, so its own umap_<name>_uncolored.html can be orbited afterward to
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


def pad_geometry(image_size, box):
    """How an image is laid into a box by ImageOps.pad: size on screen, and offset.

    Mirrors what ImageOps.pad does -- scale to fit preserving the aspect, then
    centre -- because the pixel maps below have to reproduce it exactly to put
    marks in the right place. Written out rather than assumed so the two cannot
    drift: the rounding here is PIL's own, and // 2 in place of round() would
    sit a pixel off for some sizes.
    """
    width, height = image_size
    box_width, box_height = box

    if width / height > box_width / box_height:
        new_width, new_height = box_width, round(height / width * box_width)
    else:
        new_width, new_height = round(width / height * box_height), box_height

    return (width, height, new_width, new_height,
            round((box_width - new_width) * 0.5), round((box_height - new_height) * 0.5))


def maze_geometry(maze_png):
    """How the blackout frame is laid into its square panel."""
    return pad_geometry(Image.open(maze_png).size, (PANEL, PANEL))


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


def blank_panel(text, width=PANEL):
    """An empty panel: white, `text` in the middle, a hairline round the edge.

    The border is not decoration -- without it neighbouring white panels merge
    into one white area and the grid it is meant to show cannot be seen.
    """
    image = Image.new('RGB', (width, PANEL), PLACEHOLDER_BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width - 1, PANEL - 1], outline=PLACEHOLDER_EDGE)

    font = fitted_font(text, width - 2 * INFO_X, PANEL // 12)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(((width - (right - left)) / 2 - left,
               (PANEL - (bottom - top)) / 2 - top),
              text, fill=PLACEHOLDER_INK, font=font)

    return np.asarray(image)


def placeholder(text, width=PANEL):
    """A panel function for a slot with nothing to show.

    The image is built once and handed back unchanged every frame, so an empty
    panel costs nothing per frame beyond being copied into the grid.
    """
    image = blank_panel(text, width)
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
    """Everything recorded per time bin, and what the trail is drawn from."""
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
    block_id: np.ndarray                         # every bin's block, gaps included
    in_decision_region: np.ndarray               # head between the maze lines
    head_xy: np.ndarray
    width: float                                 # seconds in one bin
    trail_bins: int                              # how many of them the trail spans
    trail_rgb: np.ndarray                        # and its colour, newest first


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


def load_bins(paths):
    """Read the per-bin csvs the video needs, and size the trail."""
    labels = paths.label_dir
    times = np.loadtxt(labels / 'bin_times.csv')

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
        block_id=np.loadtxt(labels / 'block_id.csv'),
        in_decision_region=np.loadtxt(labels / 'in_decision_region.csv').astype(bool),
        head_xy=maze_pixels(np.loadtxt(labels / 'head_positions.csv', delimiter=','),
                            paths.maze_png),
        width=width,
        trail_bins=trail_bins,
        trail_rgb=trail_ramp(trail_bins))


def bin_rows(mask):
    """Each bin's row in an embedding fitted on the bins `mask` selects, -1 if absent.

    A masked embedding's row i is the i-th True entry of its mask -- see
    run_umap.py -- so a bin's row is its rank among the set bins.
    """
    return np.where(mask, np.cumsum(mask) - 1, -1)


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
class Column:
    """One embedding, and the bins it was fitted on.

    `name` is the embedding's stem after umap_, as run_umap.py writes it, and
    `mask` selects its rows from the per-bin labels, row i being the i-th True.
    """
    name: str
    title: str
    mask: np.ndarray

    @property
    def camera_key(self):
        """The parameters.yaml entry every 3D view of this column shares."""
        return f'{self.name}_camera'


def grid_columns(bins):
    """The eight columns of the grid, left to right."""
    columns = [Column('full', 'Full session', np.ones(len(bins.times), dtype=bool)),
               Column('correct_rewarded', 'Correct rewarded, all blocks',
                      bins.correct & bins.rewarded)]
    columns += [Column(f'block_{block}_all', f'Block {block}', bins.block_id == block)
                for block in range(1, BLOCKS_SHOWN + 1)]
    columns.append(Column('decision_region', 'Decision region', bins.in_decision_region))
    return columns


@dataclass
class Grid:
    """Every rendered view, and the ones only written out.

    cells[c] holds column c's four views in VIEWS order, or is None where that
    column's embedding does not exist. extras are written as interactive plots
    but shown in no panel.
    """
    columns: list
    cells: list
    extras: list

    def all(self):
        """Each layer once: the grid column by column, then the extras."""
        return [layer for views in self.cells if views for layer in views] + self.extras


def report_labels(labels, points, what):
    """Check the reward labels pair with the cloud, and say how many are coloured."""
    assert len(labels) == len(points), \
        f'{len(labels)} labels but {len(points)} embedded bins'
    print(f'{int((np.abs(labels) <= plots.TIME_RADIUS).sum())} of {len(labels)} '
          f'{what} bins within {plots.TIME_RADIUS:g} s of a reward')


def load_points(column, paths):
    """A column's embedding, or None if run_umap.py did not write one."""
    path = paths.emb_dir / f'umap_{column.name}.npy'
    if not path.exists():
        print(f'no {path.name}; {column.title} left blank')
        return None

    points = np.load(path)
    assert len(points) == column.mask.sum(), (
        f'{path.name} has {len(points)} rows but its mask selects '
        f'{int(column.mask.sum())} bins -- re-run run_umap.py')
    return points


def column_views(column, points, bins, groups, cfg, port_colours):
    """One column's four views, in VIEWS order, each rendered once.

    Every label comes out of the per-bin files under the column's own mask, the
    same one that selected its embedding's rows, so row i and label i are the
    same bin. Only the plain view gets a pixel map, for the trail.
    """
    camera_key = column.camera_key
    camera = read_camera(cfg, camera_key)
    stem = f'umap_{column.name}'

    labels = bins.time_nearest_reward[column.mask]
    report_labels(labels, points, column.title)

    plain, plain_background, plain_pixels = plots.panel_and_pixels(
        points, column.title, camera, PANEL)
    coloured = plots.coloured_figure(points, labels, column.title, camera, PANEL)
    ports = plots.port_figure(points, bins.port_ids[column.mask], port_colours,
                              column.title, camera, PANEL)
    switch_stay = plots.switch_stay_figure(points, groups[column.mask],
                                           column.title, camera, PANEL)

    return [Layer(plain, plain_background, plain_pixels, f'{stem}_uncolored', camera_key)] + [
        Layer(figure, plots.figure_image(figure, PANEL), name=f'{stem}_{view}',
              camera_key=camera_key)
        for view, figure in zip(VIEWS[1:], [coloured, ports, switch_stay])]


def render_layers(bins, cfg, port_colours, paths):
    """Every embedding rendered once, with the pixel map its trail needs.

    These are the only kaleido renders in the whole run: a background is built
    here and every frame then paints onto a copy of it. On a cold cache that is
    five per column -- four views and the plain view's calibration figure.
    """
    groups = switch_stay_groups(bins)
    columns = grid_columns(bins)

    cells, extras = [], []
    for column in columns:
        points = load_points(column, paths)
        if points is None:
            cells.append(None)
            continue

        cells.append(column_views(column, points, bins, groups, cfg, port_colours))

        # The decision region flat too, as it was before the grid; written only.
        if column.name == 'decision_region':
            extras.append(Layer(
                plots.plain_projection(points, XY, f'{column.title} - xy projection', PANEL),
                None, name='umap_decision_region_xy_uncolored'))

    return Grid(columns=columns, cells=cells, extras=extras)


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------
#
# Each builder below does its one-off work when it is called and hands back a
# function of the frame time, so the per-frame path holds nothing but the
# painting itself.

def maze_panel(bins, maze_png):
    """The maze, with the mouse's position and its recent trail.

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


def info_panel(bins, data_file):
    """In words, what the other panels are showing at this instant.

    The session name and the label beside each row are baked into the background
    once, so a frame only has to add the values -- which is what keeps this
    panel to about a millisecond, the whole reason it is PIL text rather than a
    plotly figure.

    "block" is block_id, so a bin between trials reads the block of the trial
    before it, as every block column counts it. A bin between trials has no
    trial number.

    "reward size" reads bins.time_nearest_reward, which is signed (negative
    before a reward, positive after -- see embedding_and_labels.m), so 0 marks
    reward onset and the window checked is [0, REWARD_DISPENSING_S]. Outside
    that window, or where time_nearest_reward is NaN (no reward on either side
    of this bin), it reads None rather than bins.reward_size_ms -- that field
    is paired with time_nearest_reward but not itself windowed, so reading it
    unconditionally would show a reward's size long before or after it was
    actually dispensed.
    """
    labels = ['time', 'trial', 'block', 'reward size']
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
        block = bins.block_id[index]

        reward_offset = bins.time_nearest_reward[index]
        dispensing = 0 <= reward_offset <= REWARD_DISPENSING_S   # NaN compares False

        values = [f'{bins.times[index]:.2f} s',
                  '-' if np.isnan(trial) else f'{int(trial)}',
                  '-' if np.isnan(block) else f'{int(block)}',
                  f'{int(bins.reward_size_ms[index])} ms' if dispensing else 'None']

        for row, value in enumerate(values):
            pen.text((INFO_VALUE_X, INFO_TOP + row * INFO_STEP), value,
                     font=font, fill=INFO_INK)

        return np.asarray(image)

    return draw


def behaviour_panel(bins, paths):
    """behaviour_plot.m's figure, BEHAVIOUR_SPAN panels wide, with a line at
    the current trial.

    The figure is scaled to fit and centred, as the maze is, and the line placed
    from behaviour_axis.csv, which that script writes beside the PNG: for each
    trial it plots, the trial's SessionTrial and the pixel column it sits at,
    plus the top and bottom of the axes. The figure's x axis counts the trials
    it plotted, 1 to n, rather than SessionTrial, and the two part company
    wherever a trial was dropped, so the line is looked up by SessionTrial.

    A bin between trials holds the line at the trial before it, as block_id
    does; bins before the first trial hold it at the first.
    """
    width = BEHAVIOUR_SPAN * PANEL
    png = paths.fig_dir / 'behaviour.png'
    axis_path = paths.fig_dir / 'behaviour_axis.csv'
    if not (png.exists() and axis_path.exists()):
        print(f'no {png.name} and {axis_path.name} in {paths.fig_dir} -- run '
              'behaviour_plot.m; behaviour panel left blank')
        return placeholder('no behaviour plot', width)

    image = Image.open(png).convert('RGB')
    background = np.asarray(ImageOps.pad(image, (width, PANEL), color=BACKGROUND))
    source_width, _, new_width, _, offset_x, offset_y = pad_geometry(image.size, (width, PANEL))
    scale = new_width / source_width

    axis = np.genfromtxt(axis_path, delimiter=',', names=True)
    session_trials = axis['session_trial']

    trial = bins.trial_ids
    known = np.isfinite(trial)
    last_known = np.maximum.accumulate(np.where(known, np.arange(len(trial)), -1))
    trial = trial[np.where(last_known >= 0, last_known, np.flatnonzero(known)[0])]

    order = np.argsort(session_trials)
    found = order[np.clip(np.searchsorted(session_trials[order], trial), 0, len(order) - 1)]
    assert np.array_equal(session_trials[found], trial), \
        f'{axis_path.name} is missing trials the bins name -- re-run behaviour_plot.m'

    line_x = np.round(axis['x_px'][found] * scale + offset_x).astype(int) - BEHAVIOUR_LINE_WIDTH // 2
    top = round(axis['y_top_px'][0] * scale + offset_y)
    bottom = round(axis['y_bottom_px'][0] * scale + offset_y)
    colour = ImageColor.getrgb(BEHAVIOUR_LINE)

    def draw(time_s):
        frame = background.copy()
        x = line_x[bin_at(bins, time_s)]
        frame[top:bottom, max(x, 0):x + BEHAVIOUR_LINE_WIDTH] = colour
        return frame

    return draw


def build_panels(bins, grid, data_file, paths):
    """The grid, as rows of functions of the frame time, top row first.

    Each panel function hands back an image PANEL high and as many panels wide
    as it spans; a row's spans must add up to COLUMNS, which is checked here
    rather than discovered as a shape error on the first frame.
    """
    rows = []
    for view_index in range(len(VIEWS)):
        row = []
        for column, views in zip(grid.columns, grid.cells):
            if views is None:
                row.append((1, placeholder(f'no {column.title}')))
            elif view_index == 0:
                row.append((1, embedding_panel(views[0], bins, bin_rows(column.mask))))
            else:
                row.append((1, static_panel(views[view_index].background)))
        rows.append(row)

    last = [(1, info_panel(bins, data_file)),
            (1, maze_panel(bins, paths.maze_png)),
            (BEHAVIOUR_SPAN, behaviour_panel(bins, paths))]
    filled = sum(span for span, _ in last)
    rows.append(last + [(1, placeholder('')) for _ in range(COLUMNS - filled)])

    assert len(rows) == ROWS, f'{len(rows)} rows built but ROWS is {ROWS}'
    for number, row in enumerate(rows, start=1):
        spans = sum(span for span, _ in row)
        assert spans == COLUMNS, f'row {number} spans {spans} panels, not {COLUMNS}'

    return [[panel for _, panel in row] for row in rows]


def compose(panels, time_s):
    """One video frame: every panel drawn, each row tiled, the rows stacked."""
    return np.vstack([np.hstack([panel(time_s) for panel in row]) for row in panels])


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
    grid: Grid
    panels: list                                 # rows of panel functions


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

    bins = load_bins(paths)
    grid = render_layers(bins, cfg, params['port_colors'], paths)
    panels = build_panels(bins, grid, params['data_file'], paths)

    return Session(cfg=cfg, paths=paths, bins=bins, grid=grid, panels=panels)


def write_interactive_plots(session):
    """Every rendered embedding saved as an html plot, in the session's umap folder.

    Orbit a 3D one by hand and the readout in its corner names the camera
    position to paste into parameters.yaml.
    """
    session.paths.umap_dir.mkdir(parents=True, exist_ok=True)
    for layer in session.grid.all():
        path = session.paths.umap_dir / f'{layer.name}.html'
        print(f'wrote {plots.write_html(layer.figure, path, layer.camera_key, CAMERA_ZOOM)}')


def write_video(session, out_path=None):
    """Paint the trail onto the cached backgrounds, frame by frame, and encode.

    Without an out_path, the video lands as session_video.mp4 in the session's
    figures folder.
    """
    cfg, bins = session.cfg, session.bins
    if out_path is None:
        session.paths.fig_dir.mkdir(parents=True, exist_ok=True)
        out_path = session.paths.fig_dir / 'session_video.mp4'
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
