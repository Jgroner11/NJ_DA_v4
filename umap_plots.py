"""Plotly figures for the embedding panels, and the pixel map that goes with them.

video.py uses this for three things:

    panel_and_pixels   a rasterised cloud, plus where each of its points landed
    coloured_figure    the same cloud coloured by time to the nearest reward
    write_html         either figure saved as an interactive plot

The pixel map is what lets the video paint trails in numpy instead of asking
plotly for a fresh render every frame: a kaleido render is several seconds, so at
30 fps re-rendering would cost hours per minute of video. See fit_projection for
how the map is obtained, and why it is measured rather than derived.

Nothing here imports from video, so the dependency runs one way. Figures
take the panel size as an argument rather than reading a layout constant, and
each works out its own scene box from the points it is given -- scene_ranges is
a pure function of the data, so every figure drawn from the same points agrees
on the box without having to pass it around.
"""

import hashlib
import io
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image

CACHE_DIR = Path('.cache')                       # shared with run_umap.py

BACKGROUND = '#fcfcfb'                           # the surface a figure sits on
POINT_COLOUR = '#52514e'                         # an ordinary point in the cloud
BASE_SIZE = 1.5                                  # plotly marker size, the same for every point
SCENE_PAD = 0.02                                 # slack between the cloud and its scene box
CALIBRATION_SIZE = 10                            # marker size for the projection fit
CALIBRATION_INSET = 0.2                          # keep those markers off the scene walls

FLAT_SIZE = 3                                    # marker size in a 2D projection
AXIS_NAMES = ['UMAP 1', 'UMAP 2', 'UMAP 3']

TIME_RADIUS = 3.0                                # seconds either side of a reward that get colour
UNLABELLED = '#d8d7d2'                           # bins outside that window
COLORBAR_TITLE = 'seconds to nearest reward (- before, + after)'

# One hue per port, assigned in port order and never cycled. The first four are
# the ones the earlier alternation_plots.py used, where a patch only ever showed
# two of them; the last four are added because a whole session visits all eight.
# Eight categories is at the limit of what stays distinguishable, so the legend
# is carrying real weight here rather than decorating.
PORT_COLOURS = ['#1b8a4b', '#c9930a', '#7c4fb8', '#d1552f',
                '#2a78d6', '#c2185b', '#00838f', '#6d4c41']
NO_PORT_NAME = 'not at a port'


def pale_marker(size):
    """The recessive style for points outside the reward window."""
    return dict(size=size, color=UNLABELLED, opacity=0.35)


def scale_marker(size, labels):
    """The diverging style for points inside it, colorbar included.

    Shared by the 3D figure and the 2D projections so the scale, the symmetric
    limits and the colorbar wording are defined once. cmin and cmax are fixed at
    the window rather than taken from the data, which is what puts mrybm's gold
    exactly on the moment of reward. mrybm is cyclic, so the two tails converge
    on the same magenta at +-TIME_RADIUS rather than ending on two unrelated
    hues -- and unlike icefire, every stop stays a solid, saturated colour
    (nothing washes out near white or black against BACKGROUND). It cycles
    through four hues rather than a simple two-sided split, so it no longer
    reads as "one colour before, another after" the way a plain diverging scale
    does -- what it marks is distance from reward, via how far a colour has
    travelled from gold, not a consistent before/after hue.
    """
    return dict(
        size=size, color=labels, colorscale='mrybm',
        cmin=-TIME_RADIUS, cmax=TIME_RADIUS,
        colorbar=dict(title=dict(text=COLORBAR_TITLE, side='right'),
                      outlinewidth=0, thickness=14, len=0.7),
    )


def name_the_underlay():
    """The legend entry for the pale trace."""
    return f'No reward within {TIME_RADIUS:g} s'


def show_underlay_legend(figure):
    """Put the legend back for a two-trace figure.

    The layouts turn it off, which suits the single-trace panels. Here it is the
    only thing naming the pale underlay, so it goes back on.
    """
    figure.update_layout(showlegend=True, legend=dict(
        orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1.0,
        font=dict(size=10, color=POINT_COLOUR), bgcolor='rgba(0,0,0,0)'))
    return figure


def scene_ranges(points, pad=SCENE_PAD):
    """Explicit axis ranges for a scene, padded a little around the data.

    Set explicitly rather than left to autorange so the calibration figure and
    the panel figure get the same scene box despite holding different points.
    With autorange the two boxes would differ and the fitted projection would be
    wrong in a way nothing would flag.
    """
    low, high = points.min(axis=0), points.max(axis=0)
    margin = (high - low) * pad
    return [[lo - m, hi + m] for lo, hi, m in zip(low, high, margin)]


def apply_layout(figure, title, camera, ranges, size):
    """The layout every figure in a panel shares, calibration figures included.

    The axes are hidden but their ranges are not. visible=False drops the grid,
    the panes, the ticks and the titles, leaving the cloud on a plain surface,
    while range still pins the scene box -- which is what fit_projection relies
    on. Because the calibration figure is laid out through here too, hiding the
    axes cannot put the fitted projection out of step with what is rendered.
    """
    hidden = [dict(range=r, visible=False, showbackground=False) for r in ranges]
    figure.update_layout(
        title=dict(text=title, x=0.02, xanchor='left'),
        width=size, height=size,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=BACKGROUND,
        showlegend=False,
        scene=dict(
            camera=camera,
            xaxis=hidden[0], yaxis=hidden[1], zaxis=hidden[2],
        ),
    )
    return figure


def plain_figure(points, title, camera, size):
    """The cloud as the video panels show it: one colour, no axes, no legend."""
    figure = go.Figure(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=BASE_SIZE, color=POINT_COLOUR),
    ))
    return apply_layout(figure, title, camera, scene_ranges(points), size)


def coloured_figure(points, labels, title, camera, size):
    """The same cloud, coloured by how far each point sits from a reward.

    Two traces, as in the earlier alternation_plots.py. Points further than
    TIME_RADIUS from any reward are drawn small and pale underneath; the ones
    inside that window carry the diverging scale on top. Splitting them is the
    point of the scheme -- the pale majority would otherwise swamp the part
    being read, and the fixed symmetric cmin/cmax puts the neutral midpoint
    exactly on the moment of reward.
    """
    inside = np.abs(labels) <= TIME_RADIUS        # a NaN label compares False
    figure = go.Figure()

    figure.add_trace(go.Scatter3d(
        x=points[~inside, 0], y=points[~inside, 1], z=points[~inside, 2],
        mode='markers', marker=pale_marker(BASE_SIZE),
        name=name_the_underlay(), hoverinfo='skip',
    ))

    figure.add_trace(go.Scatter3d(
        x=points[inside, 0], y=points[inside, 1], z=points[inside, 2],
        mode='markers', marker=scale_marker(BASE_SIZE * 2, labels[inside]),
        hovertemplate='%{marker.color:.2f} s<extra></extra>',
        showlegend=False,
    ))

    return show_underlay_legend(
        apply_layout(figure, title, camera, scene_ranges(points), size))


def show_category_legend(figure):
    """Put a legend back for a figure whose traces are categories.

    Vertical and inside the plot area, unlike the underlay legend: with a trace
    per port there are too many entries to sit in a row across the top.
    """
    figure.update_layout(showlegend=True, legend=dict(
        yanchor='top', y=0.98, xanchor='right', x=0.98, itemsizing='constant',
        font=dict(size=9, color=POINT_COLOUR), bgcolor='rgba(252,252,251,0.7)'))
    return figure


def port_figure(points, ports, title, camera, size):
    """The cloud coloured by which port the mouse was at, 0 meaning none.

    One trace per port rather than one trace with an array of colours, because
    that is what gives plotly a legend entry per port -- a categorical label has
    no colorbar to explain it. Ports are drawn in numerical order, so a port
    keeps its hue whichever figure it appears in, and the palette is never
    cycled: too many ports is an error rather than two ports quietly sharing a
    colour.
    """
    present = [int(p) for p in np.unique(ports) if p > 0]
    assert len(present) <= len(PORT_COLOURS), (
        f'{len(present)} ports but only {len(PORT_COLOURS)} colours defined -- '
        'add more rather than letting them repeat')

    figure = go.Figure()

    none = ports == 0
    figure.add_trace(go.Scatter3d(
        x=points[none, 0], y=points[none, 1], z=points[none, 2],
        mode='markers', marker=pale_marker(BASE_SIZE),
        name=NO_PORT_NAME, hoverinfo='skip',
    ))

    for port in present:
        at = ports == port
        figure.add_trace(go.Scatter3d(
            x=points[at, 0], y=points[at, 1], z=points[at, 2],
            mode='markers',
            marker=dict(size=BASE_SIZE * 2, color=PORT_COLOURS[port - 1]),
            name=f'port {port}', hoverinfo='skip',
        ))

    return show_category_legend(
        apply_layout(figure, title, camera, scene_ranges(points), size))


def flat_ranges(points, dims, pad=SCENE_PAD):
    """Explicit axis ranges for a 2D projection of these columns.

    Explicit for the same reason scene_ranges is: the calibration figure holds
    different points, and on autorange it would frame a different area, so the
    fitted mapping would not describe what the real figure drew.

    Note that scaleanchor may still widen one of these to satisfy equal aspect.
    That is fine -- both figures get the identical adjustment, and the fit
    measures the result rather than predicting it.
    """
    flat = points[:, list(dims)]
    low, high = flat.min(axis=0), flat.max(axis=0)
    margin = (high - low) * pad
    return [[lo - m, hi + m] for lo, hi, m in zip(low, high, margin)]


def flat_layout(figure, title, ranges, size):
    """Layout for a 2D projection, on the same plain surface as the 3D panels.

    scaleanchor with scaleratio 1 is the only part of this that is not cosmetic.
    An embedding carries its meaning in its shape, and letting the two axes
    stretch independently to fill a square would distort it.
    """
    figure.update_layout(
        title=dict(text=title, x=0.02, xanchor='left'),
        width=size, height=size,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        showlegend=False,
        xaxis=dict(visible=False, range=ranges[0]),
        yaxis=dict(visible=False, range=ranges[1], scaleanchor='x', scaleratio=1),
    )
    return figure


def plain_projection(points, dims, title, size):
    """Two of the embedding's three columns, flat and in a single colour."""
    across, up = dims
    figure = go.Figure(go.Scattergl(
        x=points[:, across], y=points[:, up],
        mode='markers', marker=dict(size=FLAT_SIZE, color=POINT_COLOUR),
    ))
    return flat_layout(figure, title, flat_ranges(points, dims), size)


def coloured_projection(points, labels, dims, title, size):
    """The same flat view, coloured by time to the nearest reward.

    Split into a pale underlay and a coloured window exactly as the 3D version
    is, and drawing on the same shared marker specs, so the two colourings are
    read the same way.
    """
    across, up = dims
    inside = np.abs(labels) <= TIME_RADIUS        # a NaN label compares False
    figure = go.Figure()

    figure.add_trace(go.Scattergl(
        x=points[~inside, across], y=points[~inside, up],
        mode='markers', marker=pale_marker(FLAT_SIZE - 1),
        name=name_the_underlay(), hoverinfo='skip',
    ))

    figure.add_trace(go.Scattergl(
        x=points[inside, across], y=points[inside, up],
        mode='markers', marker=scale_marker(FLAT_SIZE, labels[inside]),
        hovertemplate='%{marker.color:.2f} s<extra></extra>',
        showlegend=False,
    ))

    return show_underlay_legend(
        flat_layout(figure, title, flat_ranges(points, dims), size))


def flat_calibration_targets(ranges):
    """A 3x3 grid inside the plot area: nine points, well clear of the edges."""
    (left, right), (bottom, top) = ranges
    fractions = (0.2, 0.5, 0.8)
    return np.array([[left + (right - left) * fx, bottom + (top - bottom) * fy]
                     for fx in fractions for fy in fractions])


def fit_flat_projection(points, dims, title, size):
    """The affine map taking a 2D data coordinate to a pixel in this projection.

    Measured the same way fit_projection measures the 3D one, and for the same
    reason: the plot area in pixels depends on the margins and on whatever
    scaleanchor did to satisfy equal aspect, and predicting that is more fragile
    than rendering nine known points and seeing where they land.

    Only affine here rather than projective -- a flat scatter has no
    perspective -- so three points would do and nine is insurance. Returned as
    the 3x2 matrix of a least-squares fit.
    """
    ranges = flat_ranges(points, dims)
    targets = flat_calibration_targets(ranges)

    hues = np.linspace(0, 300, len(targets))
    wanted = np.array([hsv_to_rgb(h) for h in hues])
    colours = ['#%02x%02x%02x' % tuple(rgb) for rgb in wanted]

    figure = go.Figure(go.Scattergl(
        x=targets[:, 0], y=targets[:, 1],
        mode='markers', marker=dict(size=CALIBRATION_SIZE, color=colours),
    ))
    image = figure_image(flat_layout(figure, title, ranges, size), size).astype(int)

    found, located, blobs = [], [], []
    for rgb, target in zip(wanted, targets):
        core = np.abs(image - rgb).sum(axis=2) < 30
        if not core.any():
            continue
        rows, cols = np.nonzero(core)
        found.append([cols.mean(), rows.mean()])
        located.append(target)
        blobs.append(int(core.sum()))

    # Same guard as the 3D fit: a marker another one partly covers reports the
    # centroid of a crescent, which is not where its point projects.
    blobs = np.array(blobs)
    whole = blobs >= 0.8 * np.median(blobs)

    assert whole.sum() >= 4, (
        f'only {whole.sum()} of {len(targets)} flat calibration markers came out '
        f'whole (found {len(found)}, blob sizes {sorted(blobs)})')

    found, targets = np.array(found)[whole], np.array(located)[whole]
    design = np.column_stack([targets, np.ones(len(targets))])
    matrix, *_ = np.linalg.lstsq(design, found, rcond=None)

    residual = np.abs(design @ matrix - found).max()
    assert residual < 1.0, f'flat projection fit is off by {residual:.2f} px'
    return matrix


def flat_project(matrix, points, dims):
    """Pixel coordinates, as columns of x then y, for these points' 2D view."""
    flat = points[:, list(dims)]
    return np.column_stack([flat, np.ones(len(flat))]) @ matrix


def flat_panel_and_pixels(points, dims, title, size):
    """Everything a video panel needs from one 2D projection.

    The mirror of panel_and_pixels: the figure, its rasterised background, and
    the pixel each point occupies in it.
    """
    figure = plain_projection(points, dims, title, size)
    background = figure_image(figure, size)
    pixels = flat_project(fit_flat_projection(points, dims, title, size), points, dims)
    return figure, background, pixels


def figure_image(figure, size):
    """A figure rasterised to a size x size RGB array, cached on disk.

    A kaleido render is several seconds, and nothing about these figures changes
    between runs unless a camera, a size or a styling constant does. So the
    result is keyed on the figure's own JSON together with the size: between
    them those are a complete description of what kaleido is about to draw --
    the points, the camera, the scene ranges, every marker setting. The cache
    therefore misses exactly when the picture would have differed, and hits
    otherwise. Editing a camera in parameters.yaml redraws; editing the trail
    colours, which this module knows nothing about, does not.

    Hashing costs a fraction of a second against a render's several. Delete
    .cache/ to force everything to be drawn again.

    The size is passed to to_image as well as living in the layout: otherwise it
    falls back to its own 700x500 default, which would not fit the panel.
    """
    key = hashlib.sha256(f'{size}|{figure.to_json()}'.encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f'render_{key}.png'

    if cache_path.exists():
        return np.asarray(Image.open(cache_path).convert('RGB'))

    print(f'rendering {figure.layout.title.text or "figure"} at {size}x{size}...',
          flush=True)
    png = figure.to_image(format='png', width=size, height=size)

    CACHE_DIR.mkdir(exist_ok=True)
    cache_path.write_bytes(png)
    return np.asarray(Image.open(io.BytesIO(png)).convert('RGB'))


def calibration_targets(ranges):
    """Fourteen well-spread points inside the scene: 8 corners and 6 face centres.

    Non-coplanar and far apart, which is what the fit below needs. Inset from
    the scene box rather than sitting on it: a marker on the boundary can be
    clipped by the axis walls, and one that cannot be found in the render is one
    correspondence lost.
    """
    low = np.array([r[0] for r in ranges])
    high = np.array([r[1] for r in ranges])
    span = high - low

    near, far = low + span * CALIBRATION_INSET, high - span * CALIBRATION_INSET
    middle = (low + high) / 2

    points = [[x, y, z] for x in (near[0], far[0])
              for y in (near[1], far[1])
              for z in (near[2], far[2])]
    for axis in range(3):
        for end in (near, far):
            face = middle.copy()
            face[axis] = end[axis]
            points.append(list(face))
    return np.array(points)


def fit_projection(points, title, camera, size):
    """The 3x4 matrix taking a 3D point to a pixel in this panel's rendering.

    Measured rather than derived. Plotly normalises each axis into its own scene
    box and applies a perspective camera, and reimplementing that exactly is
    fiddly to get right and silent when it is wrong. So this renders one figure
    of points whose 3D positions are known, finds each one in the image by its
    colour, and solves for the matrix that maps one to the other. Whatever
    plotly did with aspect ratio and field of view, the fit reproduces it.

    Solved by the usual direct linear transform: each correspondence gives two
    linear equations in the twelve unknowns, and the answer is the null vector
    of the stacked system, i.e. the smallest right singular vector.
    """
    ranges = scene_ranges(points)
    targets = calibration_targets(ranges)

    # Hues spread around the wheel, converted here rather than handed to plotly
    # as hsv() strings: those render as nothing at all, and asking for the exact
    # rgb we then search for removes any question of how plotly parses colours.
    hues = np.linspace(0, 300, len(targets))
    wanted = np.array([hsv_to_rgb(h) for h in hues])
    colours = ['#%02x%02x%02x' % tuple(rgb) for rgb in wanted]

    figure = go.Figure(go.Scatter3d(
        x=targets[:, 0], y=targets[:, 1], z=targets[:, 2],
        mode='markers', marker=dict(size=CALIBRATION_SIZE, color=colours),
    ))
    image = figure_image(apply_layout(figure, title, camera, ranges, size), size).astype(int)

    # Each marker's core pixels carry its colour exactly; anti-aliased edges
    # blend towards the background and are excluded by the tolerance.
    found, located, blobs = [], [], []
    for rgb, target in zip(wanted, targets):
        core = np.abs(image - rgb).sum(axis=2) < 30
        if not core.any():
            continue
        rows, cols = np.nonzero(core)
        found.append([cols.mean(), rows.mean()])
        located.append(target)
        blobs.append(int(core.sum()))

    # A marker another one partly covers shows only a crescent, and a crescent's
    # centroid is not where its point projects -- it is pulled towards whichever
    # side stayed visible. Every marker is drawn the same size, so a blob well
    # under the median is a hidden one, and including it drags the whole fit:
    # measured at 2.52 px with two such markers in, 0.08 px with them out.
    # Dropping them is safe because the fit needs six correspondences and
    # fourteen were drawn.
    blobs = np.array(blobs)
    whole = blobs >= 0.8 * np.median(blobs)

    assert whole.sum() >= 8, (
        f'only {whole.sum()} of {len(targets)} calibration markers came out whole '
        f'(found {len(found)}, blob sizes {sorted(blobs)})')
    found, targets = np.array(found)[whole], np.array(located)[whole]

    equations = []
    for (x, y, z), (u, v) in zip(targets, found):
        point = [x, y, z, 1]
        equations.append([*point, *([0] * 4), *[-u * c for c in point]])
        equations.append([*([0] * 4), *point, *[-v * c for c in point]])

    _, _, vt = np.linalg.svd(np.array(equations))
    matrix = vt[-1].reshape(3, 4)

    residual = np.abs(project(matrix, targets) - found).max()
    assert residual < 1.0, f'projection fit is off by {residual:.2f} px'
    return matrix


def hsv_to_rgb(hue):
    """The rgb plotly renders for hsv(hue, 100%, 100%), as three integers."""
    sector, offset = divmod(hue / 60.0, 1.0)
    rising, falling = round(255 * offset), round(255 * (1 - offset))
    return [[255, rising, 0], [falling, 255, 0], [0, 255, rising],
            [0, falling, 255], [rising, 0, 255], [255, 0, falling]][int(sector) % 6]


def project(matrix, points):
    """Pixel coordinates, as columns of x then y, for these 3D points."""
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = homogeneous @ matrix.T
    return projected[:, :2] / projected[:, 2:3]


def panel_and_pixels(points, title, camera, size):
    """Everything a video panel needs from one embedding.

    Returns the figure, its rasterised background, and the pixel each point
    occupies in that background. Two kaleido renders happen here -- the panel
    itself and the calibration figure behind fit_projection -- and they are the
    only ones; every frame after this is numpy.
    """
    figure = plain_figure(points, title, camera, size)
    background = figure_image(figure, size)
    pixels = project(fit_projection(points, title, camera, size), points)
    return figure, background, pixels


# Injected into the interactive HTML by write_html. Plotly substitutes the
# literal {plot_id} with the plot div's id; CAMERA_KEY is swapped per panel so
# the readout is already labelled with the right parameters.yaml key.
CAMERA_READOUT = r"""
var graph = document.getElementById('{plot_id}');

var box = document.createElement('pre');
box.style.cssText = 'position:fixed; top:8px; right:8px; margin:0;'
    + 'padding:8px 10px; background:#fff; border:1px solid #ccc;'
    + 'font:12px/1.45 monospace; white-space:pre; user-select:all;';
document.body.appendChild(box);

// Divided back out by CAMERA_ZOOM, so what is shown is the value to put in
// parameters.yaml. Reporting the zoomed eye would mean pasting it back applied
// the zoom a second time, and the plot would creep closer on every round trip.
var zoom = ZOOM_VALUE;

function show(camera) {
    var eye = camera.eye;
    box.textContent = 'CAMERA_KEY:'
        + '\n  x: ' + (eye.x / zoom).toFixed(3)
        + '\n  y: ' + (eye.y / zoom).toFixed(3)
        + '\n  z: ' + (eye.z / zoom).toFixed(3);
}

var initial = graph.layout.scene && graph.layout.scene.camera;
if (initial) { show(initial); }

graph.on('plotly_relayout', function (event) {
    if (event['scene.camera']) { show(event['scene.camera']); }
});
"""


def write_html(figure, path, camera_key=None, zoom=1.0):
    """Save a figure as an interactive plot, with a live camera readout if it has one.

    Orbit a 3D plot by hand and the readout names the camera position to paste
    into parameters.yaml under `camera_key`. A 2D projection has no camera to
    report, so it is written without one.
    """
    if camera_key is None:
        figure.write_html(path)
        return path

    script = CAMERA_READOUT.replace('CAMERA_KEY', camera_key)
    figure.write_html(path, post_script=script.replace('ZOOM_VALUE', repr(zoom)))
    return path
