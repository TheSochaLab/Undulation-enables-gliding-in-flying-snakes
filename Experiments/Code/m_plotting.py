# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:29:18 2015

@author: isaac
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox


def add_arrow_to_line2D(axes, line, arrow_locs=[.05, .275, .5, .725, .95],
                        arrowstyle='-|>', arrowsize=1, transform=None):
    """
    arrow_locs=[0.2, 0.4, 0.6, 0.8],

    http://stackoverflow.com/a/27666700


    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes:
    line: list of 1 Line2D obbject as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    if (not(isinstance(line, list)) or not(isinstance(line[0],
                                           mlines.Line2D))):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line[0].get_xdata(), line[0].get_ydata()

    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)

    color = line[0].get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw['color'] = color

    linewidth = line[0].get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    for loc in arrow_locs:
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        # arrow_tail = (np.mean(x[n - 1:n + 1]), np.mean(y[n - 1:n + 1]))
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(arrow_tail, arrow_head,
                                     transform=transform, **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows


class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).

        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor

        # Adapted from mpl_toolkits.axes_grid2
        # LICENSE: Python Software Foundation (http://docs.python.org/license.html)
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0,0), sizex, 0, fc="none"))
        if sizey:
            bars.add_artist(Rectangle((0,0), 0, sizey, fc="none"))

        if sizex and labelx:
            bars = VPacker(children=[bars, TextArea(labelx, minimumdescent=False)],
                           align="center", pad=0, sep=sep)
        if sizey and labely:
            bars = HPacker(children=[TextArea(labely), bars],
                            align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)


def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """ Add scalebars to axes

    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars

    Returns created scalebar object
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l)>1 and (l[1] - l[0])

    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])

    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)

    return sb


def plot_residuals(pr, R, fcs, inter=None, fcopt=None, markers=True):
    """Plot the residual vs cutoff frequency for all of the markers.
    """

    from itertools import cycle
    from seaborn import husl_palette, despine

    ntime, nmark, ncoord = pr.shape

    colors = husl_palette(n_colors=nmark)

    if markers:
        marks = cycle(['o', '^', '<', '>', 's', 'h', 'd'])
    else:
        marks = cycle([''])

    fig, ax = plt.subplots()
    for j in np.arange(nmark):
        mark = next(marks) + '-'
        c = colors[j]
        al = .1
        ax.plot(fcs, R[:, j, 0], mark, c=c, ms=4, alpha=al)
        ax.plot(fcs, R[:, j, 1], mark, c=c, ms=4, alpha=al)
        ax.plot(fcs, R[:, j, 2], mark, c=c, ms=4, alpha=al)

        if inter is not None and fcopt is not None:
            nn = np.zeros(3)
            ax.plot(nn, inter[j], '>', ms=5, color=c)
            ax.plot(fcopt[j], nn, '^', ms=5, color=c)
            ax.plot(fcopt[j], inter[j], mark[:-1], c=c, ms=6)

    if inter is not None:
        ax.margins(.03, .003)

    ax.set_ylim(ymax=10)
    ax.grid(True)
    ax.set(xlabel='Cutoff frequency (Hz)', ylabel='Residual (mm)')
    despine()
    fig.set_tight_layout(True)

    return fig, ax
