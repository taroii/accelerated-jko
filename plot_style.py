"""Shared paper plotting style.

Single source of truth for figure colors, fonts, and rcParams across all
JKO experiment scripts. Two algorithms per panel; Okabe-Ito blue and
vermillion are colorblind-safe and have strong luminance contrast.
"""
from __future__ import annotations

import matplotlib as mpl


_OKABE_BLUE      = "#0072B2"
_OKABE_VERMILION = "#D55E00"


ALGO_STYLE = {
    "Standard JKO":    {"color": _OKABE_BLUE,      "marker": "o"},
    "Accelerated JKO": {"color": _OKABE_VERMILION, "marker": "s"},
    "SGA":             {"color": _OKABE_BLUE,      "marker": "o"},
    "ASGA":            {"color": _OKABE_VERMILION, "marker": "s"},
}


def style_for(name):
    return ALGO_STYLE.get(name, {"color": None, "marker": "o"})


def color_for(name):
    return style_for(name)["color"]


def apply_paper_style():
    mpl.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.0,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
