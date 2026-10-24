# Copyright (c) 2026 Olegequ.
"""Author attribution at figure export; no change to numerical data."""


def save_figure(figure, *args, **kwargs):
    """Export a figure with a small author watermark, then restore the figure."""
    mark = figure.text(
        0.992, 0.006, "Olegequ", ha="right", va="bottom", fontsize=8,
        color="#374151", alpha=0.45, zorder=1000,
    )
    mark.set_gid("airwake-rotor-author-watermark")
    try:
        return figure.savefig(*args, **kwargs)
    finally:
        mark.remove()
