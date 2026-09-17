"""
theme.py

Shared look for every Bokeh figure in trading/plots.

Colour is assigned by job, not by taste:
  * identity (which forecaster / scenario)  -> fixed categorical slots, never cycled
  * order (J = 1, 2, 4, 8, 16)               -> one blue ramp, light -> dark
  * reference (perfect foresight, actuals)  -> ink, so the coloured series read
                                               against it rather than compete with it
The categorical slots were checked for colour-vision-deficiency separation and
contrast against the light surface; text never wears a series colour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    DatetimeTickFormatter,
    Div,
    FixedTicker,
    HoverTool,
    Label,
    NumberFormatter,
    Span,
    StringFormatter,
    TableColumn,
)
from bokeh.plotting import figure, output_file, save

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, PAGE, GRID, AXIS = "#fcfcfb", "#f9f9f7", "#e1e0d9", "#c3c2b7"
GOOD, BAD = "#006300", "#d03b3b"
BLUE_RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # light -> dark

FORECASTER_COLOUR = {"perfect": INK, "perfect_price": INK2, "naive": AQUA, "aemo": BLUE, "lstm": ORANGE}
FORECASTER_DASH = {"perfect_price": "dashed"}          # the two perfect variants share ink; the dash tells them apart
FORECASTER_LABEL = {
    "perfect": "Perfect foresight",
    "perfect_price": "Perfect price, forecast load",
    "naive": "Seasonal naive",
    "aemo": "AEMO pre-dispatch",
    "lstm": "LSTM",
}
PERFECT_PRICE = {"perfect", "perfect_price"}          # forecasters with zero price error


def line_style(name: str) -> dict:
    return {"color": FORECASTER_COLOUR[name], "line_dash": FORECASTER_DASH.get(name, "solid")}
SCENARIO_COLOUR = {"bess_only": BLUE, "household": ORANGE, "state_machine": MUTED}
SCENARIO_LABEL = {"bess_only": "BESS only (arbitrage)", "household": "Household (solar + load)", "state_machine": "State machine"}

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
FONT = "system-ui, sans-serif"


def datetime_formatter() -> DatetimeTickFormatter:
    """A fresh formatter per figure: Bokeh models may belong to only one document."""
    return DatetimeTickFormatter(hours="%H:%M", days="%d %b", months="%d %b")


def colour_for_j(j: int, j_values=(1, 2, 4, 8, 16)) -> str:
    """Ordinal blue step for a segment count (light for J=1, dark for J=16)."""
    steps = BLUE_RAMP[1:]
    order = sorted(j_values)
    return steps[min(order.index(j), len(steps) - 1)] if j in order else BLUE


# ---------------------------------------------------------------------------
# Figure factory and chrome
# ---------------------------------------------------------------------------

def style(p, *, legend: bool = True):
    """Recessive chrome: hairline grid, no outline, muted axis text, quiet legend."""
    p.background_fill_color = SURFACE
    p.border_fill_color = SURFACE
    p.outline_line_color = None
    p.min_border_left = 56
    p.min_border_right = 24
    for g in (p.xgrid, p.ygrid):
        g.grid_line_color = GRID
        g.grid_line_width = 1
    p.xgrid.grid_line_color = None
    for ax in (p.xaxis, p.yaxis):
        ax.axis_line_color = AXIS
        ax.major_tick_line_color = AXIS
        ax.minor_tick_line_color = None
        ax.major_label_text_color = INK2
        ax.major_label_text_font = FONT
        ax.major_label_text_font_size = "11px"
        ax.axis_label_text_color = INK2
        ax.axis_label_text_font = FONT
        ax.axis_label_text_font_size = "11px"
        ax.axis_label_text_font_style = "normal"
    p.title.text_color = INK
    p.title.text_font = FONT
    p.title.text_font_size = "13px"
    p.title.text_font_style = "bold"
    p.toolbar.autohide = True
    p.toolbar.logo = None
    if legend and p.legend:
        lg = p.legend[0]
        lg.background_fill_alpha = 0.0
        lg.border_line_color = None
        lg.label_text_font = FONT
        lg.label_text_font_size = "11px"
        lg.label_text_color = INK2
        lg.glyph_width = 18
        lg.spacing = 2
        lg.padding = 6
        lg.click_policy = "hide"
    return p


def make_figure(*, height: int = 320, title: str = "", x_axis_type=None, tools: str = TOOLS, **kw):
    if x_axis_type is not None:
        kw["x_axis_type"] = x_axis_type
    p = figure(height=height, sizing_mode="stretch_width", title=title, tools=tools, toolbar_location="above", **kw)
    if x_axis_type == "datetime":
        p.xaxis.formatter = datetime_formatter()
    return p


def zero_line(p, dimension: str = "width"):
    p.add_layout(Span(location=0, dimension=dimension, line_color=AXIS, line_width=1))


def line_hover(p, renderers, tooltips, *, formatters=None, mode: str = "vline"):
    """Crosshair-style readout: one tooltip listing every series at the hovered x."""
    p.add_tools(HoverTool(renderers=renderers, tooltips=tooltips, mode=mode, formatters=formatters or {}, line_policy="nearest"))


# ---------------------------------------------------------------------------
# Price axis: asinh so a 17,500 $/MWh spike and the 30-300 $/MWh range both read
# ---------------------------------------------------------------------------

PRICE_SCALE = 100.0
PRICE_TICKS = (-1000, -300, -100, 0, 100, 300, 1000, 3000, 10000, 17500)


def price_to_axis(rrp) -> np.ndarray:
    return np.arcsinh(np.asarray(rrp, dtype=float) / PRICE_SCALE)


def price_from_axis(y) -> np.ndarray:
    return np.sinh(np.asarray(y, dtype=float)) * PRICE_SCALE


def asinh_price_axis(p, ticks=PRICE_TICKS, axis_label: str = "Price ($/MWh, asinh scale)"):
    """Fixed ticks at real prices on a y axis that holds asinh(price/100)."""
    pos = [float(v) for v in price_to_axis(ticks)]
    p.yaxis.ticker = FixedTicker(ticks=pos)
    p.yaxis.major_label_overrides = {t: f"{v:,}" for t, v in zip(pos, ticks)}
    p.yaxis.axis_label = axis_label
    p.ygrid.ticker = FixedTicker(ticks=pos)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def end_labels(p, items, *, y_span: float, inner_height_px: float, min_gap_px: float = 14.0, x_offset: int = 6):
    """
    Direct labels at line ends. items: list of (x, y, text, colour_ignored).
    Labels are pushed apart vertically (in data units) so they never overlap;
    text stays in ink, identity comes from the line the label sits beside.
    """
    if not items:
        return
    gap = min_gap_px * y_span / max(inner_height_px, 1.0)
    order = sorted(range(len(items)), key=lambda i: items[i][1])
    ys = [items[i][1] for i in order]
    for k in range(1, len(ys)):
        if ys[k] - ys[k - 1] < gap:
            ys[k] = ys[k - 1] + gap
    # centre the pushed stack back around the originals
    shift = (sum(ys) - sum(items[i][1] for i in order)) / len(ys)
    ys = [y - shift for y in ys]
    for k, i in enumerate(order):
        x, _, text = items[i][0], items[i][1], items[i][2]
        p.add_layout(Label(x=x, y=ys[k], text=text, x_offset=x_offset, y_offset=-6,
                           text_font=FONT, text_font_size="11px", text_color=INK2, text_baseline="middle"))


def value_label(p, x, y, text, *, x_offset=0, y_offset=0, align="center", baseline="bottom", size="11px", bold=False, colour=INK2):
    p.add_layout(Label(x=x, y=y, text=text, x_offset=x_offset, y_offset=y_offset, text_align=align, text_baseline=baseline,
                       text_font=FONT, text_font_size=size, text_color=colour,
                       text_font_style="bold" if bold else "normal"))


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------

def heading(title: str, subtitle: str = "") -> Div:
    sub = f'<div style="color:{INK2};font-size:13px;margin-top:4px;max-width:1100px;line-height:1.45">{subtitle}</div>' if subtitle else ""
    return Div(text=f'<div style="font:600 20px {FONT};color:{INK};margin:8px 0 0 0">{title}</div>{sub}',
               sizing_mode="stretch_width", styles={"margin-bottom": "6px"})


def section(title: str, note: str = "") -> Div:
    n = f'<div style="color:{INK2};font-size:12px;margin-top:2px;max-width:1100px;line-height:1.45">{note}</div>' if note else ""
    return Div(text=f'<div style="font:600 15px {FONT};color:{INK};margin-top:18px;padding-top:10px;border-top:1px solid {GRID}">{title}</div>{n}',
               sizing_mode="stretch_width")


def note(text: str) -> Div:
    return Div(text=f'<div style="font:12px {FONT};color:{MUTED};line-height:1.45;max-width:1100px">{text}</div>',
               sizing_mode="stretch_width")


def stat_tiles(tiles) -> Div:
    """
    KPI row. tiles: list of dicts with label, value (str), and optional sub (str) and tone ('good'|'bad'|None).
    """
    cells = []
    for t in tiles:
        tone = {"good": GOOD, "bad": BAD}.get(t.get("tone"), INK)
        sub = f'<div style="color:{MUTED};font-size:11px;margin-top:2px">{t["sub"]}</div>' if t.get("sub") else ""
        cells.append(
            f'<div style="flex:1 1 140px;min-width:140px;padding:10px 14px;background:{SURFACE};border:1px solid {GRID};border-radius:6px">'
            f'<div style="color:{INK2};font-size:12px">{t["label"]}</div>'
            f'<div style="color:{tone};font-size:24px;font-weight:600;margin-top:2px">{t["value"]}</div>{sub}</div>'
        )
    return Div(text=f'<div style="display:flex;gap:10px;flex-wrap:wrap;font-family:{FONT};margin:4px 0 8px 0">{"".join(cells)}</div>',
               sizing_mode="stretch_width")


def summary_table(df: pd.DataFrame, columns, *, height: int = 180, text_width: int = 170, number_width: int = 92) -> DataTable:
    """
    Table view of a summary frame. columns: list of (field, title, fmt) where fmt is
    a NumberFormatter format string ('0.00', '0.0%') or None for text.
    """
    src = ColumnDataSource(df.reset_index(drop=True))
    cols = []
    for field, title, fmt in columns:
        if field not in df:
            continue
        f = NumberFormatter(format=fmt, text_align="right") if fmt else StringFormatter()
        cols.append(TableColumn(field=field, title=title, formatter=f, width=number_width if fmt else text_width))
    return DataTable(source=src, columns=cols, height=height, sizing_mode="stretch_width", index_position=None,
                     autosize_mode="none", row_height=26)


def save_page(children, *, title: str, output_path: str):
    output_file(output_path, title=title)
    root = column(*children, sizing_mode="stretch_width", styles={"padding": "8px 18px 24px 18px", "background": PAGE, "font-family": FONT})
    save(root)
    print(f"Plot saved to {output_path}")
