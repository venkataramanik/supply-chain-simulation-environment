# viz.py
# Complete Plotly animation builder + layout helper for non-animated previews.

import pandas as pd
import plotly.graph_objects as go


def warehouse_layout(width: int = 100, height: int = 60, n_docks: int = 3, n_rack_rows: int = 3):
    """Public layout helper (for quick preview)."""
    shapes = []
    docks_xy = []
    dock_h = height / (n_docks + 1)
    dock_w = 8
    for i in range(n_docks):
        y_center = (i + 1) * dock_h
        x0, x1 = 2, 2 + dock_w
        y0, y1 = y_center - dock_h / 3, y_center + dock_h / 3
        shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                           line=dict(color="black"), fillcolor="lightgray"))
        docks_xy.append(((x0 + x1) / 2, (y0 + y1) / 2))

    racks_xy = []
    rack_zone_x0, rack_zone_x1 = 30, 70
    rack_gap = (height - 10) / max(1, n_rack_rows)
    for r in range(n_rack_rows):
        y0 = 5 + r * rack_gap
        y1 = y0 + rack_gap * 0.5
        shapes.append(dict(type="rect", x0=rack_zone_x0, y0=y0, x1=rack_zone_x1, y1=y1,
                           line=dict(color="black"), fillcolor="#E8F0FE"))
        racks_xy.append(((rack_zone_x0 + rack_zone_x1) / 2, (y0 + y1) / 2))

    pack_x0, pack_x1 = 78, 95
    pack_y0, pack_y1 = height * 0.25, height * 0.75
    shapes.append(dict(type="rect", x0=pack_x0, y0=pack_y0, x1=pack_x1, y1=pack_y1,
                       line=dict(color="black"), fillcolor="#FFE8CC"))

    labels = {
        "docks_xy": docks_xy,
        "racks_xy": racks_xy,
        "pack_xy": ((pack_x0 + pack_x1) / 2, (pack_y0 + pack_y1) / 2),
        "dims": (width, height),
    }
    return shapes, labels


def _safe_label(df: pd.DataFrame, k: int) -> str:
    try:
        return str(int(df["t"].iloc[0]))
    except Exception:
        return str(k)


def build_plotly_animation(frames, shapes, labels):
    """Build an animated Plotly figure (robust to empty frames)."""
    fig = go.Figure()

    # Axes & layout
    fig.update_layout(
        shapes=shapes,
        xaxis=dict(range=[0, labels["dims"][0]], zeroline=False, showgrid=False, visible=False),
        yaxis=dict(range=[0, labels["dims"][1]], zeroline=False, showgrid=False, visible=False),
        width=900,
        height=540,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=[dict(
            type="buttons", showactive=False,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 150, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate", "transition": {"duration": 0}}]),
            ]
        )],
        sliders=[dict(
            steps=[dict(
                method="animate",
                args=[[str(k)], {"mode": "immediate",
                                 "frame": {"duration": 0, "redraw": True},
                                 "transition": {"duration": 0}}],
                label=_safe_label(frames[k], k) if len(frames) else "0"
            ) for k in range(len(frames))] or [dict(method="animate", args=[[], {}], label="0")],
            transition={"duration": 0},
            x=0, y=0, currentvalue=dict(prefix="Minute: ", visible=True), len=1.0
        )]
    )

    # Initial traces (even if empty)
    if len(frames):
        df0 = frames[0]
    else:
        df0 = pd.DataFrame([{"t": 0, "agent": "Picker-1", "type": "picker", "x": 50, "y": 30}])

    for typ, size in [("truck", 18), ("forklift", 10), ("picker", 8)]:
        sub = df0[df0["type"] == typ]
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers+text",
            text=sub["agent"], textposition="top center",
            marker=dict(size=size), name=typ.capitalize(),
            hovertemplate="%{text}<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
        ))

    # Animation frames
    fig.frames = []
    for k, df in enumerate(frames):
        data = []
        for typ, size in [("truck", 18), ("forklift", 10), ("picker", 8)]:
            sub = df[df["type"] == typ]
            data.append(go.Scatter(
                x=sub["x"], y=sub["y"], mode="markers+text",
                text=sub["agent"], textposition="top center",
                marker=dict(size=size), name=typ.capitalize(),
                hovertemplate="%{text}<br>(%{x:.1f}, %{y:.1f})<extra></extra>",
            ))
        fig.frames.append(go.Frame(data=data, name=str(k)))

    return fig
