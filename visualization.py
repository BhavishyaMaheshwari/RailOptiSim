"""
RailOptimusSim Visualization Module

This module provides comprehensive visualization capabilities for the railway simulation system.
It includes advanced plotting functions for track timelines, train journeys, and Gantt charts
with professional styling and interactive features.

Features:
- Track Timeline Visualization with real-time updates
- Train Journey Gantt Charts with status indicators
- Interactive hover information and event markers
- Professional color schemes and styling
- Enhanced visual indicators for resume/exit events
- Multi-platform and multi-station support

Author: RailOptimusSim Development Team
Version: 2.0 Professional Edition
"""

from collections import defaultdict
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import format_node, short_node

# -------------------------------
# Helpers
# -------------------------------
def _quad_bezier_path(x0, y0, x1, y1, curvature=0.35, vertical=False):
    """Return an SVG quadratic Bezier path string from (x0,y0) to (x1,y1).
    If vertical is False, bulge in Y; else bulge in X.
    """
    if vertical:
        cx = (x0 + x1) / 2.0 + curvature * (1 if y1 >= y0 else -1)
        cy = (y0 + y1) / 2.0
    else:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0 + curvature * (1 if x1 >= x0 else -1)
    return f"M {x0},{y0} Q {cx},{cy} {x1},{y1}"

def _infer_tracks_sections_from_graph(G):
    max_track, max_sec = -1, -1
    for n in G.nodes:
        if isinstance(n, tuple) and n and n[0] != "Platform":
            max_track = max(max_track, int(n[0]))
            max_sec = max(max_sec, int(n[1]))
    return (max_track + 1 if max_track >= 0 else 5), (max_sec + 1 if max_sec >= 0 else 4)

def _infer_dims_from_state(state):
    """Infer max track index and section index from state."""
    max_track, max_sec = -1, -1
    for st in state.values():
        for n in st.get("planned_path", []):
            if isinstance(n, tuple) and n and n[0] != "Platform":
                max_track = max(max_track, int(n[0]))
                max_sec = max(max_sec, int(n[1]))
        pos = st.get("pos")
        if isinstance(pos, tuple) and pos and pos[0] != "Platform":
            max_track = max(max_track, int(pos[0]))
            max_sec = max(max_sec, int(pos[1]))
    return (max_track + 1 if max_track >= 0 else 5), (max_sec + 1 if max_sec >= 0 else 4)

def enhance_for_hd(fig, scale=1.25):
    """Upscale figure fonts and canvas for high-definition presentation."""
    if fig is None:
        return fig
    base_w = fig.layout.width or 1200
    base_h = fig.layout.height or 600
    fig.update_layout(width=int(base_w * scale), height=int(base_h * scale))
    # Update global font size
    current_font = (fig.layout.font.size if fig.layout.font and fig.layout.font.size else 12)
    fig.update_layout(font=dict(size=int(current_font * scale)))
    # Legend font if present
    if fig.layout.legend and fig.layout.legend.font and fig.layout.legend.font.size:
        fig.update_layout(legend=dict(font=dict(size=int(fig.layout.legend.font.size * scale))))
    return fig

def calculate_delays(df):
    """Calculate comprehensive delay and impact statistics"""
    stats = {
        'delays': defaultdict(int),
        'reroutes': defaultdict(int),
        'affected': defaultdict(int),
        'current_delays': defaultdict(list),
        'blocked_sections': set(),
        'impact_by_event': defaultdict(lambda: {
            'affected_trains': set(),
            'rerouted_trains': set(),
            'total_delay': 0,
            'active_delays': []
        })
    }
    
    for _, row in df.iterrows():
        train = row["train"]
        action = row.get("action", "")
        event_id = row.get("event_id")
        
        if action == "runtime_plan":
            stats['reroutes'][train] += 1
            if event_id:
                stats['impact_by_event'][event_id]['rerouted_trains'].add(train)
                delay = row.get("delay", 0)
                if delay:
                    stats['impact_by_event'][event_id]['total_delay'] += delay
                    
        elif action in ["affected_by_accident", "involved_in_accident"]:
            stats['affected'][train] += 1
            duration = row.get("duration", 0)
            stats['delays'][train] += duration
            
            if event_id:
                stats['impact_by_event'][event_id]['affected_trains'].add(train)
                if duration:
                    stats['impact_by_event'][event_id]['active_delays'].append({
                        'train': train,
                        'duration': duration
                    })
            
            if isinstance(row["node"], tuple):
                stats['blocked_sections'].add(row["node"])
                
        elif action == "wait_blocked_section":
            stats['affected'][train] += 1
            stats['delays'][train] += 1
                
    return stats

def build_records_from_state(state):
    """Build DataFrame from simulation state"""
    recs = []
    for tid, st in state.items():
        info = st["info"]
        path = st["planned_path"]
        slots = st["planned_slots"]
        pos = st["pos"]
        status = st["status"]
        log = st.get("log", [])
        
        if not path or not slots:
            continue
            
        for node, slot in zip(path, slots):
            action = None
            duration = None
            event_id = None
            
            # Handle both dictionary and tuple log entries
            for log_entry in log:
                if isinstance(log_entry, dict):
                    if log_entry.get("slot") == slot:
                        action = log_entry.get("action")
                        duration = log_entry.get("duration")
                        event_id = log_entry.get("event_id")
                        break
                elif isinstance(log_entry, tuple) and len(log_entry) >= 2:
                    if log_entry[0] == slot:
                        action = log_entry[1]
                        break
                    
            recs.append({
                "train": tid,
                "type": info.type,
                "priority": info.priority,
                "node": node,
                "slot": slot,
                "status": status,
                "action": action,
                "duration": duration,
                "event_id": event_id
            })
            
    if not recs:
        return pd.DataFrame()
        
    df = pd.DataFrame(recs)
    df = df.sort_values(["train", "slot"])
    return df

def plot_track_timeline(state, trains, accident_mgr=None, current_slot=None):
    """
    Enhanced track timeline showing which train is on which track over time.
    Dynamically adjusts to network size and shows multiple platforms.
    """
    df = build_records_from_state(state)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No track timeline data (Current Slot: {current_slot})")
        return fig

    # Infer network dimensions dynamically
    num_tracks, num_sections = _infer_dims_from_state(state)
    trains_list = sorted(df["train"].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    
    # Enhanced color palette for better distinction
    unique_train_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F8C471", "#82E0AA", "#AED6F1", "#D7DBDD", "#F9E79F"
    ]
    
    color_map = {tid: unique_train_colors[i % len(unique_train_colors)] 
                for i, tid in enumerate(trains_list)}
    
    fig = go.Figure()
    # determine x-axis max for background shapes
    x_max = (current_slot + 10) if current_slot is not None else (df["slot"].max() + 5)
    
    # Draw enhanced grid with better visibility
    for tr in range(num_tracks):
        # alternating banding for readability
        if tr % 2 == 1:
            fig.add_shape(type="rect",
                          x0=-0.5, x1=x_max,
                          y0=tr-0.5, y1=tr+0.5,
                          fillcolor="rgba(0,0,0,0.03)",
                          line=dict(width=0), layer="below")
        fig.add_shape(type="line", x0=-0.5, x1=x_max, 
                      y0=tr-0.5, y1=tr-0.5,
                      line=dict(color="#E5E7E9", width=1, dash="dot"), layer="below")
    
    # Add blocked sections visualization
    blocked_sections = set()
    if accident_mgr is not None and current_slot is not None:
        actives = accident_mgr.active_summary(current_slot)
        for eid, evtype, loc, remaining, stats in actives:
            if isinstance(loc, tuple) and loc[0] != "Platform":
                track = loc[0]
                # Block entire track for accidents
                for offset in range(remaining):
                    time_slot = current_slot + offset
                    fig.add_shape(
                        type="rect",
                        x0=time_slot - 0.5, x1=time_slot + 0.5,
                        y0=track - 0.4, y1=track + 0.4,
                        fillcolor="rgba(255, 0, 0, 0.3)",
                        line=dict(color="red", width=2),
                        layer="below"
                    )
                
                # Add accident marker
                fig.add_annotation(
                    x=current_slot + remaining/2, y=track,
                    text=f"🚨 BLOCKED",
                    showarrow=False,
                    font=dict(color="red", size=12, weight="bold"),
                    bgcolor="rgba(255,255,255,0.9)"
                )

    # Plot train paths with enhanced visualization
    for tid in trains_list:
        sub = df[df["train"] == tid]
        train_type = [t.type for t in trains if t.id == tid][0]
        
        # Separate past and future if current_slot is available
        if current_slot is not None:
            past_data = sub[sub["slot"] <= current_slot]
            future_data = sub[sub["slot"] > current_slot]
        else:
            past_data = sub
            future_data = pd.DataFrame()
        
        # Plot past path (solid line)
        if not past_data.empty:
            xs, ys = [], []
            for _, row in past_data.iterrows():
                node = row["node"]
                if isinstance(node, tuple) and node[0] != "Platform":
                    xs.append(row["slot"])
                    ys.append(node[0])
            
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    line=dict(color=color_map[tid], width=4),
                    marker=dict(size=8, color=color_map[tid]),
                    name=f"{tid} ({train_type})",
                    hovertemplate=f"<b>{tid}</b><br>Track: %{{y}}<br>Time: %{{x}}<extra></extra>"
                ))
        
        # Plot future path (dashed line)
        if not future_data.empty:
            xs, ys = [], []
            for _, row in future_data.iterrows():
                node = row["node"]
                if isinstance(node, tuple) and node[0] != "Platform":
                    xs.append(row["slot"])
                    ys.append(node[0])
            
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    line=dict(color=color_map[tid], width=2, dash="dot"),
                    marker=dict(size=6, color=color_map[tid], opacity=0.6),
                    name=f"{tid} (Predicted)",
                    opacity=0.6,
                    showlegend=False,
                    hovertemplate=f"<b>{tid}</b> (Predicted)<br>Track: %{{y}}<br>Time: %{{x}}<extra></extra>"
                ))

        # Add curved arc shapes for track switches ("jumping trains")
        # Build sequence of (slot, track) for this train
        seq = []
        for _, row in sub.iterrows():
            node = row["node"]
            if isinstance(node, tuple) and node[0] != "Platform":
                seq.append((int(row["slot"]), int(node[0])))
        seq.sort()
        for i in range(1, len(seq)):
            x0, y0 = seq[i-1]
            x1, y1 = seq[i]
            if y0 != y1:  # a switch occurred
                # create a smooth curve between tracks across time
                path = _quad_bezier_path(x0, y0, x1, y1, curvature=0.5, vertical=False)
                fig.add_shape(
                    type="path",
                    path=path,
                    line=dict(color=color_map[tid], width=3),
                    opacity=0.85,
                    layer="above"
                )
        
        # Add special event markers
        train_state = state.get(tid, {})
        for log_entry in train_state.get("log", []):
            if isinstance(log_entry, tuple) and len(log_entry) >= 4:
                slot, prev_node, node, action = log_entry[0], log_entry[1], log_entry[2], log_entry[3]
                
                if action == "resume" and isinstance(node, tuple) and node[0] != "Platform":
                    fig.add_trace(go.Scatter(
                        x=[slot], y=[node[0]],
                        mode="markers+text",
                        marker=dict(symbol="star", size=20, color="#00FF7F"),
                        text=["★"],
                        name="Resume",
                        showlegend=False,
                        hovertemplate=f"<b>{tid}</b> RESUMED<br>Track: {node[0]}<br>Time: {slot}<extra></extra>"
                    ))
                elif action == "completed" and isinstance(node, tuple) and node[0] != "Platform":
                    fig.add_trace(go.Scatter(
                        x=[slot], y=[node[0]],
                        mode="markers+text",
                        marker=dict(symbol="circle", size=15, color="#32CD32"),
                        text=["✓"],
                        name="Completed",
                        showlegend=False,
                        hovertemplate=f"<b>{tid}</b> COMPLETED<br>Track: {node[0]}<br>Time: {slot}<extra></extra>"
                    ))
    
    # Add current time indicator
    if current_slot is not None:
        fig.add_vline(
            x=current_slot,
            line_width=3,
            line_color="#E74C3C",
            annotation=dict(
                text="Current Time",
                textangle=-90,
                font=dict(size=12, color="#E74C3C")
            )
        )
    
    # Calculate statistics
    completed_trains = len([t for t in trains_list if state.get(t, {}).get("status") == "completed"])
    blocked_trains = len([t for t in trains_list if state.get(t, {}).get("status") == "blocked_by_accident"])
    running_trains = len([t for t in trains_list if state.get(t, {}).get("status") == "running"])
    
    fig.update_layout(
        title=dict(
            text=f"🚂 RAILWAY TRACK TIMELINE | Current: Slot {current_slot} | ✅ Completed: {completed_trains} | 🚫 Blocked: {blocked_trains} | 🚂 Running: {running_trains}",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title="⏰ Time (slots)",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title="🛤️ Railway Tracks",
            tickmode="array",
            tickvals=list(range(num_tracks)),
            ticktext=[f"Track {i+1}" for i in range(num_tracks)],
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        height=600,
        width=1400,
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(r=50, t=80),
        hovermode="x unified"
    )
    
    return fig

def plot_network_map(G, state, platforms, current_slot=None, accident_mgr=None, lookahead_slots=6):
    """
    Render a schematic network map with tracks, sections, platform connectors and trains.
    Includes curved arcs for switches and platform entries. Positions:
    - Sections laid out horizontally (x=section), tracks vertically (y=track index)
    - Platforms placed to the right, vertically aligned to average of feeding tracks
    """
    # Infer dims
    num_tracks, num_sections = _infer_tracks_sections_from_graph(G)

    # Layout positions for section nodes
    pos = {}
    for t in range(num_tracks):
        for s in range(num_sections):
            pos[(t, s)] = (s, t)

    # Platform positions: right of last section
    plat_pos = {}
    x_plat = num_sections + 0.8
    # group platforms by their feeding tracks to position nicely
    for p in platforms:
        try:
            preds = list(G.predecessors(p))
        except Exception:
            preds = []
        ys = [pred[0] for pred in preds if isinstance(pred, tuple) and pred and pred[0] != "Platform"]
        y_avg = (sum(ys) / len(ys)) if ys else (num_tracks / 2.0)
        # small offset by platform index for separation
        offset = (p[2] if isinstance(p, tuple) and len(p) >= 3 else 0) * 0.25
        plat_pos[p] = (x_plat + offset, y_avg)

    fig = go.Figure()

    # Draw rails (tracks)
    for t in range(num_tracks):
        fig.add_shape(type="line",
                      x0=-0.25, x1=num_sections - 0.25,
                      y0=t, y1=t,
                      line=dict(color="#BDC3C7", width=6), layer="below")
        # Track labels on the left for readability
        fig.add_annotation(x=-0.8, y=t, text=f"Track {t+1}", showarrow=False,
                           font=dict(size=12, color="#2C3E50"), xanchor="right", yanchor="middle")
        # section markers
        for s in range(num_sections):
            fig.add_shape(type="line",
                          x0=s, x1=s,
                          y0=t - 0.18, y1=t + 0.18,
                          line=dict(color="#ECF0F1", width=2), layer="below")

    # Draw connectors (switches between adjacent tracks at same section)
    for t in range(num_tracks - 1):
        for s in range(num_sections):
            x0, y0 = pos[(t, s)]
            x1, y1 = pos[(t+1, s)]
            path = _quad_bezier_path(x0, y0, x1, y1, curvature=0.35, vertical=True)
            fig.add_shape(type="path", path=path, line=dict(color="#D0D3D4", width=2), layer="below")

    # Draw platform connectors from last sections
    last_s = num_sections - 1
    for p, (xp, yp) in plat_pos.items():
        try:
            preds = list(G.predecessors(p))
        except Exception:
            preds = []
        for pred in preds:
            if isinstance(pred, tuple) and pred[0] != "Platform":
                xs, ys = pos[pred]
                path = _quad_bezier_path(xs, ys, xp, yp, curvature=0.6, vertical=False)
                fig.add_shape(type="path", path=path, line=dict(color="#95A5A6", width=2, dash="dot"), layer="below")

    # Platform nodes as labeled points and station grouping
    # Group platforms by station id
    station_to_plats = {}
    for p in platforms:
        if isinstance(p, tuple) and len(p) >= 3:
            station_to_plats.setdefault(p[1], []).append(p)

    # Draw subtle grouping rectangles per station around platform cluster
    for st_id, ps in station_to_plats.items():
        ys = [plat_pos[p][1] for p in ps if p in plat_pos]
        if not ys:
            continue
        y0, y1 = min(ys) - 0.5, max(ys) + 0.5
        x0, x1 = x_plat - 0.4, x_plat + 1.0
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor="rgba(241, 196, 15, 0.06)", line=dict(color="#D4AC0D", width=1, dash="dot"),
                      layer="below")
        fig.add_annotation(x=x1, y=(y0+y1)/2.0, text=f"Station {st_id+1}", showarrow=False,
                           font=dict(size=12, color="#7D6608"), xanchor="left", yanchor="middle")

    # Platform nodes as labeled points
    for p, (xp, yp) in plat_pos.items():
        fig.add_trace(go.Scatter(
            x=[xp], y=[yp], mode="markers+text",
            marker=dict(size=14, color="#F1C40F", line=dict(width=2, color="#7D6608")),
            text=[short_node(p)], textposition="top center",
            name=format_node(p), showlegend=False,
            hovertemplate=f"{format_node(p)}<extra></extra>"
        ))

    # Color map per train
    trains_list = sorted([tid for tid in state.keys()])
    palette = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F8C471", "#82E0AA", "#AED6F1", "#D7DBDD", "#F9E79F"
    ]
    color_map = {tid: palette[i % len(palette)] for i, tid in enumerate(trains_list)}

    # Draw trains: current positions and near-future path (spline)
    for tid in trains_list:
        st = state.get(tid, {})
        path_nodes = st.get("planned_path", [])
        slots = st.get("planned_slots", [])
        # Determine start index near current_slot
        start_idx = 0
        if current_slot is not None and slots:
            for i, sl in enumerate(slots):
                if sl >= current_slot:
                    start_idx = max(0, i - 1)  # include one past node
                    break
        future_nodes = path_nodes[start_idx:start_idx + lookahead_slots]
        xs, ys = [], []
        for n in future_nodes:
            if isinstance(n, tuple) and n and n[0] == "Platform":
                x, y = plat_pos.get(n, (x_plat, num_tracks/2.0))
            elif isinstance(n, tuple):
                x, y = pos.get(n, (None, None))
            else:
                x, y = (None, None)
            if x is not None:
                xs.append(x)
                ys.append(y)
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                line=dict(color=color_map[tid], width=4, shape="spline", smoothing=1.3),
                marker=dict(size=10, color=color_map[tid]),
                name=tid,
                hovertemplate=f"<b>{tid}</b><extra></extra>"
            ))

        # Current position marker
        cur = st.get("pos")
        if isinstance(cur, tuple) and cur and cur[0] == "Platform":
            x, y = plat_pos.get(cur, (x_plat, num_tracks/2.0))
        elif isinstance(cur, tuple):
            x, y = pos.get(cur, (None, None))
        else:
            x, y = (None, None)
        if x is not None:
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text",
                marker=dict(size=16, color=color_map[tid], line=dict(width=2, color="#2C3E50")),
                text=["🚂"], textposition="bottom center",
                name=f"{tid} (now)", showlegend=False,
                hovertemplate=f"<b>{tid}</b> (current)<extra></extra>"
            ))

    # Accident overlays
    if accident_mgr is not None and current_slot is not None:
        for eid, evtype, loc, remaining, stats in accident_mgr.active_summary(current_slot):
            if isinstance(loc, tuple) and loc and loc[0] != "Platform":
                x, y = pos.get(loc, (None, None))
            elif isinstance(loc, tuple):
                x, y = plat_pos.get(loc, (None, None))
            else:
                x, y = (None, None)
            if x is not None:
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode="markers+text",
                    marker=dict(symbol="x", size=24, color="#E74C3C", line=dict(width=3, color="#922B21")),
                    text=["🚨"], textposition="top center",
                    name="Accident", showlegend=False,
                    hovertemplate=f"{evtype.upper()} at {format_node(loc)}<br>rem: {remaining}<extra></extra>"
                ))

    # Layout styling
    fig.update_layout(
        title=dict(text="🗺️ Network Map (curved jumps & live paths)", x=0.5, font=dict(size=18)),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=600, width=1400, showlegend=True,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="#FBFCFC", paper_bgcolor="#FFFFFF",
        hovermode="closest"
    )
    return fig

def plot_platform_heatmap(state, platforms, current_slot=None, selected_platforms=None):
    """Heatmap of platform occupancy over time (easier to grasp vs. bar charts)."""
    # Build rows of (platform, slot)
    rows = []
    for tid, st in state.items():
        path = st.get("planned_path", [])
        slots = st.get("planned_slots", [])
        for n, s in zip(path, slots):
            if isinstance(n, tuple) and len(n) >= 1 and n[0] == "Platform":
                rows.append({"platform": n, "slot": int(s)})
    if not rows:
        fig = go.Figure(); fig.update_layout(title="No platform activity detected"); return fig

    df = pd.DataFrame(rows)
    # Filter platforms if provided
    if selected_platforms:
        selected = set(selected_platforms)
        df = df[df["platform"].isin(selected)]

    # Determine time range
    min_slot = int(df["slot"].min())
    max_slot = int(df["slot"].max())
    # Ensure current_slot is visible
    if current_slot is not None:
        max_slot = max(max_slot, int(current_slot) + 1)

    # Prepare matrix: rows = platforms, cols = slots
    plat_values = sorted(df["platform"].unique(), key=lambda p: (p[1], p[2]))
    times = list(range(min_slot, max_slot + 1))
    occupancy = []
    for p in plat_values:
        counts = df[df["platform"] == p].groupby("slot").size().to_dict()
        row_vals = [counts.get(t, 0) for t in times]
        occupancy.append(row_vals)

    # Build Heatmap
    z = occupancy
    y_labels = [format_node(p) for p in plat_values]
    x_labels = times
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        colorscale=[[0, "#FBFCFC"], [1e-6, "#D6EAF8"], [0.5, "#76D7C4"], [1.0, "#1ABC9C"]],
        colorbar=dict(title="Trains"),
        hovertemplate="Platform: %{y}<br>Slot: %{x}<br>Trains: %{z}<extra></extra>",
        zmin=0, zmax=max(1, max(max(r) for r in z) if z else 1)
    ))

    # Current time indicator
    if current_slot is not None:
        fig.add_vline(x=int(current_slot), line_color="red", line_width=2)

    fig.update_layout(
        title=dict(text="🚉 Platform Occupancy Heatmap", x=0.5, font=dict(size=16)),
        xaxis_title="Time (slots)",
        yaxis_title="Platforms",
        height=max(400, 40 * len(plat_values)),
        width=1200,
        hovermode="x unified"
    )
    return fig

def plot_train_timeline(state, trains, accident_mgr=None):
    """
    Enhanced train position timeline showing section progression over time.
    """
    df = build_records_from_state(state)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No timeline data")
        return fig
    
    num_tracks, num_sections = _infer_dims_from_state(state)
    trains_list = sorted(df["train"].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    
    # Enhanced color palette
    unique_train_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
    ]
    
    color_map = {tid: unique_train_colors[i % len(unique_train_colors)] 
                for i, tid in enumerate(trains_list)}
    
    fig = go.Figure()
    
    # Plot each train's journey through sections with richer markers
    for tid in trains_list:
        sub = df[df["train"] == tid]
        train_type = [t.type for t in trains if t.id == tid][0]
        
        xs, ys, hover_texts, marker_colors, marker_sizes, marker_symbols = [], [], [], [], [], []
        
        for _, row in sub.iterrows():
            slot = row["slot"]
            node = row["node"]
            action = row["action"]
            
            if isinstance(node, tuple) and node[0] == "Platform":
                y = -0.5  # Platform position
                # Differentiate arrival vs generic platform presence vs departure
                if action == "enter":
                    marker_color = "#FFD700"
                    marker_size = 16
                    marker_symbol = "triangle-up"
                    hover_text = f"{tid} ARRIVED at {format_node(node)}<br>Slot: {slot}"
                elif action == "depart":
                    marker_color = "#32CD32"
                    marker_size = 16
                    marker_symbol = "triangle-down"
                    hover_text = f"{tid} DEPARTED from {format_node(node)}<br>Slot: {slot}"
                else:
                    marker_color = "#FFD700"
                    marker_size = 14
                    marker_symbol = "star"
                    hover_text = f"{tid} at {format_node(node)}<br>Slot: {slot}"
            elif isinstance(node, tuple):
                y = node[1]  # Section number
                hover_text = f"{tid}<br>Track: {node[0]+1}, Section: {node[1]+1}<br>Slot: {slot}"
                
                # Base style
                marker_color = color_map[tid]
                marker_size = 10
                marker_symbol = "circle"

                # Special actions
                if action == "runtime_plan":
                    marker_color = "#FF8C00"
                    marker_size = 12
                    marker_symbol = "diamond"
                    hover_text += "<br>🔄 REROUTED"
                elif action in ["involved_in_accident", "affected_by_accident"]:
                    marker_color = "#DC143C"
                    marker_size = 14
                    marker_symbol = "x"
                    hover_text += "<br>🚨 ACCIDENT"
                elif action == "switch":
                    marker_color = color_map[tid]
                    marker_size = 14
                    marker_symbol = "diamond-cross"
                    hover_text += "<br>⇄ SWITCHED TRACK"
            else:
                continue
                
            xs.append(slot)
            ys.append(y)
            hover_texts.append(hover_text)
            marker_colors.append(marker_color)
            marker_sizes.append(marker_size)
            marker_symbols.append(marker_symbol)
        
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines+markers+text",
                line=dict(color=color_map[tid], width=3),
                marker=dict(size=marker_sizes, color=marker_colors, symbol=marker_symbols,
                            line=dict(width=1, color="#2C3E50")),
                name=f"{tid} ({train_type})",
                hovertemplate="%{text}<extra></extra>",
                text=hover_texts
            ))
    
    # Add accident markers
    if accident_mgr is not None:
        all_slots = set(df["slot"].unique()) if not df.empty else set()
        for slot in sorted(all_slots):
            actives = accident_mgr.active_summary(slot)
            for eid, evtype, loc, rem, stats in actives:
                if isinstance(loc, tuple):
                    if loc[0] == "Platform":
                        y = -0.5
                    else:
                        y = loc[1]
                    
                    fig.add_trace(go.Scatter(
                        x=[slot], y=[y],
                        mode="markers+text",
                        marker=dict(symbol="x", size=20, color="red"),
                        text=[f"🚨"],
                        name="Accident",
                        showlegend=False,
                        hovertemplate=f"Accident at {format_node(loc)}<br>Slot: {slot}<extra></extra>"
                    ))
    
    fig.update_layout(
        title=dict(
            text="🚂 TRAIN POSITION TIMELINE",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="⏰ Time (slots)",
        yaxis=dict(
            title="📍 Track Sections",
            tickmode="array",
            tickvals=[-0.5] + list(range(num_sections)),
            ticktext=["🏁 Platform"] + [f"📍 Section {i+1}" for i in range(num_sections)],
            range=[-1, max(0.5, num_sections - 0.5)],
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        height=500,
        width=1200,
        showlegend=True,
        legend=dict(orientation="h", y=-0.2)
    )
    
    return fig

def plot_station_overview_combined(state, selected_platforms, current_slot=None):
    """
    Combined platform occupancy view.
    Stacks bars per slot by platform to see multiple platforms together.
    """
    rows = []
    for tid, st in state.items():
        path = st.get("planned_path", [])
        slots = st.get("planned_slots", [])
        for n, s in zip(path, slots):
            if isinstance(n, tuple) and len(n) >= 1 and n[0] == "Platform":
                rows.append({"platform": n, "slot": s, "train": tid})
    df = pd.DataFrame(rows)
    if df.empty:
        fig = go.Figure(); fig.update_layout(title="No platform activity detected"); return fig

    # Filter platforms
    if selected_platforms:
        df = df[df["platform"].isin(set(selected_platforms))]
    plat_values = sorted(df["platform"].unique(), key=lambda p: (p[1], p[2]))

    # Count trains per platform per slot
    cnt = df.groupby(["slot", "platform"]).size().reset_index(name="count")
    fig = go.Figure()

    # Assign colors consistently per platform
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    colormap = {p: palette[i % len(palette)] for i, p in enumerate(plat_values)}

    for p in plat_values:
        sub = cnt[cnt["platform"] == p]
        fig.add_trace(go.Bar(
            x=sub["slot"],
            y=sub["count"],
            name=format_node(p),
            marker=dict(color=colormap[p], line=dict(color="#222", width=1)),
            hovertemplate=f"{format_node(p)}<br>Trains: %{{y}}<br>Slot: %{{x}}<extra></extra>"
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(text="🚉 Combined Platform Occupancy", x=0.5, font=dict(size=16)),
        height=500,
        width=1200,
        legend=dict(orientation="h", y=-0.2),
        xaxis_title="Time (slots)",
        yaxis_title="Train count"
    )

    if current_slot is not None:
        fig.add_vline(x=current_slot, line_color="red", line_width=2)

    return fig

def plot_gantt_chart(state, trains, accident_mgr=None, current_slot=None):
    """
    Enhanced Gantt chart showing train journeys with sophisticated visualization.
    """
    df = build_records_from_state(state)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Gantt data")
        return fig

    trains_list = sorted(df["train"].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    stats = calculate_delays(df)
    
    # Enhanced color palette
    unique_train_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
    ]
    
    color_map = {tid: unique_train_colors[i % len(unique_train_colors)] 
                for i, tid in enumerate(trains_list)}
    
    fig = go.Figure()
    
    # Create Gantt bars for each train
    for tid in trains_list:
        sub = df[df["train"] == tid]
        train_type = [t.type for t in trains if t.id == tid][0]
        
        if not sub.empty:
            start_slot = sub["slot"].min()
            end_slot = sub["slot"].max()
            duration = end_slot - start_slot + 1
            
            # Determine status and color
            train_state = state.get(tid, {})
            status = train_state.get("status", "unknown")
            delays = stats['delays'][tid]
            reroutes = stats['reroutes'][tid]
            
            if status == "completed":
                bar_color = color_map[tid]
                opacity = 1.0
            elif status == "blocked_by_accident":
                bar_color = "#DC143C"
                opacity = 0.8
            else:
                bar_color = color_map[tid]
                opacity = 0.7 if delays > 0 else 1.0
            
            # Create hover text
            hover_text = (
                f"<b>{tid}</b> ({train_type})<br>"
                f"Start: Slot {start_slot}<br>"
                f"End: Slot {end_slot}<br>"
                f"Duration: {duration} slots<br>"
                f"Status: {status}<br>"
                f"Delays: {delays} slots<br>"
                f"Reroutes: {reroutes}"
            )
            
            fig.add_trace(go.Bar(
                x=[duration],
                y=[tid],
                base=start_slot,
                orientation='h',
                marker=dict(
                    color=bar_color,
                    opacity=opacity,
                    line=dict(width=2, color="#2C3E50")
                ),
                name=f"{tid} ({train_type})",
                hovertemplate=hover_text + "<extra></extra>"
            ))
    
    # Add event markers
    for tid in trains_list:
        train_state = state.get(tid, {})
        for log_entry in train_state.get("log", []):
            if isinstance(log_entry, tuple) and len(log_entry) >= 4:
                slot, prev_node, node, action = log_entry[0], log_entry[1], log_entry[2], log_entry[3]
                
                if action == "resume":
                    fig.add_trace(go.Scatter(
                        x=[slot], y=[tid],
                        mode="markers+text",
                        marker=dict(symbol="star", size=25, color="#00FF7F"),
                        text=["★"],
                        name="Resume",
                        showlegend=False,
                        hovertemplate=f"<b>{tid}</b> RESUMED at slot {slot}<extra></extra>"
                    ))
                elif action == "runtime_plan":
                    fig.add_trace(go.Scatter(
                        x=[slot], y=[tid],
                        mode="markers",
                        marker=dict(symbol="diamond", size=15, color="#FF8C00"),
                        name="Reroute",
                        showlegend=False,
                        hovertemplate=f"<b>{tid}</b> REROUTED at slot {slot}<extra></extra>"
                    ))
                elif action == "completed":
                    fig.add_trace(go.Scatter(
                        x=[slot], y=[tid],
                        mode="markers+text",
                        marker=dict(symbol="circle", size=20, color="#32CD32"),
                        text=["✓"],
                        name="Completed",
                        showlegend=False,
                        hovertemplate=f"<b>{tid}</b> COMPLETED at slot {slot}<extra></extra>"
                    ))
    
    # Add current time indicator
    if current_slot is not None:
        fig.add_vline(
            x=current_slot,
            line_width=3,
            line_color="#E74C3C",
            annotation=dict(
                text=f"Current Time: Slot {current_slot}",
                textangle=-90,
                font=dict(size=12, color="#E74C3C")
            )
        )
    
    # Calculate statistics
    completed_trains = len([t for t in trains_list if state.get(t, {}).get("status") == "completed"])
    total_delays = sum(stats['delays'].values())
    total_reroutes = sum(stats['reroutes'].values())
    
    fig.update_layout(
        title=dict(
            text=f"🚂 RAILWAY JOURNEY GANTT CHART<br>✅ Completed: {completed_trains}/{len(trains_list)} | ⏱️ Delays: {total_delays} slots | 🔄 Reroutes: {total_reroutes}",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title="⏰ Time (slots)",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title="🚂 Trains",
            categoryorder="array",
            categoryarray=trains_list
        ),
        height=600,
        width=1400,
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(r=50, t=100)
    )
    
    return fig

def plot_station_overview(state, platforms, current_slot=None):
    """
    Multi-platform occupancy overview with enhanced visualization.
    """
    # Collect platform activity data
    rows = []
    for tid, st in state.items():
        path = st.get("planned_path", [])
        slots = st.get("planned_slots", [])
        for n, s in zip(path, slots):
            if isinstance(n, tuple) and len(n) >= 1 and n[0] == "Platform":
                rows.append({"platform": n, "slot": s, "train": tid})
    
    df = pd.DataFrame(rows)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No platform activity detected")
        return fig
    
    # Create subplots for each platform
    plat_order = [p for p in platforms if p in set(df["platform"])]
    if not plat_order:
        fig = go.Figure()
        fig.update_layout(title="No active platforms")
        return fig
    
    cols = min(3, len(plat_order))  # Max 3 columns
    rows_needed = (len(plat_order) + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows_needed, cols=cols,
        subplot_titles=[format_node(p) for p in plat_order],
        shared_xaxes=True,
        vertical_spacing=0.15
    )
    
    for idx, p in enumerate(plat_order):
        sub = df[df["platform"] == p]
        row = idx // cols + 1
        col = idx % cols + 1
        
        # Count trains per slot
        slot_counts = sub.groupby("slot").size()
        
        fig.add_trace(
            go.Bar(
                x=slot_counts.index,
                y=slot_counts.values,
                marker=dict(color="#FFD700", line=dict(color="#B8860B", width=1)),
                name=format_node(p),
                showlegend=False,
                hovertemplate=f"{format_node(p)}<br>Trains: %{{y}}<br>Slot: %{{x}}<extra></extra>"
            ),
            row=row, col=col
        )
        
        # Add current time indicator
        if current_slot is not None:
            fig.add_vline(
                x=current_slot,
                line_color="red",
                line_width=2,
                row=row, col=col
            )
    
    fig.update_layout(
        height=300 * rows_needed,
        width=1200,
        title=dict(
            text="🚉 PLATFORM OCCUPANCY OVERVIEW",
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Time (slots)")
    fig.update_yaxes(title_text="Trains")
    
    return fig

## Redundant "section vs time" plot removed in favor of Network Map

def plot_stops_schedule(state, platforms=None, current_slot=None):
    """Stops Comparator: for each train, show expected vs actual stop window.
    Expected arrival = first planned platform slot; expected depart = expected arrival + dwell.
    Actual arrival/depart from logs (enter/depart). If missing, infer best effort.
    """
    trains = sorted(state.keys())
    if not trains:
        fig = go.Figure(); fig.update_layout(title="No data"); return fig

    rows = []
    for tid in trains:
        st = state.get(tid, {})
        info = st.get("info")
        dwell = getattr(info, "dwell", 2) if info is not None else 2

        # expected from current planned_path
        exp_arr = None
        if st.get("planned_path") and st.get("planned_slots"):
            for n, s in zip(st["planned_path"], st["planned_slots"]):
                if isinstance(n, tuple) and n and n[0] == "Platform":
                    exp_arr = int(s); break
        exp_dep = (exp_arr + int(dwell)) if exp_arr is not None else None

        # actual from logs
        act_arr = None
        act_dep = None
        for rec in st.get("log", []):
            if isinstance(rec, tuple) and len(rec) >= 4:
                slot, prev_node, next_node, action = rec
                if action == "enter" and isinstance(next_node, tuple) and next_node and next_node[0] == "Platform":
                    if act_arr is None:
                        act_arr = int(slot)
                elif action == "depart":
                    act_dep = int(slot)
            elif isinstance(rec, dict):
                if rec.get("action") == "involved_in_accident" and act_arr is None and isinstance(rec.get("node"), tuple) and rec.get("node")[0] == "Platform":
                    act_arr = int(rec.get("slot"))
        # fallback depart from platform_end_slot if present and not completed
        if act_dep is None and st.get("platform_end_slot") is not None:
            act_dep = int(st["platform_end_slot"]) if st.get("status") in {"at_platform", "completed"} else None

        rows.append({
            "train": tid,
            "exp_arr": exp_arr,
            "exp_dep": exp_dep,
            "act_arr": act_arr,
            "act_dep": act_dep
        })

    # Build figure with two bar layers per train
    y_order = [r["train"] for r in rows]
    fig = go.Figure()
    for r in rows:
        tid = r["train"]
        # expected window
        if r["exp_arr"] is not None and r["exp_dep"] is not None:
            fig.add_trace(go.Bar(
                x=[max(0, r["exp_dep"] - r["exp_arr"])],
                y=[tid],
                base=r["exp_arr"],
                orientation='h',
                marker=dict(color="rgba(41, 128, 185, 0.35)", line=dict(color="#1B4F72", width=1)),
                name="Expected",
                hovertemplate=f"<b>{tid}</b><br>Expected: {r['exp_arr']}–{r['exp_dep']}<extra></extra>",
                showlegend=False
            ))
        # actual window
        if r["act_arr"] is not None and r["act_dep"] is not None and r["act_dep"] >= r["act_arr"]:
            delay = (r["act_arr"] - r["exp_arr"]) if r["exp_arr"] is not None else 0
            fig.add_trace(go.Bar(
                x=[max(0, r["act_dep"] - r["act_arr"])],
                y=[tid],
                base=r["act_arr"],
                orientation='h',
                marker=dict(color="rgba(39, 174, 96, 0.75)", line=dict(color="#145A32", width=1)),
                name="Actual",
                hovertemplate=f"<b>{tid}</b><br>Actual: {r['act_arr']}–{r['act_dep']}<br>Delay: {delay:+} slots<extra></extra>",
                showlegend=False
            ))
        elif r["act_arr"] is not None and (r["act_dep"] is None):
            # Show arrival marker only
            fig.add_trace(go.Scatter(
                x=[r["act_arr"]], y=[tid], mode="markers+text",
                marker=dict(symbol="triangle-up", size=14, color="#27AE60"),
                text=["arr"], textposition="middle right",
                name="Actual Arrive",
                hovertemplate=f"<b>{tid}</b><br>Arrived: {r['act_arr']}<extra></extra>",
                showlegend=False
            ))

        # Mark expected markers if present
        if r["exp_arr"] is not None:
            fig.add_trace(go.Scatter(
                x=[r["exp_arr"]], y=[tid], mode="markers",
                marker=dict(symbol="circle-open", size=12, color="#1B4F72", line=dict(width=2)),
                name="Exp Arrive", showlegend=False,
                hovertemplate=f"<b>{tid}</b><br>Expected Arrive: {r['exp_arr']}<extra></extra>"
            ))
        if r["exp_dep"] is not None:
            fig.add_trace(go.Scatter(
                x=[r["exp_dep"]], y=[tid], mode="markers",
                marker=dict(symbol="circle-open", size=12, color="#1B4F72", line=dict(width=2)),
                name="Exp Depart", showlegend=False,
                hovertemplate=f"<b>{tid}</b><br>Expected Depart: {r['exp_dep']}<extra></extra>"
            ))

    fig.update_layout(
        title=dict(text="🕒 Stops Comparator — Expected vs Actual", x=0.5, font=dict(size=16)),
        xaxis_title="Time (slots)",
        yaxis=dict(title="Trains", categoryorder="array", categoryarray=y_order),
        height=max(400, 30 * len(y_order)),
        width=1400,
        hovermode="x unified"
    )
    if current_slot is not None:
        fig.add_vline(x=int(current_slot), line_color="red", line_width=2)
    return fig
