"""
RailOptimusSim - Advanced Railway Traffic Simulation System

This is the main application file for the RailOptimusSim system, providing a comprehensive
web-based interface for railway traffic simulation with real-time accident management,
dynamic rerouting, and advanced visualization capabilities.

Features:
- Real-time railway traffic simulation
- Interactive accident management
- Dynamic pathfinding and rerouting
- Comprehensive visualization suite
- Professional control interface

Author: RailOptimusSim Development Team
Version: 2.0 Professional Edition
"""

import dash
import ast
import json
import os
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import uuid

# Import core system components
from data import build_graph, generate_fixed_trains
from accident_manager import EmergencyEvent, AccidentManager
from simulation import Simulator
from visualization import (
    plot_gantt_chart,
    plot_train_timeline,
    plot_track_timeline,
    plot_network_map,
    enhance_for_hd,
    plot_stops_schedule,
)
from utils import format_node

def generate_accident_log(accident_mgr, current_slot):
    """
    Generate comprehensive accident log HTML with enhanced formatting
    
    Args:
        accident_mgr (AccidentManager): The accident management system
        current_slot (int): Current simulation time slot
        
    Returns:
        list: HTML elements for the accident log display
    """
    log_entries = []
    
    # Add prescheduled accidents
    for event in accident_mgr.scheduled:
        status = "🟢 ACTIVE" if event.is_active_slot(current_slot) else "⏳ SCHEDULED"
        if event.start_time > current_slot:
            status = "📅 FUTURE"
        elif event.end_time <= current_slot:
            status = "✅ RESOLVED"
            
        involved_train = accident_mgr.involved_trains.get(event.event_id, "None")
        affected_count = len(accident_mgr.affected_trains.get(event.event_id, []))
        rerouted_count = len(accident_mgr.rerouted_trains.get(event.event_id, []))
        
        log_entries.append(html.Div([
            html.Strong(f"🚨 {event.event_id} - {event.ev_type.upper()}"),
            html.Br(),
            f"📍 Location: {format_node(event.location)}",
            html.Br(),
            f"⏰ Start: Slot {event.start_time} | Duration: {event.duration_slots} slots",
            html.Br(),
            f"🚂 Involved: {involved_train} | Affected: {affected_count} | Rerouted: {rerouted_count}",
            html.Br(),
            f"📊 Status: {status}",
            html.Hr(style={"margin": "5px 0"})
        ], style={"margin-bottom": "10px", "padding": "8px", "background-color": "white", "border-radius": "3px"}))
    
    if not log_entries:
        log_entries.append(html.Div("✅ No accidents scheduled or active", 
                                  style={"text-align": "center", "color": "green", "font-style": "italic"}))
    
    return log_entries

def generate_system_stats(state, trains, accident_mgr, current_slot):
    """
    Generate comprehensive system statistics HTML with enhanced metrics
    
    Args:
        state (dict): Current simulation state
        trains (list): List of train objects
        accident_mgr (AccidentManager): Accident management system
        current_slot (int): Current simulation time slot
        
    Returns:
        list: HTML elements for the system statistics display
    """
    # Calculate train status distribution
    completed_trains = len([t for t in trains if state.get(t.id, {}).get("status") == "completed"])
    blocked_trains = len([t for t in trains if state.get(t.id, {}).get("status") == "blocked_by_accident"])
    running_trains = len([t for t in trains if state.get(t.id, {}).get("status") == "running"])
    not_arrived = len([t for t in trains if state.get(t.id, {}).get("status") == "not_arrived"])
    
    # Calculate delays and reroutes
    total_delays = 0
    total_reroutes = 0
    for train in trains:
        train_state = state.get(train.id, {})
        total_delays += train_state.get("waiting_s", 0) / 60  # Convert to minutes
        # Count reroutes from log
        for log_entry in train_state.get("log", []):
            if isinstance(log_entry, tuple) and len(log_entry) >= 4 and log_entry[3] == "runtime_plan":
                total_reroutes += 1
    
    active_accidents = len([e for e in accident_mgr.scheduled if e.is_active_slot(current_slot)])
    
    avg_delay = (total_delays / max(1, len(trains)))
    completion_pct = (completed_trains / max(1, len(trains)) * 100.0)
    # Platform utilization: approximate using Simulator.usage counts and current time
    # Fallback: derive from logs if usage not available (lightweight estimation)
    plat_busy_counts = {}
    for st in state.values():
        for rec in st.get("log", []):
            if isinstance(rec, tuple) and len(rec) >= 4 and str(rec[3]).startswith("platform_until_"):
                node = rec[1]  # tuple form: (slot, node, None, 'platform_until_X')
                if isinstance(node, tuple) and node and node[0] == "Platform":
                    end_slot_txt = rec[3].split("_")[-1]
                    try:
                        end_slot = int(end_slot_txt)
                    except Exception:
                        end_slot = (rec[0] or 0)
                    start_slot = rec[0] or 0
                    # increment occupancy per slot range
                    for s in range(int(start_slot), int(end_slot)+1):
                        plat_busy_counts[node] = plat_busy_counts.get(node, 0) + 1
    # Compute utilization percentage per platform over elapsed time
    elapsed = max(1, current_slot or 1)
    plat_util = {p: (cnt / elapsed) for p, cnt in plat_busy_counts.items()}
    busiest_pf = None
    busiest_util = 0.0
    if plat_util:
        busiest_pf, busiest_util = max(plat_util.items(), key=lambda kv: kv[1])

    stats_html = [
        html.H5("🚂 Train Status", className="mb-2"),
        html.P(f"✅ Completed: {completed_trains}/{len(trains)}"),
        html.P(f"🚫 Blocked: {blocked_trains}"),
        html.P(f"🚂 Running: {running_trains}"),
        html.P(f"⏳ Not Arrived: {not_arrived}"),
        html.Hr(),
        html.H5("📊 Performance Metrics", className="mb-2"),
        html.P(f"📈 Throughput: {completed_trains} trains ({completion_pct:.1f}%)"),
        html.P(f"⏱️ Total Delays: {total_delays:.1f} minutes"),
        html.P(f"⏱️ Avg Delay/Train: {avg_delay:.1f} minutes"),
        html.P(f"🔄 Total Reroutes: {total_reroutes}"),
        html.P(f"🚨 Active Accidents: {active_accidents}"),
        html.P(f"🏁 Busiest Platform: {format_node(busiest_pf) if busiest_pf else 'N/A'} ({busiest_util*100:.0f}% util)"),
        html.Hr(),
        html.H5("⏰ System Time", className="mb-2"),
        html.P(f"Current Slot: {current_slot}"),
        html.P(f"Time: {current_slot} minutes")
    ]
    # Append recent simulation history if available
    try:
        global SIM_HISTORY
        if SIM_HISTORY:
            items = []
            for i, h in enumerate(reversed(SIM_HISTORY[-7:]), 1):
                pct = (h['completed'] / max(1, h['total']) * 100.0)
                items.append(html.Li(
                    f"Run {i}: t={h['ts']}m • {h['completed']}/{h['total']} done ({pct:.0f}%) • total delay {h['total_delay_min']:.1f}m • incidents {h['incidents']}",
                    style={"fontSize": "12px"}
                ))
            stats_html.extend([html.Hr(), html.H6("🗂️ Recent Runs (last 10)"), html.Ul(items)])
    except Exception:
        pass
    
    return stats_html

def generate_ai_summary(state, acc_mgr, platforms, current_slot):
    """Enhanced AI summary with platform access insights."""
    total = len(state)
    completed = sum(1 for s in state.values() if s.get("status") == "completed")
    running = sum(1 for s in state.values() if s.get("status") == "running")
    blocked = sum(1 for s in state.values() if s.get("status") == "blocked_by_accident")
    
    # Platform utilization analysis
    plat_counts = {}
    for tid, st in state.items():
        for n, sl in zip(st.get("planned_path", []), st.get("planned_slots", [])):
            if isinstance(n, tuple) and n and n[0] == "Platform" and sl <= current_slot:
                plat_counts[n] = plat_counts.get(n, 0) + 1
    
    # Active incidents analysis
    active_events = acc_mgr.active_summary(current_slot)
    ev_details = []
    for _, evtype, loc, rem, stats in active_events:
        affected = stats.get("affected_trains", 0)
        rerouted = stats.get("rerouted_trains", 0)
        ev_details.append(f"{evtype}@{format_node(loc)}({affected}A,{rerouted}R,{rem}T)")
    
    # Platform efficiency
    busiest = sorted(plat_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    platform_efficiency = sum(plat_counts.values()) / max(1, len(platforms)) if platforms else 0
    
    # Convert recent operations into brief natural sentences
    recent_ops = []
    for tid, st in state.items():
        for rec in reversed(st.get("log", [])):
            if isinstance(rec, tuple) and len(rec) >= 4:
                slot, prev_node, next_node, action = rec[0], rec[1], rec[2], rec[3]
                if slot is None or (current_slot is not None and slot < current_slot - 10):
                    break
                if action in ("involved_in_accident", "affected_by_accident", "resume", "runtime_plan", "completed"):
                    if action == "resume":
                        recent_ops.append(f"{tid} resumed at {format_node(next_node)} (t={slot}).")
                    elif action == "runtime_plan":
                        recent_ops.append(f"{tid} rerouted at t={slot}.")
                    elif action == "involved_in_accident":
                        recent_ops.append(f"{tid} involved in an accident at {format_node(next_node)} (t={slot}).")
                    elif action == "affected_by_accident":
                        recent_ops.append(f"{tid} waiting due to blocked track (t={slot}).")
                    elif action == "completed":
                        recent_ops.append(f"{tid} completed (t={slot}).")
    recent_ops = recent_ops[:5]

    return [
        html.H5("🤖 AI Operations Summary"),
        html.Div([
            html.P(f"⏰ Current Time: Slot {current_slot} | 🚉 Platform Efficiency: {platform_efficiency:.1f}"),
            html.P(f"🚂 Fleet Status: {completed}/{total} completed, {running} active, {blocked} blocked"),
            html.P(f"🚨 Active Incidents: {', '.join(ev_details) if ev_details else 'None'}"),
            html.P(f"🏆 Top Platforms: {', '.join([f'{format_node(p)}({c})' for p, c in busiest]) if busiest else 'N/A'}"),
            html.P(f"📊 System Load: {'High' if blocked > 2 else 'Medium' if blocked > 0 else 'Normal'}"),
            html.Ul([html.Li(x) for x in recent_ops]) if recent_ops else html.P("No recent events in the last few minutes.")
        ], style={"fontSize": "14px"})
    ]

def generate_operations_log(state, current_slot):
    """Create a human-readable operations log from train logs."""
    entries = []
    def fmt_platform(n):
        return format_node(n) if isinstance(n, tuple) and n and n[0] == "Platform" else None
    for tid, st in state.items():
        for rec in st.get("log", []):
            # support tuple and dict log formats
            if isinstance(rec, tuple) and len(rec) >= 4:
                slot, prev_node, next_node, action = rec[0], rec[1], rec[2], rec[3]
                if action == "runtime_plan":
                    entries.append((slot, f"{slot:>3} | 🔄 {tid} rerouted"))
                elif action == "enter" and fmt_platform(next_node):
                    entries.append((slot, f"{slot:>3} | ⏺️ {tid} arrived at {format_node(next_node)}"))
                elif action == "depart" and fmt_platform(prev_node or next_node):
                    pf = prev_node if fmt_platform(prev_node) else next_node
                    entries.append((slot, f"{slot:>3} | ⏏️ {tid} departed from {format_node(pf)}"))
                elif action == "completed":
                    entries.append((slot, f"{slot:>3} | ✅ {tid} completed journey"))
                elif action == "switch":
                    entries.append((slot, f"{slot:>3} | ⇄ {tid} switched tracks"))
                elif action == "blocked_by_accident":
                    entries.append((slot, f"{slot:>3} | 🚫 {tid} waiting (accident block)"))
                elif action == "resume":
                    entries.append((slot, f"{slot:>3} | ▶️ {tid} resumed movement"))
            elif isinstance(rec, dict):
                slot = rec.get("slot")
                action = rec.get("action")
                node = rec.get("node")
                if action == "runtime_plan":
                    entries.append((slot, f"{slot:>3} | 🔄 {tid} rerouted (delay +{rec.get('delay', 0)} slots)"))
                elif action == "involved_in_accident":
                    entries.append((slot, f"{slot:>3} | 🚨 {tid} involved in accident at {format_node(node)}"))
                elif action == "affected_by_accident":
                    entries.append((slot, f"{slot:>3} | ⚠️ {tid} affected by accident (track blocked)"))
    # Sort by slot, then message
    entries.sort(key=lambda x: (x[0] if x[0] is not None else -1, x[1]))
    # Limit to last ~50 for readability
    entries = entries[-50:]
    return [html.Div(msg) for _, msg in entries]

def generate_operations_log_rows(state):
    """Build structured rows for Operations Log CSV export.
    Returns list of dicts with keys: slot, train, action, from, to, note
    """
    rows = []
    for tid, st in state.items():
        for rec in st.get("log", []):
            if isinstance(rec, tuple) and len(rec) >= 4:
                slot, prev_node, next_node, action = rec[0], rec[1], rec[2], rec[3]
                rows.append({
                    "slot": slot,
                    "train": tid,
                    "action": action,
                    "from": format_node(prev_node) if prev_node is not None else "",
                    "to": format_node(next_node) if next_node is not None else "",
                    "note": ""
                })
            elif isinstance(rec, dict):
                rows.append({
                    "slot": rec.get("slot"),
                    "train": tid,
                    "action": rec.get("action"),
                    "from": format_node(rec.get("from")) if rec.get("from") is not None else "",
                    "to": format_node(rec.get("node") or rec.get("to")) if (rec.get("node") or rec.get("to")) is not None else "",
                    "note": ", ".join([f"{k}={v}" for k, v in rec.items() if k not in {"slot","action","from","to","node"}])
                })
    # sort by slot,train
    rows.sort(key=lambda r: (r["slot"] if r["slot"] is not None else -1, r["train"]))
    return rows

def generate_now_board(state, trains, current_slot):
    """Create a simple arrivals board: upcoming platform stops in the next window."""
    import math
    rows = []
    for t in trains:
        st = state.get(t.id, {})
        path = st.get("planned_path", []) or []
        slots = st.get("planned_slots", []) or []
        eta = None
        dest = None
        for n, s in zip(path, slots):
            if isinstance(n, tuple) and n and n[0] == "Platform" and (s is None or int(s) >= int(current_slot)):
                eta = int(s)
                dest = n
                break
        if eta is not None and dest is not None:
            rows.append({
                "train": t.id,
                "type": getattr(t, "type", "-"),
                "platform": format_node(dest),
                "eta": eta,
                "status": (state.get(t.id, {}).get("status", "-")).replace("_", " ")
            })
    rows.sort(key=lambda r: (r["eta"], r["platform"]))
    header = html.Thead(html.Tr([
        html.Th("ETA"), html.Th("Train"), html.Th("Type"), html.Th("Platform"), html.Th("Status")
    ]))
    body = []
    for r in rows[:20]:
        body.append(html.Tr([
            html.Td(str(r["eta"])), html.Td(r["train"]), html.Td(r["type"]), html.Td(r["platform"]), html.Td(r["status"]) 
        ]))
    table = dbc.Table([header, html.Tbody(body)], bordered=True, hover=True, striped=True, size="sm")
    if not rows:
        return html.Div("No upcoming arrivals yet — step/run the sim.", style={"fontStyle": "italic", "color": "#7f8c8d"})
    return table

def generate_history_figure(history):
    import plotly.graph_objects as go
    fig = go.Figure()
    if not history:
        fig.update_layout(title="No history yet — complete a run and reset to record it.")
        return fig
    # Use last up to 10 runs, most recent last
    hist = history[-10:]
    xs = list(range(1, len(hist)+1))
    throughput_pct = [ (h['completed']/max(1,h['total']))*100.0 for h in hist ]
    total_delay = [ h['total_delay_min'] for h in hist ]
    incidents = [ h['incidents'] for h in hist ]
    fig.add_trace(go.Bar(x=xs, y=total_delay, name="Total Delay (min)", marker_color="#F5B041", yaxis="y"))
    fig.add_trace(go.Scatter(x=xs, y=throughput_pct, name="Throughput (%)", mode="lines+markers", marker=dict(color="#27AE60"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=xs, y=incidents, name="Incidents", mode="markers", marker=dict(color="#C0392B", size=10, symbol="x"), yaxis="y", hovertemplate="Run %{x}: %{y} incidents<extra></extra>"))
    fig.update_layout(
        title=dict(text="Run History — Throughput vs Delay", x=0.5),
        xaxis=dict(title="Run # (older → newer)", tickmode="linear"),
        yaxis=dict(title="Total Delay (min)", rangemode="tozero"),
        yaxis2=dict(title="Throughput (%)", overlaying='y', side='right', rangemode="tozero", range=[0,100]),
        legend=dict(orientation="h", y=-0.2),
        height=420, width=1200
    )
    return fig

def generate_train_overview(state, trains):
    """Build a compact train overview table with Start and Stop nodes.
    Columns: Train, Type, Start, Stop, Status, Delay (min)
    Stop is the latest platform reached (from logs or current position if platform).
    """
    def last_platform(st):
        # 1) If currently at a platform, use that
        pos = st.get("pos")
        if isinstance(pos, tuple) and pos and pos[0] == "Platform":
            return pos
        # 2) Scan logs backwards for a platform occurrence
        for rec in reversed(st.get("log", [])):
            if isinstance(rec, tuple) and len(rec) >= 4:
                slot, prev_node, next_node, action = rec[0], rec[1], rec[2], rec[3]
                # platform_until_X logs have platform in prev_node
                if isinstance(prev_node, tuple) and prev_node and prev_node[0] == "Platform":
                    return prev_node
                if isinstance(next_node, tuple) and next_node and next_node[0] == "Platform":
                    return next_node
            elif isinstance(rec, dict):
                node = rec.get("node") or rec.get("to") or rec.get("from")
                if isinstance(node, tuple) and node and node[0] == "Platform":
                    return node
        # 3) Fallback: first platform in planned path, if any
        for n in st.get("planned_path", []):
            if isinstance(n, tuple) and n and n[0] == "Platform":
                return n
        return None

    header = html.Thead(html.Tr([
        html.Th("Train"), html.Th("Type"), html.Th("Start"), html.Th("Stop"), html.Th("Status"), html.Th("Delay (min)")
    ]))
    body_rows = []
    for t in trains:
        st = state.get(t.id, {})
        start_node = format_node(getattr(t, "start", None))
        stop_node = last_platform(st)
        stop_txt = format_node(stop_node) if stop_node is not None else "—"
        status = st.get("status", "-")
        delay_min = (st.get("waiting_s", 0.0) or 0.0) / 60.0
        body_rows.append(html.Tr([
            html.Td(t.id),
            html.Td(getattr(t, "type", "-")),
            html.Td(start_node),
            html.Td(stop_txt),
            html.Td(status.replace("_", " ")),
            html.Td(f"{delay_min:.1f}")
        ]))
    table = dbc.Table([
        header,
        html.Tbody(body_rows)
    ], bordered=True, hover=True, responsive=True, striped=True, size="sm")
    return table

# =============================================================================
# SYSTEM INITIALIZATION - PROFESSIONAL RAILWAY SIMULATION SETUP
# =============================================================================

# Railway Network Configuration
NUM_TRACKS = 6          # Number of parallel tracks in the railway network
SECTIONS = 4            # Number of sections per track
NUM_STATIONS = 1        # Number of stations (single-station setup)
PLATFORMS_PER_STATION = 7   # Seven platforms at the single station
HORIZON_MINUTES = 20    # Simulation planning horizon in minutes
CURRENT_SECTIONS = SECTIONS  # Track current sections after dataset loads

# Constrained platform access: map each track to all platforms of the single station
PLATFORM_ACCESS_MAP = {}
for tr in range(NUM_TRACKS):
    PLATFORM_ACCESS_MAP[tr] = [(0, pf) for pf in range(PLATFORMS_PER_STATION)]

print("🚂 Initializing Railway Infrastructure...")
# Build demo graph only (no external JSONs)
G, PLATFORMS = build_graph(
    num_tracks=NUM_TRACKS,
    sections_per_track=SECTIONS,
    num_stations=NUM_STATIONS,
    platforms_per_station=PLATFORMS_PER_STATION,
    platform_access_map=PLATFORM_ACCESS_MAP
)
trains = generate_fixed_trains(sections_per_track=SECTIONS, num_trains=10, num_tracks=NUM_TRACKS)
print(f"✅ Demo network built: {NUM_TRACKS} tracks × {SECTIONS} sections + {NUM_STATIONS} stations × {PLATFORMS_PER_STATION} platforms")

# Initialize the accident management system
print("🚨 Initializing Emergency Management System...")
acc_mgr = AccidentManager()
# Configure accident manager with network parameters
acc_mgr.set_network(sections_per_track=CURRENT_SECTIONS)
print("✅ Emergency response system online")

# Initialize the simulation engine
print("⚙️ Initializing Simulation Engine...")
sim = Simulator(
    graph=G, 
    platform_nodes=PLATFORMS, 
    trains=trains, 
    accident_mgr=acc_mgr, 
    horizon_minutes=HORIZON_MINUTES
)

# Perform initial route planning for all trains
print("🗺️ Performing Initial Route Planning...")
sim.plan_initial()
print("✅ All trains have optimized routes planned")
print("🚀 RailOptimusSim is ready for operation!")

# Maintain last 10 simulation snapshots for quick history view
SIM_HISTORY = []  # entries: {"ts": int_slot, "wall_time": ISO-8601 str, "completed": int, "total": int, "total_delay_min": float, "avg_delay_min": float, "incidents": int, "ops": list[dict]}

# =============================================================================
# WEB APPLICATION INITIALIZATION - PROFESSIONAL DASHBOARD SETUP
# =============================================================================

# Initialize the Dash web application with Bootstrap styling
print("🌐 Initializing Web Application...")
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="RailOptimusSim - Advanced Railway Control Center",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": "Professional Railway Traffic Simulation System"}
    ]
)
server = app.server
print("✅ Web application initialized with professional styling")

app.layout = dbc.Container([
    html.Div([
        html.H1("🚂 RailOptimusSim — Advanced Railway Control Center 🚂", 
                className="text-center mb-4", 
                style={"color": "#2C3E50", "fontWeight": "bold", "textShadow": "2px 2px 4px rgba(0,0,0,0.1)"}),
        html.P("Real-time railway traffic simulation with intelligent accident management and dynamic rerouting", 
               className="text-center text-muted mb-4", 
               style={"fontSize": "16px"})
    ]),
    
    dbc.Card([
        dbc.CardBody([
                html.H4("🎯 Simulation Overview", className="card-title"),
             html.P(f"This simulation models 10 trains (Express, Passenger, Freight) on a {NUM_TRACKS}-track network with {SECTIONS} sections per track and {NUM_STATIONS} station × {PLATFORMS_PER_STATION} platforms ({NUM_STATIONS*PLATFORMS_PER_STATION} total), featuring intelligent pathfinding and real-time accident response.", 
                   className="card-text"),
            html.Hr(),
            html.H5("⚠️ Emergency Accident Interface", className="mb-3"),
            html.P("Use the controls below to trigger emergency scenarios and test the system's response capabilities:", 
                   style={"fontStyle": "italic", "color": "#7F8C8D"}),
            html.Ul([
                html.Li(html.Strong("Track Index (0-7):"), " Select the track where the emergency will occur"),
                html.Li(html.Strong("Section Index (0-3):"), " Choose the specific section on the selected track"),
                html.Li(html.Strong("Duration (1-120 slots):"), " Set how long the emergency will last (in minutes)"),
            ], className="mb-3"),
            html.P("Click 'Trigger Accident ⚠️' to activate the emergency scenario and observe real-time system response.", 
                   style={"fontStyle": "italic", "color": "#E74C3C", "fontWeight": "bold"}),
        ])
    ], className="mb-4"),
    # Removed dataset ingestion UI
    dbc.Card([
        dbc.CardBody([
            html.H5("🎬 One-Click Demos & Presets", className="card-title mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Preset Scenarios", className="fw-bold"),
                    dcc.Dropdown(
                        id="scenario-preset",
                        options=[
                            {"label": "Smooth Run (no incidents)", "value": "smooth"},
                            {"label": "Track Accident at T3-S2 (6m)", "value": "acc_t3s2"},
                            {"label": "Station 1: Platforms 1-4 blocked (8m)", "value": "st1_pf1_4"},
                            {"label": "Breakdown: Train T3 (5m)", "value": "bd_t3"},
                            {"label": "Stress: Mix of all (guided)", "value": "mix"},
                            {"label": "Rush Hour Wave", "value": "rush_wave"},
                            {"label": "Platform Wave (P1→P4)", "value": "plat_wave"},
                            {"label": "Clean Recovery", "value": "recovery"},
                        ],
                        placeholder="Pick a preset",
                        clearable=True,
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Actions", className="fw-bold"),
                    dbc.ButtonGroup([
                        dbc.Button("Apply Preset ▶", id="apply-preset", color="info", className="me-2"),
                        dbc.Button("Guided Demo 🎥", id="guided-demo", color="secondary"),
                    ])
                ], width=6)
            ])
        ])
    ], className="mb-4"),
    dbc.Card([
        dbc.CardBody([
            html.H5("🎮 Simulation Controls", className="card-title mb-3"),
            dbc.Row([
                dbc.Col(dbc.Button("Step ▶", id="step-btn", color="primary", size="lg", className="me-2"), width="auto"),
                dbc.Col(dbc.Button("Run ⏵", id="run-btn", color="success", size="lg", className="me-2"), width="auto"),
                dbc.Col(dbc.Button("Pause ⏸", id="pause-btn", color="warning", size="lg", className="me-2", disabled=True), width="auto"),
                dbc.Col(html.Span(id="run-status-badge", className="badge bg-secondary align-self-center", children="Idle"), width="auto"),
                dbc.Col(dbc.Button("Reset ↺", id="reset-btn", color="danger", size="lg", className="me-2"), width="auto"),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Speed (Run mode)", className="fw-bold"),
                    dcc.Slider(
                        id="sim-speed",
                        min=0.25, max=4.0, step=None, value=1.0,
                        marks={0.25: "0.25×", 0.5: "0.5×", 1.0: "1×", 2.0: "2×", 4.0: "4×"}
                    )
                ])
            ], className="mb-2"),
            html.Small("Use these controls to manage the simulation: Step for manual progression, Run for continuous operation, Pause to stop, and Reset to restart.", 
                      className="text-muted")
        ])
    ], className="mb-4"),
    dbc.Card([
        dbc.CardBody([
            html.H5("🚨 Emergency Scenario Trigger", className="card-title mb-3"),
            dcc.Tabs(id="accident-tabs", value="track", children=[
                dcc.Tab(label="Track/Section", value="track", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Track Index", className="fw-bold"),
                            dbc.Input(id="acc-track", placeholder=f"0-{NUM_TRACKS-1}", type="number", min=0, max=NUM_TRACKS-1, value=2, size="lg")
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Section Index", className="fw-bold"),
                            dbc.Input(id="acc-section", placeholder=f"0-{SECTIONS-1}", type="number", min=0, max=SECTIONS-1, value=2, size="lg")
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Duration (slots)", className="fw-bold"),
                            dbc.Input(id="acc-duration", placeholder="1-120", type="number", min=1, max=120, value=6, size="lg")
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Severity", className="fw-bold"),
                            dcc.Dropdown(id="acc-severity", options=[
                                {"label": "Low", "value": "low"},
                                {"label": "Medium", "value": "medium"},
                                {"label": "High", "value": "high"}
                            ], value="high")
                        ], width=3),
                    ], className="align-items-end mt-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Start Delay (slots)", className="fw-bold"),
                            dbc.Input(id="acc-delay", type="number", min=0, max=60, value=0)
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Action", className="fw-bold"),
                            dbc.Button("🚨 Trigger Emergency", id="trigger-acc", color="danger", size="lg", className="w-100")
                        ], width=4)
                    ], className="align-items-end mt-2"),
                ]),
                dcc.Tab(label="Platform(s)", value="platforms", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Platforms", className="fw-bold"),
                            dcc.Dropdown(
                                id="platform-acc-platforms",
                                options=[{"label": format_node(p), "value": str(p)} for p in PLATFORMS],
                                multi=True,
                                placeholder="Choose one or more platforms"
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Duration (slots)", className="fw-bold"),
                            dbc.Input(id="platform-acc-duration", placeholder="1-120", type="number", min=1, max=120, value=6, size="lg")
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Severity", className="fw-bold"),
                            dcc.Dropdown(id="platform-acc-severity", options=[
                                {"label": "Low", "value": "low"},
                                {"label": "Medium", "value": "medium"},
                                {"label": "High (Station-wide)", "value": "high"}
                            ], value="medium")
                        ], width=3),
                    ], className="align-items-end mt-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Start Delay (slots)", className="fw-bold"),
                            dbc.Input(id="platform-acc-delay", type="number", min=0, max=60, value=0)
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Action", className="fw-bold"),
                            dbc.Button("🚨 Trigger Platform Emergency", id="trigger-platform-acc", color="danger", size="lg", className="w-100")
                        ], width=3)
                    ], className="align-items-end mt-2"),
                ]),
                dcc.Tab(label="Train Breakdown", value="breakdown", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select Train", className="fw-bold"),
                            dcc.Dropdown(
                                id="breakdown-train",
                                options=[{"label": t.id, "value": t.id} for t in trains],
                                multi=False,
                                placeholder="Choose a train"
                            )
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Duration (slots)", className="fw-bold"),
                            dbc.Input(id="breakdown-duration", placeholder="1-120", type="number", min=1, max=120, value=6, size="lg")
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Action", className="fw-bold"),
                            dbc.Button("🧯 Trigger Breakdown", id="trigger-breakdown", color="secondary", size="lg", className="w-100")
                        ], width=3)
                    ], className="align-items-end mt-2"),
                ]),
            ]),
            html.Small("Configure emergency parameters and click to activate. The system will automatically detect affected trains and initiate rerouting procedures.", 
                      className="text-muted mt-2 d-block")
        ])
    ], className="mb-4"),
    dbc.Card([
        dbc.CardBody([
            html.H5("🔎 Views & Filters", className="card-title mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Filter Trains", className="fw-bold"),
                    dcc.Dropdown(
                        id="train-filter",
                        options=[{"label": t.id, "value": t.id} for t in trains],
                        multi=True,
                        placeholder="Select trains (optional)"
                    )
                ], width=4),
                dbc.Col([
                    dbc.Label("Filter Platforms", className="fw-bold"),
                    dcc.Dropdown(
                        id="platform-filter",
                        options=[{"label": format_node(p), "value": str(p)} for p in PLATFORMS],
                        multi=True,
                        placeholder="Select platforms (optional)"
                    )
                ], width=5),
                dbc.Col([
                    dbc.Label("Platform View", className="fw-bold"),
                    dcc.RadioItems(
                        id="platform-view-mode",
                        options=[
                            {"label": "Scatter (Platform × Train)", "value": "scatter"},
                            {"label": "Combined (stacked)", "value": "combined"},
                        ],
                        value="scatter",
                        labelStyle={"display": "block"}
                    )
                ], width=3)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Checklist(
                        options=[{"label": " High-Definition Mode", "value": "hd"}],
                        value=["hd"],
                        id="hd-mode",
                        switch=True,
                    )
                ], width="auto"),
                dbc.Col([
                    dbc.Checklist(
                        options=[{"label": " Simple Mode (hide detailed timelines)", "value": "simple"}],
                        value=[],
                        id="simple-mode",
                        switch=True,
                    )
                ], width="auto"),
                dbc.Col([
                    dbc.Checklist(
                        options=[{"label": " Dark Theme", "value": "dark"}],
                        value=[],
                        id="dark-mode",
                        switch=True,
                    )
                ], width="auto"),
            ])
        ])
    ], className="mb-3"),

    # Marker legend
    dbc.Alert([
        html.Span("Legend: "),
        html.Span("Arrive ▲ ", style={"color": "#FFD700", "fontWeight": "bold"}),
        html.Span("Depart ▼ ", style={"color": "#32CD32", "fontWeight": "bold"}),
        html.Span("Reroute ◆ ", style={"color": "#FF8C00", "fontWeight": "bold"}),
        html.Span("Accident ✖ ", style={"color": "#DC143C", "fontWeight": "bold"}),
        html.Span("Resume ★", style={"color": "#00FF7F", "fontWeight": "bold"})
    ], color="light", className="mb-3"),

    dcc.Graph(id="network-map-graph", config={
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "scale": 3},
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]
    }),
    # KPI badges row (throughput & delays)
    dbc.Row([
        dbc.Col(html.Div(id="kpi-throughput", className="badge bg-success p-2 me-2"), width="auto"),
        dbc.Col(html.Div(id="kpi-total-delay", className="badge bg-warning text-dark p-2 me-2"), width="auto"),
        dbc.Col(html.Div(id="kpi-avg-delay", className="badge bg-info text-dark p-2 me-2"), width="auto"),
        dbc.Col(html.Div(id="kpi-platform-util", className="badge bg-secondary p-2"), width="auto"),
    ], className="my-2"),
    dcc.Graph(id="timeline-graph", config={
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "scale": 3},
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]
    }),
    dcc.Graph(id="gantt-graph", config={
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "scale": 3},
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]
    }),
    dcc.Graph(id="station-graph", config={
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "scale": 3},
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]
    }),
    dcc.Graph(id="platform-train-scatter", config={
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "scale": 3},
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]
    }),
    # Human-friendly alternatives
    dcc.Graph(id="stops-graph", config={
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "scale": 3},
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]
    }),
    dbc.Card([
        dbc.CardBody([
            html.H5("🛎️ Now Board (Next Arrivals)", className="card-title mb-3"),
            html.Div(id="now-board")
        ])
    ], className="mt-3"),
    dbc.Card([
        dbc.CardBody([
            html.H5("🗂️ Run History (last 10)", className="card-title mb-3"),
            dcc.Graph(id="history-graph", config={
                "displaylogo": False,
                "toImageButtonOptions": {"format": "png", "scale": 3},
                "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]
            }),
            dbc.Button("Download Last 10 (CSV)", id="download-history-list-btn", color="secondary", size="sm", className="mt-2"),
            dcc.Download(id="download-history-list")
        ])
    ], className="mt-3"),
    # Removed geographic corridor graph
    dbc.Card([
        dbc.CardBody([
            html.H5("🧭 What am I looking at? (Network Map)", className="card-title mb-2"),
            html.Ul([
                html.Li("The long grey lines are tracks — like roads for trains."),
                html.Li("Yellow dots are platforms — the places where trains stop. They’re grouped into one big yellow zone: Station 1."),
                html.Li("Each train is a 🚂 with a colored line showing where it is going next (a short future path)."),
                html.Li("Curvy connectors show where a train can change tracks or go into a platform."),
                html.Li("A red ✖ mark means an emergency (that part of the track or a platform is blocked for a while)."),
            ], style={"marginBottom": 0})
        ])
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("🚨 Emergency Event Log", className="mb-0", style={"color": "#E74C3C"})),
                dbc.CardBody([
                    html.Div(id="accident-log", style={
                        "height": "350px", 
                        "overflow-y": "auto", 
                        "border": "2px solid #E74C3C", 
                        "padding": "15px",
                        "background-color": "#FDF2F2",
                        "border-radius": "8px",
                        "fontFamily": "monospace"
                    })
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("📊 System Performance Dashboard", className="mb-0", style={"color": "#27AE60"})),
                dbc.CardBody([
                    html.Div(id="system-stats", style={
                        "height": "350px", 
                        "border": "2px solid #27AE60", 
                        "padding": "15px",
                        "background-color": "#F0F9F0",
                        "border-radius": "8px",
                        "fontFamily": "Arial, sans-serif"
                    })
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("🧾 Operations Log", className="mb-0", style={"color": "#34495E"})),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dbc.Button("Download Current CSV", id="download-ops-btn", color="secondary", size="sm"), width="auto"),
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Export History Run"),
                                dbc.Input(id="history-index", type="number", min=1, max=10, step=1, placeholder="1..10"),
                                dbc.Button("Export", id="export-history-btn", color="primary", size="sm")
                            ])
                        ])
                    ], className="g-2 mb-2"),
                    dcc.Download(id="download-ops"),
                    dcc.Download(id="download-history"),
                    html.Div(id="ops-log", style={
                        "height": "350px",
                        "overflow-y": "auto",
                        "border": "2px solid #34495E",
                        "padding": "15px",
                        "background-color": "#F8F9FA",
                        "border-radius": "8px",
                        "fontFamily": "monospace"
                    })
                ])
            ])
        ], width=6)
    ], className="mt-4"),
    # Train Overview (Start/Stop/Status/Delay)
    dbc.Card([
        dbc.CardBody([
            html.H5("🚆 Train Overview", className="card-title mb-3"),
            html.Div(id="train-overview", style={
                "maxHeight": "380px",
                "overflowY": "auto",
                "border": "2px solid #8E44AD",
                "padding": "15px",
                "backgroundColor": "#F8F1FF",
                "borderRadius": "8px",
            })
        ])
    ], className="mt-3"),
    dbc.Card([
        dbc.CardBody([
            html.H5("🧠 AI Operations Summary", className="card-title mb-3"),
            html.Div(id="ai-summary")
        ])
    ], className="mt-3"),
    # Removed dataset-loaded store
    dbc.Card([
        dbc.CardBody([
            html.H5("🗣️ Plain English Summary", className="card-title mb-3"),
            html.Div(id="plain-summary", style={"fontFamily": "Arial, sans-serif", "fontSize": "14px"})
        ])
    ], className="mt-3"),
    dcc.Interval(id="interval", interval=1000, n_intervals=0, disabled=True),
    dcc.Store(id="is-running", data=False),
    
    dbc.Card([
        dbc.CardBody([
            html.H5("📡 System Status", className="card-title mb-3"),
            html.Div(id="sim-status", style={
                "padding": "15px",
                "border": "2px solid #3498DB",
                "borderRadius": "8px",
                "backgroundColor": "#EBF3FD",
                "fontFamily": "Arial, sans-serif",
                "fontSize": "16px",
                "fontWeight": "bold"
            })
        ])
    ], className="mt-4"),
], fluid=True)

# Callback: run/pause with status badge and button states
@app.callback(
    Output("interval", "disabled"),
    Output("is-running", "data"),
    Output("run-btn", "children"),
    Output("run-btn", "color"),
    Output("pause-btn", "disabled"),
    Output("run-status-badge", "children"),
    Input("run-btn", "n_clicks"),
    Input("pause-btn", "n_clicks"),
    State("is-running", "data")
)
def run_pause(run_clicks, pause_clicks, running):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, False, "Run ⏵", "success", True, "Idle"
    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    if trig == "run-btn":
        return False, True, "Running ⏵", "secondary", False, "▶ Running"
    if trig == "pause-btn":
        return True, False, "Run ⏵", "success", True, "⏸ Paused"
    return True, False, "Run ⏵", "success", True, "Idle"


# Callback: step, interval tick, trigger accident, reset
@app.callback(
    Output("timeline-graph", "figure"),
    Output("gantt-graph", "figure"),
    Output("station-graph", "figure"),
    Output("stops-graph", "figure"),
    Output("platform-train-scatter", "figure"),
    Output("network-map-graph", "figure"),
    Output("kpi-throughput", "children"),
    Output("kpi-total-delay", "children"),
    Output("kpi-avg-delay", "children"),
    Output("kpi-platform-util", "children"),
    Output("timeline-graph", "style"),
    Output("sim-status", "children"),
    Output("accident-log", "children"),
    Output("system-stats", "children"),
    Output("train-overview", "children"),
    Output("ai-summary", "children"),
    Output("ops-log", "children"),
    Output("plain-summary", "children"),
    Output("now-board", "children"),
    Output("history-graph", "figure"),
    Input("step-btn", "n_clicks"),
    Input("interval", "n_intervals"),
    Input("trigger-acc", "n_clicks"),
    Input("trigger-platform-acc", "n_clicks"),
    Input("trigger-breakdown", "n_clicks"),
    Input("reset-btn", "n_clicks"),
    Input("apply-preset", "n_clicks"),
    Input("guided-demo", "n_clicks"),
    Input("train-filter", "value"),
    Input("platform-filter", "value"),
    Input("platform-view-mode", "value"),
    Input("hd-mode", "value"),
    Input("dark-mode", "value"),
    Input("simple-mode", "value"),
    State("acc-track", "value"),
    State("acc-section", "value"),
    State("acc-duration", "value"),
    State("acc-severity", "value"),
    State("acc-delay", "value"),
    State("platform-acc-platforms", "value"),
    State("platform-acc-duration", "value"),
    State("platform-acc-severity", "value"),
    State("platform-acc-delay", "value"),
    State("breakdown-train", "value"),
    State("breakdown-duration", "value"),
    State("scenario-preset", "value"),
)
def control(step_clicks, n_intervals, trigger_clicks, trigger_platform_clicks, trigger_breakdown_clicks, reset_clicks, apply_preset_clicks, guided_demo_clicks, train_filter, platform_filter, platform_view, hd_mode, dark_mode, simple_mode, acc_track, acc_section, acc_duration, acc_severity, acc_delay, platform_nodes, platform_acc_duration, platform_acc_severity, platform_acc_delay, breakdown_train, breakdown_duration, scenario_value):
    global G, PLATFORMS, trains, acc_mgr, sim, GEO_PATH, STATION_POSITIONS, CURRENT_SECTIONS
    ctx = dash.callback_context
    trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    status = "Idle"

    if trig == "reset-btn":
        # snapshot current KPIs into history, then rebuild sim
        try:
            completed_snap = sum(1 for s in sim.state.values() if s.get("status") == "completed")
            total_delay_snap = sum((s.get("waiting_s", 0.0) or 0.0) for s in sim.state.values()) / 60.0
            incidents_snap = len([e for e in acc_mgr.scheduled if e.is_active_slot(sim.current_slot)])
            from datetime import datetime
            SIM_HISTORY.append({
                "ts": sim.current_slot,
                "wall_time": datetime.now().isoformat(timespec='seconds'),
                "completed": int(completed_snap),
                "total": int(len(sim.state)),
                "total_delay_min": float(total_delay_snap),
                "avg_delay_min": float(total_delay_snap / max(1, len(sim.state))),
                "incidents": int(incidents_snap),
                "ops": generate_operations_log_rows(sim.state)
            })
            if len(SIM_HISTORY) > 10:
                SIM_HISTORY[:] = SIM_HISTORY[-10:]
        except Exception:
            pass
        # rebuild sim
        acc_mgr = AccidentManager()
        acc_mgr.set_network(sections_per_track=CURRENT_SECTIONS)
        sim = Simulator(
            graph=G,
            platform_nodes=PLATFORMS,
            trains=trains,
            accident_mgr=acc_mgr,
            horizon_minutes=HORIZON_MINUTES
        )
        sim.plan_initial()
        status = "Simulator reset."
    elif trig == "trigger-acc":
        try:
            if None in [acc_track, acc_section, acc_duration]:
                raise ValueError("All accident parameters must be specified")
            
            node = (int(acc_track), int(acc_section))
            duration = int(acc_duration)
            sev = acc_severity or "medium"
            delay = int(acc_delay or 0)
            
            # Validate inputs
            if not (0 <= acc_track < NUM_TRACKS):
                raise ValueError(f"Track index must be between 0 and {NUM_TRACKS-1}")
            if not (0 <= acc_section < CURRENT_SECTIONS):
                raise ValueError(f"Section index must be between 0 and {CURRENT_SECTIONS-1}")
            if not (1 <= duration <= 120):
                raise ValueError("Duration must be between 1 and 120 slots")
                
            # Create and schedule accident (single track/section)
            loc = (int(acc_track), int(acc_section))
            ev = EmergencyEvent(
                event_id=str(uuid.uuid4())[:8],
                ev_type="accident",
                location=loc,
                start_time=sim.current_slot + delay,
                duration_slots=duration,
                info={"severity": sev}
            )
            acc_mgr.schedule(ev)
            if delay == 0:
                sim.handle_accident(loc, duration)
            status = f"🚨 Emergency: Track {acc_track}, Section {acc_section} blocked for {duration} slots (sev={sev}, delay={delay})"
        except Exception as e:
            status = f"⚠️ Failed to schedule accident: {str(e)}"
    elif trig == "trigger-platform-acc":
        try:
            if not platform_nodes or platform_acc_duration is None:
                raise ValueError("Select platforms and duration")
            duration = int(platform_acc_duration)
            sev = platform_acc_severity or "medium"
            delay = int(platform_acc_delay or 0)
            selected_plats = []
            for p in platform_nodes:
                try:
                    selected_plats.append(ast.literal_eval(p))
                except Exception:
                    pass
            if not selected_plats:
                raise ValueError("No valid platforms selected")
            # Expand to full station if severity is high
            target_platforms = set(selected_plats)
            if sev == "high":
                stations = {p[1] for p in selected_plats if isinstance(p, tuple) and len(p) >= 3}
                for st_id in stations:
                    for p in PLATFORMS:
                        if isinstance(p, tuple) and len(p) >= 3 and p[1] == st_id:
                            target_platforms.add(p)
            created = 0
            for pnode in sorted(target_platforms):
                ev = EmergencyEvent(
                    event_id=str(uuid.uuid4())[:8],
                    ev_type="accident",
                    location=pnode,
                    start_time=sim.current_slot + delay,
                    duration_slots=duration,
                    info={"severity": sev, "station_wide": (sev == "high")}
                )
                acc_mgr.schedule(ev)
                # Reroute/mark affected
                if delay == 0:
                    sim.handle_accident(pnode, duration)
                created += 1
            status = f"🚨 Platform emergency: {created} platform(s) blocked for {duration} slots (sev={sev}, delay={delay})"
        except Exception as e:
            status = f"⚠️ Failed to schedule platform emergency: {str(e)}"
    elif trig == "trigger-breakdown":
        try:
            if not breakdown_train or breakdown_duration is None:
                raise ValueError("Select a train and duration")
            duration = int(breakdown_duration)
            st = sim.state.get(breakdown_train)
            if not st:
                raise ValueError("Unknown train")
            if st.get("pos") is None:
                raise ValueError("Train not yet on network; step/run until it enters, then trigger")
            node = st.get("pos")
            ev = EmergencyEvent(
                event_id=str(uuid.uuid4())[:8],
                ev_type="breakdown",
                location=node,
                start_time=sim.current_slot,
                duration_slots=duration,
                info={"severity": "high", "train": breakdown_train}
            )
            acc_mgr.schedule(ev)
            # Explicitly block the chosen train
            st["status"] = "blocked_by_accident"
            st["accident_blocked_until"] = sim.current_slot + duration
            st.setdefault("log", []).append({
                "slot": sim.current_slot,
                "action": "involved_in_accident",
                "node": node,
                "duration": duration,
                "event_id": ev.event_id,
                "blocked_until": sim.current_slot + duration
            })
            acc_mgr.add_affected_train(ev.event_id, breakdown_train, sim.current_slot)
            acc_mgr.set_involved_train(ev.event_id, breakdown_train)
            # Reroute others if needed
            sim.handle_accident(node, duration)
            status = f"🧯 Breakdown: {breakdown_train} disabled at {format_node(node)} for {duration} slots"
        except Exception as e:
            status = f"⚠️ Failed to trigger breakdown: {str(e)}"
    elif trig == "apply-preset":
        try:
            if not scenario_value:
                raise ValueError("Pick a preset first")
            created = []
            if scenario_value == "smooth":
                status = "✅ Smooth Run preset applied (no incidents)."
            elif scenario_value == "acc_t3s2":
                ev = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="accident", location=(2, 1), start_time=sim.current_slot, duration_slots=6, info={"severity": "high"})
                acc_mgr.schedule(ev)
                sim.handle_accident((2, 1), 6)
                created.append("Track T3-S2 (6)")
                status = "🚨 Track accident preset applied."
            elif scenario_value == "st1_pf1_4":
                max_pf = min(4, PLATFORMS_PER_STATION)
                plats = [("Platform", 0, pf) for pf in range(0, max_pf)]
                for p in plats:
                    ev = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="accident", location=p, start_time=sim.current_slot, duration_slots=8, info={"severity": "medium"})
                    acc_mgr.schedule(ev)
                    sim.handle_accident(p, 8)
                created.append("Station1 P1-4 (8)")
                status = "🚨 Station 1 platform block preset applied."
            elif scenario_value == "bd_t3":
                tid = "T3"
                st = sim.state.get(tid)
                if st and st.get("pos") is not None:
                    node = st.get("pos")
                else:
                    # fallback: use its start node
                    node = next((t.start for t in trains if t.id == tid), None)
                ev = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="breakdown", location=node, start_time=sim.current_slot, duration_slots=5, info={"severity": "high", "train": tid})
                acc_mgr.schedule(ev)
                if st:
                    st["status"] = "blocked_by_accident"
                    st["accident_blocked_until"] = sim.current_slot + 5
                    st.setdefault("log", []).append({"slot": sim.current_slot, "action": "involved_in_accident", "node": node, "duration": 5, "event_id": ev.event_id})
                    acc_mgr.add_affected_train(ev.event_id, tid, sim.current_slot)
                    acc_mgr.set_involved_train(ev.event_id, tid)
                sim.handle_accident(node, 5)
                created.append("T3 breakdown (5)")
                status = "🧯 Train T3 breakdown preset applied."
            elif scenario_value == "mix":
                # Mix: schedule future events as a guided sequence adapted to single-station
                # Now: small platform block at Station1 P1-4 (or fewer if not available) for 6
                max_pf = min(4, PLATFORMS_PER_STATION)
                for pf in range(0, max_pf):
                    p = ("Platform", 0, pf)
                    ev = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="accident", location=p, start_time=sim.current_slot, duration_slots=6, info={"severity": "medium"})
                    acc_mgr.schedule(ev)
                    sim.handle_accident(p, 6)
                # +2: track accident T2-S3 for 6
                ev2 = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="accident", location=(1, 2), start_time=sim.current_slot + 2, duration_slots=6, info={"severity": "high"})
                acc_mgr.schedule(ev2)
                # +4: T2 breakdown (5)
                ev3 = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="breakdown", location=(0, 0), start_time=sim.current_slot + 4, duration_slots=5, info={"severity": "high", "train": "T2"})
                acc_mgr.schedule(ev3)
                status = "🎥 Guided mix preset queued: Station1 platforms now, track in +2, breakdown in +4."
            elif scenario_value == "rush_wave":
                # Rush hour: compress schedules of first half trains and small staggered platform blocks
                # Move arrival times earlier to create a wave
                for i, t in enumerate(trains[:max(1, len(trains)//2)]):
                    stt = sim.state.get(t.id)
                    if stt:
                        # Pull planned start earlier if possible
                        if stt.get("planned_slots"):
                            delta = min(2, stt["planned_slots"][0])
                            stt["planned_slots"] = [max(0, s - delta) for s in stt["planned_slots"]]
                # Light platform contention: block P0 then P1 briefly
                for offset, pf in enumerate(range(0, min(2, PLATFORMS_PER_STATION))):
                    p = ("Platform", 0, pf)
                    ev = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="accident", location=p, start_time=sim.current_slot + offset, duration_slots=3, info={"severity": "low"})
                    acc_mgr.schedule(ev)
                    if offset == 0:
                        sim.handle_accident(p, 3)
                status = "🚇 Rush Hour Wave: earlier arrivals and brief platform blocks queued."
            elif scenario_value == "plat_wave":
                # Platform wave: sequentially block P0->P3 to create a wave effect
                max_pf = min(4, PLATFORMS_PER_STATION)
                for i in range(max_pf):
                    p = ("Platform", 0, i)
                    ev = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="accident", location=p, start_time=sim.current_slot + i*2, duration_slots=4, info={"severity": "medium"})
                    acc_mgr.schedule(ev)
                    if i == 0:
                        sim.handle_accident(p, 4)
                status = "🌊 Platform Wave preset queued across P1→P4."
            elif scenario_value == "recovery":
                # Clean recovery: start with a couple of incidents that expire soon
                ev1 = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="accident", location=(0, 1), start_time=sim.current_slot, duration_slots=3, info={"severity": "low"})
                acc_mgr.schedule(ev1)
                sim.handle_accident((0, 1), 3)
                ev2 = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="accident", location=(2, 2), start_time=sim.current_slot + 1, duration_slots=3, info={"severity": "low"})
                acc_mgr.schedule(ev2)
                status = "🧽 Clean Recovery: short incidents scheduled to clear quickly."
            else:
                status = "ℹ️ Preset not recognized."
        except Exception as e:
            status = f"⚠️ Failed to apply preset: {str(e)}"
    elif trig == "guided-demo":
        try:
            # Sequence: immediate T3 breakdown 4m, +2 accident (2,1) 6m, +4 Station1 P1-4 8m
            tid = "T3"
            st = sim.state.get(tid)
            node = st.get("pos") if st and st.get("pos") is not None else next((t.start for t in trains if t.id == tid), (0, 0))
            ev1 = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="breakdown", location=node, start_time=sim.current_slot, duration_slots=4, info={"severity": "high", "train": tid})
            acc_mgr.schedule(ev1)
            if st:
                st["status"] = "blocked_by_accident"
                st["accident_blocked_until"] = sim.current_slot + 4
                st.setdefault("log", []).append({"slot": sim.current_slot, "action": "involved_in_accident", "node": node, "duration": 4, "event_id": ev1.event_id})
                acc_mgr.add_affected_train(ev1.event_id, tid, sim.current_slot)
                acc_mgr.set_involved_train(ev1.event_id, tid)
            sim.handle_accident(node, 4)
            ev2 = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="accident", location=(2, 1), start_time=sim.current_slot + 2, duration_slots=6, info={"severity": "high"})
            acc_mgr.schedule(ev2)
            plats = [("Platform", 0, pf) for pf in range(0, 4)]
            for p in plats:
                ev3 = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="accident", location=p, start_time=sim.current_slot + 4, duration_slots=8, info={"severity": "medium"})
                acc_mgr.schedule(ev3)
            status = "🎬 Guided demo queued: breakdown now, track accident in +2, station block in +4."
        except Exception as e:
            status = f"⚠️ Failed to queue guided demo: {str(e)}"
    elif trig == "unused":
        pass
    
    elif trig == "step-btn" or trig == "interval":
        sim.step_slot()
        status = f"Advanced to slot {sim.current_slot}"

    # Apply optional train filter
    filtered_state = sim.state
    filtered_trains = trains
    if train_filter:
        sel = set(train_filter)
        filtered_state = {tid: st for tid, st in sim.state.items() if tid in sel}
        filtered_trains = [t for t in trains if t.id in sel]

    # Parse platform_filter values back to tuples early (used by multiple views)
    selected_platforms = None
    if platform_filter:
        try:
            selected_platforms = [ast.literal_eval(p) for p in platform_filter]
        except Exception:
            selected_platforms = None

    # Figures
    # Replace timeline with track usage vs time (human readable)
    timeline_fig = plot_track_timeline(filtered_state, filtered_trains, accident_mgr=acc_mgr, current_slot=sim.current_slot)
    # Apply platform filter to Gantt as well (only trains using selected platforms)
    g_state = filtered_state
    g_trains = filtered_trains
    if selected_platforms:
        sel_set = set(selected_platforms)
        vis_trains = []
        for t in g_trains:
            st = g_state.get(t.id, {})
            uses = False
            for n in st.get("planned_path", []) or []:
                if isinstance(n, tuple) and n and n[0] == "Platform" and n in sel_set:
                    uses = True
                    break
            if uses:
                vis_trains.append(t)
        g_trains = vis_trains
        g_state = {t.id: g_state.get(t.id, {}) for t in g_trains}
    gantt_fig = plot_gantt_chart(g_state, g_trains, accident_mgr=acc_mgr, current_slot=sim.current_slot)
    # Human friendly: Stops Comparator (expected vs actual)
    stops_fig = plot_stops_schedule(filtered_state, platforms=selected_platforms, current_slot=sim.current_slot)

    if platform_view == "grid":
        from visualization import plot_station_overview
        station_fig = plot_station_overview(filtered_state, platforms=selected_platforms, current_slot=sim.current_slot)
    elif platform_view == "combined":
        from visualization import plot_station_overview_combined
        station_fig = plot_station_overview_combined(filtered_state, selected_platforms, current_slot=sim.current_slot)
    else:
        # Default: scatter view
        from visualization import plot_platform_train_scatter
        station_fig = plot_platform_train_scatter(filtered_state, platforms=(selected_platforms or PLATFORMS), current_slot=sim.current_slot)
        plat_train_scatter_fig = station_fig

    # Network map view
    network_fig = plot_network_map(G, filtered_state, PLATFORMS, current_slot=sim.current_slot, accident_mgr=acc_mgr)
    # Removed geo figure

    # Apply HD enhancements if enabled
    scale = 1.3 if (isinstance(hd_mode, list) and "hd" in hd_mode) else 1.0
    if scale != 1.0:
        timeline_fig = enhance_for_hd(timeline_fig, scale=scale)
        gantt_fig = enhance_for_hd(gantt_fig, scale=scale)
        station_fig = enhance_for_hd(station_fig, scale=scale)
        stops_fig = enhance_for_hd(stops_fig, scale=scale)
        network_fig = enhance_for_hd(network_fig, scale=scale)

    # Dark theme toggle
    if isinstance(dark_mode, list) and "dark" in dark_mode:
        for f in (timeline_fig, gantt_fig, station_fig, stops_fig, network_fig):
            try:
                f.update_layout(template="plotly_dark")
            except Exception:
                pass

    # Panels
    accident_log = generate_accident_log(acc_mgr, sim.current_slot)
    system_stats = generate_system_stats(sim.state, trains, acc_mgr, sim.current_slot)
    train_overview = generate_train_overview(sim.state, trains)
    # KPI badges text
    completed_trains = sum(1 for s in sim.state.values() if s.get("status") == "completed")
    total_delays_min = sum((s.get("waiting_s", 0.0) or 0.0) for s in sim.state.values()) / 60.0
    avg_delay_min = (total_delays_min / max(1, len(sim.state)))
    pct = (completed_trains / max(1, len(sim.state)) * 100.0)
    # Count reroutes from logs and active incidents
    reroutes = 0
    for st in sim.state.values():
        for lg in st.get("log", []):
            if isinstance(lg, tuple) and len(lg) >= 4 and lg[3] == "runtime_plan":
                reroutes += 1
    active_incidents = len([e for e in acc_mgr.scheduled if e.is_active_slot(sim.current_slot)])
    kpi_throughput = f"Throughput: {completed_trains}/{len(sim.state)} ({pct:.0f}%)"
    kpi_total = f"Total Delay: {total_delays_min:.1f} min | Reroutes: {reroutes}"
    kpi_avg = f"Avg Delay/Train: {avg_delay_min:.1f} min | Incidents: {active_incidents}"
    ai_summary = generate_ai_summary(sim.state, acc_mgr, PLATFORMS, sim.current_slot)
    ops_log = generate_operations_log(sim.state, sim.current_slot)
    now_board = generate_now_board(sim.state, trains, sim.current_slot)
    history_fig = generate_history_figure(SIM_HISTORY)

    # Simple mode hides detailed timelines
    simple = isinstance(simple_mode, list) and "simple" in simple_mode
    timeline_style = ({"display": "none"} if simple else {})

    # Plain English summary for judges
    completed = sum(1 for s in sim.state.values() if s.get("status") == "completed")
    blocked = sum(1 for s in sim.state.values() if s.get("status") == "blocked_by_accident")
    running = sum(1 for s in sim.state.values() if s.get("status") == "running")
    plain = [
        html.P(f"Time now is minute {sim.current_slot}."),
        html.P(f"Out of {len(trains)} trains, {completed} finished, {running} are moving, and {blocked} are waiting because of emergencies."),
        html.P("Trains follow colored lines: they move along grey tracks, switch curves to change track, and go to yellow platforms to stop."),
        html.P("If you see a red X, that piece of the track or a platform is blocked for some time. The system smartly tries other paths."),
    ]

    # Platform utilization KPI badge: estimate busy% and busiest platform
    # Reuse the same logic as in generate_system_stats (but lightweight)
    plat_busy_counts = {}
    for st in sim.state.values():
        for rec in st.get("log", []):
            if isinstance(rec, tuple) and len(rec) >= 4 and str(rec[3]).startswith("platform_until_"):
                node = rec[1]
                if isinstance(node, tuple) and node and node[0] == "Platform":
                    try:
                        end_slot = int(str(rec[3]).split("_")[-1])
                    except Exception:
                        end_slot = (rec[0] or 0)
                    start_slot = rec[0] or 0
                    for s in range(int(start_slot), int(end_slot)+1):
                        plat_busy_counts[node] = plat_busy_counts.get(node, 0) + 1
    elapsed = max(1, sim.current_slot or 1)
    if plat_busy_counts:
        busiest_pf, busiest_cnt = max(plat_busy_counts.items(), key=lambda kv: kv[1])
        util_pct = (sum(plat_busy_counts.values()) / (elapsed * max(1, len(PLATFORMS)))) * 100.0
        kpi_util = f"Platform Util: {util_pct:.0f}% | Busiest: {format_node(busiest_pf)}"
    else:
        kpi_util = "Platform Util: 0% | Busiest: N/A"

    return timeline_fig, gantt_fig, station_fig, stops_fig, plat_train_scatter_fig, network_fig, kpi_throughput, kpi_total, kpi_avg, kpi_util, timeline_style, status, accident_log, system_stats, train_overview, ai_summary, ops_log, plain, now_board, history_fig

# Removed dataset-related callbacks and dynamic section max

# Speed slider -> set interval period (ms). 1× = 1000ms per tick
@app.callback(Output("interval", "interval"), Input("sim-speed", "value"))
def set_speed(speed):
    # Guard and map to ms (inverse relation)
    try:
        val = float(speed) if speed is not None else 1.0
    except Exception:
        val = 1.0
    base_ms = 1000.0
    ms = max(50, int(base_ms / max(0.01, val)))
    return ms

# Download operations log as CSV
@app.callback(
    Output("download-ops", "data"),
    Input("download-ops-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_ops(n_clicks):
    import csv
    import io
    rows = generate_operations_log_rows(sim.state)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["slot","train","action","from","to","note"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return dict(content=output.getvalue(), filename="operations_log.csv")

# Download a selected history run as CSV
@app.callback(
    Output("download-history", "data"),
    Input("export-history-btn", "n_clicks"),
    State("history-index", "value"),
    prevent_initial_call=True
)
def download_history(n_clicks, idx):
    import csv, io
    try:
        if not idx or not SIM_HISTORY:
            return dash.no_update
        i = max(1, min(10, int(idx)))
        hist = SIM_HISTORY[-i]
        rows = hist.get("ops", []) or []
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["slot","train","action","from","to","note"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        return dict(content=output.getvalue(), filename=f"history_run_{i}.csv")
    except Exception:
        return dash.no_update

# Download the last 10 simulations (history list) as CSV with timestamps
@app.callback(
    Output("download-history-list", "data"),
    Input("download-history-list-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_history_list(n_clicks):
    import csv, io
    try:
        if not SIM_HISTORY:
            return dash.no_update
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["index","ts","wall_time","completed","total","throughput_pct","total_delay_min","avg_delay_min","incidents"]) 
        writer.writeheader()
        hist = SIM_HISTORY[-10:]
        for i, h in enumerate(hist, 1):
            pct = (h['completed'] / max(1, h['total'])) * 100.0
            writer.writerow({
                "index": i,
                "ts": h.get("ts"),
                "wall_time": h.get("wall_time"),
                "completed": h.get("completed"),
                "total": h.get("total"),
                "throughput_pct": f"{pct:.1f}",
                "total_delay_min": h.get("total_delay_min"),
                "avg_delay_min": h.get("avg_delay_min"),
                "incidents": h.get("incidents")
            })
        return dict(content=output.getvalue(), filename="simulation_history_last10.csv")
    except Exception:
        return dash.no_update

# =============================================================================
# APPLICATION EXECUTION - PROFESSIONAL DEPLOYMENT
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 RAILOPTIMUSSIM - ADVANCED RAILWAY CONTROL CENTER")
    print("="*80)
    print("🌐 Starting web server...")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050")
    print("🎯 System ready for professional railway simulation!")
    print("="*80 + "\n")
    
    # Launch the application with professional settings
    app.run(
        debug=True,
        host='127.0.0.1',
        port=8050,
        dev_tools_hot_reload=True
    )
