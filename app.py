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
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import uuid

# Import core system components
from data import build_graph, generate_fixed_trains
from accident_manager import EmergencyEvent, AccidentManager
from simulation import Simulator
from visualization import (
    plot_track_timeline,
    plot_gantt_chart,
    plot_train_timeline,
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
    
    stats_html = [
        html.H5("🚂 Train Status", className="mb-2"),
        html.P(f"✅ Completed: {completed_trains}/{len(trains)}"),
        html.P(f"🚫 Blocked: {blocked_trains}"),
        html.P(f"🚂 Running: {running_trains}"),
        html.P(f"⏳ Not Arrived: {not_arrived}"),
        html.Hr(),
        html.H5("📊 Performance Metrics", className="mb-2"),
        html.P(f"⏱️ Total Delays: {total_delays:.1f} minutes"),
        html.P(f"🔄 Total Reroutes: {total_reroutes}"),
        html.P(f"🚨 Active Accidents: {active_accidents}"),
        html.P(f"📈 Completion Rate: {(completed_trains/len(trains)*100):.1f}%"),
        html.Hr(),
        html.H5("⏰ System Time", className="mb-2"),
        html.P(f"Current Slot: {current_slot}"),
        html.P(f"Time: {current_slot} minutes")
    ]
    
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
    
    return [
        html.H5("🤖 AI Operations Summary"),
        html.Div([
            html.P(f"⏰ Current Time: Slot {current_slot} | 🚉 Platform Efficiency: {platform_efficiency:.1f}"),
            html.P(f"🚂 Fleet Status: {completed}/{total} completed, {running} active, {blocked} blocked"),
            html.P(f"🚨 Active Incidents: {', '.join(ev_details) if ev_details else 'None'}"),
            html.P(f"🏆 Top Platforms: {', '.join([f'{format_node(p)}({c})' for p, c in busiest]) if busiest else 'N/A'}"),
            html.P(f"📊 System Load: {'High' if blocked > 2 else 'Medium' if blocked > 0 else 'Normal'}")
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

# =============================================================================
# SYSTEM INITIALIZATION - PROFESSIONAL RAILWAY SIMULATION SETUP
# =============================================================================

# Railway Network Configuration
NUM_TRACKS = 8          # Number of parallel tracks in the railway network
SECTIONS = 4            # Number of sections per track
NUM_STATIONS = 2        # Number of stations
PLATFORMS_PER_STATION = 16  # Platforms per station (total = 32)
HORIZON_MINUTES = 20    # Simulation planning horizon in minutes

# Constrained platform access (richer mapping): each track connects to up to 4 platforms per station
PLATFORM_ACCESS_MAP = {}
for tr in range(NUM_TRACKS):
    choices = []
    max_links = min(4, PLATFORMS_PER_STATION)
    for st in range(NUM_STATIONS):
        for k in range(max_links):
            pf = (tr + k) % PLATFORMS_PER_STATION
            choices.append((st, pf))
    PLATFORM_ACCESS_MAP[tr] = choices

# Build the railway infrastructure graph
print("🚂 Initializing Railway Infrastructure...")
G, PLATFORMS = build_graph(
    num_tracks=NUM_TRACKS,
    sections_per_track=SECTIONS,
    num_stations=NUM_STATIONS,
    platforms_per_station=PLATFORMS_PER_STATION,
    platform_access_map=PLATFORM_ACCESS_MAP
)
print(f"✅ Railway network built: {NUM_TRACKS} tracks × {SECTIONS} sections + {NUM_STATIONS} stations × {PLATFORMS_PER_STATION} platforms")

# Generate the train fleet
print("🚂 Generating Train Fleet...")
trains = generate_fixed_trains(sections_per_track=SECTIONS)
print(f"✅ {len(trains)} trains generated and ready for deployment")

# Initialize the accident management system
print("🚨 Initializing Emergency Management System...")
acc_mgr = AccidentManager()
# Configure accident manager with network parameters
acc_mgr.set_network(sections_per_track=SECTIONS)
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
            html.P("This advanced simulation models 10 trains (Express, Passenger, Freight) on an 8-track network with 4 sections per track and 2 stations × 16 platforms (32 total), featuring intelligent pathfinding and real-time accident response.", 
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
    dbc.Card([
        dbc.CardBody([
            html.H5("� One-Click Demos & Presets", className="card-title mb-3"),
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
            html.H5("�🎮 Simulation Controls", className="card-title mb-3"),
            dbc.Row([
                dbc.Col(dbc.Button("Step ▶", id="step-btn", color="primary", size="lg", className="me-2"), width="auto"),
                dbc.Col(dbc.Button("Run ⏵", id="run-btn", color="success", size="lg", className="me-2"), width="auto"),
                dbc.Col(dbc.Button("Pause ⏸", id="pause-btn", color="warning", size="lg", className="me-2"), width="auto"),
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
                            dbc.Input(id="acc-track", placeholder="0-4", type="number", min=0, max=NUM_TRACKS-1, value=2, size="lg")
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Section Index", className="fw-bold"),
                            dbc.Input(id="acc-section", placeholder="0-3", type="number", min=0, max=SECTIONS-1, value=2, size="lg")
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Duration (slots)", className="fw-bold"),
                            dbc.Input(id="acc-duration", placeholder="1-120", type="number", min=1, max=120, value=6, size="lg")
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Action", className="fw-bold"),
                            dbc.Button("🚨 Trigger Emergency", id="trigger-acc", color="danger", size="lg", className="w-100")
                        ], width=3)
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
                            {"label": "Per Platform (grid)", "value": "grid"},
                            {"label": "Combined (stacked)", "value": "combined"},
                        ],
                        value="grid",
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
    dcc.Graph(id="track-timeline-graph", config={
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "scale": 3},
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]
    }),
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
    dbc.Card([
        dbc.CardBody([
            html.H5("🧭 What am I looking at? (Network Map)", className="card-title mb-2"),
            html.Ul([
                html.Li("The long grey lines are tracks — like roads for trains."),
                html.Li("Yellow dots are platforms — the places where trains stop. They’re grouped into two big yellow zones: Station 1 and Station 2."),
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
                    dbc.Button("Download CSV", id="download-ops-btn", color="secondary", size="sm", className="mb-2"),
                    dcc.Download(id="download-ops"),
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
    dbc.Card([
        dbc.CardBody([
            html.H5("🧠 AI Operations Summary", className="card-title mb-3"),
            html.Div(id="ai-summary")
        ])
    ], className="mt-3"),
    dbc.Card([
        dbc.CardBody([
            html.H5("🗣️ Plain English Summary", className="card-title mb-3"),
            html.Div(id="plain-summary", style={"fontFamily": "Arial, sans-serif", "fontSize": "14px"})
        ])
    ], className="mt-3"),
    dcc.Interval(id="interval", interval=1000, n_intervals=0, disabled=True),
    
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

# Callback: run/pause
@app.callback(Output("interval", "disabled"), Input("run-btn", "n_clicks"), Input("pause-btn", "n_clicks"), State("interval", "disabled"))
def run_pause(run_clicks, pause_clicks, is_disabled):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True
    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    if trig == "run-btn":
        return False
    if trig == "pause-btn":
        return True
    return is_disabled


# Callback: step, interval tick, trigger accident, reset
@app.callback(
    Output("track-timeline-graph", "figure"),
    Output("timeline-graph", "figure"),
    Output("gantt-graph", "figure"),
    Output("station-graph", "figure"),
    Output("network-map-graph", "figure"),
    Output("track-timeline-graph", "style"),
    Output("timeline-graph", "style"),
    Output("sim-status", "children"),
    Output("accident-log", "children"),
    Output("system-stats", "children"),
    Output("ai-summary", "children"),
    Output("ops-log", "children"),
    Output("plain-summary", "children"),
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
    State("platform-acc-platforms", "value"),
    State("platform-acc-duration", "value"),
    State("breakdown-train", "value"),
    State("breakdown-duration", "value"),
    State("scenario-preset", "value"),
)
def control(step_clicks, n_intervals, trigger_clicks, trigger_platform_clicks, trigger_breakdown_clicks, reset_clicks, apply_preset_clicks, guided_demo_clicks, train_filter, platform_filter, platform_view, hd_mode, dark_mode, simple_mode, acc_track, acc_section, acc_duration, platform_nodes, platform_acc_duration, breakdown_train, breakdown_duration, scenario_value):
    global sim, acc_mgr
    ctx = dash.callback_context
    trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    status = "Idle"

    if trig == "reset-btn":
        # rebuild sim
        acc_mgr = AccidentManager()
        acc_mgr.set_network(sections_per_track=SECTIONS)
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
            
            # Validate inputs
            if not (0 <= acc_track < NUM_TRACKS):
                raise ValueError(f"Track index must be between 0 and {NUM_TRACKS-1}")
            if not (0 <= acc_section < SECTIONS):
                raise ValueError(f"Section index must be between 0 and {SECTIONS-1}")
            if not (1 <= duration <= 120):
                raise ValueError("Duration must be between 1 and 120 slots")
                
            # Create and schedule accident
            ev = EmergencyEvent(
                event_id=str(uuid.uuid4())[:8],
                ev_type="accident",
                location=node,
                start_time=sim.current_slot,
                duration_slots=duration,
                info={"severity": "high"}
            )
            acc_mgr.schedule(ev)
            
            # Force reroute for affected trains
            sim.handle_accident(node, duration)
            
            status = f"🚨 Emergency: Track {acc_track}, Section {acc_section} blocked for {duration} slots"
        except Exception as e:
            status = f"⚠️ Failed to schedule accident: {str(e)}"
    elif trig == "trigger-platform-acc":
        try:
            if not platform_nodes or platform_acc_duration is None:
                raise ValueError("Select platforms and duration")
            duration = int(platform_acc_duration)
            selected_plats = []
            for p in platform_nodes:
                try:
                    selected_plats.append(ast.literal_eval(p))
                except Exception:
                    pass
            if not selected_plats:
                raise ValueError("No valid platforms selected")
            created = 0
            for pnode in selected_plats:
                ev = EmergencyEvent(
                    event_id=str(uuid.uuid4())[:8],
                    ev_type="accident",
                    location=pnode,
                    start_time=sim.current_slot,
                    duration_slots=duration,
                    info={"severity": "medium"}
                )
                acc_mgr.schedule(ev)
                # Reroute/mark affected
                sim.handle_accident(pnode, duration)
                created += 1
            status = f"🚨 Platform emergency: {created} platform(s) blocked for {duration} slots"
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
                plats = [("Platform", 0, pf) for pf in range(0, 4)]
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
                # Mix: schedule future events as a guided sequence
                # Now: small platform block at Station2 P9-12 for 6
                for pf in range(8, 12):
                    p = ("Platform", 1, pf)
                    ev = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="accident", location=p, start_time=sim.current_slot, duration_slots=6, info={"severity": "medium"})
                    acc_mgr.schedule(ev)
                    sim.handle_accident(p, 6)
                # +2: track accident T2-S3 for 6
                ev2 = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="accident", location=(1, 2), start_time=sim.current_slot + 2, duration_slots=6, info={"severity": "high"})
                acc_mgr.schedule(ev2)
                # +4: T2 breakdown (5)
                ev3 = EmergencyEvent(event_id=str(uuid.uuid4())[:8], ev_type="breakdown", location=(0, 0), start_time=sim.current_slot + 4, duration_slots=5, info={"severity": "high", "train": "T2"})
                acc_mgr.schedule(ev3)
                status = "🎥 Guided mix preset queued: platforms now, track in +2, breakdown in +4."
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

    # Figures
    track_fig = plot_track_timeline(filtered_state, filtered_trains, accident_mgr=acc_mgr, current_slot=sim.current_slot)
    timeline_fig = plot_train_timeline(filtered_state, filtered_trains, accident_mgr=acc_mgr)
    gantt_fig = plot_gantt_chart(filtered_state, filtered_trains, accident_mgr=acc_mgr, current_slot=sim.current_slot)

    # Platform view
    # Parse platform_filter values back to tuples
    selected_platforms = None
    if platform_filter:
        try:
            selected_platforms = [ast.literal_eval(p) for p in platform_filter]
        except Exception:
            selected_platforms = None
    # Replace occupancy with Stops Comparator (expected vs actual)
    station_fig = plot_stops_schedule(filtered_state, selected_platforms, current_slot=sim.current_slot)

    # Network map view
    network_fig = plot_network_map(G, filtered_state, PLATFORMS, current_slot=sim.current_slot, accident_mgr=acc_mgr)

    # Apply HD enhancements if enabled
    scale = 1.3 if (isinstance(hd_mode, list) and "hd" in hd_mode) else 1.0
    if scale != 1.0:
        track_fig = enhance_for_hd(track_fig, scale=scale)
        timeline_fig = enhance_for_hd(timeline_fig, scale=scale)
        gantt_fig = enhance_for_hd(gantt_fig, scale=scale)
        station_fig = enhance_for_hd(station_fig, scale=scale)
        network_fig = enhance_for_hd(network_fig, scale=scale)

    # Dark theme toggle
    if isinstance(dark_mode, list) and "dark" in dark_mode:
        for f in (track_fig, timeline_fig, gantt_fig, station_fig, network_fig):
            try:
                f.update_layout(template="plotly_dark")
            except Exception:
                pass

    # Panels
    accident_log = generate_accident_log(acc_mgr, sim.current_slot)
    system_stats = generate_system_stats(sim.state, trains, acc_mgr, sim.current_slot)
    ai_summary = generate_ai_summary(sim.state, acc_mgr, PLATFORMS, sim.current_slot)
    ops_log = generate_operations_log(sim.state, sim.current_slot)

    # Simple mode hides detailed timelines
    simple = isinstance(simple_mode, list) and "simple" in simple_mode
    track_style = ({"display": "none"} if simple else {})
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

    return track_fig, timeline_fig, gantt_fig, station_fig, network_fig, track_style, timeline_style, status, accident_log, system_stats, ai_summary, ops_log, plain

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
