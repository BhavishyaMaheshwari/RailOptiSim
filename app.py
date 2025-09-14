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
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import uuid

# Import core system components
from data import build_graph, generate_fixed_trains
from accident_manager import EmergencyEvent, AccidentManager
from simulation import Simulator
from visualization import plot_track_timeline, plot_gantt_chart, plot_train_timeline
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

# =============================================================================
# SYSTEM INITIALIZATION - PROFESSIONAL RAILWAY SIMULATION SETUP
# =============================================================================

# Railway Network Configuration
NUM_TRACKS = 5          # Number of parallel tracks in the railway network
SECTIONS = 4            # Number of sections per track
HORIZON_MINUTES = 20    # Simulation planning horizon in minutes

# Build the railway infrastructure graph
print("🚂 Initializing Railway Infrastructure...")
G, PLATFORM = build_graph(num_tracks=NUM_TRACKS, sections_per_track=SECTIONS)
print(f"✅ Railway network built: {NUM_TRACKS} tracks × {SECTIONS} sections + Platform")

# Generate the train fleet
print("🚂 Generating Train Fleet...")
trains = generate_fixed_trains(sections_per_track=SECTIONS)
print(f"✅ {len(trains)} trains generated and ready for deployment")

# Initialize the accident management system
print("🚨 Initializing Emergency Management System...")
acc_mgr = AccidentManager()
print("✅ Emergency response system online")

# System starts with a clean slate - no predefined accidents
# All accidents will be triggered through the user interface for maximum control
print("🎯 System ready for user-controlled emergency scenarios")

# Initialize the simulation engine
print("⚙️ Initializing Simulation Engine...")
sim = Simulator(
    graph=G, 
    platform_node=PLATFORM, 
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
            html.P("This advanced simulation models 10 trains (Express, Passenger, Freight) operating on a 5-track railway network with 4 sections each, featuring intelligent pathfinding and real-time accident response.", 
                   className="card-text"),
            html.Hr(),
            html.H5("⚠️ Emergency Accident Interface", className="mb-3"),
            html.P("Use the controls below to trigger emergency scenarios and test the system's response capabilities:", 
                   style={"fontStyle": "italic", "color": "#7F8C8D"}),
            html.Ul([
                html.Li(html.Strong("Track Index (0-4):"), " Select the track where the emergency will occur"),
                html.Li(html.Strong("Section Index (0-3):"), " Choose the specific section on the selected track"),
                html.Li(html.Strong("Duration (1-120 slots):"), " Set how long the emergency will last (in minutes)"),
            ], className="mb-3"),
            html.P("Click 'Trigger Accident ⚠️' to activate the emergency scenario and observe real-time system response.", 
                   style={"fontStyle": "italic", "color": "#E74C3C", "fontWeight": "bold"}),
        ])
    ], className="mb-4"),
    dbc.Card([
        dbc.CardBody([
            html.H5("🎮 Simulation Controls", className="card-title mb-3"),
            dbc.Row([
                dbc.Col(dbc.Button("Step ▶", id="step-btn", color="primary", size="lg", className="me-2"), width="auto"),
                dbc.Col(dbc.Button("Run ⏵", id="run-btn", color="success", size="lg", className="me-2"), width="auto"),
                dbc.Col(dbc.Button("Pause ⏸", id="pause-btn", color="warning", size="lg", className="me-2"), width="auto"),
                dbc.Col(dbc.Button("Reset ↺", id="reset-btn", color="danger", size="lg", className="me-2"), width="auto"),
            ], className="mb-3"),
            html.Small("Use these controls to manage the simulation: Step for manual progression, Run for continuous operation, Pause to stop, and Reset to restart.", 
                      className="text-muted")
        ])
    ], className="mb-4"),
    dbc.Card([
        dbc.CardBody([
            html.H5("🚨 Emergency Scenario Trigger", className="card-title mb-3"),
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
            ], className="align-items-end"),
            html.Small("Configure emergency parameters and click to activate. The system will automatically detect affected trains and initiate rerouting procedures.", 
                      className="text-muted mt-2")
        ])
    ], className="mb-4"),
    dcc.Graph(id="track-timeline-graph"),
    dcc.Graph(id="timeline-graph"),
    dcc.Graph(id="gantt-graph"),
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
        ], width=6)
    ], className="mt-4"),
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
    Output("sim-status", "children"),
    Output("accident-log", "children"),
    Output("system-stats", "children"),
    Input("step-btn", "n_clicks"),
    Input("interval", "n_intervals"),
    Input("trigger-acc", "n_clicks"),
    Input("reset-btn", "n_clicks"),
    State("acc-track", "value"),
    State("acc-section", "value"),
    State("acc-duration", "value"),
)
def control(step_clicks, n_intervals, trigger_clicks, reset_clicks, acc_track, acc_section, acc_duration):
    global sim, acc_mgr
    ctx = dash.callback_context
    trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    status = "Idle"

    if trig == "reset-btn":
        # rebuild sim
        acc_mgr = AccidentManager()
        sim = Simulator(graph=G, platform_node=PLATFORM, trains=trains, accident_mgr=acc_mgr, horizon_minutes=SECTIONS*NUM_TRACKS)
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

    elif trig == "step-btn" or trig == "interval":
        sim.step_slot()
        status = f"Advanced to slot {sim.current_slot}"

    # Pass accident manager and current slot for dynamic updates
    track_fig = plot_track_timeline(sim.state, trains, accident_mgr=acc_mgr, current_slot=sim.current_slot)
    timeline_fig = plot_train_timeline(sim.state, trains, accident_mgr=acc_mgr)
    gantt_fig = plot_gantt_chart(sim.state, trains, accident_mgr=acc_mgr, current_slot=sim.current_slot)
    
    # Generate accident log
    accident_log = generate_accident_log(acc_mgr, sim.current_slot)
    
    # Generate system statistics
    system_stats = generate_system_stats(sim.state, trains, acc_mgr, sim.current_slot)
    
    return track_fig, timeline_fig, gantt_fig, status, accident_log, system_stats

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
