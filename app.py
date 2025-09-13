# app.py
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import uuid

from data import build_graph, generate_fixed_trains
from accident_manager import EmergencyEvent, AccidentManager
from simulation import Simulator
from visualization import plot_track_timeline, plot_gantt_chart, plot_train_timeline

# Build infrastructure + trains
NUM_TRACKS = 5
SECTIONS = 4
G, PLATFORM = build_graph(num_tracks=NUM_TRACKS, sections_per_track=SECTIONS)
trains = generate_fixed_trains(sections_per_track=SECTIONS)
acc_mgr = AccidentManager()

# Add test accidents
test_accidents = [
    EmergencyEvent(event_id="test1", ev_type="accident", location=(2, 1), start_time=5, duration_slots=4),  # Track 2, Section 1
    EmergencyEvent(event_id="test2", ev_type="accident", location=(3, 2), start_time=8, duration_slots=3),  # Track 3, Section 2
    EmergencyEvent(event_id="test3", ev_type="accident", location=("Platform", 0), start_time=12, duration_slots=2)  # Platform accident
]
for ev in test_accidents:
    acc_mgr.schedule(ev)

sim = Simulator(graph=G, platform_node=PLATFORM, trains=trains, accident_mgr=acc_mgr, horizon_minutes=SECTIONS*NUM_TRACKS)

# initial planning
sim.plan_initial()

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H2("RailOptimusSim — Live Control Panel"),
    html.Div([
        html.P("""
        This simulation models 10 trains running on 5 tracks, each with 4 sections.
        """),
        html.Ul([
            html.Li("Track index (0..4): The track where the accident will occur."),
            html.Li("Section index (0..3): The section on the selected track for the accident."),
            html.Li("Duration slots (minutes): How long the accident lasts (1-120)."),
        ]),
        html.P("Fill these fields and click 'Trigger Accident ⚠' to schedule an accident at the specified location and duration.", style={"fontStyle": "italic"}),
        html.Hr()
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("Step ▶", id="step-btn", color="primary"), width="auto"),
        dbc.Col(dbc.Button("Run ⏵", id="run-btn", color="success"), width="auto"),
        dbc.Col(dbc.Button("Pause ⏸", id="pause-btn", color="warning"), width="auto"),
        dbc.Col(dbc.Button("Reset ↺", id="reset-btn", color="danger"), width="auto"),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(dbc.Input(id="acc-track", placeholder="Track index (0..4)", type="number", min=0, max=NUM_TRACKS-1, value=2), width=3),
        dbc.Col(dbc.Input(id="acc-section", placeholder="Section index (0..3)", type="number", min=0, max=SECTIONS-1, value=2), width=3),
        dbc.Col(dbc.Input(id="acc-duration", placeholder="Duration slots (minutes)", type="number", min=1, max=120, value=6), width=3),
        dbc.Col(dbc.Button("Trigger Accident ⚠", id="trigger-acc", color="danger"), width=3)
    ], className="mb-2"),
    dcc.Graph(id="track-timeline-graph"),
    dcc.Graph(id="timeline-graph"),
    dcc.Graph(id="gantt-graph"),
    dcc.Interval(id="interval", interval=1000, n_intervals=0, disabled=True),
    html.Div(id="sim-status", style={"marginTop": "10px"}),
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
    return track_fig, timeline_fig, gantt_fig, status

if __name__ == "__main__":
    app.run(debug=True)
