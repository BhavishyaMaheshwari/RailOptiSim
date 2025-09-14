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

Author: RailOptimusSim Development Team
Version: 2.0 Professional Edition
"""

from collections import defaultdict
import pandas as pd
import plotly.graph_objects as go
from utils import format_node, short_node

def calculate_delays(df):
    """Calculate comprehensive delay and impact statistics"""
    stats = {
        'delays': defaultdict(int),          # Train -> total delay slots
        'reroutes': defaultdict(int),        # Train -> number of reroutes
        'affected': defaultdict(int),        # Train -> number of times affected
        'current_delays': defaultdict(list),  # Train -> list of active delays
        'blocked_sections': set(),           # Set of currently blocked sections
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
                    
        elif action == "affected_by_accident":
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
                
        elif action == "involved_in_accident":
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
            stats['delays'][train] += 1  # Each wait slot counts as delay
            
        elif action == "accident_resolved":
            # Train has resumed after accident
            stats['affected'][train] += 1
            
        elif action == "completed":
            # Train has completed its journey
            stats['affected'][train] += 1
                
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

def plot_interactive(state, trains, accident_mgr, graph, platform_node, sim_slot, kpis=None):
    df = build_records_from_state(state)
    if df.empty:
        fig = go.Figure(); fig.update_layout(title="No events to plot"); return fig

    trains_list = sorted(df["train"].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    palette = [
        "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231",
        "#911EB4", "#46F0F0", "#F032E6", "#BCF60C", "#17BECF"
    ]
    color_map = {tid: palette[i % len(palette)] for i, tid in enumerate(trains_list)}
    fig = go.Figure()

    # Draw grid for tracks and sections
    for tr in range(5):
        fig.add_shape(type="line", x0=-0.5, x1=3.5, y0=tr-0.5, y1=tr-0.5, line=dict(color="#bbb", width=1, dash="dot"), layer="below")
    for sec in range(4):
        fig.add_shape(type="line", x0=sec, x1=sec, y0=-0.5, y1=4.5, line=dict(color="#bbb", width=1, dash="dot"), layer="below")
    # Platform line
    fig.add_shape(type="line", x0=-0.5, x1=-0.5, y0=-0.5, y1=4.5, line=dict(color="#222", width=2, dash="solid"), layer="below")

    for i, tid in enumerate(trains_list):
        sub = df[df["train"] == tid]
        xs, ys, hover, dirs, labels = [], [], [], [], []
        nodes = sub["node"].tolist(); slots = sub["slot"].tolist()
        for j, (node, slot) in enumerate(zip(nodes, slots)):
            if node is None:
                x = None
            elif isinstance(node, tuple) and node[0] == "Platform":
                x = -0.5
            elif isinstance(node, tuple):
                x = node[1]
            else:
                x = None
            xs.append(x); ys.append(i); hover.append(f"{tid}<br>{format_node(node)}<br>slot={slot}")
            labels.append(tid)
        for j in range(len(xs)):
            if j == 0: dirs.append("→")
            else:
                prev, cur = xs[j-1], xs[j]
                if prev is None or cur is None: dirs.append("→")
                elif cur > prev: dirs.append("→")
                elif cur < prev: dirs.append("←")
                else: dirs.append("•")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers+text", line=dict(color=color_map[tid], width=3),
            marker=dict(size=12, color=color_map[tid], line=dict(width=2, color="#222")),
            text=labels, textposition="bottom center",
            name=f"{tid} ({[t.type for t in trains if t.id==tid][0]})",
            hoverinfo="text", hovertext=hover, showlegend=True
        ))
        # switch markers
        node_list = nodes
        for j in range(len(node_list)-1):
            a = node_list[j]; b = node_list[j+1]
            if isinstance(a, tuple) and isinstance(b, tuple) and a[0] != b[0]:
                xm = xs[j+1]; ym = i
                fig.add_trace(go.Scatter(
                    x=[xm], y=[ym], mode="markers+text",
                    marker=dict(symbol="diamond", size=18, color=color_map[tid], line=dict(color="black", width=2)),
                    text=[f"Switch!"], textposition="top center",
                    hoverinfo="text", hovertext=[f"Switch: {short_node(a)} → {short_node(b)}"], name="Switch", showlegend=False
                ))

    # Highlight platform node
    fig.add_trace(go.Scatter(
        x=[-0.5], y=[-1], mode="markers+text",
        marker=dict(symbol="star", size=22, color="#FFD700", line=dict(width=2, color="#222")),
        text=["Platform"], textposition="bottom center",
        name="Platform", showlegend=True
    ))

    # annotate accidents
    actives = accident_mgr.active_summary(sim_slot)
    if actives:
        txt = "🚨 Active events:<br>" + "<br>".join([f"{eid} {evtype} @{format_node(loc)} ({rem} slots left)" for eid, evtype, loc, rem in actives])
        fig.add_annotation(xref="paper", yref="paper", x=1.02, y=0.95, text=txt, showarrow=False, bgcolor="rgba(255,200,200,0.9)")
        # Add accident markers on the plot
        accident_xs = []
        accident_ys = []
        accident_texts = []
        for eid, evtype, loc, rem in actives:
            if isinstance(loc, tuple) and loc[0] == "Platform":
                x = -0.5
                y = -1  # Place platform accidents below tracks
            elif isinstance(loc, tuple):
                x = loc[1]
                y = loc[0]
            else:
                continue
            accident_xs.append(x)
            accident_ys.append(y)
            accident_texts.append(f"🚨 {evtype} ({rem} left)")
        if accident_xs:
            fig.add_trace(go.Scatter(
                x=accident_xs,
                y=accident_ys,
                mode="markers+text",
                marker=dict(symbol="x", size=22, color="red", line=dict(width=3, color="black")),
                text=accident_texts,
                textposition="top center",
                name="Accident",
                showlegend=True
            ))
    if kpis:
        kpi_txt = f"📊 KPIs<br>Avg wait (s): {kpis['avg_wait_s']:.1f}<br>Throughput: {kpis['throughput']}/{len(trains)}"
        fig.add_annotation(xref="paper", yref="paper", x=1.02, y=0.75, text=kpi_txt, showarrow=False, bgcolor="white", bordercolor="black")
    fig.update_layout(
        title=f"Train Timeline (slot ≈ {sim_slot})",
        xaxis_title="Section (Platform = -0.5)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(trains_list))),
            ticktext=[f"{tid} ({[t.type for t in trains if t.id==tid][0]})" for tid in trains_list],
            range=[-1.5, len(trains_list)-0.5],
            title="Trains"
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=[-0.5,0,1,2,3],
            ticktext=["Platform", "Sec1", "Sec2", "Sec3", "Sec4"],
            range=[-1, 3.5],
            title="Section"
        ),
        height=700, width=1200, margin=dict(r=320), legend=dict(orientation="h", y=-0.15)
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(dtick=1)
    return fig

def plot_track_timeline(state, trains, accident_mgr=None, current_slot=None):
    """
    Plots which train is on which track as a function of time (slot).
    Shows current time and differentiates between past/future predictions.
    """
    import plotly.graph_objects as go
    import pandas as pd
    from collections import defaultdict
    
    df = build_records_from_state(state)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No track timeline data (Current Slot: {current_slot})")
        return fig
        
    # Split data into past and future
    if current_slot is not None:
        df["is_future"] = df["slot"] > current_slot
        df["is_past"] = df["slot"] <= current_slot
        # Mark trains currently in blocked sections
        if accident_mgr:
            blocked = accident_mgr.blocked_nodes(current_slot)
            df["is_blocked"] = df.apply(lambda x: x["node"] in blocked and x["slot"] == current_slot, axis=1)

    trains_list = sorted(df["train"].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    
    # Create unique colors for each train - ensuring maximum visual distinction
    unique_train_colors = [
        "#FF6B6B",  # T1 - Vibrant Red
        "#4ECDC4",  # T2 - Teal
        "#45B7D1",  # T3 - Blue
        "#96CEB4",  # T4 - Mint Green
        "#FFEAA7",  # T5 - Yellow
        "#DDA0DD",  # T6 - Plum
        "#98D8C8",  # T7 - Seafoam
        "#F7DC6F",  # T8 - Gold
        "#BB8FCE",  # T9 - Light Purple
        "#85C1E9"   # T10 - Light Blue
    ]
    
    # Create color mapping - each train gets a unique color
    color_map = {}
    for i, tid in enumerate(trains_list):
        color_map[tid] = unique_train_colors[i % len(unique_train_colors)]
    
    fig = go.Figure()
    
    # Draw blocked sections and accident effects
    blocked_sections = []
    accident_log = []
    if accident_mgr is not None and current_slot is not None:
        actives = accident_mgr.active_summary(current_slot)
        for eid, evtype, loc, remaining, stats in actives:
            if isinstance(loc, tuple):
                track = -1 if loc[0] == "Platform" else loc[0]
                severity = stats.get("severity", "normal")
                affected = stats.get("affected_trains", 0)
                rerouted = stats.get("rerouted_trains", 0)
                total_delay = stats.get("total_delay", 0)
                
                # Draw blocked track rectangle (entire track is blocked)
                for offset in range(remaining):
                    opacity = 0.8 - (offset * 0.1)  # Fade out into the future
                    fig.add_shape(
                        type="rect",
                        x0=current_slot + offset - 0.5,
                        x1=current_slot + offset + 0.5,
                        y0=track - 0.4,
                        y1=track + 0.4,
                        fillcolor="black" if severity == "high" else "darkred",
                        opacity=max(0.3, opacity),
                        line=dict(color="red", width=3),
                        layer="below"
                    )
                
                # Add warning text with severity and impact
                warning_text = f"{'🚨' if severity == 'high' else '⚠️'} TRACK BLOCKED"
                fig.add_annotation(
                    x=current_slot + remaining/2,
                    y=track,
                    text=warning_text,
                    showarrow=False,
                    font=dict(color="red", size=12, weight="bold"),
                    bgcolor="rgba(255,255,255,0.9)"
                )
                
                # Add impact marker
                if affected > 0:
                    fig.add_annotation(
                        x=current_slot,
                        y=track + 0.5,
                        text=f"↯ {affected} affected",
                        showarrow=False,
                        yanchor="bottom",
                        font=dict(color="red", size=10),
                        bgcolor="rgba(255,255,255,0.9)"
                    )
                
                # Get involved train info
                involved_train = accident_mgr.involved_trains.get(eid, "Unknown")
                
                # Get rerouted trains list
                rerouted_trains = accident_mgr.rerouted_trains.get(eid, [])
                rerouted_list = ", ".join(rerouted_trains) if rerouted_trains else "None"
                
                # Detailed accident log
                accident_log.append(
                    f"{'🚨' if severity == 'high' else '⚠️'} {evtype.upper()} at {format_node(loc)}<br>"
                    f"⏰ {remaining} slots remaining<br>"
                    f"🚂 Involved: {involved_train}<br>"
                    f"🔄 Rerouted: {rerouted_list}<br>"
                    f"⏱️ Total delay: {total_delay} slots"
                )
                
                blocked_sections.append((current_slot, track))    # Plot train paths with enhanced visualization and status tracking
    for tid in trains_list:
        train_type = [t.type for t in trains if t.id == tid][0]
        sub = df[df["train"] == tid]
        past_xs, past_ys, past_hover = [], [], []
        future_xs, future_ys, future_hover = [], [], []
        reroute_xs, reroute_ys = [], []
        platform_xs, platform_ys = [], []
        delay_xs, delay_ys = [], []
        
        for idx, row in sub.iterrows():
            slot = row["slot"]
            node = row["node"]
            action = row.get("action", "")
            is_future = current_slot is not None and slot > current_slot
            
            if node is None:
                y = None
            elif isinstance(node, tuple) and node[0] == "Platform":
                y = -1
                platform_xs.append(slot)
                platform_ys.append(y)
            elif isinstance(node, tuple):
                y = node[0]
            else:
                y = None
                
            hover_text = (
                f"{tid} ({train_type})<br>"
                f"Slot: {slot}<br>"
                f"Location: {format_node(node)}"
            )
            if action:
                hover_text += f"<br>Action: {action}"
                
            if is_future:
                future_xs.append(slot)
                future_ys.append(y)
                future_hover.append(hover_text + " (Predicted)")
            else:
                past_xs.append(slot)
                past_ys.append(y)
                past_hover.append(hover_text)
            
            if action == "runtime_plan":
                reroute_xs.append(slot)
                reroute_ys.append(y)
            elif action == "affected_by_accident":
                delay_xs.append(slot)
                delay_ys.append(y)
            elif action == "involved_in_accident":
                delay_xs.append(slot)
                delay_ys.append(y)
            elif action == "accident_resolved":
                delay_xs.append(slot)
                delay_ys.append(y)
            elif action == "completed":
                delay_xs.append(slot)
                delay_ys.append(y)

        # Past path (solid line)
        if past_xs:
            fig.add_trace(go.Scatter(
                x=past_xs, y=past_ys,
                mode="lines+markers",
                line=dict(color=color_map[tid], width=3),
                marker=dict(size=8, color=color_map[tid]),
                name=f"{tid} ({train_type})",
                hoverinfo="text",
                hovertext=past_hover
            ))
        
        # Future path (dashed line)
        if future_xs:
            fig.add_trace(go.Scatter(
                x=future_xs, y=future_ys,
                mode="lines+markers",
                line=dict(color=color_map[tid], width=2, dash="dot"),
                marker=dict(size=6, color=color_map[tid], opacity=0.7),
                name=f"{tid} ({train_type}) - Predicted",
                hoverinfo="text",
                hovertext=future_hover,
                opacity=0.6,
                showlegend=True
            ))
        
        # Platform visits (gold stars)
        if platform_xs:
            fig.add_trace(go.Scatter(
                x=platform_xs, y=platform_ys,
                mode="markers",
                marker=dict(symbol="star", size=15, color="gold", line=dict(color="black", width=1)),
                name=f"{tid} at Platform",
                showlegend=False
            ))
            
        # Reroute events (orange diamonds)
        if reroute_xs:
            fig.add_trace(go.Scatter(
                x=reroute_xs, y=reroute_ys,
                mode="markers+text",
                marker=dict(symbol="diamond", size=12, color="orange", line=dict(color="black", width=1)),
                text=["↺"] * len(reroute_xs),
                textposition="top center",
                name=f"{tid} Rerouted",
                showlegend=False
            ))
            
        # Delay events (red triangles)
        if delay_xs:
            fig.add_trace(go.Scatter(
                x=delay_xs, y=delay_ys,
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=10, color="red", line=dict(color="black", width=1)),
                text=["⏱"] * len(delay_xs),
                textposition="top center",
                name=f"{tid} Delayed",
                showlegend=False
            ))
            
        # Check if this train is currently blocked by accident
        train_state = state.get(tid, {})
        if train_state.get("status") == "blocked_by_accident":
            blocked_until = train_state.get("accident_blocked_until", current_slot)
            if current_slot is not None and current_slot < blocked_until:
                pos = train_state.get("pos")
                y_pos = pos[0] if isinstance(pos, tuple) and pos[0] != "Platform" else -1
                fig.add_trace(go.Scatter(
                    x=[current_slot], y=[y_pos],
                    mode="markers+text",
                    marker=dict(symbol="square", size=20, color="darkred", line=dict(color="black", width=2)),
                    text=["🚫 BLOCKED"],
                    textposition="top center",
                    name=f"{tid} Blocked",
                    showlegend=False
                ))
        
        # Add resume event markers
        resume_xs, resume_ys = [], []
        for log_entry in train_state.get("log", []):
            if isinstance(log_entry, tuple) and len(log_entry) >= 4 and log_entry[3] == "resume":
                slot = log_entry[0]
                pos = log_entry[2] if len(log_entry) > 2 else None
                if pos is not None:
                    if isinstance(pos, tuple) and pos[0] != "Platform":
                        resume_xs.append(slot)
                        resume_ys.append(pos[0])
        
        if resume_xs:
            fig.add_trace(go.Scatter(
                x=resume_xs, y=resume_ys,
                mode="markers+text",
                marker=dict(
                    symbol="star", 
                    size=25, 
                    color="#00FF7F", 
                    line=dict(width=3, color="#006400")
                ),
                text=["★ RESUME"] * len(resume_xs),
                textposition="top center",
                name=f"{tid} Resume",
                showlegend=False,
                hovertemplate=f"<b>{tid}</b><br>🚀 RESUMED at slot %{{x}}<br>Track: %{{y}}<br><extra></extra>"
            ))
        
        # Add journey completion markers
        completion_xs, completion_ys = [], []
        for log_entry in train_state.get("log", []):
            if isinstance(log_entry, tuple) and len(log_entry) >= 4 and log_entry[3] == "completed":
                slot = log_entry[0]
                pos = log_entry[2] if len(log_entry) > 2 else None
                if pos is not None:
                    if isinstance(pos, tuple) and pos[0] != "Platform":
                        completion_xs.append(slot)
                        completion_ys.append(pos[0])
        
        if completion_xs:
            fig.add_trace(go.Scatter(
                x=completion_xs, y=completion_ys,
                mode="markers+text",
                marker=dict(
                    symbol="x", 
                    size=28, 
                    color="#32CD32", 
                    line=dict(width=4, color="#228B22")
                ),
                text=["✓ COMPLETED"] * len(completion_xs),
                textposition="top center",
                name=f"{tid} Completed",
                showlegend=False,
                hovertemplate=f"<b>{tid}</b><br>🎯 COMPLETED at slot %{{x}}<br>Track: %{{y}}<br><extra></extra>"
            ))

    # Add current time marker if provided
    if current_slot is not None:
        fig.add_vline(
            x=current_slot,
            line_width=2,
            line_dash="solid",
            line_color="gray",
            annotation=dict(
                text="Current Time",
                textangle=-90,
                yanchor="bottom",
                font=dict(size=12)
            )
        )

    # Calculate comprehensive statistics
    completed_trains = len([t for t in trains_list if state.get(t, {}).get("status") == "completed"])
    blocked_trains = len([t for t in trains_list if state.get(t, {}).get("status") == "blocked_by_accident"])
    running_trains = len([t for t in trains_list if state.get(t, {}).get("status") == "running"])
    
    # Update layout with enhanced information
    title_text = [
        "🚂 RAILWAY TRACK TIMELINE",
        f"⏰ Current Time: Slot {current_slot}" if current_slot is not None else None,
        f"🚨 Active Accidents: {len(accident_log)}" if accident_log else None,
        f"✅ Completed: {completed_trains} | 🚫 Blocked: {blocked_trains} | 🚂 Running: {running_trains}"
    ]
    title_text = " | ".join(filter(None, title_text))
        
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title="Time (slot)",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='gray'
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[-1] + list(range(5)),
            ticktext=["Platform"] + [f"Track {i}" for i in range(5)],
            title="Track",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='gray'
        ),
        height=600,  # Increased height for better visibility
        width=1200,  # Increased width for better spacing
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(r=300)  # Extra right margin for accident log
    )
    
    # Add accident log as enhanced annotation with icons and formatting
    if accident_log:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.02, y=1,
            text="<br>".join(accident_log),
            showarrow=False,
            bordercolor="red",
            borderwidth=2,
            bgcolor="rgba(255,255,255,0.95)",
            align="left",
            font=dict(size=11)
        )
    
    return fig
    import pandas as pd
    import plotly.graph_objects as go
    df = build_records_from_state(state)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No track timeline data")
        return fig

    # Setup initial data
    trains_list = sorted(df["train"].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    palette = [
        "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231",
        "#911EB4", "#46F0F0", "#F032E6", "#BCF60C", "#17BECF"
    ]
    color_map = {tid: palette[i % len(palette)] for i, tid in enumerate(trains_list)}
    
    # Calculate delays and get all time slots
    stats = calculate_delays(df)
    delays = stats['delays']
    reroutes = stats['reroutes']
    total_delay = sum(delays.values())
    all_slots = set(df["slot"].unique())
    
    # Initialize visualization data
    accident_slots = set()
    accident_points = []
    accident_log = []
    blocked_sections = {}
    
    fig = go.Figure()

    # Process accidents and blocked sections
    if accident_mgr is not None:
        for slot in sorted(all_slots):
            actives = accident_mgr.active_summary(slot)
            for eid, evtype, loc, rem in actives:
                if isinstance(loc, tuple):
                    track = -1 if loc[0] == "Platform" else loc[0]
                    section = -0.5 if loc[0] == "Platform" else loc[1]
                    blocked_sections[(track, section)] = (slot, rem)
                    accident_log.append(f"🚨 Slot {slot}: Track {track}, Section {section} - {evtype} ({rem} slots left)")
                    if loc[0] == "Platform":
                        y = -1
                    else:
                        y = loc[0]
                    accident_points.append((slot, y, evtype, rem, loc[0] == "Platform"))
                    accident_slots.add((slot, y))

    # Add blocked section visualization
    for (track, section), (start_slot, duration) in blocked_sections.items():
        # Black rectangle for blocked section
        fig.add_shape(
            type="rect",
            x0=start_slot - 0.5,
            x1=start_slot + duration + 0.5,
            y0=track - 0.5,
            y1=track + 0.5,
            fillcolor="rgba(0,0,0,0.8)",
            line=dict(color="red", width=2, dash="dot"),
            layer="below"
        )
        # Warning text
        fig.add_annotation(
            x=start_slot + duration/2,
            y=track,
            text="⚠️ BLOCKED ⚠️",
            showarrow=False,
            font=dict(color="white", size=14, weight="bold"),
            bgcolor="rgba(255,0,0,0.7)"
        )

    # Add accident time slots (red stripes)
    for slot, y in accident_slots:
        fig.add_shape(
            type="rect",
            x0=slot-0.5,
            x1=slot+0.5,
            y0=y-0.5 if y != -1 else -1.5,
            y1=y+0.5 if y != -1 else -0.5,
            fillcolor="rgba(255,0,0,0.2)",
            line_width=0,
            layer="below"
        )

    # Plot trains with rerouting visualization
    for i, tid in enumerate(trains_list):
        sub = df[df["train"] == tid]
        xs, ys, hover, marker_syms, marker_cols, marker_sizes, texts = [], [], [], [], [], [], []
        
        # Track train status
        is_delayed = delays[tid] > 0
        is_rerouted = reroutes[tid] > 0
        
        for _, row in sub.iterrows():
            slot = row["slot"]
            node = row["node"]
            action = row["action"]
            
            # Process node coordinates and markers
            if node is None:
                y = None
            elif isinstance(node, tuple) and node[0] == "Platform":
                y = -1
            elif isinstance(node, tuple):
                y = node[0]
            else:
                y = None
                
            xs.append(slot)
            ys.append(y)
            
            # Enhanced markers for different events
            if action == "enter":
                hover.append(f"{tid}<br>slot={slot}<br>track={y}<br>STARTED journey")
                marker_syms.append("circle-open-dot")
                marker_cols.append(color_map[tid])
                marker_sizes.append(16)
                texts.append("Start")
            elif action == "depart":
                hover.append(f"{tid}<br>slot={slot}<br>track={y}<br>DEPARTED platform")
                marker_syms.append("star-diamond")
                marker_cols.append("#FFD700")
                marker_sizes.append(20)
                texts.append("Departed")
            elif action == "completed":
                hover.append(f"{tid}<br>slot={slot}<br>track={y}<br>Journey Ended")
                marker_syms.append("x-thin-open")
                marker_cols.append("#222")
                marker_sizes.append(22)
                texts.append("Ended")
            elif action == "runtime_plan":
                hover.append(f"{tid}<br>slot={slot}<br>track={y}<br>Rerouted!")
                marker_syms.append("diamond-cross")
                marker_cols.append("orange")
                marker_sizes.append(20)
                texts.append("↗️")
            elif action == "wait_accident":
                hover.append(f"{tid}<br>slot={slot}<br>track={y}<br>WAITING (accident)")
                marker_syms.append("hourglass")
                marker_cols.append("#d62728")
                marker_sizes.append(18)
                texts.append("⌛")
            else:
                hover.append(f"{tid}<br>slot={slot}<br>track={y}")
                marker_syms.append("circle")
                marker_cols.append(color_map[tid])
                marker_sizes.append(10)
                texts.append("")

        # Add train path with status-dependent style
        line_style = dict(color=color_map[tid], width=3)
        if is_rerouted:
            line_style.update(dash="dot")
        if is_delayed:
            line_style.update(width=2)
        
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers+text",
            line=line_style,
            marker=dict(
                size=marker_sizes,
                color=marker_cols,
                symbol=marker_syms,
                line=dict(width=2, color="#222")
            ),
            text=texts,
            textposition="top center",
            name=f"{tid}",
            hoverinfo="text",
            hovertext=hover,
            showlegend=True
        ))

    # Add accidents as warning markers
    if accident_points:
        fig.add_trace(go.Scatter(
            x=[pt[0] for pt in accident_points],
            y=[pt[1] for pt in accident_points],
            mode="markers+text",
            marker=dict(
                symbol=["star" if pt[4] else "warning" for pt in accident_points],
                size=26,
                color="red",
                line=dict(width=3, color="black")
            ),
            text=["🚨" for _ in accident_points],
            textposition="top center",
            name="Accident",
            showlegend=True
        ))

    # Add KPI annotation
    kpi_txt = (
        f"📊 KPIs<br>Avg wait (s): {stats.get('avg_wait_s', 0):.1f}<br>"
        f"Throughput: {stats.get('throughput', 0)}/{len(trains_list)}"
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.02,
        y=0.95,
        text=kpi_txt,
        showarrow=False,
        bgcolor="white",
        bordercolor="black"
    )

    # Add accident log
    if accident_log:
        log_text = "<br>".join(accident_log[-5:])
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1.02,
            y=0.5,
            text=f"🚨 Active Events<br>{log_text}",
            showarrow=False,
            bgcolor="rgba(255,200,200,0.9)",
            bordercolor="red",
            borderwidth=2
        )

    fig.update_layout(
        title=dict(
            text="Track Occupancy vs Time (slot)",
            x=0.5,
            y=0.95
        ),
        xaxis_title="Time (slot)",
        yaxis=dict(
            tickmode="array",
            tickvals=[-1,0,1,2,3,4],
            ticktext=["Platform", "Track0", "Track1", "Track2", "Track3", "Track4"],
            range=[-1.5, 4.5],
            title="Track"
        ),
        height=400,
        width=1100,
        margin=dict(r=200, t=50),
        showlegend=True,
        legend=dict(orientation="h", y=-0.2)
    )
    
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(dtick=1)
    return fig

def plot_gantt_chart(state, trains, accident_mgr=None, current_slot=None):
    """
    Plots a Gantt chart of each train's journey with enhanced delay visualization and sophisticated color coding.
    """
    import pandas as pd
    import plotly.graph_objects as go
    df = build_records_from_state(state)
    if df.empty:
        fig = go.Figure(); fig.update_layout(title="No Gantt data"); return fig

    # Setup and calculate delays
    trains_list = sorted(df["train"].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    stats = calculate_delays(df)
    delays = stats['delays']
    reroutes = stats['reroutes']
    total_delay = sum(delays.values())
    
    # Create unique colors for each train - ensuring maximum visual distinction
    unique_train_colors = [
        "#FF6B6B",  # T1 - Vibrant Red
        "#4ECDC4",  # T2 - Teal
        "#45B7D1",  # T3 - Blue
        "#96CEB4",  # T4 - Mint Green
        "#FFEAA7",  # T5 - Yellow
        "#DDA0DD",  # T6 - Plum
        "#98D8C8",  # T7 - Seafoam
        "#F7DC6F",  # T8 - Gold
        "#BB8FCE",  # T9 - Light Purple
        "#85C1E9"   # T10 - Light Blue
    ]
    
    # Create color mapping - each train gets a unique color
    color_map = {}
    for i, tid in enumerate(trains_list):
        color_map[tid] = unique_train_colors[i % len(unique_train_colors)]
    
    # Status-based color variations (darker versions for different states)
    status_colors = {
        "completed": {},  # Will be filled dynamically
        "blocked": {},    # Will be filled dynamically
        "delayed": {},    # Will be filled dynamically
        "normal": color_map
    }
    
    # Create status variations for each train's base color
    for tid in trains_list:
        base_color = color_map[tid]
        # Convert hex to RGB for manipulation
        hex_color = base_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Create darker versions for different statuses
        completed_rgb = tuple(max(0, int(c * 0.7)) for c in rgb)
        blocked_rgb = tuple(max(0, int(c * 0.5)) for c in rgb)
        delayed_rgb = (255, 165, 0)  # Orange for all delayed trains
        
        status_colors["completed"][tid] = f"#{completed_rgb[0]:02x}{completed_rgb[1]:02x}{completed_rgb[2]:02x}"
        status_colors["blocked"][tid] = f"#{blocked_rgb[0]:02x}{blocked_rgb[1]:02x}{blocked_rgb[2]:02x}"
        status_colors["delayed"][tid] = "#FFA500"  # Orange for delayed
    
    # Collect data
    gantt_data = []
    reroute_events = []
    accident_log = []
    blocked_sections = {}

    # Build a DataFrame for each train's journey, with proper gap handling for blocked trains
    for tid in trains_list:
        train_state = state.get(tid, {})
        train_log = train_state.get("log", [])
        
        if not train_log:
            continue
            
        # Process the train's log to create continuous timeline with gaps for blocked periods
        current_slot = 0
        # Get max slot from log entries, skip if no valid entries
        log_slots = [entry[0] for entry in train_log if isinstance(entry, tuple) and len(entry) >= 3]
        if not log_slots:
            continue  # Skip this train if no valid log entries
        max_slot = max(log_slots)
        
        # Create timeline segments
        segments = []
        current_segment = None
        
        for slot in range(max_slot + 1):
            # Find log entries for this slot
            slot_entries = [entry for entry in train_log if isinstance(entry, tuple) and entry[0] == slot]
            
            # Check if train is blocked at this slot
            is_blocked = False
            is_moving = False
            current_node = None
            
            for entry in slot_entries:
                if len(entry) >= 4:
                    action = entry[3]
                    if action in ["blocked_by_accident", "wait_blocked_section", "wait_blocked_new"]:
                        is_blocked = True
                    elif action in ["move", "enter", "switch", "accident_resolved", "resume", "completed"]:
                        is_moving = True
                        current_node = entry[2] if len(entry) > 2 else None
            
            # Check if train is currently blocked by accident
            if train_state.get("status") == "blocked_by_accident":
                blocked_until = train_state.get("accident_blocked_until", slot)
                if slot < blocked_until:
                    is_blocked = True
            
            # Handle segment transitions
            if is_blocked:
                # End current moving segment if exists
                if current_segment and current_segment["type"] == "moving":
                    current_segment["end"] = slot
                    segments.append(current_segment)
                    current_segment = None
                
                # Start or continue blocked segment
                if not current_segment or current_segment["type"] != "blocked":
                    if current_segment:
                        current_segment["end"] = slot
                        segments.append(current_segment)
                    current_segment = {
                        "type": "blocked",
                        "start": slot,
                        "end": slot + 1,
                        "node": current_node
                    }
                else:
                    current_segment["end"] = slot + 1
                    
            elif is_moving:
                # End current blocked segment if exists
                if current_segment and current_segment["type"] == "blocked":
                    current_segment["end"] = slot
                    segments.append(current_segment)
                    current_segment = None
                
                # Start or continue moving segment
                if not current_segment or current_segment["type"] != "moving":
                    if current_segment:
                        current_segment["end"] = slot
                        segments.append(current_segment)
                    current_segment = {
                        "type": "moving",
                        "start": slot,
                        "end": slot + 1,
                        "node": current_node
                    }
                else:
                    current_segment["end"] = slot + 1
                    current_segment["node"] = current_node
        
        # Close final segment
        if current_segment:
            segments.append(current_segment)
        
        # Convert segments to Gantt data
        for segment in segments:
            if segment["type"] == "moving" and segment["node"] is not None:
                if isinstance(segment["node"], tuple) and segment["node"][0] != "Platform":
                    section = segment["node"][1]
                    # Check if this is a completion segment
                    is_completion = False
                    for log_entry in train_log:
                        if (isinstance(log_entry, tuple) and len(log_entry) >= 4 and 
                            log_entry[0] == segment["end"] - 1 and log_entry[3] == "completed"):
                            is_completion = True
                            break
                    
                    gantt_data.append(dict(
                        Task=tid,
                        Start=segment["start"],
                        Finish=segment["end"],
                        Section=section,
                        Color=color_map[tid],
                        Delay=0,
                        Reroutes=0,
                        Accident=False,
                        Type=[t.type for t in trains if t.id == tid][0],
                        Status="completed" if is_completion else "moving"
                    ))
            elif segment["type"] == "blocked":
                # Add blocked segment with different color
                gantt_data.append(dict(
                    Task=tid,
                    Start=segment["start"],
                    Finish=segment["end"],
                    Section="BLOCKED",
                    Color="darkred",
                    Delay=segment["end"] - segment["start"],
                    Reroutes=0,
                    Accident=True,
                    Type=[t.type for t in trains if t.id == tid][0],
                    Status="blocked"
                ))

    if not gantt_data:
        fig = go.Figure()
        fig.update_layout(title="No Gantt data")
        return fig

    gantt_df = pd.DataFrame(gantt_data)
    fig = go.Figure()

    # Plot train bars with enhanced visual styling and sophisticated color coding
    for _, row in gantt_df.iterrows():
        train_id = row['Task']
        train_type = row['Type']
        status = row['Status']
        
        # Enhanced color and styling based on status and train type
        if status == 'blocked':
            # Blocked segments - darker version of train's unique color
            bar_color = status_colors["blocked"].get(train_id, "#8B0000")
            bar_opacity = 0.9
            marker_pattern = dict(
                color=bar_color,
                line=dict(width=3, color="#8B0000")
            )
            hover_text = (
                f"🚫 <b>{train_id}</b> ({train_type})<br>"
                f"Status: <span style='color:red'>BLOCKED BY ACCIDENT</span><br>"
                f"Duration: {row['Finish'] - row['Start']} slots<br>"
                f"Start: Slot {row['Start']}<br>"
                f"Finish: Slot {row['Finish']}<br>"
                f"Impact: High Priority Block"
            )
        elif status == 'completed':
            # Completed journey segments - darker version of train's unique color
            bar_color = status_colors["completed"].get(train_id, "#2E8B57")
            bar_opacity = 1.0
            marker_pattern = dict(
                color=bar_color,
                line=dict(width=3, color="#2E8B57")
            )
            hover_text = (
                f"✅ <b>{train_id}</b> ({train_type})<br>"
                f"Status: <span style='color:green'>JOURNEY COMPLETED</span><br>"
                f"Section: {row['Section']}<br>"
                f"Start: Slot {row['Start']}<br>"
                f"Finish: Slot {row['Finish']}<br>"
                f"Total Duration: {row['Finish'] - row['Start']} slots"
            )
        else:
            # Moving segments - use train's unique color
            base_color = color_map.get(train_id, row['Color'])
            delay_count = delays.get(train_id, 0)
            reroute_count = reroutes.get(train_id, 0)
            
            # Adjust opacity and color intensity based on delays
            if delay_count > 0:
                bar_opacity = 0.8
                marker_pattern = dict(
                    color=base_color,
                    line=dict(width=2, color="#FF8C00")
                )
            else:
                bar_opacity = 1.0
                marker_pattern = dict(
                    color=base_color,
                    line=dict(width=2, color="#2C3E50")
                )
            
            hover_text = (
                f"🚂 <b>{train_id}</b> ({train_type})<br>"
                f"Status: <span style='color:blue'>MOVING</span><br>"
                f"Section: {row['Section']}<br>"
                f"Start: Slot {row['Start']}<br>"
                f"Finish: Slot {row['Finish']}<br>"
                f"Delays: {delay_count} slots<br>"
                f"Reroutes: {reroute_count}"
            )
            
        fig.add_trace(go.Bar(
            x=[row['Finish'] - row['Start']],
            y=[row['Task']],
            base=row['Start'],
            orientation='h',
            marker=marker_pattern,
            opacity=bar_opacity,
            name=f"{train_id} ({train_type})",
            hovertemplate=hover_text + "<extra></extra>",
            showlegend=True
        ))
    # Add enhanced visual markers for special events
    for tid in gantt_df['Task'].unique():
        st = state[tid]
        train_type = [t.type for t in trains if t.id == tid][0]
        
        # Resume events - bright green star markers with enhanced visibility
        for log_entry in st.get('log', []):
            if (isinstance(log_entry, (tuple, list)) and len(log_entry) >= 4 and log_entry[3] == 'resume'):
                slot = log_entry[0]
                fig.add_trace(go.Scatter(
                    x=[slot], y=[tid],
                    mode="markers+text",
                    marker=dict(
                        symbol="star", 
                        size=30, 
                        color="#00FF7F", 
                        line=dict(width=4, color="#006400")
                    ),
                    text=["★ RESUME"],
                    textposition="middle right",
                    name="Resume Event",
                    showlegend=False,
                    hovertemplate=f"<b>{tid}</b> ({train_type})<br>🚀 RESUMED OPERATIONS at slot {slot}<br>Status: Back to normal operation<br><extra></extra>"
                ))
        
        # Reroute events - diamond markers
        for log_entry in st.get('log', []):
            if (isinstance(log_entry, (tuple, list)) and len(log_entry) >= 4 and log_entry[3] == 'runtime_plan'):
                slot = log_entry[0]
                fig.add_trace(go.Scatter(
                    x=[slot], y=[tid],
                    mode="markers+text",
                    marker=dict(
                        symbol="diamond", 
                        size=20, 
                        color="#FF8C00", 
                        line=dict(width=2, color="#FF4500")
                    ),
                    text=["↺ REROUTE"],
                    textposition="top center",
                    name="Reroute Event",
                    showlegend=False,
                    hovertemplate=f"<b>{tid}</b> ({train_type})<br>Rerouted at slot {slot}<br><extra></extra>"
                ))
        
        # Completion events - bright green checkmark markers
        for log_entry in st.get('log', []):
            if (isinstance(log_entry, (tuple, list)) and len(log_entry) >= 4 and log_entry[3] == 'completed'):
                slot = log_entry[0]
                fig.add_trace(go.Scatter(
                    x=[slot], y=[tid],
                    mode="markers+text",
                    marker=dict(
                        symbol="x", 
                        size=28, 
                        color="#32CD32", 
                        line=dict(width=4, color="#228B22")
                    ),
                    text=["✓ COMPLETE"],
                    textposition="bottom center",
                    name="Completion Event",
                    showlegend=False,
                    hovertemplate=f"<b>{tid}</b> ({train_type})<br>🎯 JOURNEY COMPLETED at slot {slot}<br>Status: Mission accomplished!<br><extra></extra>"
                ))
    # Calculate statistics for title and annotations
    completed_trains = len([t for t in trains_list if state.get(t, {}).get("status") == "completed"])
    blocked_trains = len([t for t in trains_list if state.get(t, {}).get("status") == "blocked_by_accident"])
    total_delays = sum(stats['delays'].values())
    total_reroutes = sum(stats['reroutes'].values())
    
    # Enhanced layout with better styling and organization
    fig.update_layout(
        title=dict(
            text=f"🚂 <b>RAILWAY JOURNEY GANTT CHART</b> 🚂<br>"
                 f"<span style='font-size:14px'>✅ Completed: {completed_trains}/{len(trains_list)} | "
                 f"🚫 Blocked: {blocked_trains} | ⏱️ Total Delays: {total_delays} slots | "
                 f"🔄 Reroutes: {total_reroutes}</span>",
            x=0.5,
            xanchor="center",
            font=dict(size=18, color="#2C3E50")
        ),
        height=700,  # Increased height for better visibility
        width=1600,  # Increased width for better spacing
        margin=dict(r=350, t=120, b=80, l=100),  # Better margins
        showlegend=True,
        legend=dict(
            orientation="v", 
            y=1, 
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#2C3E50",
            borderwidth=1,
            font=dict(size=12)
        ),
        font=dict(size=13, family="Arial, sans-serif"),
        plot_bgcolor="rgba(248,249,250,0.8)",
        paper_bgcolor="white",
        xaxis=dict(
            title=dict(text="⏰ Time (slots)", font=dict(size=14, color="#2C3E50")),
            gridcolor="rgba(128,128,128,0.2)",
            showgrid=True,
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)"
        ),
        yaxis=dict(
            title=dict(text="🚂 Trains", font=dict(size=14, color="#2C3E50")),
            gridcolor="rgba(128,128,128,0.2)",
            showgrid=True,
            categoryorder="array",
            categoryarray=trains_list
        )
    )
    
    # Enhanced KPI display with better styling and more information
    completion_rate = (completed_trains/len(trains_list)*100) if trains_list else 0
    avg_delay = total_delays / len(trains_list) if trains_list else 0
    
    kpi_txt = (
        f"<b>📊 SYSTEM PERFORMANCE</b><br>"
        f"<span style='color:#27AE60'>✅ Completed:</span> {completed_trains}/{len(trains_list)}<br>"
        f"<span style='color:#E74C3C'>🚫 Blocked:</span> {blocked_trains}<br>"
        f"<span style='color:#F39C12'>⏱️ Total Delays:</span> {total_delays} slots<br>"
        f"<span style='color:#3498DB'>🔄 Reroutes:</span> {total_reroutes}<br>"
        f"<span style='color:#9B59B6'>📈 Completion Rate:</span> {completion_rate:.1f}%<br>"
        f"<span style='color:#E67E22'>⏰ Avg Delay:</span> {avg_delay:.1f} slots"
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.02, y=0.95,
        text=kpi_txt,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#2C3E50",
        borderwidth=2,
        font=dict(size=11, family="Arial, sans-serif"),
        align="left"
    )
    
    # Add current time indicator if available
    if current_slot is not None:
        fig.add_vline(
            x=current_slot,
            line_width=3,
            line_dash="solid",
            line_color="#E74C3C",
            annotation=dict(
                text=f"<b>Current Time: Slot {current_slot}</b>",
                textangle=-90,
                yanchor="bottom",
                font=dict(size=12, color="#E74C3C"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#E74C3C",
                borderwidth=1
            )
        )
    return fig
    for tid in trains_list:
        sub = df[df["train"] == tid]
        start_slot = sub["slot"].min()
        end_slot = sub["slot"].max()
        if current_slot is not None and end_slot < current_slot:
            continue
            
        status = sub.iloc[-1]["action"] if not sub.empty else ""
        gantt_data.append({
            "Train": tid,
            "Start": start_slot,
            "Finish": end_slot,
            "Type": [t.type for t in trains if t.id == tid][0],
            "Color": color_map[tid],
            "Status": status,
            "Delay": delays[tid],
            "Reroutes": reroutes[tid]
        })
        
        # Collect reroute events
        reroutes_df = sub[sub["action"] == "runtime_plan"]
        for _, row in reroutes_df.iterrows():
            reroute_events.append({"Train": tid, "Slot": row["slot"]})

    if not gantt_data:
        fig = go.Figure()
        fig.update_layout(title="No Gantt data")
        return fig

    gantt_df = pd.DataFrame(gantt_data)
    fig = go.Figure()

    # Plot train bars with delay information
    for _, row in gantt_df.iterrows():
        bar_opacity = 1.0 if row["Delay"] == 0 else 0.7
        fig.add_trace(go.Bar(
            x=[row["Finish"] - row["Start"] + 1],
            y=[row["Train"]],
            base=row["Start"],
            orientation='h',
            marker_color=row["Color"],
            marker_line_width=2,
            marker_line_color="#222",
            opacity=bar_opacity,
            name=row["Train"],
            hovertemplate=(
                f"Train: {row['Train']}<br>"
                f"Type: {row['Type']}<br>"
                f"Start: {row['Start']}<br>"
                f"Finish: {row['Finish']}<br>"
                f"Delay: {row['Delay']} slots<br>"
                f"Reroutes: {row['Reroutes']}"
            )
        ))

        # Add delay indicator if delayed
        if row["Delay"] > 0:
            fig.add_trace(go.Scatter(
                x=[row["Finish"]], y=[row["Train"]],
                mode="markers+text",
                marker=dict(symbol="hourglass", size=20, color="red"),
                text=f"↔️ {row['Delay']} slots",
                textposition="middle right",
                name="Delay",
                showlegend=False
            ))

    # Process and visualize accidents
    if accident_mgr is not None:
        all_slots = set(df["slot"].unique())
        for slot in sorted(all_slots):
            actives = accident_mgr.active_summary(slot)
            for eid, evtype, loc, rem, stats in actives:
                if isinstance(loc, tuple):
                    # Add red background for accident duration
                    fig.add_vrect(
                        x0=slot-0.5,
                        x1=slot+rem+0.5,
                        fillcolor="rgba(255,0,0,0.1)",
                        line_width=0,
                        layer="below"
                    )
                    # Add accident label
                    fig.add_annotation(
                        x=slot+rem/2,
                        y=len(trains_list)-1,
                        text=f"🚨 {format_node(loc)} blocked",
                        showarrow=False,
                        font=dict(color="red"),
                        bgcolor="white"
                    )
                    # Update blocked sections
                    if isinstance(loc, tuple):
                        track = -1 if loc[0] == "Platform" else loc[0]
                        section = -0.5 if loc[0] == "Platform" else loc[1]
                        blocked_sections[(track, section)] = (slot, rem)
                        accident_log.append(
                            f"Slot {slot}: {format_node(loc)}<br>"
                            f"🚫 {evtype} ({rem} slots left)"
                        )

    # Add comprehensive status display
    status_txt = (
        f"📊 Journey Stats<br>"
        f"Total Delay: {total_delay} slots<br>"
        f"Delayed Trains: {len([d for d in delays.values() if d > 0])}<br>"
        f"Rerouted Trains: {len([r for r in reroutes.values() if r > 0])}<br>"
        f"Blocked Sections: {len(blocked_sections)}"
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.02, y=0.95,
        text=status_txt,
        showarrow=False,
        bgcolor="white",
        bordercolor="red" if total_delay > 0 else "black"
    )

    # Add accident log if exists
    if accident_log:
        log_text = "<br>".join(accident_log[-5:])
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.02, y=0.5,
            text=f"🚨 Active Events<br>{log_text}",
            showarrow=False,
            bgcolor="rgba(255,200,200,0.9)",
            bordercolor="red"
        )

    # Annotate reroute events
    for ev in reroute_events:
        fig.add_vline(
            x=ev["Slot"],
            line_width=2,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"Reroute {ev['Train']}",
            annotation_position="top right"
        )

    fig.update_layout(
        title="Train Journey Timeline",
        xaxis_title="Time (slot)",
        yaxis_title="Train",
        barmode='stack',
        height=400,
        width=1100,
        margin=dict(r=200, t=50),
        showlegend=True,
        legend=dict(orientation="h", y=-0.2)
    )
    return fig
def plot_train_timeline(state, trains, accident_mgr=None):
    """
    Plots each train's position (section/platform) as a function of time (slot).
    Y-axis: Section (Platform = -0.5, Sec1 = 0, ...)
    X-axis: Time (slot)
    Shows accidents and reroutes.
    """
    import pandas as pd
    import plotly.graph_objects as go
    df = build_records_from_state(state)
    if df.empty:
        fig = go.Figure(); fig.update_layout(title="No timeline data"); return fig
    trains_list = sorted(df["train"].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    
    # Create unique colors for each train - ensuring maximum visual distinction
    unique_train_colors = [
        "#FF6B6B",  # T1 - Vibrant Red
        "#4ECDC4",  # T2 - Teal
        "#45B7D1",  # T3 - Blue
        "#96CEB4",  # T4 - Mint Green
        "#FFEAA7",  # T5 - Yellow
        "#DDA0DD",  # T6 - Plum
        "#98D8C8",  # T7 - Seafoam
        "#F7DC6F",  # T8 - Gold
        "#BB8FCE",  # T9 - Light Purple
        "#85C1E9"   # T10 - Light Blue
    ]
    
    # Create color mapping - each train gets a unique color
    color_map = {}
    for i, tid in enumerate(trains_list):
        color_map[tid] = unique_train_colors[i % len(unique_train_colors)]
    fig = go.Figure()
    for i, tid in enumerate(trains_list):
        sub = df[df["train"] == tid]
        xs, ys, hover, marker_syms, marker_cols, marker_sizes, texts = [], [], [], [], [], [], []
        for _, row in sub.iterrows():
            slot = row["slot"]
            node = row["node"]
            action = row["action"]
            if node is None:
                y = None
            elif isinstance(node, tuple) and node[0] == "Platform":
                y = -0.5
            elif isinstance(node, tuple):
                y = node[1]
            else:
                y = None
            xs.append(slot)
            ys.append(y)
            hover.append(f"{tid}<br>slot={slot}<br>pos={node}")
            # Highlight at platform
            if isinstance(node, tuple) and node[0] == "Platform":
                marker_syms.append("star")
                marker_cols.append("gold")
                marker_sizes.append(18)
                texts.append("At Platform")
            elif action == "runtime_plan":
                marker_syms.append("diamond")
                marker_cols.append("orange")
                marker_sizes.append(16)
                texts.append("Rerouted!")
            elif action == "involved_in_accident":
                marker_syms.append("square")
                marker_cols.append("darkred")
                marker_sizes.append(18)
                texts.append("🚫 BLOCKED")
            elif action == "accident_resolved":
                marker_syms.append("star")
                marker_cols.append("#00FF7F")
                marker_sizes.append(25)
                texts.append("★ RESUME")
            elif action == "completed":
                marker_syms.append("x")
                marker_cols.append("#32CD32")
                marker_sizes.append(28)
                texts.append("✓ COMPLETED")
            else:
                marker_syms.append("circle")
                marker_cols.append(color_map[tid])
                marker_sizes.append(10)
                texts.append("")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers+text", line=dict(color=color_map[tid], width=3),
            marker=dict(size=marker_sizes, color=marker_cols, symbol=marker_syms, line=dict(width=2, color="#222")),
            text=texts, textposition="top center",
            name=f"{tid}", hoverinfo="text", hovertext=hover, showlegend=True
        ))
    # Accidents as red X markers
    if accident_mgr is not None:
        all_slots = set(df["slot"].unique())
        for slot in sorted(all_slots):
            actives = accident_mgr.active_summary(slot)
            for eid, evtype, loc, rem, stats in actives:
                if isinstance(loc, tuple) and loc[0] == "Platform":
                    y = -0.5
                elif isinstance(loc, tuple):
                    y = loc[1]
                else:
                    continue
                severity = stats.get("severity", "normal")
                fig.add_trace(go.Scatter(
                    x=[slot], y=[y], mode="markers+text",
                    marker=dict(symbol="x", size=22, color="red", line=dict(width=3, color="black")),
                    text=[f"🚨 {evtype}"], textposition="top center",
                    name="Accident", showlegend=False
                ))
    # Calculate delays and initialize tracking
    stats = calculate_delays(df)
    delays = stats['delays']
    reroutes = stats['reroutes']
    accident_log = []
    blocked_sections = {}
    
    # Process accidents and collect data
    if accident_mgr is not None:
        for slot in sorted(all_slots):
            actives = accident_mgr.active_summary(slot)
            for eid, evtype, loc, rem, stats in actives:
                if isinstance(loc, tuple):
                    track = -1 if loc[0] == "Platform" else loc[0]
                    section = -0.5 if loc[0] == "Platform" else loc[1]
                    blocked_sections[(track, section)] = (slot, rem)
                    affected = stats.get("affected_trains", 0)
                    rerouted = stats.get("rerouted_trains", 0)
                    accident_log.append(
                        f"Slot {slot}: {format_node(loc)} - {evtype}<br>"
                        f"⏰ {rem} slots remaining<br>"
                        f"🚂 {affected} trains affected<br>"
                        f"🔄 {rerouted} trains rerouted"
                    )

    # Enhanced train position plotting
    for tid in trains_list:
        sub = df[df["train"] == tid]
        xs, ys, hover, marker_syms, marker_cols, marker_sizes, texts = [], [], [], [], [], [], []
        for _, row in sub.iterrows():
            slot = row["slot"]
            node = row["node"]
            action = row["action"]
            if node is None:
                y = None
            elif isinstance(node, tuple) and node[0] == "Platform":
                y = -0.5
            elif isinstance(node, tuple):
                y = node[1]
            else:
                y = None
            xs.append(slot)
            ys.append(y)
            hover.append(f"{tid}<br>slot={slot}<br>pos={node}")
            # Highlight at platform
            if isinstance(node, tuple) and node[0] == "Platform":
                marker_syms.append("star")
                marker_cols.append("gold")
                marker_sizes.append(18)
                texts.append("At Platform")
            elif action == "runtime_plan":
                marker_syms.append("diamond")
                marker_cols.append("orange")
                marker_sizes.append(16)
                texts.append("Rerouted!")
            else:
                marker_syms.append("circle")
                marker_cols.append(color_map[tid])
                marker_sizes.append(10)
                texts.append("")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers+text", line=dict(color=color_map[tid], width=3),
            marker=dict(size=marker_sizes, color=marker_cols, symbol=marker_syms, line=dict(width=2, color="#222")),
            text=texts, textposition="top center",
            name=f"{tid}", hoverinfo="text", hovertext=hover, showlegend=True
        ))

        # Add delay/reroute indicators
        if delays[tid] > 0:
            texts.append(f"⌛ +{delays[tid]}")
        if reroutes[tid] > 0:
            texts.append(f"↗️ x{reroutes[tid]}")

    # Add comprehensive accident log
    if accident_log:
        log_text = "<br>".join(accident_log[-5:])  # Show last 5 events
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.02, y=0.95,
            text=f"<b>🚨 Active Accidents & Impact</b><br>{log_text}",
            showarrow=False,
            bgcolor="rgba(255,200,200,0.9)",
            bordercolor="red",
            borderwidth=2
        )

    # Add KPI annotation
    kpi_txt = (
        f"📊 KPIs<br>Avg wait (s): {stats.get('avg_wait_s', 0):.1f}<br>"
        f"Throughput: {stats.get('throughput', 0)}/{len(trains_list)}"
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.02, y=0.5,
        text=kpi_txt,
        showarrow=False,
        bgcolor="white",
        bordercolor="black"
    )

    # Calculate train statistics
    completed_trains = len([t for t in trains_list if state.get(t, {}).get("status") == "completed"])
    blocked_trains = len([t for t in trains_list if state.get(t, {}).get("status") == "blocked_by_accident"])
    
    fig.update_layout(
        title=dict(
            text=f"🚂 TRAIN POSITION TIMELINE | ✅ Completed: {completed_trains} | 🚫 Blocked: {blocked_trains}",
            x=0.5,
            xanchor="center"
        ),
        xaxis_title="⏰ Time (slots)",
        yaxis=dict(
            tickmode="array",
            tickvals=[-0.5,0,1,2,3],
            ticktext=["🏁 Platform", "📍 Section 1", "📍 Section 2", "📍 Section 3", "📍 Section 4"],
            range=[-1, 3.5],
            title="📍 Track Sections"
        ),
        height=500, width=1200, margin=dict(r=200), 
        legend=dict(orientation="h", y=-0.2),
        font=dict(size=12)
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(dtick=1)
    return fig
def plot_section_vs_track(state, trains, accident_mgr=None):
    """
    Plots section vs track with enhanced accident and delay visualization.
    """
    import pandas as pd
    import plotly.graph_objects as go
    df = build_records_from_state(state)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No section vs track data")
        return fig

    # Setup
    trains_list = sorted(df["train"].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    palette = [
        "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231",
        "#911EB4", "#46F0F0", "#F032E6", "#BCF60C", "#17BECF"
    ]
    color_map = {tid: palette[i % len(palette)] for i, tid in enumerate(trains_list)}
    
    # Initialize tracking variables
    blocked_sections = {}
    accident_points = []
    accident_log = []
    all_slots = set(df["slot"].unique())
    
    # Calculate delays
    stats = calculate_delays(df)
    delays = stats['delays']
    reroutes = stats['reroutes']
    total_delay = sum(delays.values())
    
    fig = go.Figure()

    # Process accidents and blocked sections
    if accident_mgr is not None:
        for slot in sorted(all_slots):
            actives = accident_mgr.active_summary(slot)
            for eid, evtype, loc, rem in actives:
                if isinstance(loc, tuple):
                    # Process accident location
                    track = -1 if loc[0] == "Platform" else loc[0]
                    section = -0.5 if loc[0] == "Platform" else loc[1]
                    
                    # Track blocked sections
                    blocked_sections[(track, section)] = (slot, rem)
                    
                    # Add to accident points for markers
                    accident_points.append((section, track, evtype, rem, loc[0] == "Platform"))
                    
                    # Add to accident log
                    affected_trains = [tid for tid in trains_list if delays[tid] > 0]
                    accident_log.append(
                        f"Slot {slot}: {format_node(loc)}<br>"
                        f"🚫 {evtype} ({rem} slots left)<br>"
                        f"Affected: {', '.join(affected_trains) if affected_trains else 'None'}"
                    )

    # Add blocked section visualization
    for (track, section), (slot, rem) in blocked_sections.items():
        # Solid red rectangle
        fig.add_shape(
            type="rect",
            x0=section-0.5 if section != -0.5 else -1,
            x1=section+0.5 if section != -0.5 else 0,
            y0=track-0.5,
            y1=track+0.5,
            fillcolor="rgba(255,0,0,0.3)",
            line=dict(color="red", width=2),
            layer="below"
        )
        # "BLOCKED" text
        fig.add_annotation(
            x=section,
            y=track,
            text=f"BLOCKED<br>({rem} slots left)",
            showarrow=False,
            font=dict(color="red", size=12),
            bgcolor="white"
        )

    # Plot trains with delay/reroute indicators
    for tid in trains_list:
        sub = df[df["train"] == tid]
        xs, ys, marker_syms, marker_cols, marker_sizes, texts = [], [], [], [], [], []
        
        for _, row in sub.iterrows():
            node = row["node"]
            action = row["action"]
            
            if node is None:
                continue
            
            if isinstance(node, tuple) and node[0] == "Platform":
                x = -0.5
                y = -1
            elif isinstance(node, tuple):
                x = node[1]
                y = node[0]
            else:
                continue
                
            xs.append(x)
            ys.append(y)
            
            if action == "runtime_plan":
                marker_syms.append("diamond-cross")
                marker_cols.append("orange")
                marker_sizes.append(20)
                texts.append(f"Rerouted! (+{reroutes[tid]})")
            elif action == "wait_accident":
                marker_syms.append("hourglass")
                marker_cols.append("#d62728")
                marker_sizes.append(18)
                texts.append(f"Wait (+{delays[tid]})")
            else:
                marker_syms.append("circle")
                marker_cols.append(color_map[tid])
                marker_sizes.append(14)
                texts.append(tid)
        
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(
                size=marker_sizes,
                color=marker_cols,
                symbol=marker_syms,
                line=dict(width=2, color="#222")
            ),
            text=texts,
            textposition="top center",
            name=f"{tid}",
            showlegend=True
        ))

    # Add accidents as red X/star markers
    if accident_points:
        fig.add_trace(go.Scatter(
            x=[pt[0] for pt in accident_points],
            y=[pt[1] for pt in accident_points],
            mode="markers+text",
            marker=dict(
                symbol=["star" if pt[4] else "x" for pt in accident_points],
                size=26,
                color="red",
                line=dict(width=3, color="black")
            ),
            text=[f"🚨 {pt[2]}<br>({pt[3]} left)" for pt in accident_points],
            textposition="top center",
            name="Accident",
            showlegend=True
        ))

    # Add accident log
    if accident_log:
        log_text = "<br>".join(accident_log[-5:])  # Show last 5 events
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1.02,
            y=0.95,
            text=f"<b>🚨 Active Accidents & Impact</b><br>{log_text}",
            showarrow=False,
            bgcolor="rgba(255,200,200,0.9)",
            bordercolor="red",
            borderwidth=2
        )

    # Add comprehensive status display
    status_txt = (
        f"📊 Impact Stats<br>"
        f"Total Delay: {total_delay} slots<br>"
        f"Delayed Trains: {len([d for d in delays.values() if d > 0])}<br>"
        f"Rerouted Trains: {len([r for r in reroutes.values() if r > 0])}<br>"
        f"Blocked Sections: {len(blocked_sections)}"
    )
    
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.02,
        y=0.5,
        text=status_txt,
        showarrow=False,
        bgcolor="white",
        bordercolor="red" if total_delay > 0 else "black"
    )

    fig.update_layout(
        title="Section vs Track (Accident Visibility)",
        xaxis=dict(
            tickmode="array",
            tickvals=[-0.5,0,1,2,3],
            ticktext=["Platform", "Sec1", "Sec2", "Sec3", "Sec4"],
            range=[-1, 3.5],
            title="Section"
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[-1,0,1,2,3,4],
            ticktext=["Platform", "Track0", "Track1", "Track2", "Track3", "Track4"],
            range=[-1.5, 4.5],
            title="Track"
        ),
        height=500,
        width=900,
        margin=dict(r=200),
        showlegend=True,
        legend=dict(orientation="h", y=-0.2)
    )
    
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(dtick=1)
    return fig
