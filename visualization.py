# visualization.py
import pandas as pd
import plotly.graph_objects as go
from utils import format_node, short_node

def build_records_from_state(state):
    recs = []
    for tid, st in state.items():
        for rec in st["log"]:
            slot, frm, to, action = rec
            node = to if to is not None else frm
            recs.append({
                "train": tid,
                "slot": slot,
                "time_min": slot,
                "from": frm,
                "to": to,
                "node": node,
                "action": action
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
    for i, tid in enumerate(trains_list):
        sub = df[df["train"] == tid]
        xs, ys, hover, dirs = [], [], [], []
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
            marker=dict(size=10, color=color_map[tid]), text=dirs, textposition="top center",
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
                    marker=dict(symbol="diamond", size=14, color=color_map[tid], line=dict(color="black", width=1)),
                    text=[f"{short_node(a)}→{short_node(b)}"], textposition="bottom center",
                    hoverinfo="text", hovertext=[f"Switch: {short_node(a)} → {short_node(b)}"], name="Switch", showlegend=False
                ))
    # annotate accidents
    actives = accident_mgr.active_summary(sim_slot)
    if actives:
        txt = "🚨 Active events:<br>" + "<br>".join([f"{eid} {evtype} @{format_node(loc)} ({rem} slots left)" for eid, evtype, loc, rem in actives])
        fig.add_annotation(xref="paper", yref="paper", x=1.02, y=0.95, text=txt, showarrow=False, bgcolor="rgba(255,200,200,0.9)")
    if kpis:
        kpi_txt = f"📊 KPIs<br>Avg wait (s): {kpis['avg_wait_s']:.1f}<br>Throughput: {kpis['throughput']}/{len(trains)}"
        fig.add_annotation(xref="paper", yref="paper", x=1.02, y=0.75, text=kpi_txt, showarrow=False, bgcolor="white", bordercolor="black")
    fig.update_layout(title=f"Train Timeline (slot ≈ {sim_slot})", xaxis_title="Section (Platform = -0.5)", yaxis=dict(tickmode="array", tickvals=list(range(len(trains_list))), ticktext=trains_list), height=700, width=1100, margin=dict(r=300), legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(dtick=1)
    return fig
