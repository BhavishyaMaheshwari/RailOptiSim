# main.py
from data import build_graph, generate_fixed_trains
from accident_manager import EmergencyEvent, AccidentManager
from simulation import Simulator
from visualization import plot_interactive
from utils import format_node

def main():
    # build infra and trains
    G, PLATFORM = build_graph(num_tracks=5, sections_per_track=4)
    trains = generate_fixed_trains(sections_per_track=4)

    # accident manager: schedule a demo accident
    acc = AccidentManager()
    # schedule accident at Track 2 Section 3 starting at slot 4 lasting 6 slots
    ev = EmergencyEvent(event_id="EV_demo_1", ev_type="accident", location=(1,2), start_time=4, duration_slots=6)
    acc.schedule(ev)

    # create simulator
    sim = Simulator(graph=G, platform_node=PLATFORM, trains=trains, accident_mgr=acc,
                    time_step_s=60, horizon_minutes=30)

    final_state, kpis = sim.run(max_slots=60)  # run up to 60 slots (60 minutes)
    print("KPIs:", kpis)

    # visualize final logs (slot index used as time)
    plot_interactive(final_state, trains, acc, G, PLATFORM, sim.current_slot, kpis=kpis, export_csv="event_log.csv")

if __name__ == "__main__":
    main()
