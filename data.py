# data.py
from collections import namedtuple
import networkx as nx

# Train tuple
Train = namedtuple("Train", ["id", "type", "priority", "start", "goal", "sched_arrival", "dwell"])

PRIORITY_MAP = {"Express": 0, "Passenger": 1, "Freight": 2}  # lower number = higher priority

def build_graph(num_tracks=5, sections_per_track=4, num_stations=1, platforms_per_station=1, platform_access_map=None):
    """
    Build directed graph of tracks & sections with enhanced platform access control.
    """
    G = nx.DiGraph()

    # nodes
    for tr in range(num_tracks):
        for sec in range(sections_per_track):
            G.add_node((tr, sec))

    # Build platform nodes
    platforms = []
    for st in range(num_stations):
        for pf in range(platforms_per_station):
            pnode = ("Platform", st, pf)
            G.add_node(pnode)
            platforms.append(pnode)

    # forward edges (same track)
    for tr in range(num_tracks):
        for sec in range(sections_per_track - 1):
            G.add_edge((tr, sec), (tr, sec + 1), travel=1.0)

    # lateral switches: adjacent tracks same section (faster than section travel)
    for tr in range(num_tracks - 1):
        for sec in range(sections_per_track):
            G.add_edge((tr, sec), (tr + 1, sec), travel=0.5)
            G.add_edge((tr + 1, sec), (tr, sec), travel=0.5)

    # connect last sections to station platforms with access constraints
    # platform_access_map: dict[int track] -> list[(station_id, platform_id)]
    for tr in range(num_tracks):
        last = (tr, sections_per_track - 1)
        if platform_access_map and tr in platform_access_map:
            for (st, pf) in platform_access_map[tr]:
                if 0 <= st < num_stations and 0 <= pf < platforms_per_station:
                    G.add_edge(last, ("Platform", st, pf), travel=0.5)
        else:
            # More realistic fallback: not all tracks connect to all platforms
            st = tr % max(1, num_stations)
            pf = tr % max(1, platforms_per_station)
            G.add_edge(last, ("Platform", st, pf), travel=0.5)

    return G, platforms

def generate_fixed_trains(sections_per_track=4):
    """
    Returns a list of 10 deterministic Train objects (fixed dataset)
    sched_arrival is in minutes (slot index since time_step_s=60s)
    NOTE: Goals remain as last section nodes; the simulator routes to any platform.
    """
    fixed = [
        ("T1", "Express",  (0, 0), (2, sections_per_track - 1), 0, 4),
        ("T2", "Passenger",(1, 0), (0, sections_per_track - 1), 1, 3),
        ("T3", "Freight", (4, 0), (3, sections_per_track - 1), 2, 6),
        ("T4", "Passenger",(2, 0), (1, sections_per_track - 1), 2, 3),
        ("T5", "Express", (3, 0), (4, sections_per_track - 1), 3, 4),
        ("T6", "Passenger",(0, 0), (1, sections_per_track - 1), 4, 3),
        ("T7", "Freight", (2, 0), (2, sections_per_track - 1), 5, 6),
        ("T8", "Passenger",(4, 0), (0, sections_per_track - 1), 6, 3),
        ("T9", "Express", (1, 0), (3, sections_per_track - 1), 7, 4),
        ("T10","Passenger",(3, 0), (4, sections_per_track - 1), 8, 3),
    ]
    trains = []
    for tid, ttype, start, goal, arr, dwell in fixed:
        trains.append(Train(tid, ttype, PRIORITY_MAP[ttype], start, goal, arr, dwell))
    return trains
