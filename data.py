# data.py
from collections import namedtuple
import networkx as nx

# Train tuple (platform_required added)
Train = namedtuple("Train", ["id", "type", "priority", "start", "goal", "sched_arrival", "dwell", "platform_required"])

PRIORITY_MAP = {"Express": 0, "Passenger": 1, "Freight": 2}  # lower number = higher priority

def build_graph(num_tracks=5, sections_per_track=4, num_stations=1, platforms_per_station=1, platform_access_map=None, station_section_map=None):
    """
    Build directed graph of tracks & sections with enhanced platform access control.
    
    Args:
        num_tracks: number of parallel tracks
        sections_per_track: sections along each track
        num_stations: station count
        platforms_per_station: number of platforms per station
        platform_access_map: dict[int track] -> list[(station_id, platform_id)] mapping which platforms a track can access
        station_section_map: optional dict[int station_id] -> int section_index to attach platforms near that section
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

    # connect sections to station platforms with access constraints
    # platform_access_map: dict[int track] -> list[(station_id, platform_id)]
    for tr in range(num_tracks):
        # default attachment section is last section unless station_section_map overrides
        if platform_access_map and tr in platform_access_map:
            for (st, pf) in platform_access_map[tr]:
                if 0 <= st < num_stations and 0 <= pf < platforms_per_station:
                    sec = int(station_section_map.get(st, sections_per_track - 1)) if station_section_map else (sections_per_track - 1)
                    G.add_edge((tr, sec), ("Platform", st, pf), travel=0.5)
        else:
            # More realistic fallback: not all tracks connect to all platforms
            st = tr % max(1, num_stations)
            pf = tr % max(1, platforms_per_station)
            sec = int(station_section_map.get(st, sections_per_track - 1)) if station_section_map else (sections_per_track - 1)
            G.add_edge((tr, sec), ("Platform", st, pf), travel=0.5)

    return G, platforms

def generate_fixed_trains(sections_per_track=4, num_trains=32, num_tracks=8, platform_optional_ratio=0.25):
    """
    Generate a deterministic set of trains for demo.
    - num_trains: total trains to create (default 32)
    - num_tracks: to distribute starts/goals (default 8)
    - sections_per_track: last section index for goals

    Staggers schedules by 1 minute; types cycle through Express/Passenger/Freight.
    """
    types_cycle = ["Express", "Passenger", "Freight", "Passenger"]
    dwell_map = {"Express": 3, "Passenger": 4, "Freight": 6}
    trains = []
    import random
    random.seed(42)
    for i in range(int(num_trains)):
        tid = f"T{i+1}"
        ttype = types_cycle[i % len(types_cycle)]
        start_track = i % max(1, int(num_tracks))
        goal_track = (start_track + 3) % max(1, int(num_tracks))  # spread directions
        start = (start_track, 0)
        goal = (goal_track, max(0, sections_per_track - 1))
        sched_arrival = i  # stagger by 1 minute
        dwell = dwell_map[ttype]
        # Some trains do not require platform access (e.g., through-freight)
        platform_required = not (random.random() < platform_optional_ratio and ttype in ("Freight", "Express"))
        trains.append(Train(tid, ttype, PRIORITY_MAP[ttype], start, goal, sched_arrival, dwell, platform_required))
    return trains
