# data.py
from collections import namedtuple
import networkx as nx

# Train tuple
Train = namedtuple("Train", ["id", "type", "priority", "start", "goal", "sched_arrival", "dwell"])

PRIORITY_MAP = {"Express": 0, "Passenger": 1, "Freight": 2}  # lower number = higher priority

def build_graph(num_tracks=5, sections_per_track=4):
    """
    Build directed graph of tracks & sections.
    Node format: (track_index, section_index) zero-indexed
    Special platform node: ("Platform", 0)
    Edges have 'travel' in minutes.
    """
    G = nx.DiGraph()
    PLATFORM = ("Platform", 0)
    # nodes
    for tr in range(num_tracks):
        for sec in range(sections_per_track):
            G.add_node((tr, sec))
    G.add_node(PLATFORM)
    # forward edges (same track)
    for tr in range(num_tracks):
        for sec in range(sections_per_track - 1):
            # travel=1.0 means 1 minute between sections on same track
            G.add_edge((tr, sec), (tr, sec + 1), travel=1.0)
    # lateral switches: adjacent tracks same section (faster than section travel)
    for tr in range(num_tracks - 1):
        for sec in range(sections_per_track):
            G.add_edge((tr, sec), (tr + 1, sec), travel=0.5)
            G.add_edge((tr + 1, sec), (tr, sec), travel=0.5)
    # connect last sections to platform
    for tr in range(num_tracks):
        last = (tr, sections_per_track - 1)
        G.add_edge(last, PLATFORM, travel=0.5)
    return G, PLATFORM

def generate_fixed_trains(sections_per_track=4):
    """
    Returns a list of 10 deterministic Train objects (fixed dataset)
    sched_arrival is in minutes (slot index since time_step_s=60s)
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
