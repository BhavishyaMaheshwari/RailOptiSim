# simulation.py
import heapq
import math
from collections import defaultdict
from copy import deepcopy
import numpy as np

from utils import format_node
from accident_manager import EmergencyEvent, AccidentManager

# Defaults & tunables
TIME_STEP_S = 60           # seconds per slot (1 minute)
MIN_HEADWAY_S = 60         # minimum headway (1 minute)
CONGESTION_ALPHA = 30.0    # seconds penalty per queued train (soft)
REROUTE_PENALTY_S = 120.0  # penalty for reroute (soft)
PLATFORM_CAPACITY = 1
DWELL_DEFAULT_S = 60.0
SAFE_BRAKE_MARGIN_M = 100.0

def secs_to_slots(secs):
    return int(math.ceil(secs / TIME_STEP_S))

def edge_travel_time_seconds(G, u, v):
    if G.has_edge(u, v) and "travel" in G[u][v]:
        return float(G[u][v]["travel"]) * 60.0
    return 60.0

class NodeReservationTable:
    def __init__(self, nodes, horizon, capacities=None):
        self.res = {node: {} for node in nodes}
        self.horizon = horizon
        self.capacities = capacities if capacities else {}

    def is_free(self, node, slot):
        return len(self.res[node].get(slot, [])) < self.capacities.get(node, 1)

    def reserve(self, node, slot, train_id):
        if slot < 0:
            return False
        if len(self.res[node].get(slot, [])) < self.capacities.get(node, 1):
            self.res[node].setdefault(slot, []).append(train_id)
            return True
        return False

    def occupancy_count(self, node, slot):
        return len(self.res[node].get(slot, []))

    def get_reserved_trains(self, node, slot):
        return list(self.res[node].get(slot, []))

    def clear_old(self, current_slot):
        for node in self.res:
            for s in list(self.res[node].keys()):
                if s < current_slot - self.horizon:
                    del self.res[node][s]

def dijkstra_dynamic(G, source, target, start_slot, res_table, blocked_pairs):
    import heapq

    # priority queue: (cost_so_far, counter, node, time_slot, parent)
    pq = []
    counter = 0
    heapq.heappush(pq, (0, counter, source, start_slot, None))

    dist = {(source, start_slot): 0}
    parent = {}

    def is_goal(node):
        # target can be a single node or a collection
        if target is None:
            return False
        if isinstance(target, (set, list, tuple)):
            return node in set(target)
        return node == target

    while pq:
        g, _, u, slot, p = heapq.heappop(pq)

        if is_goal(u):
            # reconstruct path
            path = []
            cur = (u, slot)
            while cur in parent:
                path.append(cur[0])
                cur = parent[cur]
            path.append(source)
            path.reverse()
            # Also reconstruct slots
            slots = []
            cur = (u, slot)
            while cur in parent:
                slots.append(cur[1])
                cur = parent[cur]
            slots.append(start_slot)
            slots.reverse()
            return path, slots

        for v in G.successors(u):
            edge = (u, v)
            travel_time = secs_to_slots(G[u][v].get("travel", 1.0) * 60.0)
            arrival_slot = slot + travel_time

            # Skip if resource conflict or blocked pair
            if (u, v) in blocked_pairs:
                continue
            # Prevent planning through blocked nodes/slots
            if (v, arrival_slot) in blocked_pairs:
                continue
            # For node-based reservation, check if node v is free at arrival_slot
            if not res_table.is_free(v, arrival_slot):
                continue

            newg = g + travel_time
            state = (v, arrival_slot)

            if state not in dist or newg < dist[state]:
                dist[state] = newg
                parent[state] = (u, slot)

                counter += 1
                heapq.heappush(pq, (newg, counter, v, arrival_slot, u))

    return None, None  # no path found


class Simulator:
    def __init__(self, graph, platform_nodes, trains, accident_mgr: AccidentManager,
                 horizon_minutes=60, platform_capacity=PLATFORM_CAPACITY, **kwargs):
        self.G = graph
        # Multiple platforms supported
        self.platform_nodes = list(platform_nodes) if isinstance(platform_nodes, (list, tuple, set)) else [platform_nodes]
        self.trains = deepcopy(trains)
        self.acc = accident_mgr
        self.current_slot = 0
        self.horizon_slots = int(math.ceil(horizon_minutes))
        nodes = list(self.G.nodes())
        # Per-platform capacity map
        caps = {p: platform_capacity for p in self.platform_nodes}
        self.res_table = NodeReservationTable(nodes, self.horizon_slots, capacities=caps)

        # state: per train
        self.state = {}
        for t in self.trains:
            self.state[t.id] = {
                "info": t,
                "pos": None,
                "slot": None,
                "status": "not_arrived",
                "planned_path": [],
                "planned_slots": [],
                "log": [],
                "waiting_s": 0.0,
                "switches": 0
            }
        self.usage = defaultdict(int)

    def is_platform_node(self, node):
        return isinstance(node, tuple) and len(node) >= 1 and node[0] == "Platform"

    def blocked_set(self):
        """Returns set of (node, slot) tuples that are blocked"""
        s = set()
        for delta in range(0, self.horizon_slots+1):
            slot = self.current_slot + delta
            nodes = self.acc.blocked_nodes(slot)
            for n in nodes:
                s.add((n, slot))
        return s

    def try_reserve(self, train_id, path, slots):
        # check capacity and headway
        min_headway_slots = secs_to_slots(MIN_HEADWAY_S)
        to_commit = []
        blocked = self.blocked_set()
        for node, slot in zip(path, slots):
            if (node, slot) in blocked:
                return False
            if not self.res_table.is_free(node, slot):
                return False
            # check headway neighborhood
            for s in range(max(0, slot - min_headway_slots), slot + min_headway_slots + 1):
                others = [tid for tid in self.res_table.get_reserved_trains(node, s) if tid != train_id]
                if others:
                    return False
            to_commit.append((node, slot))
        for node, slot in to_commit:
            self.res_table.reserve(node, slot, train_id)
        return True

    def _targets_for_train(self, train_info):
        # If goal is a platform node, go there; otherwise route to any platform
        goal = train_info.goal
        if self.is_platform_node(goal):
            return {goal}
        return set(self.platform_nodes)

    def plan_initial(self):
        for t in self.trains:
            st = self.state[t.id]
            start_slot = max(self.current_slot, int(t.sched_arrival))
            blocked_pairs = self.blocked_set()
            targets = self._targets_for_train(t)
            path, slots = dijkstra_dynamic(self.G, t.start, targets, start_slot, self.res_table, blocked_pairs)
            if path and slots:
                ok = self.try_reserve(t.id, path, slots)
                if ok:
                    st["planned_path"] = path
                    st["planned_slots"] = slots
                    st["log"].append((start_slot, None, path[0], "planned_start"))
                else:
                    st["planned_path"] = path
                    st["planned_slots"] = slots

    def attempt_runtime_plan(self, tid):
        st = self.state[tid]
        info = st["info"]
        start_node = st["pos"] if st["pos"] else info.start
        start_slot = max(self.current_slot, st["slot"] if st["slot"] is not None else self.current_slot)
        targets = self._targets_for_train(info)
        path, slots = dijkstra_dynamic(self.G, start_node, targets, start_slot, self.res_table, self.blocked_set())
        if path and slots and self.try_reserve(tid, path, slots):
            st["planned_path"] = path
            st["planned_slots"] = slots
            st["log"].append((self.current_slot, None, None, "runtime_plan"))
            return True
        return False

    def step_slot(self):
        cur = self.current_slot
        blocked = self.blocked_set()
        for t in self.trains:
            st = self.state[t.id]
            info = st["info"]
            was_blocked = st.get("was_blocked", False)
            if st["status"] == "not_arrived":
                if cur >= int(info.sched_arrival):
                    if st["planned_slots"] and st["planned_slots"][0] == cur:
                        node = st["planned_path"][0]
                        if (node, cur) in blocked:
                            st["waiting_s"] += TIME_STEP_S
                            st["log"].append((cur, None, node, "wait_blocked_entry"))
                            st["was_blocked"] = True
                        else:
                            st["pos"] = node
                            st["slot"] = cur
                            st["status"] = "running"
                            st["log"].append((cur, None, st["pos"], "enter"))
                            if st.get("was_blocked", False):
                                st["log"].append((cur, None, st["pos"], "resume"))
                                st["was_blocked"] = False
            elif st["status"] == "running":
                if not st["planned_slots"]:
                    if not self.attempt_runtime_plan(t.id):
                        st["waiting_s"] += TIME_STEP_S
                        st["log"].append((cur, st["pos"], st["pos"], "wait_noplan"))
                    continue
                try:
                    idx = st["planned_slots"].index(st["slot"])
                except ValueError:
                    idx = None
                    for ii, s in enumerate(st["planned_slots"]):
                        if s >= cur:
                            idx = ii; break
                    if idx is None:
                        if not self.attempt_runtime_plan(t.id):
                            st["waiting_s"] += TIME_STEP_S
                            st["log"].append((cur, st["pos"], st["pos"], "wait_outsync"))
                        continue
                if idx + 1 < len(st["planned_slots"]):
                    next_slot = st["planned_slots"][idx+1]
                    next_node = st["planned_path"][idx+1]
                    # Enforce: if a section is blocked at next_slot, train must wait
                    if (next_node, next_slot) in blocked:
                        st["waiting_s"] += TIME_STEP_S
                        st["log"].append((cur, st["pos"], st["pos"], "wait_blocked_section"))
                        st["was_blocked"] = True
                        continue
                    reserved = self.res_table.get_reserved_trains(next_node, next_slot)
                    if reserved and st["info"].id not in reserved:
                        st["waiting_s"] += TIME_STEP_S
                        st["log"].append((cur, st["pos"], st["pos"], "wait_conflict"))
                        continue
                    prev = st["pos"]
                    st["pos"] = next_node
                    st["slot"] = next_slot
                    st["log"].append((next_slot, prev, next_node, "move"))
                    self.usage[next_node] += 1
                    if prev is not None and isinstance(prev, tuple) and isinstance(next_node, tuple) and prev[0] != next_node[0]:
                        st["switches"] += 1
                        st["log"].append((next_slot, prev, next_node, "switch"))
                    if st.get("was_blocked", False):
                        st["log"].append((next_slot, prev, next_node, "resume"))
                        st["was_blocked"] = False
                    # Any platform node completes the trip after dwell
                    if self.is_platform_node(next_node):
                        st["status"] = "at_platform"
                        dwell_slots = secs_to_slots(getattr(info, "dwell", DWELL_DEFAULT_S))
                        st["platform_end_slot"] = next_slot + dwell_slots
                        st["log"].append((next_slot, next_node, None, f"platform_until_{st['platform_end_slot']}"))
                else:
                    if self.is_platform_node(st["pos"]):
                        st["status"] = "completed"
                        st["log"].append((cur, st["pos"], None, "completed"))
                    else:
                        if not self.attempt_runtime_plan(t.id):
                            st["waiting_s"] += TIME_STEP_S
                            st["log"].append((cur, st["pos"], st["pos"], "wait_no_next"))
            elif st["status"] == "at_platform":
                if st.get("platform_end_slot", 0) <= cur:
                    st["status"] = "completed"
                    st["log"].append((cur, st["pos"], None, "depart"))
            elif st["status"] == "blocked_by_accident":
                # Train is blocked due to accident - check if accident duration has ended
                if st.get("accident_blocked_until", 0) <= cur:
                    st["status"] = "running"
                    st["log"].append((cur, st["pos"], st["pos"], "accident_resolved"))
                    st["log"].append((cur, st["pos"], st["pos"], "resume"))
                    if not self.attempt_runtime_plan(t.id):
                        st["waiting_s"] += TIME_STEP_S
                        st["log"].append((cur, st["pos"], st["pos"], "wait_noplan_after_accident"))
                else:
                    st["waiting_s"] += TIME_STEP_S
                    st["log"].append((cur, st["pos"], st["pos"], "blocked_by_accident"))
        self.current_slot += 1
        self.res_table.clear_old(self.current_slot)

    def run(self, max_slots=120):
        self.plan_initial()
        slots = 0
        while slots < max_slots:
            all_done = all(self.state[t.id]["status"] == "completed" for t in self.trains)
            if all_done:
                break
            self.step_slot()
            slots += 1
        return self.state, self.compute_kpis()

    def compute_kpis(self):
        waits = []
        completed = 0
        per_train = {}
        for t in self.trains:
            st = self.state[t.id]
            waits.append(st["waiting_s"])
            if st["status"] == "completed": completed += 1
            path = [rec[2] for rec in st["log"] if rec[2] is not None]
            condensed = []
            for n in path:
                if not condensed or condensed[-1] != n:
                    condensed.append(n)
            per_train[t.id] = {"waiting_s": st["waiting_s"], "switches": st["switches"], "status": st["status"], "path": condensed}
        avg_wait = float(np.mean(waits)) if waits else 0.0
        util = {node: cnt / max(1, self.current_slot) for node, cnt in self.usage.items()}
        return {"per_train": per_train, "avg_wait_s": avg_wait, "throughput": completed, "util": util}

    def handle_accident(self, node, duration):
        """Handle accident by blocking the involved train and rerouting others"""
        # Find all trains affected by this accident
        involved_train = None  # The train actually involved in the accident
        affected_trains = []   # Other trains that need rerouting
        active_events = self.acc.active_summary(self.current_slot)
        current_event = None
        
        # Find the relevant event
        for ev_id, evtype, loc, rem, stats in active_events:
            if loc == node:
                current_event = ev_id
                break
                
        if not current_event:
            return 0, 0
            
        # First, identify the train actually involved in the accident (currently at the location)
        for tid, st in self.state.items():
            if st["status"] in ["running", "at_platform"] and st["pos"] == node:
                involved_train = tid
                # Block this train for the duration of the accident
                st["status"] = "blocked_by_accident"
                st["accident_blocked_until"] = self.current_slot + duration
                st["log"].append({
                    "slot": self.current_slot,
                    "action": "involved_in_accident",
                    "node": node,
                    "duration": duration,
                    "event_id": current_event,
                    "blocked_until": self.current_slot + duration
                })
                self.acc.add_affected_train(current_event, tid, self.current_slot)
                self.acc.set_involved_train(current_event, tid)
                break
        
        # If no train is currently at the accident location, find the first train that would reach it
        if involved_train is None:
            earliest_arrival = float('inf')
            for tid, st in self.state.items():
                if st["status"] != "completed" and st["planned_path"]:
                    try:
                        # Find when this train would reach the accident location
                        for i, planned_node in enumerate(st["planned_path"]):
                            if planned_node == node and i < len(st["planned_slots"]):
                                arrival_slot = st["planned_slots"][i]
                                if arrival_slot < earliest_arrival:
                                    earliest_arrival = arrival_slot
                                    involved_train = tid
                                break
                    except (IndexError, ValueError):
                        continue
            
            # If we found a train that would reach the accident location, block it
            if involved_train is not None:
                st = self.state[involved_train]
                st["status"] = "blocked_by_accident"
                st["accident_blocked_until"] = self.current_slot + duration
                st["log"].append({
                    "slot": self.current_slot,
                    "action": "involved_in_accident",
                    "node": node,
                    "duration": duration,
                    "event_id": current_event,
                    "blocked_until": self.current_slot + duration
                })
                self.acc.add_affected_train(current_event, involved_train, self.current_slot)
                self.acc.set_involved_train(current_event, involved_train)
        
        # Now identify other trains that need rerouting (those that would pass through the blocked track)
        blocked_track = node[0] if isinstance(node, tuple) and node[0] != "Platform" else None
        
        for tid, st in self.state.items():
            if st["status"] != "completed" and tid != involved_train:
                path = st["planned_path"]
                if not path:
                    continue
                    
                # Check if train's path goes through the blocked track
                needs_reroute = False
                for planned_node in path:
                    if isinstance(planned_node, tuple) and planned_node[0] == blocked_track:
                        needs_reroute = True
                        break
                
                if needs_reroute:
                    affected_trains.append(tid)
                    self.acc.add_affected_train(current_event, tid, self.current_slot)
                    st["log"].append({
                        "slot": self.current_slot,
                        "action": "affected_by_accident",
                        "node": node,
                        "duration": duration,
                        "event_id": current_event,
                        "blocked_track": blocked_track
                    })
        
        # Try to reroute each affected train (excluding the involved train)
        rerouted = 0
        for tid in affected_trains:
            st = self.state[tid]
            old_path = st["planned_path"]
            old_arrival = st["planned_slots"][-1] if st["planned_slots"] else self.current_slot
            
            if self.attempt_runtime_plan(tid):
                rerouted += 1
                new_arrival = st["planned_slots"][-1]
                delay = max(0, new_arrival - old_arrival)
                
                # Log the reroute with stats
                self.acc.add_rerouted_train(current_event, tid, self.current_slot, delay)
                st["log"].append({
                    "slot": self.current_slot,
                    "action": "runtime_plan",
                    "success": True,
                    "old_path": old_path,
                    "delay": delay,
                    "event_id": current_event
                })
            else:
                st["log"].append({
                    "slot": self.current_slot,
                    "action": "runtime_plan",
                    "success": False,
                    "reason": "no_valid_path",
                    "event_id": current_event
                })
        
        return len(affected_trains) + (1 if involved_train else 0), rerouted
