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

    while pq:
        g, _, u, slot, p = heapq.heappop(pq)

        if u == target:
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
    def __init__(self, graph, platform_node, trains, accident_mgr: AccidentManager,
                 horizon_minutes=60):
        self.G = graph
        self.platform = platform_node
        self.trains = deepcopy(trains)
        self.acc = accident_mgr
        self.current_slot = 0
        self.horizon_slots = int(math.ceil(horizon_minutes))
        nodes = list(self.G.nodes())
        caps = {self.platform: PLATFORM_CAPACITY}
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

    def blocked_set(self):
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

    def plan_initial(self):
        for t in self.trains:
            st = self.state[t.id]
            start_slot = max(self.current_slot, int(t.sched_arrival))
            blocked_pairs = self.blocked_set()
            path, slots = dijkstra_dynamic(self.G, t.start, t.goal, start_slot, self.res_table, blocked_pairs)
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
        path, slots = dijkstra_dynamic(self.G, start_node, info.goal, start_slot, self.res_table, self.blocked_set())
        if path and slots and self.try_reserve(tid, path, slots):
            st["planned_path"] = path
            st["planned_slots"] = slots
            st["log"].append((self.current_slot, None, None, "runtime_plan"))
            return True
        return False

    def step_slot(self):
        cur = self.current_slot
        blocked_nodes = self.acc.blocked_nodes(cur)
        for t in self.trains:
            st = self.state[t.id]
            info = st["info"]
            if st["status"] == "not_arrived":
                if cur >= int(info.sched_arrival):
                    if st["planned_slots"] and st["planned_slots"][0] == cur:
                        st["pos"] = st["planned_path"][0]
                        st["slot"] = cur
                        st["status"] = "running"
                        st["log"].append((cur, None, st["pos"], "enter"))
            elif st["status"] == "running":
                if not st["planned_slots"]:
                    if not self.attempt_runtime_plan(t.id):
                        st["waiting_s"] += TIME_STEP_S
                        st["log"].append((cur, st["pos"], st["pos"], "wait_noplan"))
                    continue
                # find current index
                try:
                    idx = st["planned_slots"].index(st["slot"])
                except ValueError:
                    # sync to next planned slot >= current
                    idx = None
                    for ii, s in enumerate(st["planned_slots"]):
                        if s >= cur:
                            idx = ii; break
                    if idx is None:
                        if not self.attempt_runtime_plan(t.id):
                            st["waiting_s"] += TIME_STEP_S
                            st["log"].append((cur, st["pos"], st["pos"], "wait_outsync"))
                        continue
                # attempt move to next
                if idx + 1 < len(st["planned_slots"]):
                    next_slot = st["planned_slots"][idx+1]
                    next_node = st["planned_path"][idx+1]
                    if next_node in blocked_nodes or (next_node, next_slot) in self.blocked_set():
                        st["waiting_s"] += TIME_STEP_S
                        st["log"].append((cur, st["pos"], st["pos"], "wait_blocked"))
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
                    # switch detection (track change)
                    if prev is not None and isinstance(prev, tuple) and isinstance(next_node, tuple) and prev[0] != next_node[0]:
                        st["switches"] += 1
                        st["log"].append((next_slot, prev, next_node, "switch"))
                    # platform handling
                    if next_node == self.platform:
                        st["status"] = "at_platform"
                        dwell_slots = secs_to_slots(getattr(info, "dwell", DWELL_DEFAULT_S))
                        st["platform_end_slot"] = next_slot + dwell_slots
                        st["log"].append((next_slot, next_node, None, f"platform_until_{st['platform_end_slot']}"))
                else:
                    if st["pos"] == info.goal:
                        st["status"] = "completed"
                        st["log"].append((cur, st["pos"], None, "completed"))
                    else:
                        if not self.attempt_runtime_plan(t.id):
                            st["waiting_s"] += TIME_STEP_S
                            st["log"].append((cur, st["pos"], st["pos"], "wait_no_next"))
            elif st["status"] == "at_platform":
                if st.get("platform_end_slot", 0) <= cur:
                    st["status"] = "completed"
                    st["log"].append((cur, self.platform, None, "depart"))
        # advance
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
