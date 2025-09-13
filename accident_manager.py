# accident_manager.py
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class EmergencyEvent:
    event_id: str
    ev_type: str         # 'accident', 'breakdown', 'signal'
    location: object     # node (track,section) or train id
    start_time: int      # slot index
    duration_slots: int
    info: dict = None

    @property
    def end_time(self):
        return self.start_time + self.duration_slots

    def is_active_slot(self, slot):
        return self.start_time <= slot < self.end_time

class AccidentManager:
    def __init__(self):
        self.scheduled = []  # list of EmergencyEvent
        self.affected_trains = defaultdict(list)  # event_id -> list of affected trains
        self.rerouted_trains = defaultdict(list)  # event_id -> list of rerouted trains
        self.accident_stats = defaultdict(lambda: {
            "total_delay": 0,
            "trains_affected": 0,
            "trains_rerouted": 0,
            "resolution_time": None
        })

    def schedule(self, event: EmergencyEvent):
        """Schedule a new emergency event"""
        self.scheduled.append(event)
        # Initialize stats for this event
        self.accident_stats[event.event_id] = {
            "total_delay": 0,
            "trains_affected": 0,
            "trains_rerouted": 0,
            "start_time": event.start_time,
            "end_time": event.end_time,
            "location": event.location,
            "type": event.ev_type,
            "resolution_time": None
        }

    def add_affected_train(self, event_id, train_id, current_slot):
        """Record a train affected by an accident"""
        if train_id not in self.affected_trains[event_id]:
            self.affected_trains[event_id].append(train_id)
            self.accident_stats[event_id]["trains_affected"] += 1

    def add_rerouted_train(self, event_id, train_id, current_slot, delay):
        """Record a successfully rerouted train"""
        if train_id not in self.rerouted_trains[event_id]:
            self.rerouted_trains[event_id].append(train_id)
            self.accident_stats[event_id]["trains_rerouted"] += 1
            self.accident_stats[event_id]["total_delay"] += delay

    def blocked_nodes(self, slot_index):
        """Return set of blocked nodes at given slot index."""
        blocked = set()
        for e in self.scheduled:
            if e.is_active_slot(slot_index):
                if e.ev_type in ("accident", "breakdown", "signal"):
                    blocked.add(e.location)
                    # Also block adjacent nodes if it's a major accident
                    if e.info and e.info.get("severity") == "high":
                        if isinstance(e.location, tuple) and e.location[0] != "Platform":
                            track, section = e.location
                            # Block adjacent sections on same track
                            if section > 0:
                                blocked.add((track, section - 1))
                            if section < 3:  # assuming 4 sections (0-3)
                                blocked.add((track, section + 1))
        return blocked

    def active_summary(self, slot_index):
        """Get detailed summary of active accidents"""
        out = []
        for e in self.scheduled:
            if e.is_active_slot(slot_index):
                remaining = int(max(0, e.end_time - slot_index))
                stats = self.accident_stats[e.event_id]
                out.append((
                    e.event_id, 
                    e.ev_type, 
                    e.location, 
                    remaining,
                    {
                        "affected_trains": len(self.affected_trains[e.event_id]),
                        "rerouted_trains": len(self.rerouted_trains[e.event_id]),
                        "total_delay": stats["total_delay"],
                        "severity": e.info.get("severity", "normal") if e.info else "normal"
                    }
                ))
        return out

    def get_event_stats(self, event_id):
        """Get detailed statistics for a specific event"""
        return self.accident_stats.get(event_id, None)
