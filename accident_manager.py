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
    involved_train: str = None  # The train involved in the accident
    info: dict = None

    @property
    def end_time(self):
        return self.start_time + self.duration_slots

    def is_active_slot(self, slot):
        return self.start_time <= slot < self.end_time

class AccidentManager:
    """
    Advanced Accident Management System
    
    Handles emergency events with comprehensive tracking and impact assessment.
    Features intelligent train classification and detailed statistics collection.
    """
    def __init__(self):
        self.scheduled = []  # List of EmergencyEvent objects
        self.affected_trains = defaultdict(list)  # event_id -> list of affected trains
        self.rerouted_trains = defaultdict(list)  # event_id -> list of rerouted trains
        self.involved_trains = defaultdict(str)  # event_id -> involved train id
        self.accident_stats = defaultdict(lambda: {
            "total_delay": 0,
            "trains_affected": 0,
            "trains_rerouted": 0,
            "resolution_time": None,
            "severity_level": "normal",
            "impact_radius": 0
        })
        # Network config (for blocking whole tracks)
        self.sections_per_track = 4

    def set_network(self, sections_per_track: int):
        """Set network configuration for accident effects."""
        self.sections_per_track = max(1, int(sections_per_track))

    def schedule(self, event: EmergencyEvent):
        """
        Schedule a new emergency event with comprehensive initialization
        
        Args:
            event (EmergencyEvent): The emergency event to schedule
        """
        self.scheduled.append(event)
        
        # Initialize comprehensive stats for this event
        self.accident_stats[event.event_id] = {
            "total_delay": 0,
            "trains_affected": 0,
            "trains_rerouted": 0,
            "start_time": event.start_time,
            "end_time": event.end_time,
            "location": event.location,
            "type": event.ev_type,
            "resolution_time": None,
            "severity_level": event.info.get("severity", "normal") if event.info else "normal",
            "impact_radius": 0,
            "created_at": event.start_time
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

    def set_involved_train(self, event_id, train_id):
        """Record the train involved in the accident"""
        self.involved_trains[event_id] = train_id

    def blocked_nodes(self, slot_index):
        """Return set of blocked nodes at given slot index."""
        blocked = set()
        for e in self.scheduled:
            if e.is_active_slot(slot_index):
                if e.ev_type in ("accident", "breakdown", "signal"):
                    if isinstance(e.location, tuple) and e.location[0] != "Platform":
                        # Block the ENTIRE track where the accident occurs
                        track, section = e.location
                        # Block all sections on the same track
                        for sec in range(self.sections_per_track):
                            blocked.add((track, sec))
                    else:
                        # For platform accidents, just block the platform
                        blocked.add(e.location)
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
