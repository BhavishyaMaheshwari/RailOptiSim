# accident_manager.py
from dataclasses import dataclass

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

    def schedule(self, event: EmergencyEvent):
        self.scheduled.append(event)

    def blocked_nodes(self, slot_index):
        """Return set of blocked nodes at given slot index."""
        blocked = set()
        for e in self.scheduled:
            if e.is_active_slot(slot_index):
                if e.ev_type in ("accident", "breakdown", "signal"):
                    blocked.add(e.location)
        return blocked

    def active_summary(self, slot_index):
        out = []
        for e in self.scheduled:
            if e.is_active_slot(slot_index):
                remaining = int(max(0, e.end_time - slot_index))
                out.append((e.event_id, e.ev_type, e.location, remaining))
        return out
