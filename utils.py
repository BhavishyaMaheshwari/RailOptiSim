# utils.py
# Optional station labels for prettier display (id -> code/name)
STATION_LABELS = {}

def set_station_labels(mapping):
    """Set station id -> label mapping used by formatters.
    mapping: dict[int -> str]
    """
    global STATION_LABELS
    try:
        STATION_LABELS = {int(k): str(v) for k, v in mapping.items()}
    except Exception:
        # best-effort fallback
        STATION_LABELS = mapping or {}

def _station_label(st_id):
    if isinstance(st_id, int) and st_id in STATION_LABELS:
        return STATION_LABELS[st_id]
    return f"Station{(st_id + 1) if isinstance(st_id, int) else st_id}"

def format_node(node):
    if node is None:
        return "None"
    if isinstance(node, tuple) and node[0] == "Platform":
        # ("Platform", station_id, platform_id)
        if len(node) >= 3:
            label = _station_label(node[1])
            return f"{label}-Plat{node[2]+1}"
        return "Platform"
    if isinstance(node, tuple):
        tr, sec = node
        return f"Track{tr+1}-Sec{sec+1}"
    return str(node)

def short_node(node):
    if node is None:
        return "NONE"
    if isinstance(node, tuple) and node[0] == "Platform":
        if len(node) >= 3:
            label = _station_label(node[1])
            return f"{label}:{node[2]+1}"
        return "PLAT"
    if isinstance(node, tuple):
        tr, sec = node
        return f"T{tr+1}S{sec+1}"
    return str(node)
