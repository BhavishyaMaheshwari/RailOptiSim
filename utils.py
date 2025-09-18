# utils.py
def format_node(node):
    if node is None:
        return "None"
    if isinstance(node, tuple) and node[0] == "Platform":
        # ("Platform", station_id, platform_id)
        if len(node) >= 3:
            return f"Station{node[1]+1}-Plat{node[2]+1}"
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
            return f"P{node[1]+1}:{node[2]+1}"
        return "PLAT"
    if isinstance(node, tuple):
        tr, sec = node
        return f"T{tr+1}S{sec+1}"
    return str(node)
