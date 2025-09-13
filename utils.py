# utils.py
def format_node(node):
    if node is None:
        return "None"
    if isinstance(node, tuple) and node[0] == "Platform":
        return "Platform"
    if isinstance(node, tuple):
        tr, sec = node
        return f"Track{tr+1}-Sec{sec+1}"
    return str(node)

def short_node(node):
    if node is None:
        return "NONE"
    if isinstance(node, tuple) and node[0] == "Platform":
        return "PLAT"
    if isinstance(node, tuple):
        tr, sec = node
        return f"T{tr+1}S{sec+1}"
    return str(node)
