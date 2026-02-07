import os

from screeninfo import get_monitors


def get_primary_monitor():
    """Get the primary monitor, or first available."""
    monitors = get_monitors()

    # Debug: print monitor info
    if os.environ.get("EYETRAX_DEBUG"):
        for i, m in enumerate(monitors):
            print(f"Monitor {i}: {m.width}x{m.height} at ({m.x},{m.y}) primary={m.is_primary}")

    # First try to find the explicitly marked primary
    for m in monitors:
        if m.is_primary:
            return m

    # Fallback: use monitor at position (0,0) or closest to it
    monitors_sorted = sorted(monitors, key=lambda m: (m.x, m.y))
    return monitors_sorted[0] if monitors_sorted else None


def get_screen_size():
    m = get_primary_monitor()
    return m.width, m.height


def get_screen_geometry():
    """Get screen x, y, width, height for the primary monitor."""
    m = get_primary_monitor()
    return m.x, m.y, m.width, m.height
