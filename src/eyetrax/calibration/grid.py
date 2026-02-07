"""Grid-aware calibration for terminal grid applications."""

import cv2
import numpy as np

from eyetrax.calibration.common import (
    _pulse_and_capture,
    show_start_prompt,
    wait_for_face_and_countdown,
)
from eyetrax.utils.screen import get_screen_geometry


def run_grid_calibration(gaze_estimator, rows: int, cols: int, camera_index: int = 0):
    """
    Grid-aware calibration that places calibration points at cell centers.

    For a 2x2 grid, this calibrates at 4 cell centers plus screen corners
    to ensure good coverage across the entire screen.
    """
    sx, sy, sw, sh = get_screen_geometry()

    if not show_start_prompt("Calibration"):
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2, sx, sy):
        cap.release()
        cv2.destroyAllWindows()
        return

    # Calculate cell centers
    cell_width = sw / cols
    cell_height = sh / rows

    pts = []

    # Add cell centers
    for r in range(rows):
        for c in range(cols):
            x = int((c + 0.5) * cell_width)
            y = int((r + 0.5) * cell_height)
            pts.append((x, y))

    # Add screen corners and edges for better coverage
    margin = 0.1  # 10% margin from edge
    mx, my = int(sw * margin), int(sh * margin)

    # Corners
    corners = [
        (mx, my),  # Top-left
        (sw - mx, my),  # Top-right
        (mx, sh - my),  # Bottom-left
        (sw - mx, sh - my),  # Bottom-right
    ]

    # Add corners that aren't too close to existing cell centers
    min_dist = min(cell_width, cell_height) * 0.3
    for corner in corners:
        too_close = False
        for pt in pts:
            dist = np.hypot(corner[0] - pt[0], corner[1] - pt[1])
            if dist < min_dist:
                too_close = True
                break
        if not too_close:
            pts.append(corner)

    # Add center of screen if not already covered
    center = (sw // 2, sh // 2)
    center_covered = any(
        np.hypot(center[0] - pt[0], center[1] - pt[1]) < min_dist for pt in pts
    )
    if not center_covered:
        pts.append(center)

    res = _pulse_and_capture(gaze_estimator, cap, pts, sw, sh)
    cap.release()
    cv2.destroyAllWindows()

    if res is None:
        return

    feats, targs = res
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))
