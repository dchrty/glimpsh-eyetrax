import cv2
import numpy as np

from eyetrax.calibration.common import (
    _pulse_and_capture,
    close_all_windows,
    compute_grid_points,
    show_start_prompt,
    wait_for_face_and_countdown,
)
from eyetrax.utils.screen import get_screen_geometry


def run_5_point_calibration(gaze_estimator, camera_index: int = 0, cap=None):
    """
    Faster five-point calibration
    """
    sx, sy, sw, sh = get_screen_geometry()

    if not show_start_prompt("Calibration"):
        close_all_windows()
        return

    own_cap = cap is None
    if own_cap:
        cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2, sx, sy):
        if own_cap:
            cap.release()
        close_all_windows()
        return

    order = [(1, 1), (0, 0), (2, 0), (0, 2), (2, 2)]
    pts = compute_grid_points(order, sw, sh)

    res = _pulse_and_capture(gaze_estimator, cap, pts, sw, sh)
    if own_cap:
        cap.release()
    close_all_windows()
    if res is None:
        return
    feats, targs = res
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))
