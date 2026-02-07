import cv2
import numpy as np

from eyetrax.calibration.common import (
    _pulse_and_capture,
    compute_grid_points,
    show_start_prompt,
    wait_for_face_and_countdown,
)
from eyetrax.utils.screen import get_screen_geometry


def run_9_point_calibration(gaze_estimator, camera_index: int = 0):
    """
    Standard nine-point calibration
    """
    sx, sy, sw, sh = get_screen_geometry()

    # Show start prompt and wait for space
    if not show_start_prompt("Calibration"):
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2, sx, sy):
        cap.release()
        cv2.destroyAllWindows()
        return

    order = [
        (1, 1),
        (0, 0),
        (2, 0),
        (0, 2),
        (2, 2),
        (1, 0),
        (0, 1),
        (2, 1),
        (1, 2),
    ]
    pts = compute_grid_points(order, sw, sh)

    res = _pulse_and_capture(gaze_estimator, cap, pts, sw, sh)
    cap.release()
    cv2.destroyAllWindows()
    if res is None:
        return
    feats, targs = res
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))
