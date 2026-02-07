import time

import cv2
import numpy as np


def compute_grid_points(order, sw: int, sh: int, margin_ratio: float = 0.10):
    """
    Translate grid (row, col) indices into absolute pixel locations
    """
    if not order:
        return []

    max_r = max(r for r, _ in order)
    max_c = max(c for _, c in order)

    mx, my = int(sw * margin_ratio), int(sh * margin_ratio)
    gw, gh = sw - 2 * mx, sh - 2 * my

    step_x = 0 if max_c == 0 else gw / max_c
    step_y = 0 if max_r == 0 else gh / max_r

    return [(mx + int(c * step_x), my + int(r * step_y)) for r, c in order]


def wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, dur: int = 2) -> bool:
    """
    Waits for a face to be detected (not blinking), then shows a countdown ellipse
    """
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_TOPMOST, 1)  # Bring to front
    fd_start = None
    countdown = False
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        f, blink = gaze_estimator.extract_features(frame)
        face = f is not None and not blink
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        now = time.time()
        if face:
            if not countdown:
                fd_start = now
                countdown = True
            elapsed = now - fd_start
            if elapsed >= dur:
                return True
            t = elapsed / dur
            e = t * t * (3 - 2 * t)
            ang = 360 * (1 - e)
            cv2.ellipse(
                canvas,
                (sw // 2, sh // 2),
                (50, 50),
                0,
                -90,
                -90 + ang,
                (0, 255, 0),
                -1,
            )
        else:
            countdown = False
            fd_start = None
            txt = "Face not detected"
            fs = 2
            thick = 3
            size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
            tx = (sw - size[0]) // 2
            ty = (sh + size[1]) // 2
            cv2.putText(
                canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), thick
            )
        cv2.imshow("Calibration", canvas)
        if cv2.waitKey(1) == 27:
            return False


def _pulse_and_capture(
    gaze_estimator,
    cap,
    pts,
    sw: int,
    sh: int,
    pulse_d: float = 1.0,
    cd_d: float = 1.0,
):
    """
    Shared pulse-and-capture loop for each calibration point
    """
    feats, targs = [], []

    for x, y in pts:
        # pulse
        ps = time.time()
        final_radius = 20
        while True:
            e = time.time() - ps
            if e > pulse_d:
                break
            ok, frame = cap.read()
            if not ok:
                continue
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            radius = 15 + int(15 * abs(np.sin(2 * np.pi * e)))
            final_radius = radius
            cv2.circle(canvas, (x, y), radius, (0, 255, 0), -1)
            cv2.imshow("Calibration", canvas)
            if cv2.waitKey(1) == 27:
                return None
        # capture
        cs = time.time()
        while True:
            e = time.time() - cs
            if e > cd_d:
                break
            ok, frame = cap.read()
            if not ok:
                continue
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            cv2.circle(canvas, (x, y), final_radius, (0, 255, 0), -1)
            t = e / cd_d
            ease = t * t * (3 - 2 * t)
            ang = 360 * (1 - ease)
            cv2.ellipse(canvas, (x, y), (40, 40), 0, -90, -90 + ang, (255, 255, 255), 4)
            cv2.imshow("Calibration", canvas)
            if cv2.waitKey(1) == 27:
                return None
            ft, blink = gaze_estimator.extract_features(frame)
            if ft is not None and not blink:
                feats.append(ft)
                targs.append([x, y])

    return feats, targs
