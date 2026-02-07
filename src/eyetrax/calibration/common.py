import platform
import time

import cv2
import numpy as np

from eyetrax.utils.screen import get_screen_geometry

# Detect platform
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


def close_all_windows() -> None:
    """Destroy all OpenCV windows and flush the event loop."""
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(100)


def make_fullscreen(window_name: str, sx: int, sy: int, sw: int, sh: int, canvas: np.ndarray) -> None:
    """Make a window fill the screen."""
    # Create resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Show content first (required by some window managers)
    cv2.imshow(window_name, canvas)
    cv2.waitKey(1)

    # Position and resize to fill the screen
    cv2.moveWindow(window_name, sx, sy)
    cv2.resizeWindow(window_name, sw, sh)
    cv2.waitKey(1)

    if not IS_MACOS:
        # On Linux, use true fullscreen mode
        for _ in range(3):
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.waitKey(50)


def show_start_prompt(window_name: str = "Calibration") -> bool:
    """Show a 'press space to start' screen. Returns False if ESC pressed."""
    sx, sy, sw, sh = get_screen_geometry()

    # Create canvas
    canvas = np.zeros((sh, sw, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Different message based on window name
    if "Kalman" in window_name:
        text = "Press SPACE to start Kalman calibration"
    else:
        text = "Press SPACE to start calibration"

    text_size, _ = cv2.getTextSize(text, font, 2, 3)
    tx = (sw - text_size[0]) // 2
    ty = (sh + text_size[1]) // 2
    cv2.putText(canvas, text, (tx, ty), font, 2, (255, 255, 255), 3)

    sub_text = "(Press ESC to cancel)"
    sub_size, _ = cv2.getTextSize(sub_text, font, 1, 2)
    cv2.putText(canvas, sub_text, ((sw - sub_size[0]) // 2, ty + 60), font, 1, (128, 128, 128), 2)

    # Setup fullscreen window (pass canvas so it can show content first)
    make_fullscreen(window_name, sx, sy, sw, sh, canvas)

    while True:
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(100) & 0xFF
        if key == 32:  # Space
            return True
        elif key == 27:  # ESC
            return False


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


def wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, dur: int = 2, sx: int = 0, sy: int = 0) -> bool:
    """
    Waits for a face to be detected (not blinking), then shows a countdown ellipse
    """
    # Window should already exist from show_start_prompt, but ensure it's set up
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Calibration", sx, sy)
    cv2.resizeWindow("Calibration", sw, sh)
    if not IS_MACOS:
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
