"""WebSocket server for streaming gaze coordinates."""

import atexit
import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import cv2
import numpy as np

from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_grid_calibration,
    run_lissajous_calibration,
)
from eyetrax.filters import KalmanSmoother, KDESmoother, NoSmoother, make_kalman
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, iter_frames

try:
    import websockets
    from websockets.server import serve
except ImportError:
    print("Error: websockets library required. Install with: pip install websockets")
    sys.exit(1)


def get_default_model_path() -> Path:
    """Get default path for saved calibration model."""
    xdg_data = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return Path(xdg_data) / "eyetrax" / "model.pkl"


def parse_serve_args():
    """Parse command line arguments for serve mode."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="eyetrax",
        description="Eye tracking WebSocket server for glimpsh integration",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="WebSocket server port (default: 8001)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="WebSocket server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--filter",
        choices=["kalman", "kde", "none"],
        default="none",
        help="Smoothing filter (default: none)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--calibration",
        choices=["9p", "5p", "lissajous", "grid"],
        default="9p",
        help="Calibration method if no saved model (default: 9p, use 'grid' with --grid)",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        metavar="RxC",
        help="Grid mode for terminal grids (e.g., '2x2'). Enables cell-based output with hysteresis.",
    )
    parser.add_argument(
        "--model",
        default="ridge",
        help="ML model for gaze estimation (default: ridge)",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help="Path to saved model file (default: ~/.local/share/eyetrax/model.pkl)",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Force recalibration even if saved model exists",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="KDE confidence level (default: 0.5)",
    )
    return parser.parse_args()


def parse_grid(grid_str):
    """Parse grid string like '2x2' into (rows, cols)."""
    if not grid_str:
        return None
    try:
        parts = grid_str.lower().split("x")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        print(f"Invalid grid format: {grid_str}. Expected format: RxC (e.g., 2x2)")
        sys.exit(1)


class GridTracker:
    """Tracks which grid cell the gaze is in with hysteresis."""

    def __init__(self, rows: int, cols: int, hysteresis: float = 0.15):
        self.rows = rows
        self.cols = cols
        self.hysteresis = hysteresis  # Extra margin needed to leave current cell
        self.current_cell = None  # (row, col)

    def update(self, x_norm: float, y_norm: float) -> tuple:
        """
        Update with normalized coordinates (0-1).
        Returns (row, col) of current cell with hysteresis.
        """
        # Calculate which cell the raw gaze is in
        col = int(x_norm * self.cols)
        row = int(y_norm * self.rows)

        # Clamp to valid range
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))

        new_cell = (row, col)

        if self.current_cell is None:
            self.current_cell = new_cell
            return self.current_cell

        # Calculate cell boundaries with hysteresis
        curr_row, curr_col = self.current_cell
        cell_width = 1.0 / self.cols
        cell_height = 1.0 / self.rows

        # Current cell boundaries (expanded by hysteresis)
        left = curr_col * cell_width - self.hysteresis * cell_width
        right = (curr_col + 1) * cell_width + self.hysteresis * cell_width
        top = curr_row * cell_height - self.hysteresis * cell_height
        bottom = (curr_row + 1) * cell_height + self.hysteresis * cell_height

        # Only switch cell if gaze is clearly outside current cell
        if x_norm < left or x_norm > right or y_norm < top or y_norm > bottom:
            self.current_cell = new_cell

        return self.current_cell

    def get_cell_center(self, row: int, col: int) -> tuple:
        """Get normalized center coordinates of a cell."""
        cell_width = 1.0 / self.cols
        cell_height = 1.0 / self.rows
        x = (col + 0.5) * cell_width
        y = (row + 0.5) * cell_height
        return x, y


# Global state for the gaze loop
clients = set()
latest_gaze = {"x": None, "y": None, "cell": None}  # Normalized coords + optional cell
running = True
camera_cap = None  # Global reference for cleanup
grid_tracker = None  # Optional GridTracker for grid mode


def _cleanup_camera():
    """Last-resort camera cleanup on exit."""
    global camera_cap
    if camera_cap is not None:
        try:
            camera_cap.release()
            print("Camera released (atexit)")
        except Exception:
            pass


atexit.register(_cleanup_camera)


async def handle_client(websocket):
    """Handle a WebSocket client connection."""
    clients.add(websocket)
    print(f"Client connected ({len(clients)} total)")

    # Send hello message per glimpsh protocol
    await websocket.send(json.dumps({
        "type": "hello",
        "name": "EyeTrax",
        "version": "0.3.0",
    }))

    try:
        # Keep connection alive, client only receives
        async for _ in websocket:
            pass
    except websockets.ConnectionClosed:
        pass
    finally:
        clients.discard(websocket)
        print(f"Client disconnected ({len(clients)} total)")


async def broadcast_gaze():
    """Broadcast latest gaze coordinates to all connected clients."""
    if not clients:
        return

    # Only broadcast if we have valid gaze data
    if latest_gaze["x"] is None or latest_gaze["y"] is None:
        return

    # Build message - include cell if in grid mode
    msg = {"x": latest_gaze["x"], "y": latest_gaze["y"]}
    if latest_gaze["cell"] is not None:
        msg["cell"] = latest_gaze["cell"]

    message = json.dumps(msg)
    await asyncio.gather(
        *[client.send(message) for client in clients],
        return_exceptions=True,
    )


def run_gaze_loop(gaze_estimator, smoother, camera_index, screen_width, screen_height):
    """Run the gaze capture loop (blocking, runs in thread)."""
    global latest_gaze, running, camera_cap, grid_tracker

    cap = cv2.VideoCapture(camera_index)
    camera_cap = cap  # Store globally for cleanup

    try:
        while running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            features, blink_detected = gaze_estimator.extract_features(frame)

            if features is not None and not blink_detected:
                gaze_point = gaze_estimator.predict(np.array([features]))[0]
                x, y = map(int, gaze_point)
                x_pred, y_pred = smoother.step(x, y)

                # Output normalized coordinates (0-1) for screen-agnostic protocol
                x_norm = round(x_pred / screen_width, 4)
                y_norm = round(y_pred / screen_height, 4)

                # Clamp to valid range
                x_norm = max(0.0, min(1.0, x_norm))
                y_norm = max(0.0, min(1.0, y_norm))

                latest_gaze["x"] = x_norm
                latest_gaze["y"] = y_norm

                # If grid mode, also track cell with hysteresis
                if grid_tracker is not None:
                    row, col = grid_tracker.update(x_norm, y_norm)
                    latest_gaze["cell"] = {"row": row, "col": col}
                else:
                    latest_gaze["cell"] = None
    finally:
        cap.release()
        camera_cap = None
        print("Camera released")


async def broadcast_loop():
    """Continuously broadcast gaze data to clients."""
    global running
    while running:
        await broadcast_gaze()
        await asyncio.sleep(1 / 60)  # ~60 Hz


def run_serve():
    """Main entry point for WebSocket server mode."""
    global running, grid_tracker

    args = parse_serve_args()

    # Parse grid if specified
    grid = parse_grid(args.grid)
    if grid:
        rows, cols = grid
        grid_tracker = GridTracker(rows, cols)
        print(f"Grid mode enabled: {rows}x{cols}")

    # Determine model path - use grid-specific path if in grid mode
    if args.model_file:
        model_path = Path(args.model_file)
    elif grid:
        base_path = get_default_model_path()
        model_path = base_path.parent / f"model_grid_{grid[0]}x{grid[1]}.pkl"
    else:
        model_path = get_default_model_path()

    # Initialize gaze estimator
    gaze_estimator = GazeEstimator(model_name=args.model)

    # Load or calibrate
    if not args.recalibrate and model_path.exists():
        gaze_estimator.load_model(model_path)
        print(f"Loaded calibration from {model_path}")
    else:
        if args.calibration == "grid" or (grid and args.calibration == "9p"):
            # Use grid calibration if --grid specified or --calibration grid
            if grid:
                print(f"Starting GRID calibration for {grid[0]}x{grid[1]} layout...")
                run_grid_calibration(gaze_estimator, grid[0], grid[1], camera_index=args.camera)
            else:
                print("Warning: --calibration grid requires --grid RxC. Using 9p instead.")
                run_9_point_calibration(gaze_estimator, camera_index=args.camera)
        elif args.calibration == "9p":
            print("Starting 9-point calibration...")
            run_9_point_calibration(gaze_estimator, camera_index=args.camera)
        elif args.calibration == "5p":
            print("Starting 5-point calibration...")
            run_5_point_calibration(gaze_estimator, camera_index=args.camera)
        else:
            print("Starting lissajous calibration...")
            run_lissajous_calibration(gaze_estimator, camera_index=args.camera)

        # Auto-save model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        gaze_estimator.save_model(model_path)
        print(f"Saved calibration to {model_path}")

    # Setup smoother
    screen_width, screen_height = get_screen_size()
    if args.filter == "kalman":
        kalman = make_kalman()
        smoother = KalmanSmoother(kalman)
        kalman_path = model_path.parent / (model_path.stem + "_kalman.npy")
        if not args.recalibrate and kalman_path.exists():
            noise_cov = np.load(kalman_path)
            smoother.kf.measurementNoiseCov = noise_cov
            print(f"Loaded Kalman params from {kalman_path}")
        else:
            smoother.tune(gaze_estimator, camera_index=args.camera)
            if smoother.kf.measurementNoiseCov is not None:
                np.save(kalman_path, smoother.kf.measurementNoiseCov)
                print(f"Saved Kalman params to {kalman_path}")
    elif args.filter == "kde":
        smoother = KDESmoother(screen_width, screen_height, confidence=args.confidence)
    else:
        smoother = NoSmoother()

    # Start gaze capture in background thread
    import threading
    gaze_thread = threading.Thread(
        target=run_gaze_loop,
        args=(gaze_estimator, smoother, args.camera, screen_width, screen_height),
    )
    gaze_thread.start()

    # Handle shutdown
    def shutdown(sig, frame):
        global running, camera_cap
        print("\nShutting down...")
        running = False
        # Force release camera if thread is stuck
        if camera_cap is not None:
            camera_cap.release()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Run WebSocket server
    async def main():
        async with serve(handle_client, args.host, args.port):
            print(f"EyeTrax server running on ws://{args.host}:{args.port}/")
            print("Waiting for clients...")
            await broadcast_loop()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        running = False
        if camera_cap is not None:
            camera_cap.release()

    # Wait for gaze thread to finish
    gaze_thread.join(timeout=2.0)
    print("Server stopped")


if __name__ == "__main__":
    run_serve()
