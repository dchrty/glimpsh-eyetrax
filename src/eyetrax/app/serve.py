"""WebSocket server for streaming gaze coordinates."""

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
        choices=["9p", "5p", "lissajous"],
        default="9p",
        help="Calibration method if no saved model (default: 9p)",
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


# Global state for the gaze loop
clients = set()
latest_gaze = {"x_px": None, "y_px": None}
running = True


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

    message = json.dumps(latest_gaze)
    await asyncio.gather(
        *[client.send(message) for client in clients],
        return_exceptions=True,
    )


def run_gaze_loop(gaze_estimator, smoother, camera_index):
    """Run the gaze capture loop (blocking, runs in thread)."""
    global latest_gaze, running

    with camera(camera_index) as cap:
        for frame in iter_frames(cap):
            if not running:
                break

            features, blink_detected = gaze_estimator.extract_features(frame)

            if features is not None and not blink_detected:
                gaze_point = gaze_estimator.predict(np.array([features]))[0]
                x, y = map(int, gaze_point)
                x_pred, y_pred = smoother.step(x, y)
                latest_gaze["x_px"] = int(x_pred)
                latest_gaze["y_px"] = int(y_pred)
            else:
                # Keep last known position during blinks
                pass


async def broadcast_loop():
    """Continuously broadcast gaze data to clients."""
    global running
    while running:
        await broadcast_gaze()
        await asyncio.sleep(1 / 60)  # ~60 Hz


def run_serve():
    """Main entry point for WebSocket server mode."""
    global running

    args = parse_serve_args()

    # Determine model path
    model_path = Path(args.model_file) if args.model_file else get_default_model_path()

    # Initialize gaze estimator
    gaze_estimator = GazeEstimator(model_name=args.model)

    # Load or calibrate
    if not args.recalibrate and model_path.exists():
        gaze_estimator.load_model(model_path)
        print(f"Loaded calibration from {model_path}")
    else:
        print("Starting calibration...")
        if args.calibration == "9p":
            run_9_point_calibration(gaze_estimator, camera_index=args.camera)
        elif args.calibration == "5p":
            run_5_point_calibration(gaze_estimator, camera_index=args.camera)
        else:
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
        smoother.tune(gaze_estimator, camera_index=args.camera)
    elif args.filter == "kde":
        smoother = KDESmoother(screen_width, screen_height, confidence=args.confidence)
    else:
        smoother = NoSmoother()

    # Handle shutdown
    def shutdown(sig, frame):
        global running
        print("\nShutting down...")
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start gaze capture in background thread
    import threading
    gaze_thread = threading.Thread(
        target=run_gaze_loop,
        args=(gaze_estimator, smoother, args.camera),
        daemon=True,
    )
    gaze_thread.start()

    # Run WebSocket server
    async def main():
        async with serve(handle_client, args.host, args.port):
            print(f"EyeTrax server running on ws://{args.host}:{args.port}/")
            print("Waiting for clients...")
            await broadcast_loop()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

    print("Server stopped")


if __name__ == "__main__":
    run_serve()
