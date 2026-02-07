# glimpsh-eyetrax

Webcam-based eye tracking for [glimpsh](https://github.com/dchrty/glimpsh). Fork of [EyeTrax](https://github.com/ck-zhang/EyeTrax) with WebSocket server support.

## For glimpsh users

You don't need to install this separately - it's included when you install glimpsh:

```bash
git clone https://github.com/dchrty/glimpsh
cd glimpsh
uv sync --extra eyetrax
uv run glimpsh
```

## Standalone Usage

If you want to use eyetrax independently:

```bash
git clone https://github.com/dchrty/glimpsh-eyetrax
cd glimpsh-eyetrax
uv sync
uv run eyetrax --filter kalman
```

This starts a WebSocket server on `ws://127.0.0.1:8001/` that streams gaze coordinates.

### Options

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--filter` | `kalman`, `kde`, `none` | `none` | Smoothing filter |
| `--grid` | `RxC` (e.g., `2x2`) | — | Grid mode with cell-based output |
| `--camera` | int | `0` | Webcam index |
| `--calibration` | `9p`, `5p`, `lissajous`, `grid` | `9p` | Calibration method |
| `--recalibrate` | — | — | Force fresh calibration |
| `--port` | int | `8001` | WebSocket port |

### Examples

```bash
# Kalman smoothing (recommended)
uv run eyetrax --filter kalman

# Grid mode for 2x2 terminal layout
uv run eyetrax --filter kalman --grid 2x2

# Force recalibration
uv run eyetrax --filter kalman --recalibrate

# Different camera
uv run eyetrax --camera 1
```

## Calibration

Calibration data is saved automatically:
- Standard: `~/.local/share/eyetrax/model.pkl`
- Grid mode: `~/.local/share/eyetrax/model_grid_2x2.pkl`

To recalibrate, delete the model file:

```bash
rm ~/.local/share/eyetrax/model*.pkl
```

## WebSocket Protocol

The server sends JSON messages:

```json
{"type": "hello", "name": "EyeTrax", "version": "0.3.0"}
{"x": 0.5, "y": 0.3}
{"x": 0.5, "y": 0.3, "cell": {"row": 0, "col": 1}}
```

- `x`, `y`: Normalized coordinates (0-1)
- `cell`: Only present in grid mode, includes hysteresis

## Other Commands

| Command | Purpose |
|---------|---------|
| `eyetrax` | WebSocket server for glimpsh |
| `eyetrax-demo` | On-screen gaze overlay demo |
| `eyetrax-virtualcam` | Stream overlay to virtual webcam |

## Requirements

- Linux or macOS
- Python 3.10+
- Webcam
- OpenCV with GUI support

## Credits

Based on [EyeTrax](https://github.com/ck-zhang/EyeTrax) by Chenkai Zhang.

## License

MIT
