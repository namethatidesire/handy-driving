# handy driving

Control driving games with your hands in front of your webcam. Uses computer vision to track hand gestures and translate them into a virtual Xbox 360 controller.

| Hand | Gesture | Input |
|------|---------|-------|
| Steering hand (default: right) | Point index finger left / right | Steering (left stick X) |
| Throttle hand (default: left) | Show palm to camera, fingers up | Throttle (right trigger) |
| Throttle hand (default: left) | Show back of hand to camera, fingers down | Reverse (left trigger) |

## Requirements

- Windows 10/11
- Python **3.9–3.12** (mediapipe has no Windows wheels for 3.13+; see note below)
- A webcam
- [ViGEmBus](https://github.com/nefarius/ViGEmBus/releases) kernel driver (for real controller output)

## Setup

### If you have Python 3.12 or earlier

```
pip install -r requirements.txt
python main.py --dry-run
```

### If you have Python 3.13+ (venv with side-by-side install)

mediapipe does not publish Windows wheels for Python 3.13+. You do **not** need to uninstall your existing Python — Windows supports multiple versions at once via the `py` launcher.

1. Download and install **Python 3.12** from <https://www.python.org/downloads/>
   (leave your existing Python untouched)
2. Confirm it registered: `py -3.12 --version`
3. Create a project virtual environment using 3.12:
   ```
   py -3.12 -m venv .venv
   ```
4. Activate it and install dependencies:
   ```
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
5. Run:
   ```
   python main.py --dry-run
   ```

The `.venv` folder is isolated to this project — your system Python is unaffected.

Install ViGEmBus from its [releases page](https://github.com/nefarius/ViGEmBus/releases) for real controller output. Without it the app automatically falls back to dry-run mode.

## Usage

```
python main.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--steer-hand left/right` | `right` | Hand used for steering (index finger direction) |
| `--throttle-hand left/right` | `left` | Hand used for throttle/reverse (palm orientation) |
| `--camera N` | `0` | Camera device index |
| `--alpha 0–1` | `0.25` | EMA smoothing (lower = smoother, more lag) |
| `--no-calibration` | off | Load saved `calibration.json` instead of recalibrating |
| `--calib-file PATH` | `calibration.json` | Path to calibration file |
| `--dry-run` | off | Print values to console, no virtual controller |
| `--no-flip` | off | Disable camera mirroring |

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Q` / `Esc` | Quit |
| `R` | Restart calibration |
| `S` | Save calibration to file |
| `P` | Pause / resume controller output |

## Calibration

On first run, the app steps through four positions:

1. **Steer left** — point index finger as far left as comfortable, hold
2. **Steer right** — point index finger as far right as comfortable, hold
3. **Throttle** — show palm to camera with fingers pointing up, hold
4. **Reverse** — show back of hand to camera with fingers pointing down, hold

A progress bar fills as you hold each pose. Calibration is saved automatically on completion. On subsequent runs use `--no-calibration` to skip straight to driving.

## Verifying the virtual controller

Open **Game Controllers** (`Win + R` → `joy.cpl`). After launch, a device named *Xbox 360 Controller* should appear. Move your hands and confirm the axes respond before opening a game.
