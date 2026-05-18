from __future__ import annotations

import sys

if sys.version_info >= (3, 13):
    print(
        "ERROR: Python 3.13+ is not supported.\n"
        "mediapipe only publishes Windows wheels for Python 3.9–3.12.\n\n"
        "Fix: install Python 3.12 from https://python.org/downloads/ then run:\n"
        "  py -3.12 -m venv .venv\n"
        "  .venv\\Scripts\\activate\n"
        "  pip install -r requirements.txt\n"
        "  python main.py",
        file=sys.stderr,
    )
    sys.exit(1)

import os
os.environ.setdefault("GLOG_minloglevel", "3")   # suppress MediaPipe C++ log noise
os.environ.setdefault("GRPC_VERBOSITY", "NONE")  # suppress gRPC internals used by telemetry uploader

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from calibration import CalibrationLoadError, CalibrationManager
from config import AppConfig, AppState, SmoothingConfig
from controller import XInputController
from mapper import DriveState, GestureMapper
from settings import Settings, load_settings, save_settings
from settings_panel import SettingsPanel
from tracker import HandResult, HandTracker
from visualizer import HUDVisualizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_WIN_NAME = "handy driving"

_CALIB_POSITION = {
    AppState.CALIB_STEER_LEFT:   "STEER: Point index finger as far LEFT as comfortable",
    AppState.CALIB_STEER_RIGHT:  "STEER: Point index finger as far RIGHT as comfortable",
    AppState.CALIB_THROTTLE_MAX: "THROTTLE: Show PALM to camera, fingers pointing UP",
    AppState.CALIB_BRAKE_MAX:    "REVERSE: Show BACK of hand to camera, fingers pointing DOWN",
}
_CALIB_HOLDING = {
    AppState.CALIB_STEER_LEFT:   "STEER LEFT: Hold that position...",
    AppState.CALIB_STEER_RIGHT:  "STEER RIGHT: Hold that position...",
    AppState.CALIB_THROTTLE_MAX: "THROTTLE: Hold that position...",
    AppState.CALIB_BRAKE_MAX:    "REVERSE: Hold that position...",
}

_CALIBRATION_STATES = set(_CALIB_POSITION.keys())
_STEER_STATES = {AppState.CALIB_STEER_LEFT, AppState.CALIB_STEER_RIGHT}

# Mouse state (written by callback, read by main loop)
_mouse_pos   = [0, 0]
_mouse_event = [cv2.EVENT_MOUSEMOVE]


def _on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
    _mouse_pos[0], _mouse_pos[1] = x, y
    if event != cv2.EVENT_MOUSEMOVE:
        _mouse_event[0] = event


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="handy driving — control driving games with hand gestures"
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print values to console instead of writing to virtual controller")
    p.add_argument("--calib-file", default="calibration.json",
                   help="Path to calibration JSON (default: calibration.json)")
    return p.parse_args()


def _config_from_settings(s: Settings, calib_file: str) -> AppConfig:
    return AppConfig(
        camera_index=s.camera_index,
        calib_file=calib_file,
        smoothing=SmoothingConfig(alpha=s.alpha),
        steer_hand=s.steer_hand,
        throttle_hand=s.throttle_hand,
        flip_camera=s.flip_camera,
        width=s.width,
        height=s.height,
        fps=s.fps,
    )


def _open_camera(s: Settings) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(s.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  s.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.height)
    cap.set(cv2.CAP_PROP_FPS,          s.fps)
    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info("Camera %d: %dx%d @ %.0f fps (requested %dx%d @ %d fps)",
                s.camera_index, actual_w, actual_h, actual_fps,
                s.width, s.height, s.fps)
    return cap


def _load_or_create_calibration(config: AppConfig) -> CalibrationManager:
    calib_path = Path(config.calib_file)
    if calib_path.exists():
        try:
            mgr = CalibrationManager.load(str(calib_path), config)
            logger.info("Loaded calibration from %s. Press R to recalibrate.", calib_path)
            return mgr
        except CalibrationLoadError as e:
            logger.warning("Could not load calibration (%s) — running fresh calibration.", e)
    return CalibrationManager(config)


def _render_overlay(
    frame: np.ndarray,
    state: AppState,
    drive_state: DriveState,
    calibration: CalibrationManager,
    hands: dict[str, HandResult],
    tracker: HandTracker,
    fps: float,
    is_dry_run: bool,
    no_throttle_frames: int,
    hud: HUDVisualizer,
    panel_open: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]

    for result in hands.values():
        frame = tracker.draw_landmarks(frame, result)

    # State label
    if state in _CALIBRATION_STATES:
        if calibration.waiting_for_ready:
            label, color = _CALIB_POSITION[state], (0, 220, 255)
        else:
            label, color = _CALIB_HOLDING[state], (0, 220, 255)
    elif state == AppState.RUNNING:
        label, color = "RUNNING", (0, 255, 0)
    elif state == AppState.PAUSED:
        label, color = "PAUSED", (0, 165, 255)
    else:
        label, color = str(state.name), (255, 255, 255)

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # No-hand warning
    if not hands:
        warn_color = (0, 0, 255) if no_throttle_frames > 60 else (0, 165, 255)
        cv2.putText(frame, "NO HANDS DETECTED", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "NO HANDS DETECTED", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, warn_color, 2, cv2.LINE_AA)

    # Calibration bottom area
    if state in _CALIBRATION_STATES:
        bx, by, bw_bar, bh = 10, h - 30, w - 20, 16
        if calibration.waiting_for_ready:
            prompt = "Press SPACE when you're in position"
            (tw, _), _ = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            px = w // 2 - tw // 2
            cv2.putText(frame, prompt, (px, by + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, prompt, (px, by + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
        else:
            progress = calibration.hold_progress
            cv2.rectangle(frame, (bx, by), (bx + bw_bar, by + bh), (50, 50, 50), -1)
            filled = int(bw_bar * progress)
            if filled > 0:
                cv2.rectangle(frame, (bx, by), (bx + filled, by + bh), (0, 200, 255), -1)
            cv2.rectangle(frame, (bx, by), (bx + bw_bar, by + bh), (180, 180, 180), 1)
            pct = f"{int(progress * 100)}%"
            cv2.putText(frame, pct, (bx + bw_bar // 2 - 12, by + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            if state in _STEER_STATES:
                raw_text = f"finger angle: {drive_state.ema_steering:.3f} rad"
            else:
                raw_text = f"palm metric: {drive_state.ema_throttle:.4f}"
            cv2.putText(frame, raw_text, (10, by - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    # HUD while running
    if state == AppState.RUNNING:
        norm_steer = calibration.normalize_steering(drive_state.ema_steering)
        norm_thr   = calibration.normalize_throttle(drive_state.ema_throttle)
        throttle   = max(0.0,  norm_thr)
        brake      = max(0.0, -norm_thr)
        hud.draw(frame, norm_steer, throttle, brake)

    # FPS / dry-run label
    fps_label = f"{fps:.0f} FPS" + (" [DRY RUN]" if is_dry_run else "")
    cv2.putText(frame, fps_label, (w - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, fps_label, (w - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Key hints (bottom-left; shift right when panel is open to avoid overlap)
    hint_x = 310 if panel_open else 10
    if state in _CALIBRATION_STATES and calibration.waiting_for_ready:
        hints = "[SPACE] Ready  [Q] Quit  [R] Recalibrate  [Tab] Settings"
    else:
        hints = "[Q] Quit  [R] Recalibrate  [S] Save  [P] Pause  [Tab] Settings"
    cv2.putText(frame, hints, (hint_x, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

    return frame


def main() -> None:
    args = _parse_args()
    settings = load_settings()
    config   = _config_from_settings(settings, args.calib_file)

    cap = _open_camera(settings)
    if cap is None:
        logger.error("Could not open camera %d.", settings.camera_index)
        sys.exit(1)

    cv2.namedWindow(_WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(_WIN_NAME, _on_mouse)

    tracker    = HandTracker(flip_camera=config.flip_camera)
    mapper     = GestureMapper(config.smoothing)
    calibration = _load_or_create_calibration(config)
    controller = XInputController(dry_run=args.dry_run)
    hud        = HUDVisualizer()
    panel      = SettingsPanel(settings)

    logger.info(
        "Steer: %s hand (index finger).  Throttle: %s hand (palm orientation).  "
        "Press Tab for settings, Q to quit.",
        config.steer_hand.upper(), config.throttle_hand.upper(),
    )

    app_state         = calibration.current_state
    drive_state       = mapper.get_last()
    fps               = 0.0
    prev_time         = time.perf_counter()
    read_failures     = 0
    no_throttle_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                read_failures += 1
                if read_failures >= 10:
                    logger.error("Camera read failed 10 consecutive times. Exiting.")
                    break
                continue
            read_failures = 0

            if config.flip_camera:
                frame = cv2.flip(frame, 1)

            fh, fw = frame.shape[:2]

            # --- translate raw mouse coords to frame coords ---
            rect = cv2.getWindowImageRect(_WIN_NAME)
            if rect[2] > 0 and rect[3] > 0:
                fx = int((_mouse_pos[0] - rect[0]) * fw / rect[2])
                fy = int((_mouse_pos[1] - rect[1]) * fh / rect[3])
            else:
                fx, fy = _mouse_pos[0], _mouse_pos[1]
            panel.on_mouse(_mouse_event[0], fx, fy)
            _mouse_event[0] = cv2.EVENT_MOUSEMOVE  # consume click

            # --- apply live settings changes from panel ---
            if panel.pending_live:
                panel.pending_live = False
                s = panel.settings
                config.steer_hand    = s.steer_hand
                config.throttle_hand = s.throttle_hand
                config.flip_camera   = s.flip_camera
                config.smoothing.alpha = s.alpha
                mapper._alpha          = s.alpha
                save_settings(s)

            # --- apply camera settings changes from panel ---
            if panel.pending_camera:
                panel.pending_camera = False
                s = panel.settings
                new_cap = _open_camera(s)
                if new_cap is not None:
                    cap.release()
                    cap = new_cap
                    config.camera_index = s.camera_index
                    config.width        = s.width
                    config.height       = s.height
                    config.fps          = s.fps
                    read_failures       = 0
                    save_settings(s)
                else:
                    logger.warning("Could not open camera %d — keeping current.", s.camera_index)

            # --- hand tracking ---
            hands           = tracker.process(frame)
            steer_result    = hands.get(config.steer_hand.capitalize())
            throttle_result = hands.get(config.throttle_hand.capitalize())

            if throttle_result is not None:
                no_throttle_frames = 0
            else:
                no_throttle_frames += 1

            drive_state = mapper.update(steer_result, throttle_result)

            # --- calibration state machine ---
            if app_state != AppState.PAUSED:
                if app_state in _CALIBRATION_STATES:
                    hand_present = (steer_result is not None
                                    if app_state in _STEER_STATES
                                    else throttle_result is not None)
                    app_state = calibration.update(drive_state, hand_present=hand_present)
                    if app_state == AppState.RUNNING:
                        calibration.save(config.calib_file)

            # --- controller output ---
            if app_state == AppState.RUNNING:
                norm_steer = calibration.normalize_steering(drive_state.ema_steering)
                norm_thr   = calibration.normalize_throttle(drive_state.ema_throttle)
                throttle   = max(0.0,  norm_thr)
                brake      = max(0.0, -norm_thr)
                if no_throttle_frames > 60:
                    throttle = 0.0
                controller.write(norm_steer, throttle, brake)
            else:
                controller.write(0.0, 0.0, 0.0)

            # --- FPS ---
            now      = time.perf_counter()
            dt       = max(now - prev_time, 1e-6)
            fps      = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = now

            # --- render ---
            frame = _render_overlay(
                frame, app_state, drive_state, calibration,
                hands, tracker, fps, controller.is_dry_run,
                no_throttle_frames, hud, panel.is_open,
            )
            if panel.is_open:
                panel.draw(frame)

            cv2.imshow(_WIN_NAME, frame)

            # --- keyboard ---
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == 9:  # Tab
                panel.toggle()
            elif key == ord(" "):
                if app_state in _CALIBRATION_STATES and calibration.waiting_for_ready:
                    calibration.confirm_ready()
            elif key in (ord("r"), ord("R")):
                logger.info("Restarting calibration.")
                calibration.reset()
                app_state = calibration.current_state
            elif key in (ord("s"), ord("S")):
                if calibration.current_state == AppState.RUNNING:
                    calibration.save(config.calib_file)
                else:
                    logger.info("Calibration not complete yet — cannot save.")
            elif key in (ord("p"), ord("P")):
                if app_state == AppState.PAUSED:
                    app_state = calibration.current_state
                    logger.info("Resumed.")
                elif app_state == AppState.RUNNING:
                    app_state = AppState.PAUSED
                    logger.info("Paused.")

    finally:
        controller.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
