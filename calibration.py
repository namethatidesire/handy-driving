from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from config import AppConfig, AppState, CalibrationData
from mapper import DriveState

logger = logging.getLogger(__name__)


class CalibrationLoadError(Exception):
    pass


class CalibrationManager:
    SCHEMA_VERSION = 1

    _PHASES = [
        AppState.CALIB_STEER_LEFT,
        AppState.CALIB_STEER_RIGHT,
        AppState.CALIB_THROTTLE_MAX,
        AppState.CALIB_BRAKE_MAX,
    ]

    def __init__(
        self,
        config: AppConfig,
        data: Optional[CalibrationData] = None,
        start_running: bool = False,
    ):
        self._config = config
        self._data = data or CalibrationData(
            steer_hand=config.steer_hand,
            throttle_hand=config.throttle_hand,
        )
        self._phase_index = len(self._PHASES) if start_running else 0
        self._hold_count = 0
        self._accumulated: list[float] = []
        self._waiting_for_ready: bool = True

    @property
    def current_state(self) -> AppState:
        if self._phase_index >= len(self._PHASES):
            return AppState.RUNNING
        return self._PHASES[self._phase_index]

    @property
    def waiting_for_ready(self) -> bool:
        return self._waiting_for_ready

    @property
    def hold_progress(self) -> float:
        return min(1.0, self._hold_count / max(1, self._config.calib_hold_frames))

    def confirm_ready(self) -> None:
        """Call when the user signals they are in position for the current phase."""
        self._waiting_for_ready = False
        self._hold_count = 0
        self._accumulated.clear()

    def update(self, drive_state: DriveState, hand_present: bool = True) -> AppState:
        if self.current_state == AppState.RUNNING:
            return AppState.RUNNING

        if self._waiting_for_ready:
            return self.current_state

        if not hand_present:
            # Reset hold counter so the user must restart the hold gesture
            self._hold_count = 0
            self._accumulated.clear()
            return self.current_state

        phase = self.current_state
        val = drive_state.ema_steering if phase in (
            AppState.CALIB_STEER_LEFT, AppState.CALIB_STEER_RIGHT
        ) else drive_state.ema_throttle

        self._hold_count += 1
        self._accumulated.append(val)

        if self._hold_count >= self._config.calib_hold_frames:
            # Use median of accumulated values for robustness
            locked = sorted(self._accumulated)[len(self._accumulated) // 2]
            self._apply(phase, locked)
            self._advance()

        return self.current_state

    def _apply(self, phase: AppState, value: float) -> None:
        if phase == AppState.CALIB_STEER_LEFT:
            self._data.steer_min = value
        elif phase == AppState.CALIB_STEER_RIGHT:
            self._data.steer_max = value
        elif phase == AppState.CALIB_THROTTLE_MAX:
            self._data.throttle_max = value
        elif phase == AppState.CALIB_BRAKE_MAX:
            self._data.throttle_min = value

    def _advance(self) -> None:
        self._phase_index += 1
        self._hold_count = 0
        self._accumulated.clear()
        self._waiting_for_ready = True
        if self.current_state == AppState.RUNNING:
            logger.info("Calibration complete: %s", self._data)

    def reset(self) -> None:
        self._phase_index = 0
        self._hold_count = 0
        self._accumulated.clear()
        self._waiting_for_ready = True
        self._data = CalibrationData(
            steer_hand=self._config.steer_hand,
            throttle_hand=self._config.throttle_hand,
        )

    def normalize_steering(self, raw: float) -> float:
        return _normalize(raw, self._data.steer_min, self._data.steer_max, -1.0, 1.0)

    def normalize_throttle(self, raw: float) -> float:
        return _normalize(raw, self._data.throttle_min, self._data.throttle_max, -1.0, 1.0)

    def save(self, path: str) -> None:
        d = asdict(self._data)
        d["schema_version"] = self.SCHEMA_VERSION
        Path(path).write_text(json.dumps(d, indent=2))
        logger.info("Calibration saved to %s", path)

    @classmethod
    def load(cls, path: str, config: AppConfig) -> CalibrationManager:
        try:
            raw = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise CalibrationLoadError(f"Cannot read calibration file: {e}") from e

        try:
            data = CalibrationData(
                steer_min=raw["steer_min"],
                steer_max=raw["steer_max"],
                throttle_min=raw["throttle_min"],
                throttle_max=raw["throttle_max"],
                steer_hand=raw.get("steer_hand", config.steer_hand),
                throttle_hand=raw.get("throttle_hand", config.throttle_hand),
            )
        except KeyError as e:
            raise CalibrationLoadError(f"Calibration file missing key: {e}") from e

        return cls(config=config, data=data, start_running=True)


def _normalize(raw: float, raw_min: float, raw_max: float, out_min: float, out_max: float) -> float:
    if raw_max == raw_min:
        return (out_min + out_max) / 2.0
    t = (raw - raw_min) / (raw_max - raw_min)
    t = max(0.0, min(1.0, t))
    return out_min + t * (out_max - out_min)
