from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from config import SmoothingConfig
from tracker import HandResult


@dataclass
class DriveState:
    steering_raw: float
    throttle_raw: float
    ema_steering: float
    ema_throttle: float


def _steering_raw(lm: list) -> float:
    """Angle of index finger (MCP→tip) from horizontal, in radians."""
    return math.atan2(-(lm[8][1] - lm[5][1]), lm[8][0] - lm[5][0])


def _throttle_raw(lm: list) -> float:
    """
    Z-component of (wrist→index_MCP) × (wrist→pinky_MCP).
    Positive when palm faces the camera; negative when back of hand faces it.
    The calibration maps the extremes to [-1, 1].
    """
    v1x = lm[5][0] - lm[0][0]
    v1y = lm[5][1] - lm[0][1]
    v2x = lm[17][0] - lm[0][0]
    v2y = lm[17][1] - lm[0][1]
    return v1x * v2y - v1y * v2x


class GestureMapper:
    def __init__(self, config: SmoothingConfig):
        self._alpha = config.alpha
        self._ema_steer = 0.0
        self._ema_throttle = 0.0

    def update(
        self,
        steer_hand: Optional[HandResult],
        throttle_hand: Optional[HandResult],
    ) -> DriveState:
        if steer_hand is not None:
            raw = _steering_raw(steer_hand.landmarks)
            self._ema_steer = self._alpha * raw + (1 - self._alpha) * self._ema_steer
        else:
            raw = self._ema_steer  # hold last value

        if throttle_hand is not None:
            traw = _throttle_raw(throttle_hand.landmarks)
            self._ema_throttle = self._alpha * traw + (1 - self._alpha) * self._ema_throttle
        else:
            traw = self._ema_throttle
            self._ema_throttle *= (1 - self._alpha)  # decay toward 0

        return DriveState(
            steering_raw=raw,
            throttle_raw=traw,
            ema_steering=self._ema_steer,
            ema_throttle=self._ema_throttle,
        )

    def get_last(self) -> DriveState:
        return DriveState(
            steering_raw=self._ema_steer,
            throttle_raw=self._ema_throttle,
            ema_steering=self._ema_steer,
            ema_throttle=self._ema_throttle,
        )
