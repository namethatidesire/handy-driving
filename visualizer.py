"""Lightweight driving HUD: steering arc + throttle/brake bars."""

from __future__ import annotations

import math

import cv2
import numpy as np

# BGR colours
_CYAN    = (255, 210,  20)   # steering active sweep
_GREEN   = ( 30, 200,  60)   # throttle
_RED     = ( 30,  60, 220)   # brake
_GRAY    = (160, 160, 160)
_DARK    = ( 40,  40,  40)
_DIMMED  = ( 75,  75,  75)
_WHITE   = (230, 230, 230)

# Steering arc spans ±_ARC_HALF degrees from straight-up (270° in OpenCV CW convention)
_ARC_HALF  = 70   # degrees — total 140° sweep
_ARC_START = 270 - _ARC_HALF   # 200°
_ARC_END   = 270 + _ARC_HALF   # 340°


class HUDVisualizer:
    """
    Renders a panel at the bottom of the frame showing:
    - Steering arc with needle
    - Throttle bar (right of arc, green)
    - Brake bar (left of arc, red)
    """

    def __init__(self, radius: int = 75, panel_alpha: float = 0.60, bar_width: int = 22):
        self._r = radius
        self._alpha = panel_alpha
        self._bw = bar_width

    def draw(
        self,
        frame: np.ndarray,
        steering: float,   # normalised [-1, 1]
        throttle: float,   # normalised [0, 1]
        brake: float,      # normalised [0, 1]
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        r, bw = self._r, self._bw

        # --- semi-transparent background panel ---
        panel_h = r + 52
        panel_top = h - panel_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_top), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, self._alpha, frame, 1.0 - self._alpha, 0, frame)
        cv2.line(frame, (0, panel_top), (w, panel_top), (55, 55, 55), 1)

        # arc centre sits right at the bottom edge; arc opens upward
        cx = w // 2
        cy = h - 14

        gap = 14   # gap between arc edge and bar
        bar_x_left  = cx - r - bw - gap      # brake bar left edge
        bar_x_right = cx + r + gap            # throttle bar left edge

        self._draw_arc(frame, cx, cy, r, steering)
        self._draw_bar(frame, bar_x_left,  cy, r, brake,    label="B", color=_RED)
        self._draw_bar(frame, bar_x_right, cy, r, throttle, label="T", color=_GREEN)

        return frame

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_arc(
        self,
        frame: np.ndarray,
        cx: int,
        cy: int,
        r: int,
        steering: float,
    ) -> None:
        # Background track
        cv2.ellipse(frame, (cx, cy), (r, r), 0, _ARC_START, _ARC_END,
                    _DARK, 10, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (r, r), 0, _ARC_START, _ARC_END,
                    _DIMMED, 1, cv2.LINE_AA)

        # Active sweep from neutral (270°) toward current angle
        steer_cw = 270.0 + steering * _ARC_HALF   # maps [-1,1] → [200,340]
        if abs(steering) > 0.01:
            sa, ea = (int(steer_cw), 270) if steering < 0 else (270, int(steer_cw))
            cv2.ellipse(frame, (cx, cy), (r, r), 0, sa, ea, _CYAN, 8, cv2.LINE_AA)

        # Centre-tick marks at each end and at neutral
        for tick_angle in (_ARC_START, 270, _ARC_END):
            self._tick(frame, cx, cy, r, tick_angle, length=6)

        # Needle
        angle_rad = math.radians(steer_cw)
        nx = cx + int(r * math.cos(angle_rad))
        ny = cy + int(r * math.sin(angle_rad))
        cv2.line(frame, (cx, cy), (nx, ny), _WHITE, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, _WHITE, -1, cv2.LINE_AA)

        # Direction & percentage label centred above the arc
        direction = "L" if steering < -0.03 else ("R" if steering > 0.03 else "—")
        label = f"{direction}  {int(abs(steering) * 100):3d}%"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        lx = cx - tw // 2
        ly = cy - r - 10
        cv2.putText(frame, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _CYAN, 1, cv2.LINE_AA)

    @staticmethod
    def _tick(
        frame: np.ndarray,
        cx: int,
        cy: int,
        r: int,
        angle_cw: float,
        length: int,
    ) -> None:
        rad = math.radians(angle_cw)
        ox, oy = math.cos(rad), math.sin(rad)
        x0 = cx + int((r - length) * ox)
        y0 = cy + int((r - length) * oy)
        x1 = cx + int((r + length) * ox)
        y1 = cy + int((r + length) * oy)
        cv2.line(frame, (x0, y0), (x1, y1), _GRAY, 1, cv2.LINE_AA)

    def _draw_bar(
        self,
        frame: np.ndarray,
        x: int,
        cy: int,
        height: int,
        value: float,
        label: str,
        color: tuple,
    ) -> None:
        bw = self._bw
        top = cy - height

        # Track
        cv2.rectangle(frame, (x, top), (x + bw, cy), _DARK, -1)
        cv2.rectangle(frame, (x, top), (x + bw, cy), _DIMMED, 1)

        # Fill from bottom up
        fill_h = int(max(0.0, min(1.0, value)) * height)
        if fill_h > 0:
            cv2.rectangle(frame, (x, cy - fill_h), (x + bw, cy), color, -1)

        # Label letter
        cv2.putText(frame, label, (x + bw // 2 - 5, top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Percentage
        pct = f"{int(value * 100)}%"
        (tw, _), _ = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.putText(frame, pct, (x + bw // 2 - tw // 2, top - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, _GRAY, 1, cv2.LINE_AA)
