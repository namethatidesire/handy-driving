"""OpenCV-drawn settings panel rendered as an overlay on the video frame."""
from __future__ import annotations

import dataclasses
from typing import Callable

import cv2
import numpy as np

from settings import Settings

# --- layout constants (frame pixels) ---
_W      = 300   # panel width
_PAD    = 14    # horizontal inner padding
_ROW    = 34    # standard control-row height
_BTN_H  = 26    # button height
_HDR_H  = 42    # header height
_SEC_H  = 26    # section-label height
_DIV_H  = 10    # divider height

# --- colours (BGR) ---
_BG       = (22,  22,  22)
_LINE     = (55,  55,  55)
_SEC_CLR  = (90,  90,  90)
_ACTIVE   = (0,  200, 255)
_INACTIVE = (55,  55,  55)
_HOVER    = (80,  80,  80)
_WHITE    = (220, 220, 220)
_DIM      = (130, 130, 130)
_APPLY_C  = (30, 140,  60)


class _Btn:
    __slots__ = ("x", "y", "w", "h", "action")

    def __init__(self, x: int, y: int, w: int, h: int, action: Callable):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.action = action

    def hit(self, mx: int, my: int) -> bool:
        return self.x <= mx < self.x + self.w and self.y <= my < self.y + self.h


class SettingsPanel:
    """
    Draws a settings overlay on the right edge of the video frame.
    Toggle with Tab; interact with the mouse.
    Pending changes are signalled via `pending_live` and `pending_camera`.
    """

    def __init__(self, settings: Settings):
        self._s     = dataclasses.replace(settings)
        self._open  = False
        self._mx    = -1
        self._my    = -1
        self._btns: list[_Btn] = []
        self.pending_live   = False
        self.pending_camera = False

    # ------------------------------------------------------------------ public

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def settings(self) -> Settings:
        return dataclasses.replace(self._s)

    def toggle(self) -> None:
        self._open = not self._open

    def close(self) -> None:
        self._open = False

    def on_mouse(self, event: int, fx: int, fy: int) -> None:
        """Call every frame with frame-space coordinates."""
        self._mx, self._my = fx, fy
        if event == cv2.EVENT_LBUTTONDOWN and self._open:
            for btn in self._btns:
                if btn.hit(fx, fy):
                    btn.action()
                    return

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if not self._open:
            return frame
        self._btns = []
        fh, fw = frame.shape[:2]
        x0 = fw - _W
        self._render(frame, x0, fh)
        return frame

    # ------------------------------------------------------------------ render

    def _render(self, frame: np.ndarray, x0: int, fh: int) -> None:
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, 0), (x0 + _W, fh), _BG, -1)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
        cv2.line(frame, (x0, 0), (x0, fh), _LINE, 1)

        cy = 0

        cy = self._header(frame, x0, cy)
        cy = self._divider(frame, x0, cy)

        cy = self._section(frame, x0, cy, "CAMERA")
        cy = self._stepper(frame, x0, cy, "Index",
                           str(self._s.camera_index),
                           self._dec_idx, self._inc_idx)
        cy = self._res_buttons(frame, x0, cy)
        cy = self._choice_row(frame, x0, cy, "FPS",
                              [("30", 30), ("60", 60), ("120", 120)],
                              self._s.fps, self._set_fps,
                              btn_w=56)
        cy = self._divider(frame, x0, cy)

        cy = self._section(frame, x0, cy, "CONTROLS")
        cy = self._choice_row(frame, x0, cy, "Steer hand",
                              [("Left", "left"), ("Right", "right")],
                              self._s.steer_hand, self._set_steer)
        cy = self._choice_row(frame, x0, cy, "Throttle hand",
                              [("Left", "left"), ("Right", "right")],
                              self._s.throttle_hand, self._set_throttle)
        cy = self._choice_row(frame, x0, cy, "Mirror",
                              [("On", True), ("Off", False)],
                              self._s.flip_camera, self._set_flip)
        cy = self._divider(frame, x0, cy)

        cy = self._section(frame, x0, cy, "SMOOTHING")
        cy = self._stepper(frame, x0, cy, "Alpha",
                           f"{self._s.alpha:.2f}",
                           self._dec_alpha, self._inc_alpha)
        cy = self._divider(frame, x0, cy)

        self._apply_button(frame, x0, cy)

    # ------------------------------------------------------------------ widgets

    def _header(self, frame: np.ndarray, x0: int, cy: int) -> int:
        cv2.putText(frame, "SETTINGS", (x0 + _PAD, cy + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, _WHITE, 1, cv2.LINE_AA)
        bx, by, bsz = x0 + _W - 38, cy + 8, 26
        hov = self._hov(bx, by, bsz, bsz)
        cv2.rectangle(frame, (bx, by), (bx + bsz, by + bsz),
                      _HOVER if hov else _INACTIVE, -1)
        cv2.putText(frame, "X", (bx + 8, by + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, _WHITE, 1, cv2.LINE_AA)
        self._btns.append(_Btn(bx, by, bsz, bsz, self.close))
        return cy + _HDR_H

    def _divider(self, frame: np.ndarray, x0: int, cy: int) -> int:
        y = cy + 4
        cv2.line(frame, (x0 + _PAD, y), (x0 + _W - _PAD, y), _LINE, 1)
        return cy + _DIV_H

    def _section(self, frame: np.ndarray, x0: int, cy: int, label: str) -> int:
        cv2.putText(frame, label, (x0 + _PAD, cy + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, _SEC_CLR, 1, cv2.LINE_AA)
        return cy + _SEC_H

    def _stepper(self, frame: np.ndarray, x0: int, cy: int,
                 label: str, value: str,
                 dec_fn: Callable, inc_fn: Callable) -> int:
        cv2.putText(frame, label, (x0 + _PAD, cy + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, _DIM, 1, cv2.LINE_AA)
        bw = 28
        val_w = 54
        right_edge = x0 + _W - _PAD
        by = cy + (_ROW - _BTN_H) // 2

        # inc [>] on far right
        ix = right_edge - bw
        self._draw_btn(frame, ix, by, bw, _BTN_H, ">", inc_fn)

        # value text
        vx = ix - val_w
        (tw, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.putText(frame, value, (vx + (val_w - tw) // 2, by + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, _WHITE, 1, cv2.LINE_AA)

        # dec [<]
        dx = vx - bw
        self._draw_btn(frame, dx, by, bw, _BTN_H, "<", dec_fn)

        return cy + _ROW

    def _choice_row(self, frame: np.ndarray, x0: int, cy: int,
                    label: str, options: list, active,
                    setter: Callable, btn_w: int = 64) -> int:
        cv2.putText(frame, label, (x0 + _PAD, cy + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, _DIM, 1, cv2.LINE_AA)
        gap = 4
        n = len(options)
        total = n * btn_w + (n - 1) * gap
        bx0 = x0 + _W - _PAD - total
        by = cy + (_ROW - _BTN_H) // 2
        for i, (lbl, val) in enumerate(options):
            bx = bx0 + i * (btn_w + gap)
            is_on = (val == active)
            hov = self._hov(bx, by, btn_w, _BTN_H)
            fill = _ACTIVE if is_on else (_HOVER if hov else _INACTIVE)
            cv2.rectangle(frame, (bx, by), (bx + btn_w, by + _BTN_H), fill, -1)
            tc = (20, 20, 20) if is_on else _WHITE
            (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.putText(frame, lbl, (bx + (btn_w - tw) // 2, by + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, tc, 1, cv2.LINE_AA)
            v = val
            self._btns.append(_Btn(bx, by, btn_w, _BTN_H, lambda v=v: setter(v)))
        return cy + _ROW

    def _res_buttons(self, frame: np.ndarray, x0: int, cy: int) -> int:
        """2×2 grid of resolution preset buttons."""
        cv2.putText(frame, "Resolution", (x0 + _PAD, cy + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, _DIM, 1, cv2.LINE_AA)
        presets = [("720p",  1280,  720),
                   ("1080p", 1920, 1080),
                   ("1440p", 2560, 1440),
                   ("4K",    3840, 2160)]
        bw, bh, gap = 60, _BTN_H, 4
        # right-align the 2-wide grid
        right_edge = x0 + _W - _PAD
        col1_x = right_edge - bw
        col0_x = col1_x - gap - bw
        row0_y = cy + 4
        row1_y = row0_y + bh + 5

        for i, (lbl, pw, ph) in enumerate(presets):
            bx = col0_x if i % 2 == 0 else col1_x
            by = row0_y if i < 2 else row1_y
            is_on = (self._s.width == pw and self._s.height == ph)
            hov = self._hov(bx, by, bw, bh)
            fill = _ACTIVE if is_on else (_HOVER if hov else _INACTIVE)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), fill, -1)
            tc = (20, 20, 20) if is_on else _WHITE
            (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(frame, lbl, (bx + (bw - tw) // 2, by + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, tc, 1, cv2.LINE_AA)
            w_, h_ = pw, ph
            self._btns.append(_Btn(bx, by, bw, bh, lambda w=w_, h=h_: self._set_res(w, h)))

        return cy + 4 + bh + 5 + bh + 6

    def _apply_button(self, frame: np.ndarray, x0: int, cy: int) -> None:
        bx = x0 + _PAD
        bw = _W - 2 * _PAD
        bh = 32
        by = cy + 8
        hov = self._hov(bx, by, bw, bh)
        fill = tuple(min(255, int(c * 1.25)) for c in _APPLY_C) if hov else _APPLY_C
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), fill, -1)
        lbl = "APPLY CAMERA SETTINGS"
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(frame, lbl, (bx + (bw - tw) // 2, by + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1, cv2.LINE_AA)
        self._btns.append(_Btn(bx, by, bw, bh, self._apply_camera))

    def _draw_btn(self, frame: np.ndarray, bx: int, by: int,
                  bw: int, bh: int, label: str, action: Callable) -> None:
        hov = self._hov(bx, by, bw, bh)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),
                      _HOVER if hov else _INACTIVE, -1)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.putText(frame, label, (bx + (bw - tw) // 2, by + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, _WHITE, 1, cv2.LINE_AA)
        self._btns.append(_Btn(bx, by, bw, bh, action))

    # ------------------------------------------------------------------ helpers

    def _hov(self, bx: int, by: int, bw: int, bh: int) -> bool:
        return bx <= self._mx < bx + bw and by <= self._my < by + bh

    # ------------------------------------------------------------------ actions (live)

    def _set_steer(self, val: str) -> None:
        self._s.steer_hand = val
        self.pending_live = True

    def _set_throttle(self, val: str) -> None:
        self._s.throttle_hand = val
        self.pending_live = True

    def _set_flip(self, val: bool) -> None:
        self._s.flip_camera = val
        self.pending_live = True

    def _dec_alpha(self) -> None:
        self._s.alpha = max(0.05, round(self._s.alpha - 0.05, 2))
        self.pending_live = True

    def _inc_alpha(self) -> None:
        self._s.alpha = min(1.00, round(self._s.alpha + 0.05, 2))
        self.pending_live = True

    # ------------------------------------------------------------------ actions (camera, staged)

    def _dec_idx(self) -> None:
        if self._s.camera_index > 0:
            self._s.camera_index -= 1

    def _inc_idx(self) -> None:
        self._s.camera_index += 1

    def _set_res(self, w: int, h: int) -> None:
        self._s.width, self._s.height = w, h

    def _set_fps(self, fps: int) -> None:
        self._s.fps = fps

    def _apply_camera(self) -> None:
        self.pending_camera = True
