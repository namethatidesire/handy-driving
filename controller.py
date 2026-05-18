from __future__ import annotations

import logging

from config import XINPUT_STICK_MAX, XINPUT_STICK_MIN, XINPUT_TRIGGER_MAX

logger = logging.getLogger(__name__)

_vg = None
_VIGEM_AVAILABLE = False

try:
    import vgamepad as _vg
    _VIGEM_AVAILABLE = True
except Exception:
    pass


class XInputController:
    def __init__(self, dry_run: bool = False):
        self._dry_run = dry_run
        self._pad = None

        if not dry_run and not _VIGEM_AVAILABLE:
            logger.warning(
                "vgamepad could not be imported. Falling back to dry-run mode.\n"
                "To enable XInput output:\n"
                "  1. Install ViGEmBus: https://github.com/nefarius/ViGEmBus/releases\n"
                "  2. Run: pip install vgamepad\n"
                "Or start with --dry-run to suppress this message."
            )
            self._dry_run = True

        if not self._dry_run:
            try:
                self._pad = _vg.VX360Gamepad()
                # Send an immediate neutral state so the controller appears active
                # to games that enumerate XInput slots at startup.
                self._pad.reset()
                self._pad.update()
                logger.info("Virtual Xbox 360 controller created.")
            except Exception as e:
                logger.warning(
                    "Failed to create virtual controller (%s). Falling back to dry-run.", e
                )
                self._dry_run = True

    @property
    def is_dry_run(self) -> bool:
        return self._dry_run

    def write(self, steering: float, throttle: float, brake: float) -> None:
        steering_int = int(max(XINPUT_STICK_MIN, min(XINPUT_STICK_MAX, steering * XINPUT_STICK_MAX)))
        throttle_int = int(max(0, min(XINPUT_TRIGGER_MAX, throttle * XINPUT_TRIGGER_MAX)))
        brake_int = int(max(0, min(XINPUT_TRIGGER_MAX, brake * XINPUT_TRIGGER_MAX)))

        if self._dry_run:
            print(
                f"\rSTEER: {steering:+.3f}  THROTTLE: {throttle:.3f}  BRAKE: {brake:.3f}   ",
                end="",
                flush=True,
            )
            return

        self._pad.left_joystick(x_value=steering_int, y_value=0)
        self._pad.right_trigger(value=throttle_int)
        self._pad.left_trigger(value=brake_int)
        self._pad.update()

    def close(self) -> None:
        if self._dry_run:
            print()
            return
        if self._pad is not None:
            try:
                self._pad.reset()
                self._pad.update()
            except Exception:
                pass
