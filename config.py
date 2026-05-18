from dataclasses import dataclass, field
from enum import Enum, auto


class AppState(Enum):
    CALIB_STEER_LEFT = auto()
    CALIB_STEER_RIGHT = auto()
    CALIB_THROTTLE_MAX = auto()
    CALIB_BRAKE_MAX = auto()
    RUNNING = auto()
    PAUSED = auto()


@dataclass
class CalibrationData:
    steer_min: float = -1.0
    steer_max: float = 1.0
    throttle_min: float = -0.05
    throttle_max: float = 0.05
    steer_hand: str = "right"
    throttle_hand: str = "left"


@dataclass
class SmoothingConfig:
    alpha: float = 0.25


@dataclass
class AppConfig:
    camera_index: int = 0
    calib_hold_frames: int = 30
    calib_file: str = "calibration.json"
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    steer_hand: str = "right"
    throttle_hand: str = "left"
    flip_camera: bool = True
    width: int = 2560
    height: int = 1440
    fps: int = 60


XINPUT_STICK_MIN = -32768
XINPUT_STICK_MAX = 32767
XINPUT_TRIGGER_MAX = 255
