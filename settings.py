from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, fields
from pathlib import Path

logger = logging.getLogger(__name__)

_PATH = Path(__file__).parent / "settings.json"


@dataclass
class Settings:
    camera_index: int = 0
    width: int = 2560
    height: int = 1440
    fps: int = 60
    steer_hand: str = "right"
    throttle_hand: str = "left"
    flip_camera: bool = True
    alpha: float = 0.25


def load_settings() -> Settings:
    try:
        raw = json.loads(_PATH.read_text())
        valid_keys = {f.name for f in fields(Settings)}
        return Settings(**{k: v for k, v in raw.items() if k in valid_keys})
    except Exception:
        return Settings()


def save_settings(s: Settings) -> None:
    try:
        _PATH.write_text(json.dumps(asdict(s), indent=2))
    except Exception as e:
        logger.warning("Could not save settings: %s", e)
