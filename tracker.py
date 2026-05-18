from __future__ import annotations

import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# Connection list for drawing — (start_index, end_index) pairs
_CONNECTIONS = [
    (c.start, c.end)
    for c in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
]


def _ensure_model() -> str:
    """Download the hand landmark model bundle (~8 MB) if not already present."""
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)
    print(f"Downloading hand landmark model (~8 MB) to {_MODEL_PATH.name} ...", flush=True)
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not download the hand landmark model: {e}\n"
            f"Download it manually from:\n  {_MODEL_URL}\n"
            f"and place it at: {_MODEL_PATH}"
        ) from e
    print("Model ready.", flush=True)
    return str(_MODEL_PATH)


@dataclass
class HandResult:
    landmarks: list  # 21 (x, y, z) tuples, normalized [0, 1]
    handedness: str  # "Left" or "Right"


class HandTracker:
    def __init__(self, flip_camera: bool = True):
        self._t0 = time.perf_counter()

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_ensure_model()),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _ts(self) -> int:
        """Monotonically increasing timestamp in milliseconds."""
        return int((time.perf_counter() - self._t0) * 1000)

    def process(self, bgr_frame: np.ndarray) -> dict[str, HandResult]:
        """Returns all detected hands keyed by 'Left' or 'Right'."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._ts())

        hands: dict[str, HandResult] = {}
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
            hands[label] = HandResult(landmarks=landmarks, handedness=label)
        return hands

    def draw_landmarks(self, bgr_frame: np.ndarray, result: HandResult) -> np.ndarray:
        h, w = bgr_frame.shape[:2]
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in result.landmarks]

        for start, end in _CONNECTIONS:
            cv2.line(bgr_frame, pts[start], pts[end], (0, 220, 0), 2, cv2.LINE_AA)

        for pt in pts:
            cv2.circle(bgr_frame, pt, 5, (255, 255, 255), -1)
            cv2.circle(bgr_frame, pt, 5, (0, 150, 0), 1)

        return bgr_frame

    def close(self) -> None:
        self._landmarker.close()
