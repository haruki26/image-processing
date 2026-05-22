from pathlib import Path
from typing import Final

import cv2

VIDEO_PATH: Final[Path] = Path(__file__).parent.parent.parent / "movies" / "moving_vehicles.mp4"

STOP_KEY: Final[int] = ord("s")
PLAY_KEY: Final[int] = ord("p")
ESC_KEY: Final[int] = 27

WINDOW_NAME: Final[str] = "camera"


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    rect: tuple[int, int, int, int] | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if rect is not None:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), thickness=2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(int(1000 / frame_rate))
        if key == ESC_KEY:  # ESC
            break
        if key == STOP_KEY:  # stopping
            x, y, w, h = cv2.selectROI(WINDOW_NAME, frame, fromCenter=False, showCrosshair=False)
            rect = (x, y, w, h)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
