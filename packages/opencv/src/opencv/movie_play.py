from pathlib import Path
from typing import Final

import cv2

VIDEO_PATH: Final[Path] = Path(__file__).parent.parent.parent / "movies" / "people_move.mp4"

STOP_KEY: Final[int] = ord("s")
PLAY_KEY: Final[int] = ord("p")
ESC_KEY: Final[int] = 27


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("camera", frame)

        key = cv2.waitKey(int(1000 / frame_rate))
        if key == ESC_KEY:  # ESC
            break
        if key == STOP_KEY:  # stopping
            while True:
                key2 = cv2.waitKey(10)
                if key2 == PLAY_KEY:  # playing
                    break
                if key2 == ESC_KEY:  # ESC
                    return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
