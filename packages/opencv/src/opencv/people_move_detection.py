from pathlib import Path
from typing import Final

import cv2

from shared.color import RGBColor

VIDEO_PATH: Final[Path] = Path(__file__).parent.parent.parent / "movies" / "people_move.mp4"

BOX_COLOR: Final[RGBColor] = RGBColor(255, 255, 0)
DETECTION_AREA_THRESHOLD: Final[int] = 500

ESC_KEY: Final[int] = 27


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    avg = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if avg is None:
            avg = gray_frame.copy().astype("float")

        cv2.accumulateWeighted(gray_frame, avg, 0.5)

        frame_delta = cv2.absdiff(gray_frame, cv2.convertScaleAbs(avg))
        bin_frame_delta = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(bin_frame_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < DETECTION_AREA_THRESHOLD:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR.get_tuple(), thickness=2)

        cv2.imshow("camera", frame)

        key = cv2.waitKey(int(1000 / frame_rate))
        if key == ESC_KEY:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
