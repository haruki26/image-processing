from enum import Enum
from pathlib import Path
from typing import Final

import cv2

VIDEO_PATH: Final[Path] = Path(__file__).parent.parent.parent / "movies" / "moving_vehicles.mp4"

STOP_KEY: Final[int] = ord("s")
PLAY_KEY: Final[int] = ord("p")
ESC_KEY: Final[int] = 27

WINDOW_NAME: Final[str] = "camera"


class TrackerType(Enum):
    BOOSTING = cv2.legacy.TrackerBoosting.create
    MIL = cv2.legacy.TrackerMIL.create
    KCF = cv2.legacy.TrackerKCF.create
    TLD = cv2.legacy.TrackerTLD.create
    MEDIANFLOW = cv2.legacy.TrackerMedianFlow.create
    GOTURN = cv2.TrackerGOTURN.create
    MOSSE = cv2.legacy.TrackerMOSSE.create
    CSRT = cv2.legacy.TrackerCSRT.create

    def get_tracker(self) -> cv2.Tracker:
        if self == TrackerType.GOTURN:
            return self.value("goturn.caffemodel", "goturn.prototxt")
        return self.value()


AVAILABLE_TRACKERS: Final[list[str]] = [tracker.name for tracker in TrackerType]


def get_tracker_from_input() -> cv2.Tracker:
    while True:
        print("Select tracker type:")
        for idx, tracker_name in enumerate(AVAILABLE_TRACKERS):
            print(f"{idx}: {tracker_name}")
        try:
            tracker_id = int(input("Enter tracker ID: "))
            tracker_type = TrackerType[AVAILABLE_TRACKERS[tracker_id]]
            return tracker_type.get_tracker()
        except ValueError, KeyError:
            print("Invalid input. Please enter a valid tracker ID.")


def main() -> None:
    tracker = get_tracker_from_input()

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    is_tracking = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if is_tracking:
            is_tracking, raw_rect = tracker.update(frame)
            x, y, w, h = map(int, raw_rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), thickness=2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(int(1000 / frame_rate))
        if key == ESC_KEY:  # ESC
            break
        if key == STOP_KEY:  # stopping
            rect = cv2.selectROI(WINDOW_NAME, frame, fromCenter=False, showCrosshair=False)
            tracker.init(frame, rect)
            is_tracking = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
