from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from shared.color import RGBColor

nrg = np.random.default_rng()

VIDEO_PATH: Final[Path] = Path(__file__).parent.parent.parent / "movies" / "moving_peoples.mp4"

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

TRACKING_VIEWER_COLOR: Final[RGBColor] = RGBColor(255, 255, 0)

CASCADE_PATH: Final[Path] = Path(__file__).parent / "cascade" / "haarcascade_fullbody.xml"

SCALE_FACTOR: Final[float] = 1.1
MIN_NEIGHBORS: Final[int] = 2
MIN_SIZE: Final[tuple[int, int]] = (10, 10)


def get_tracker_from_input() -> TrackerType:
    while True:
        print("Select tracker type:")
        for idx, tracker_name in enumerate(AVAILABLE_TRACKERS):
            print(f"{idx}: {tracker_name}")
        try:
            tracker_id = int(input("Enter tracker ID: "))
            return TrackerType[AVAILABLE_TRACKERS[tracker_id]]
        except ValueError, KeyError:
            print("Invalid input. Please enter a valid tracker ID.")


def generate_color() -> RGBColor:
    return RGBColor(*map(int, nrg.integers(0, 256, size=3)))


def detect_bodies(frame: np.ndarray, cascade: cv2.CascadeClassifier) -> Sequence[Sequence[int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=MIN_SIZE)


def main() -> None:
    tracker_type = get_tracker_from_input()
    multi_tracker = cv2.legacy.MultiTracker.create()
    rects: list[list[Sequence[int]]] = []
    colors = []

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    ret, frame = cap.read()

    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    bodies = detect_bodies(frame, cascade)
    for rect in bodies:
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), TRACKING_VIEWER_COLOR.get_tuple(), thickness=2)
        rects.append([rect])
        colors.append(generate_color())

        tracker = tracker_type.get_tracker()
        multi_tracker.add(tracker, frame, rect)

    print(f"Tracking {len(rects)} bodies")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        _, bbox = multi_tracker.update(frame)
        for i, rect in enumerate(bbox):
            x, y, w, h = map(int, rect)
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color.get_tuple(), thickness=2)

            rects[i].append((x, y, w, h))
            for k, rect in enumerate(rects[i][:-1]):
                next_rect = rects[i][k + 1]
                cv2.line(
                    frame,
                    (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2),
                    (next_rect[0] + next_rect[2] // 2, next_rect[1] + next_rect[3] // 2),
                    color.get_tuple(),
                    thickness=2,
                )

        cv2.imshow("frame", frame)

        key = cv2.waitKey(int(1000 / frame_rate))
        if key == ESC_KEY:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
