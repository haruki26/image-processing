from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from shared.color import RGBColor

from opencv.utils.select_rois import select_rois

if TYPE_CHECKING:
    from collections.abc import Sequence

rnd = np.random.default_rng()

VIDEO_PATH: Final[Path] = Path(__file__).parent.parent.parent / "movies" / "moving_peoples.mp4"

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


def main() -> None:
    tracker_type = get_tracker_from_input()
    multi_tracker = cv2.legacy.MultiTracker.create()
    rects: list[list[Sequence[int]]] = []

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    is_tracking = False
    colors: list[RGBColor] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if is_tracking:
            track, bbox = multi_tracker.update(frame)
            if not track:
                is_tracking = False
                print("Tracking failed. Press 's' to select ROIs again.")
            else:
                for i, (rect, color) in enumerate(zip(bbox, colors, strict=True)):
                    x, y, w, h = map(int, rect)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color.get_tuple(), thickness=2)

                    rects[i].append(list(map(int, rect)))
                    for k, prev_rect in enumerate(rects[i][:-1]):
                        next_rect = rects[i][k + 1]

                        cv2.line(
                            frame,
                            (prev_rect[0] + prev_rect[2] // 2, prev_rect[1] + prev_rect[3] // 2),
                            (next_rect[0] + next_rect[2] // 2, next_rect[1] + next_rect[3] // 2),
                            color=color.get_tuple(),
                            thickness=2,
                        )
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(int(1000 / frame_rate))
        if key == ESC_KEY:  # ESC
            break
        if key == STOP_KEY:  # stopping
            added_trackers = 0
            for roi in select_rois(frame):
                tracker = tracker_type.get_tracker()
                if not multi_tracker.add(tracker, frame, roi):
                    continue
                rects.append([list(map(int, roi))])
                colors.append(RGBColor(int(rnd.integers(0, 256)), int(rnd.integers(0, 256)), int(rnd.integers(0, 256))))
                added_trackers += 1
            if added_trackers > 0:
                is_tracking = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
