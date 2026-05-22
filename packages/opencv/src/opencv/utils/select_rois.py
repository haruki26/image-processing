from typing import TYPE_CHECKING, Final

import cv2

if TYPE_CHECKING:
    from shared import ImageArray

type ROI = tuple[int, int, int, int]

SELECT_WINDOW_NAME: Final = "Select ROIS"


def select_rois(img: ImageArray, *, window_name: str = SELECT_WINDOW_NAME) -> list[ROI]:
    rois: list[ROI] = []

    cv2.imshow(window_name, img)
    while True:
        for roi in rois:
            cv2.rectangle(img, roi, (0, 255, 255), 2)

        (x, y, w, h) = cv2.selectROI(window_name, img)

        if all(v == 0 for v in (x, y, w, h)):
            break

        rois.append((x, y, w, h))

    return rois
