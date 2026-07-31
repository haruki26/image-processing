from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from shared.pipeline import ProcessedImageInfo

from assignment.utils.pipe import pipe

if TYPE_CHECKING:
    from shared import ImageArray


@dataclass
class TargetArea:
    x: int
    y: int
    w: int
    h: int


SEED_HSV_LOWER = np.array([16, 75, 185], dtype=np.uint8)
SEED_HSV_UPPER = np.array([19, 175, 255], dtype=np.uint8)

TARGET_AREA: Final = TargetArea(x=832, y=335, w=451, h=383)


@pipe.process("seeds.jpg", is_save_image=False, is_show_image=True)
def main(img: ImageArray) -> ProcessedImageInfo:
    target_img = img[TARGET_AREA.y : TARGET_AREA.y + TARGET_AREA.h, TARGET_AREA.x : TARGET_AREA.x + TARGET_AREA.w]

    hsv_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV)
    bin_img = cv2.inRange(hsv_img, SEED_HSV_LOWER, SEED_HSV_UPPER)

    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(bin_img, kernel, iterations=3)
    dilated_img = cv2.dilate(eroded_img, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = len(contours)
    print(f"Number of seeds: {count}")

    return ProcessedImageInfo(image_name="dilated.jpg", image_array=dilated_img)


if __name__ == "__main__":
    main()
