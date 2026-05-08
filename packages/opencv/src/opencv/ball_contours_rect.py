from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from shared.color import RGBColor

from opencv.utils.pipe import pipe

if TYPE_CHECKING:
    from shared import ImageArray
    from shared.pipeline import ProcessedImageInfo

type Bound = tuple[int, int, int]


@dataclass
class BallMaskInfo:
    lower_bound: Final[Bound]
    upper_bound: Final[Bound]


BASE_SATURATION_LOWER: Final[int] = 32
BASE_SATURATION_UPPER: Final[int] = 255
BASE_VALUE_LOWER: Final[int] = 0
BASE_VALUE_UPPER: Final[int] = 255


def create_mask_info(h_lower: int, h_upper: int) -> BallMaskInfo:
    lower_bound = (h_lower, BASE_SATURATION_LOWER, BASE_VALUE_LOWER)
    upper_bound = (h_upper, BASE_SATURATION_UPPER, BASE_VALUE_UPPER)
    return BallMaskInfo(lower_bound=lower_bound, upper_bound=upper_bound)


class MaskInfos(Enum):
    PINK = create_mask_info(h_lower=157, h_upper=167)
    GREEN = create_mask_info(h_lower=86, h_upper=95)
    BLUE = create_mask_info(h_lower=100, h_upper=110)


KSIZE: Final[tuple[int, int]] = (5, 5)
LINE_COLOR: Final[RGBColor] = RGBColor(255, 255, 0)

COLORS: Final[set[str]] = {"pink", "green", "blue"}

BALL_ASPECT_RATIO_THRESHOLD: Final[float] = 1.5
BALL_SIZE_THRESHOLD: Final[int] = 10 * 1000


@pipe.process("ball.jpg")
def main(img: ImageArray) -> ProcessedImageInfo:
    color = input("Enter the target color (pink, green, blue): ").strip().lower()
    if color not in COLORS:
        msg = f"Invalid color: {color}. Must be one of {COLORS}."
        raise ValueError(msg)

    mask_info = MaskInfos[color.upper()].value

    blurred_img = cv2.blur(img, KSIZE)
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    upperb = np.array(mask_info.upper_bound, dtype=np.uint8)
    lowerb = np.array(mask_info.lower_bound, dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lowerb, upperb)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > BALL_ASPECT_RATIO_THRESHOLD:
            continue

        if w * h < BALL_SIZE_THRESHOLD:
            continue

        cv2.rectangle(img, (x, y), (x + w, y + h), LINE_COLOR.get_tuple(), thickness=2)

    return pipe.create_processed_image_info(f"ball_{color}_contours_rect.jpg", img)


if __name__ == "__main__":
    main()
