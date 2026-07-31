from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from shared import ImageArray
from shared.pipeline import ProcessedImageInfo

from assignment.utils.pipe import pipe

if TYPE_CHECKING:
    from shared import ImageArray

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
    PINK = create_mask_info(h_lower=160, h_upper=180)
    GREEN = create_mask_info(h_lower=86, h_upper=100)
    BLUE = create_mask_info(h_lower=100, h_upper=110)


KSIZE: Final[tuple[int, int]] = (5, 5)


@pipe.process("ball.jpg")
def main(img: ImageArray) -> ProcessedImageInfo:
    blurred = cv2.blur(img.copy(), KSIZE)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)

    for mask_info in MaskInfos:
        color_mask = cv2.inRange(
            hsv_img,
            np.array(mask_info.value.lower_bound, dtype=np.uint8),
            np.array(mask_info.value.upper_bound, dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask, color_mask)

    result = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    result[mask > 0] = img[mask > 0]

    return ProcessedImageInfo(
        image_name="emphasized_balls.jpg",
        image_array=result,
    )


if __name__ == "__main__":
    main()
