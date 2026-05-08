from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from shared.color import RGBColor

from opencv.utils.pipe import pipe

if TYPE_CHECKING:
    from shared import ImageArray
    from shared.pipeline import ProcessedImageInfo

type Bound = tuple[int, int, int]


BLUE_MASK_INFO_UPPER: Final[Bound] = (110, 255, 255)
BLUE_MASK_INFO_LOWER: Final[Bound] = (100, 32, 0)

KSIZE: Final[tuple[int, int]] = (5, 5)
LINE_COLOR: Final[RGBColor] = RGBColor(255, 255, 0)


@pipe.process("ball.jpg")
def main(img: ImageArray) -> ProcessedImageInfo:
    blurred_img = cv2.blur(img, KSIZE)
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    upperb = np.array(BLUE_MASK_INFO_UPPER, dtype=np.uint8)
    lowerb = np.array(BLUE_MASK_INFO_LOWER, dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lowerb, upperb)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, LINE_COLOR.get_tuple(), thickness=2)

    return pipe.create_processed_image_info("ball_blue_contours.jpg", img)


if __name__ == "__main__":
    main()
