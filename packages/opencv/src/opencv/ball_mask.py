from dataclasses import dataclass
from enum import Enum
from typing import Final

import cv2
import numpy as np

from opencv.utils.image import image_manager


class BallMaskType(Enum):
    PINK = "pink"
    GREEN = "green"
    BLUE = "blue"


@dataclass
class BallMaskInfo:
    lower_bound: Final[tuple[int, int, int]]
    upper_bound: Final[tuple[int, int, int]]


KSIZE: Final[tuple[int, int]] = (5, 5)


def generate_mask_range(hue: int, saturation: int, value: int) -> BallMaskInfo:
    lower_bound = (hue - 10, saturation - 20, 0)
    upper_bound = (hue + 10, saturation + 20, value + 20)
    return BallMaskInfo(lower_bound=lower_bound, upper_bound=upper_bound)


PINK_MASK_INFO: Final[BallMaskInfo] = generate_mask_range(hue=167, saturation=61, value=225)
GREEN_MASK_INFO: Final[BallMaskInfo] = generate_mask_range(hue=96, saturation=87, value=228)
BLUE_MASK_INFO: Final[BallMaskInfo] = generate_mask_range(hue=102, saturation=125, value=234)


def get_target_color() -> BallMaskType:
    while True:
        user_input = input("Enter the target color (pink, green, blue): ").strip().lower()
        if user_input in BallMaskType._value2member_map_:
            return BallMaskType(user_input)
        print("Invalid input. Please enter 'pink', 'green', or 'blue'.")  # noqa: T201


def get_mask_info(color: BallMaskType) -> BallMaskInfo:
    if color == BallMaskType.PINK:
        return PINK_MASK_INFO
    if color == BallMaskType.GREEN:
        return GREEN_MASK_INFO
    return BLUE_MASK_INFO


def main() -> None:
    color_type = get_target_color()

    img = image_manager.read_image("ball.jpg")
    blurred_img = cv2.blur(img, KSIZE)

    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    mask_info = get_mask_info(color_type)
    upperb = np.array(mask_info.upper_bound, dtype=np.uint8)
    lowerb = np.array(mask_info.lower_bound, dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lowerb, upperb)

    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
