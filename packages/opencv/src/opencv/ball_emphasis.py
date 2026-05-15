from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final, TypedDict

import cv2
import numpy as np

from shared.color import HSVColor

from opencv.utils.image import image_manager

if TYPE_CHECKING:
    from collections.abc import Sequence

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

EXIT_KEY_CODE: Final[int] = 27  # ESC key


class MouseCallbackParams(TypedDict):
    img: ImageArray
    blurred_img: ImageArray
    hsv_img: ImageArray
    gray_img: ImageArray
    display_img: ImageArray


def check_mask_info(hsv_color: HSVColor) -> BallMaskInfo | None:
    for mask_info in MaskInfos:
        lower = mask_info.value.lower_bound
        upper = mask_info.value.upper_bound
        if (
            lower[0] <= hsv_color.hue <= upper[0]
            and lower[1] <= hsv_color.saturation <= upper[1]
            and lower[2] <= hsv_color.value <= upper[2]
        ):
            return mask_info.value
    return None


def get_contours_in_mask(
    contours: Sequence[np.ndarray],
    x: int,
    y: int,
) -> np.ndarray | None:
    for contour in contours:
        retval = cv2.pointPolygonTest(contour, (x, y), measureDist=False)
        if retval < 0:
            continue
        return contour
    return None


def handler(
    event: cv2.MouseEventTypes,
    x: int,
    y: int,
    _: cv2.MouseEventFlags,
    params: MouseCallbackParams,
) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_color = HSVColor(
            hue=int(params["hsv_img"][y, x][0]),
            saturation=int(params["hsv_img"][y, x][1]),
            value=int(params["hsv_img"][y, x][2]),
        )

        if (mask_info := check_mask_info(hsv_color)) is None:
            return
        mask = cv2.inRange(
            params["hsv_img"],
            np.array(mask_info.lower_bound, dtype=np.uint8),
            np.array(mask_info.upper_bound, dtype=np.uint8),
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # ty:ignore[invalid-assignment]

        if (contour := get_contours_in_mask(contours, x, y)) is None:
            return

        selection_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(selection_mask, [contour], contourIdx=-1, color=255, thickness=cv2.FILLED)

        gray_bgr_img = cv2.cvtColor(params["gray_img"], cv2.COLOR_GRAY2BGR)
        params["display_img"][:] = gray_bgr_img
        params["display_img"][selection_mask > 0] = params["img"][selection_mask > 0]


def main() -> None:
    img = image_manager.read_image("ball.jpg")
    blurred_img = cv2.blur(img, KSIZE)
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    display_img = img.copy()

    cv2.namedWindow("Ball Emphasis")
    cv2.setMouseCallback(
        "Ball Emphasis",
        handler,  # ty:ignore[invalid-argument-type]
        {
            "img": img,
            "blurred_img": blurred_img,
            "hsv_img": hsv_img,
            "gray_img": gray_img,
            "display_img": display_img,
        },
    )

    while True:
        cv2.imshow("Ball Emphasis", display_img)
        if cv2.waitKey(1) & 0xFF == EXIT_KEY_CODE:  # ESC key
            break
    image_manager.save_image(display_img, "ball_emphasis_result.jpg")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
